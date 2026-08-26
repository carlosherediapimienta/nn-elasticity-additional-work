"""Multiproduct MLP experiment: nested Optuna, frozen SKU universe, raw bootstrap.

The mask is not a network input. Early stopping uses the last 20% of outer
train, never outer val. Search MAE is the mean of inner expanding folds.
"""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np
import pandas as pd
from icdn.data.splits import TemporalSplitter

from src.benchmarks.constants import (
    HIDDEN_CHOICES,
    HOLDOUT_TRAIN_FRAC,
    INNER_TRAIN_FRAC,
    MIN_INNER_FRAC,
    N_BOOT_MLP,
    N_FOLDS,
    N_INNER_FOLDS,
    N_TRIALS_MLP,
    PERIOD_COL,
    PRUNER_STARTUP_MLP,
    SEED,
)
from src.benchmarks.demand_mlp import DemandMLPPipeline
from src.benchmarks.features import ICDNFeaturePipeline
from src.benchmarks.predict import (
    attach_pred,
    compute_row,
    n_torch_params,
    summary_series,
    tag_series,
    val_cells,
)
from src.benchmarks.protocol import (
    block_sampler,
    expanding_folds,
    holdout_split,
    load_panel,
    model_datasets,
    print_fold_banner,
    print_holdout_banner,
    project_root,
    run_all_datasets,
    save_bootstrap_report,
    save_kfold_tables,
    save_table,
    summarize_kind,
)
from src.benchmarks.search import dump_best, make_study, report_and_maybe_prune
from src.benchmarks.universe import UniverseError, assert_layout, freeze_products, require_products


class MLPExperiment:
    """Nested-CV MLP benchmark. Constants: N_TRIALS_MLP, N_BOOT_MLP."""
    def __init__(self):
        self.model = "mlp"
        self.n_folds = N_FOLDS
        self.n_boot = N_BOOT_MLP
        self.n_trials = N_TRIALS_MLP
        self.root = project_root()
        self.datasets = model_datasets(self.root, self.model)

    def split_inner(self, train_raw):
        return TemporalSplitter(period_col=PERIOD_COL).single_split(
            train_raw, train_frac=INNER_TRAIN_FRAC
        )

    def featurize_inner(self, spec, inner_raw, es_raw, val_raw):
        feats = ICDNFeaturePipeline(schema=spec["schema"])
        inner = feats.fit(inner_raw).transform(inner_raw)
        es = feats.transform_val(es_raw)
        val = feats.transform_val(val_raw)
        return inner, es, val, feats.shared_cols, feats.product_cols

    def suggest(self, trial):
        hidden_key = trial.suggest_categorical("hidden", list(HIDDEN_CHOICES))
        return dict(
            hidden=HIDDEN_CHOICES[hidden_key],
            dropout=trial.suggest_float("dropout", 0.05, 0.40),
            lr=trial.suggest_float("lr", 5e-4, 3e-3, log=True),
            act="gelu",
            weight_decay=1e-5,
            d_store=16,
            huber_delta=1.0,
            n_epochs=80,
            es_patience=15,
            seed=SEED,
        )

    def mae_one(self, inner, es, val, shared_cols, product_cols, products, params):
        mlp = DemandMLPPipeline(shared_cols, product_cols, products=products, **params)
        with contextlib.redirect_stdout(io.StringIO()):
            mlp.fit(inner, es)
            metrics, _, _ = mlp.evaluate(val)
        assert_layout(mlp.products, products, "mlp")
        return float(metrics["mae_val"])

    def search(self, train_raw, spec, out_dir, products):
        """Inner expanding folds of outer train only. Missing frozen SKUs skip that inner fold."""
        splitter = TemporalSplitter(period_col=PERIOD_COL)
        inner_folds = splitter.expanding_splits(
            train_raw, n_folds=N_INNER_FOLDS, min_train_frac=MIN_INNER_FRAC
        )
        prepared = []
        for i, (tr, va) in enumerate(inner_folds):
            try:
                tr = require_products(tr, products, f"inner{i} train")
                va = require_products(va, products, f"inner{i} val")
                fit_raw, es_raw = self.split_inner(tr)
                fit_raw = require_products(fit_raw, products, f"inner{i} fit")
                es_raw = require_products(es_raw, products, f"inner{i} es")
            except UniverseError as e:
                print("  SKIP inner fold", i, e)
                continue
            prepared.append(self.featurize_inner(spec, fit_raw, es_raw, va))
        if not prepared:
            raise UniverseError("no valid inner folds for frozen product universe")
        study = make_study(
            f"mlp_inner_{out_dir.name}", out_dir, seed=SEED, n_startup_trials=PRUNER_STARTUP_MLP,
        )

        def objective(trial):
            params = self.suggest(trial)
            maes = []
            for k, (inner, es, val, shared_cols, product_cols) in enumerate(prepared):
                mae = self.mae_one(inner, es, val, shared_cols, product_cols, products, params)
                maes.append(mae)
                report_and_maybe_prune(trial, maes, k)
            return float(np.mean(maes))

        study.optimize(objective, n_trials=self.n_trials, gc_after_trial=True)
        return dump_best(study, out_dir)

    def fit(self, inner, es, val, shared_cols, product_cols, products, hp):
        """Final fit with early stopping on `es`. Elasticities averaged over val weeks."""
        mlp = DemandMLPPipeline(
            shared_cols,
            product_cols,
            products=products,
            hidden=hp["hidden"],
            dropout=hp["dropout"],
            lr=hp["lr"],
            act="gelu",
            weight_decay=1e-5,
            d_store=16,
            huber_delta=1.0,
            seed=SEED,
        )
        mlp.fit(inner, es)
        assert_layout(mlp.products, products, "mlp")
        metrics, elast, cells = mlp.evaluate(val)
        summary = elast.groupby(
            ["store_code", "product_i", "product_j", "kind"], as_index=False
        ).agg(
            elasticity=("elasticity", "mean"),
            n_val=("week_id", "size"),
        )
        n_parameters = n_torch_params(mlp.model)
        used_gpu = str(mlp.device).startswith("cuda")
        return metrics, elast, summary, cells, n_parameters, used_gpu

    def summarize(self, metrics, summary, **extra):
        return summarize_kind(metrics, summary, self.model, n_cells_required=False, **extra)

    def print_fit(self, metrics, elast, summary) -> None:
        own = elast[elast.kind == "own"]
        cross = elast[elast.kind == "cross"]
        print(
            "  mae/rmse/r2", metrics["mae_val"], metrics["rmse_val"], metrics["r2_val"],
            "n_cells", metrics.get("n_cells"),
        )
        if len(own):
            print("  own  mean/min/max", own.elasticity.mean(), own.elasticity.min(), own.elasticity.max())
        if len(cross):
            print("  cross mean/min/max", cross.elasticity.mean(), cross.elasticity.min(), cross.elasticity.max())
        own_s = summary[summary.kind == "own"]
        cross_s = summary[summary.kind == "cross"]
        if len(own_s):
            print(
                "  own  (store,i) mean/min/max",
                own_s.elasticity.mean(), own_s.elasticity.min(), own_s.elasticity.max(),
            )
        if len(cross_s):
            print(
                "  cross (store,i,j) mean/min/max",
                cross_s.elasticity.mean(), cross_s.elasticity.min(), cross_s.elasticity.max(),
            )

    def run_kfold(self, name, spec):
        """Outer expanding CV. Optuna + early-stop split use only the outer train."""
        panel = load_panel(spec)
        products = freeze_products(panel)
        print("  frozen products", len(products), products)
        folds = expanding_folds(panel, n_folds=self.n_folds)
        rows, series = [], []
        for k, (train_raw, val_raw) in enumerate(folds, 1):
            print_fold_banner(name, k, self.n_folds, train_raw, val_raw)
            try:
                train_raw = require_products(train_raw, products, "fold train")
                val_raw = require_products(val_raw, products, "fold val")
                t0 = time.perf_counter()
                hp, study = self.search(train_raw, spec, spec["out"] / f"fold{k}", products)
                fit_raw, es_raw = self.split_inner(train_raw)
                fit_raw = require_products(fit_raw, products, "fold fit")
                es_raw = require_products(es_raw, products, "fold es")
            except UniverseError as e:
                print("  SKIP fold", k, e)
                continue
            inner, es, val, shared_cols, product_cols = self.featurize_inner(
                spec, fit_raw, es_raw, val_raw
            )
            print("  shared:", len(shared_cols), "product:", len(product_cols))
            metrics, elast, summary, cells, n_parameters, used_gpu = self.fit(
                inner, es, val, shared_cols, product_cols, products, hp
            )
            compute = compute_row(n_parameters, time.perf_counter() - t0, used_gpu=used_gpu, study=study)
            self.print_fit(metrics, elast, summary)
            save_table(attach_pred(val_cells(val, name, k), cells), spec["out"], f"fold{k}_pred_cells.csv")
            rows.append(self.summarize(metrics, summary, dataset=name, fold=k, **compute))
            series.append(tag_series(summary_series(summary), name, self.model, outer_fold=k))
        return save_kfold_tables(spec["out"], rows, series, allow_empty=True)

    def run_bootstrap(self, name, spec):
        """Search once on holdout train; each replicate resamples raw train and refits."""
        out_dir = spec["out"]
        panel = load_panel(spec)
        products = freeze_products(panel)
        print("  frozen products", len(products), products)
        train_raw, val_raw = holdout_split(panel, train_frac=HOLDOUT_TRAIN_FRAC)
        try:
            train_raw = require_products(train_raw, products, "holdout train")
            val_raw = require_products(val_raw, products, "holdout val")
            t0 = time.perf_counter()
            hp, study = self.search(train_raw, spec, out_dir / "holdout", products)
            fit_raw, es_raw = self.split_inner(train_raw)
            fit_raw = require_products(fit_raw, products, "holdout fit")
            es_raw = require_products(es_raw, products, "holdout es")
        except UniverseError as e:
            print("  SKIP holdout", e)
            return pd.DataFrame()
        inner, es, val, shared_cols, product_cols = self.featurize_inner(
            spec, fit_raw, es_raw, val_raw
        )

        print_holdout_banner(name, train_raw, val_raw)
        print("  shared:", len(shared_cols), shared_cols)
        print("  product:", len(product_cols), product_cols)
        metrics, elast, summary, cells, n_parameters, used_gpu = self.fit(
            inner, es, val, shared_cols, product_cols, products, hp
        )
        compute = compute_row(n_parameters, time.perf_counter() - t0, used_gpu=used_gpu, study=study)
        self.print_fit(metrics, elast, summary)
        holdout = tag_series(summary_series(summary), name, self.model)
        save_table(holdout, out_dir, "holdout_elasticities.csv")
        save_table(attach_pred(val_cells(val, name, "holdout"), cells), out_dir, "holdout_pred_cells.csv")
        print("  compute", compute)

        periods = sorted(train_raw["week_id"].unique())
        sampler = block_sampler()
        rows, replicates = [], []
        for b in range(self.n_boot):
            print(f"  boot {b + 1}/{self.n_boot}")
            train_b_raw = sampler.sample(train_raw, periods)
            try:
                train_b_raw = require_products(train_b_raw, products, f"boot{b} train")
                fit_raw_b, es_raw_b = self.split_inner(train_b_raw)
                fit_raw_b = require_products(fit_raw_b, products, f"boot{b} fit")
                es_raw_b = require_products(es_raw_b, products, f"boot{b} es")
            except UniverseError as e:
                print("  SKIP boot", b + 1, e)
                continue
            inner_b, es_b, val_b, shared_b, product_b = self.featurize_inner(
                spec, fit_raw_b, es_raw_b, val_raw
            )
            t0 = time.perf_counter()
            metrics_b, _, summary_b, _, n_parameters_b, used_gpu_b = self.fit(
                inner_b, es_b, val_b, shared_b, product_b, products, hp
            )
            compute_b = compute_row(n_parameters_b, time.perf_counter() - t0, used_gpu=used_gpu_b)
            rows.append(self.summarize(metrics_b, summary_b, dataset=name, boot=b, **compute_b))
            replicates.append(tag_series(summary_series(summary_b), name, self.model, bootstrap_id=b))
            save_table(pd.DataFrame(rows), out_dir, "bootstrap.csv")
            save_table(pd.concat(replicates, ignore_index=True), out_dir, "bootstrap_replicates.csv")
        if not replicates:
            return pd.DataFrame()
        return save_bootstrap_report(out_dir, rows, replicates, holdout)

    def run_all(self):
        return run_all_datasets(self.datasets, self.run_kfold, self.run_bootstrap)
