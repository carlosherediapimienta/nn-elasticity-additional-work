"""Multiproduct MLP experiment: nested Optuna, frozen SKU universe, non-overlapping block bootstrap.

The mask is not a network input. Early stopping uses the last 20% of outer
train, never outer val. Search MAE is the mean of inner expanding folds.
Trials and final fits share MLP_MAX_EPOCHS / MLP_PATIENCE. Outer-val lags
use the full outer train as history, including the early-stop window.
"""

from __future__ import annotations

import contextlib
import io
import time

import numpy as np
import pandas as pd
from icdn.data.splits import TemporalSplitter

from src.benchmarks.constants import (
    BLOCK_SIZE,
    HIDDEN_CHOICES,
    INNER_TRAIN_FRAC,
    MIN_INNER_FRAC,
    MLP_MAX_EPOCHS,
    MLP_PATIENCE,
    N_BOOT_MLP,
    N_FOLDS,
    N_INNER_FOLDS,
    N_TRIALS_MLP,
    PERIOD_COL,
    PRUNER_STARTUP_MLP,
    SEED,
)
from src.benchmarks.demand_mlp import DemandMLPPipeline
from src.benchmarks.bootstrap import BootstrapError, BootstrapPlan, save_bootstrap_manifest
from src.benchmarks.features import ICDNFeaturePipeline, frozen_calendar
from src.benchmarks.predict import (
    compute_row,
    n_torch_params,
    native_metrics_from_cells,
    summary_series,
    tag_series,
)
from src.benchmarks.protocol import (
    SplitPlan,
    dataset_bootstrap_plan,
    dataset_split_plan,
    first_inner_train,
    load_panel,
    model_datasets,
    print_fold_banner,
    print_holdout_banner,
    project_root,
    run_all_datasets,
    save_bootstrap_report,
    save_elasticities_long,
    save_kfold_tables,
    save_pred_grid,
    save_table,
    summarize_kind,
)
from src.benchmarks.reporting import append_run_manifest, record_failure, run_manifest_row
from src.benchmarks.search import dump_best, make_study, report_and_maybe_prune
from src.benchmarks.universe import (
    UniverseError,
    allow_missing_validation_products,
    assert_layout,
    freeze_products,
    require_training_products,
)


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

    def featurize_inner(self, spec, fit_raw, es_raw, val_raw, history_raw):
        """Fit feature stats on `fit_raw`. Val lags use `history_raw` (fit + early-stop)."""
        feats = ICDNFeaturePipeline(schema=spec["schema"])
        inner = feats.fit(fit_raw).transform(fit_raw)
        es = feats.transform(es_raw, history=fit_raw)
        val = feats.transform(val_raw, history=history_raw)
        return inner, es, val, feats.shared_cols, feats.product_cols

    def pipeline_kwargs(self, hp):
        """Searched HPs plus the shared epoch/patience protocol."""
        return dict(
            hidden=hp["hidden"],
            dropout=hp["dropout"],
            lr=hp["lr"],
            act="gelu",
            weight_decay=1e-5,
            d_store=16,
            huber_delta=1.0,
            n_epochs=MLP_MAX_EPOCHS,
            es_patience=MLP_PATIENCE,
            seed=SEED,
        )

    def suggest(self, trial):
        hidden_key = trial.suggest_categorical("hidden", list(HIDDEN_CHOICES))
        return self.pipeline_kwargs({
            "hidden": HIDDEN_CHOICES[hidden_key],
            "dropout": trial.suggest_float("dropout", 0.05, 0.40),
            "lr": trial.suggest_float("lr", 5e-4, 3e-3, log=True),
        })

    def mae_one(self, inner, es, val, shared_cols, product_cols, products, params):
        mlp = DemandMLPPipeline(shared_cols, product_cols, products=products, **params)
        with contextlib.redirect_stdout(io.StringIO()):
            mlp.fit(inner, es)
            metrics, _, _ = mlp.evaluate(val)
        assert_layout(mlp.products, products, "mlp")
        return float(metrics["mae_val"])

    def search(self, train_raw, spec, out_dir, products):
        """Inner expanding folds of outer train only.

        Every inner fold must be usable. A missing SKU in the fit slice aborts
        the outer fold rather than scoring Optuna on a subset of folds.
        Inner val and early-stopping may omit a SKU (mask zero).
        """
        splitter = TemporalSplitter(period_col=PERIOD_COL)
        inner_folds = splitter.expanding_splits(
            train_raw, n_folds=N_INNER_FOLDS, min_train_frac=MIN_INNER_FRAC
        )
        if len(inner_folds) != N_INNER_FOLDS:
            raise UniverseError(
                f"inner folds {len(inner_folds)}/{N_INNER_FOLDS} for frozen product universe"
            )
        prepared = []
        for i, (tr, va) in enumerate(inner_folds):
            fit_raw, es_raw = self.split_inner(tr)
            fit_raw = require_training_products(fit_raw, products, f"inner{i} fit")
            va = allow_missing_validation_products(va, products)
            es_raw = allow_missing_validation_products(es_raw, products)
            prepared.append(self.featurize_inner(spec, fit_raw, es_raw, va, history_raw=tr))
        if len(prepared) != N_INNER_FOLDS:
            raise UniverseError(
                f"inner folds {len(prepared)}/{N_INNER_FOLDS} for frozen product universe"
            )
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
            **self.pipeline_kwargs(hp),
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
            "coverage", metrics.get("prediction_coverage"),
            "n_val", metrics.get("n_val_cells"),
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

    def run_kfold(self, name, spec, plan: SplitPlan | None = None):
        """Outer expanding CV. Optuna + early-stop split use only the outer train."""
        panel = load_panel(spec)
        products = freeze_products(panel)
        print("  frozen products", len(products), products)
        if plan is None:
            plan = dataset_split_plan(panel, spec, name, products=products)
        folds = plan.materialize_folds(panel)
        n_folds = len(folds)
        require_training_products(
            self.split_inner(first_inner_train(folds[0][0]))[0],
            products,
            "shortest nested fit",
        )
        rows, series = [], []
        n_folds_attempted = 0
        n_folds_ok = 0
        for k, (train_raw, val_raw) in enumerate(folds, 1):
            print_fold_banner(name, k, n_folds, train_raw, val_raw)
            n_folds_attempted += 1
            try:
                fit_raw, es_raw = self.split_inner(train_raw)
                fit_raw = require_training_products(fit_raw, products, "fold fit")
                t_search = time.perf_counter()
                hp, study = self.search(train_raw, spec, spec["out"] / f"fold{k}", products)
                tuning_seconds = time.perf_counter() - t_search
            except UniverseError as e:
                print("  SKIP fold", k, e)
                record_failure(
                    spec["out"].parent,
                    dataset=name, model=self.model, stage="kfold", fold_or_boot_id=k,
                    error_type="UniverseError", error_message=str(e),
                    n_attempted=n_folds_attempted, n_successful=n_folds_ok,
                )
                continue
            val_raw = allow_missing_validation_products(val_raw, products)
            es_raw = allow_missing_validation_products(es_raw, products)
            inner, es, val, shared_cols, product_cols = self.featurize_inner(
                spec, fit_raw, es_raw, val_raw, history_raw=train_raw
            )
            print("  shared:", len(shared_cols), "product:", len(product_cols))
            t_fit = time.perf_counter()
            metrics, elast, summary, cells, n_parameters, used_gpu = self.fit(
                inner, es, val, shared_cols, product_cols, products, hp
            )
            fit_seconds = time.perf_counter() - t_fit
            metrics = {**metrics, **save_pred_grid(val, cells, name, k, spec["out"])}
            save_elasticities_long(elast, name, self.model, k, spec["out"])
            compute = compute_row(
                n_parameters, tuning_seconds + fit_seconds, used_gpu=used_gpu, study=study,
                tuning_seconds=tuning_seconds, fit_seconds=fit_seconds,
            )
            self.print_fit(metrics, elast, summary)
            rows.append(self.summarize(metrics, summary, dataset=name, fold=k, **compute))
            series.append(tag_series(summary_series(summary), name, self.model, outer_fold=k))
            n_folds_ok += 1
            append_run_manifest(
                spec["out"].parent,
                run_manifest_row(
                    dataset=name, model=self.model, stage="kfold", outer_fold=k,
                    train_raw=train_raw, val_raw=val_raw, products=products,
                    early_stop=es_raw,
                ),
            )
        return save_kfold_tables(spec["out"], rows, series, allow_empty=True)

    def run_bootstrap(
        self,
        name,
        spec,
        plan: SplitPlan | None = None,
        boot_plan: BootstrapPlan | None = None,
    ):
        """Search once on holdout train; each replicate resamples raw train and refits."""
        out_dir = spec["out"]
        panel = load_panel(spec)
        products = freeze_products(panel)
        print("  frozen products", len(products), products)
        if plan is None:
            plan = dataset_split_plan(panel, spec, name, products=products)
        train_raw, val_raw = plan.materialize_holdout(panel)
        if boot_plan is None:
            boot_plan = dataset_bootstrap_plan(train_raw, val_raw, spec, name)
        try:
            fit_raw, es_raw = self.split_inner(train_raw)
            fit_raw = require_training_products(fit_raw, products, "holdout fit")
            require_training_products(
                self.split_inner(first_inner_train(train_raw))[0],
                products,
                "shortest nested holdout fit",
            )
            t_search = time.perf_counter()
            hp, study = self.search(train_raw, spec, out_dir / "holdout", products)
            tuning_seconds = time.perf_counter() - t_search
        except UniverseError as e:
            print("  SKIP holdout", e)
            record_failure(
                out_dir.parent,
                dataset=name, model=self.model, stage="holdout", fold_or_boot_id="holdout",
                error_type="UniverseError", error_message=str(e),
                n_attempted=1, n_successful=0,
            )
            return pd.DataFrame()
        val_raw = allow_missing_validation_products(val_raw, products)
        es_raw = allow_missing_validation_products(es_raw, products)
        with frozen_calendar(boot_plan.calendar):
            inner, es, val, shared_cols, product_cols = self.featurize_inner(
                spec, fit_raw, es_raw, val_raw, history_raw=train_raw
            )

        print_holdout_banner(name, train_raw, val_raw)
        print("  shared:", len(shared_cols), shared_cols)
        print("  product:", len(product_cols), product_cols)
        t_fit = time.perf_counter()
        metrics, elast, summary, cells, n_parameters, used_gpu = self.fit(
            inner, es, val, shared_cols, product_cols, products, hp
        )
        fit_seconds = time.perf_counter() - t_fit
        metrics = {**metrics, **save_pred_grid(val, cells, name, "holdout", out_dir)}
        compute = compute_row(
            n_parameters, tuning_seconds + fit_seconds, used_gpu=used_gpu, study=study,
            tuning_seconds=tuning_seconds, fit_seconds=fit_seconds,
        )
        self.print_fit(metrics, elast, summary)
        holdout = tag_series(summary_series(summary), name, self.model)
        save_table(holdout, out_dir, "holdout_elasticities.csv")
        append_run_manifest(
            out_dir.parent,
            run_manifest_row(
                dataset=name, model=self.model, stage="holdout", outer_fold="holdout",
                train_raw=train_raw, val_raw=val_raw, products=products,
                early_stop=es_raw,
            ),
        )
        print("  compute", compute)

        n_boot = min(self.n_boot, boot_plan.n_boot)
        rows, replicates, manifests = [], [], []
        for b in range(n_boot):
            print(f"  boot {b + 1}/{n_boot}")
            try:
                draw = boot_plan.draw(train_raw, val_raw, b)
            except BootstrapError as e:
                rec = dict(boot_plan.replicates[b]) if b < len(boot_plan.replicates) else {}
                manifests.append({"bootstrap_id": b, "accepted": False, "skip": str(e), **rec})
                save_bootstrap_manifest(out_dir, manifests, block_size=BLOCK_SIZE, seed=SEED)
                print("  SKIP boot", b + 1, e)
                record_failure(
                    out_dir.parent,
                    dataset=name, model=self.model, stage="bootstrap", fold_or_boot_id=b,
                    error_type="BootstrapError", error_message=str(e),
                    n_attempted=len(manifests), n_successful=len(replicates),
                )
                continue
            train_b_raw, val_b_raw = draw.train, draw.val
            try:
                train_b_raw = require_training_products(train_b_raw, products, f"boot{b} train")
                fit_raw_b, es_raw_b = self.split_inner(train_b_raw)
                fit_raw_b = require_training_products(fit_raw_b, products, f"boot{b} fit")
            except UniverseError as e:
                manifests.append({"bootstrap_id": b, "accepted": False, "skip": str(e), **draw.manifest})
                save_bootstrap_manifest(out_dir, manifests, block_size=BLOCK_SIZE, seed=SEED)
                print("  SKIP boot", b + 1, e)
                record_failure(
                    out_dir.parent,
                    dataset=name, model=self.model, stage="bootstrap", fold_or_boot_id=b,
                    error_type="UniverseError", error_message=str(e),
                    n_attempted=len(manifests), n_successful=len(replicates),
                )
                continue
            manifests.append({"bootstrap_id": b, "accepted": True, **draw.manifest})
            es_raw_b = allow_missing_validation_products(es_raw_b, products)
            val_b_raw = allow_missing_validation_products(val_b_raw, products)
            with frozen_calendar(boot_plan.calendar):
                inner_b, es_b, val_b, shared_b, product_b = self.featurize_inner(
                    spec, fit_raw_b, es_raw_b, val_b_raw, history_raw=train_b_raw
                )
            t0 = time.perf_counter()
            metrics_b, _, summary_b, cells_b, n_parameters_b, used_gpu_b = self.fit(
                inner_b, es_b, val_b, shared_b, product_b, products, hp
            )
            elapsed_b = time.perf_counter() - t0
            metrics_b = {**metrics_b, **native_metrics_from_cells(val_b, cells_b, name, b)}
            compute_b = compute_row(
                n_parameters_b, elapsed_b, used_gpu=used_gpu_b,
                tuning_seconds=0.0, fit_seconds=elapsed_b,
                n_attempted=len(manifests), n_successful=len(replicates) + 1,
            )
            rows.append(self.summarize(metrics_b, summary_b, dataset=name, boot=b, **compute_b))
            replicates.append(tag_series(summary_series(summary_b), name, self.model, bootstrap_id=b))
            save_table(pd.DataFrame(rows), out_dir, "bootstrap.csv")
            save_table(pd.concat(replicates, ignore_index=True), out_dir, "bootstrap_replicates.csv")
            save_bootstrap_manifest(out_dir, manifests, block_size=BLOCK_SIZE, seed=SEED)
        if not replicates:
            return pd.DataFrame()
        save_bootstrap_manifest(out_dir, manifests, block_size=BLOCK_SIZE, seed=SEED)
        for rec in rows:
            rec["n_attempted"] = len(manifests)
            rec["n_successful"] = len(replicates)
        return save_bootstrap_report(out_dir, rows, replicates, holdout)

    def run_all(self):
        return run_all_datasets(self.datasets, self.run_kfold, self.run_bootstrap)
