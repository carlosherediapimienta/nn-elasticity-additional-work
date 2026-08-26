"""ICDN experiment: nested Optuna, frozen SKU universe, causal prices, raw bootstrap.

`patch_panel_builder()` replaces ICDN's price fill (which could bfill) with
CausalPriceFill. `patch_icdn_universe` forces the same product order in every
split. Elasticities come from `model.elasticities(..., aggregate=True)`.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from icdn import ICDNConfig, ICDNModel
from icdn.data.splits import TemporalSplitter

from src.benchmarks.constants import (
    HIDDEN_CHOICES,
    HOLDOUT_TRAIN_FRAC,
    MIN_INNER_FRAC,
    N_BOOT_ICDN,
    N_FOLDS,
    N_INNER_FOLDS,
    N_TRIALS_ICDN,
    PERIOD_COL,
    PRUNER_STARTUP_ICDN,
    SEED,
)
from src.benchmarks.predict import (
    attach_pred,
    compute_row,
    metrics_from_cells,
    n_torch_params,
    summary_series,
    tag_series,
    val_cells,
)
from src.benchmarks.prices import CausalPriceFill, patch_panel_builder, set_active
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
from src.benchmarks.universe import (
    UniverseError,
    assert_layout,
    freeze_products,
    patch_icdn_universe,
    require_products,
)

# Must run before any ICDNModel.fit so PanelBuilder never backfills prices.
patch_panel_builder()

ICDN_EXTRAS = {
    "walmart": {
        "own_elasticity_bounds": (-3.5, 0.0),
        "cross_elasticity_bounds": (-0.4, 0.8),
        "beta_prior": -2.0,
        "same_category_first": True,
    },
    "one_c": {
        "own_elasticity_bounds": (-3.0, 0.0),
        "cross_elasticity_bounds": (-0.2, 0.5),
        "beta_prior": -1.5,
        "same_category_first": True,
    },
}


class ICDNExperiment:
    """Nested-CV ICDN benchmark. Dataset extras (bounds, k, coverage) stay in ICDN_EXTRAS."""
    def __init__(self):
        self.model = "icdn"
        self.n_folds = N_FOLDS
        self.n_boot = N_BOOT_ICDN
        self.n_trials = N_TRIALS_ICDN
        self.root = project_root()
        self.datasets = model_datasets(self.root, self.model, extras=ICDN_EXTRAS)

    def suggest(self, trial, n_products):
        hidden_key = trial.suggest_categorical("hidden", list(HIDDEN_CHOICES))
        return dict(
            hidden=HIDDEN_CHOICES[hidden_key],
            dropout=trial.suggest_float("dropout", 0.05, 0.40),
            lr=trial.suggest_float("lr", 5e-4, 3e-3, log=True),
            warmup_lr=trial.suggest_float("warmup_lr", 5e-4, 3e-3, log=True),
            lambda_smooth=trial.suggest_float("lambda_smooth", 0.01, 0.08, log=True),
            lambda_elast=trial.suggest_float("lambda_elast", 0.01, 0.08, log=True),
            k_neighbors=min(trial.suggest_int("k_neighbors", 2, 4), max(n_products - 1, 1)),
            n_knots=trial.suggest_int("n_knots", 3, 5),
        )

    def make_config(self, spec, n_products, searched, verbose=False):
        return ICDNConfig(
            schema=spec["schema"],
            n_products=n_products,
            k_neighbors=searched["k_neighbors"],
            same_category_first=spec["same_category_first"],
            own_elasticity_bounds=spec["own_elasticity_bounds"],
            cross_elasticity_bounds=spec["cross_elasticity_bounds"],
            beta_prior=spec["beta_prior"],
            # Frozen universe already chose the SKUs. Coverage must not drop a slot
            # from the Jacobian (same role as the MLP mask: sparse ≠ absent).
            min_coverage=0.0,
            hidden=searched["hidden"],
            dropout=searched["dropout"],
            lr=searched["lr"],
            warmup_lr=searched["warmup_lr"],
            lambda_smooth=searched["lambda_smooth"],
            lambda_elast=searched["lambda_elast"],
            n_knots=int(searched["n_knots"]),
            enforce_negative_beta=True,
            warmup_epochs=25,
            epochs=50,
            early_stopping_patience=12,
            seed=SEED,
            verbose=verbose,
        )

    def cells(self, model, val_raw):
        scored = model.score(val_raw)
        scored = scored[scored["demand"].notna()]
        return scored.rename(columns={
            "log_demand": "y_true",
            "predicted_log_demand": "y_pred",
        })[["store_code", "product_code", "week_id", "y_true", "y_pred"]]

    def compute_stats(self, model):
        net = getattr(model, "model", None) or getattr(model, "_model", None)
        device = getattr(model, "device", None) or getattr(model, "_device", None)
        n_parameters = n_torch_params(net) if net is not None else 0
        used_gpu = str(device).startswith("cuda") if device is not None else False
        return n_parameters, used_gpu

    def mae_one(self, train_raw, val_raw, spec, searched, products):
        train_raw = require_products(train_raw, products, "icdn inner train")
        val_raw = require_products(val_raw, products, "icdn inner val")
        set_active(CausalPriceFill().fit(train_raw))
        model = ICDNModel(self.make_config(spec, len(products), searched, verbose=False))
        model.fit(train_raw)
        assert_layout(model.products, products, "icdn")
        mae = float(model.evaluate(val_raw)["mae"])
        set_active(None)
        return mae

    def search(self, train_raw, spec, out_dir, products):
        """Inner expanding folds of outer train. Causal fill is fit per inner train slice."""
        splitter = TemporalSplitter(period_col=PERIOD_COL)
        inner_folds = splitter.expanding_splits(
            train_raw, n_folds=N_INNER_FOLDS, min_train_frac=MIN_INNER_FRAC
        )
        prepared = []
        for i, (tr, va) in enumerate(inner_folds):
            try:
                tr = require_products(tr, products, f"inner{i} train")
                va = require_products(va, products, f"inner{i} val")
            except UniverseError as e:
                print("  SKIP inner fold", i, e)
                continue
            prepared.append((tr, va))
        if not prepared:
            raise UniverseError("no valid inner folds for frozen product universe")
        study = make_study(
            f"icdn_inner_{out_dir.name}", out_dir, seed=SEED, n_startup_trials=PRUNER_STARTUP_ICDN,
        )

        def objective(trial):
            searched = self.suggest(trial, len(products))
            maes = []
            for k, (tr, va) in enumerate(prepared):
                mae = self.mae_one(tr, va, spec, searched, products)
                maes.append(mae)
                report_and_maybe_prune(trial, maes, k)
            return float(np.mean(maes))

        study.optimize(objective, n_trials=self.n_trials, gc_after_trial=True)
        return dump_best(study, out_dir)

    def fit(self, train_raw, val_raw, spec, hp, products, verbose=True):
        """Fit ICDN on train with causal prices; score val cells and aggregated elasticities."""
        train_raw = require_products(train_raw, products, "icdn train")
        val_raw = require_products(val_raw, products, "icdn val")
        set_active(CausalPriceFill().fit(train_raw))
        model = ICDNModel(self.make_config(spec, len(products), hp, verbose=verbose))
        model.fit(train_raw)
        assert_layout(model.products, products, "icdn")
        cells = self.cells(model, val_raw)
        metrics = metrics_from_cells(cells)
        elast = model.elasticities(val_raw, aggregate=True).rename(columns={
            "product_code": "product_i",
            "competitor": "product_j",
            "n_obs": "n_val",
        })
        n_parameters, used_gpu = self.compute_stats(model)
        set_active(None)
        return metrics, elast, cells, n_parameters, used_gpu

    def summarize(self, metrics, elast, **extra):
        return summarize_kind(metrics, elast, self.model, n_cells_required=True, **extra)

    def print_fit(self, metrics, elast) -> None:
        own = elast[elast.kind == "own"]
        cross = elast[elast.kind == "cross"]
        print(
            "  mae/rmse/r2", metrics["mae_val"], metrics["rmse_val"], metrics["r2_val"],
            "n_cells", metrics["n_cells"],
        )
        if len(own):
            print("  own  mean/min/max", own.elasticity.mean(), own.elasticity.min(), own.elasticity.max())
        if len(cross):
            print("  cross mean/min/max", cross.elasticity.mean(), cross.elasticity.min(), cross.elasticity.max())

    def run_kfold(self, name, spec):
        """Outer expanding CV. Search and fit stay inside outer train; val is evaluation only."""
        panel = load_panel(spec)
        products = freeze_products(panel)
        patch_icdn_universe(products)
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
                metrics, elast, cells, n_parameters, used_gpu = self.fit(
                    train_raw, val_raw, spec, hp, products, verbose=True
                )
                compute = compute_row(n_parameters, time.perf_counter() - t0, used_gpu=used_gpu, study=study)
            except UniverseError as e:
                print("  SKIP fold", k, e)
                continue
            self.print_fit(metrics, elast)
            print("  compute", compute)
            save_table(attach_pred(val_cells(val_raw, name, k), cells), spec["out"], f"fold{k}_pred_cells.csv")
            rows.append(self.summarize(metrics, elast, dataset=name, fold=k, **compute))
            series.append(tag_series(summary_series(elast), name, self.model, outer_fold=k))
        return save_kfold_tables(spec["out"], rows, series, allow_empty=True)

    def run_bootstrap(self, name, spec):
        """Search once on holdout train; each replicate resamples raw train and refits."""
        out_dir = spec["out"]
        panel = load_panel(spec)
        products = freeze_products(panel)
        patch_icdn_universe(products)
        print("  frozen products", len(products), products)
        train_raw, val_raw = holdout_split(panel, train_frac=HOLDOUT_TRAIN_FRAC)
        try:
            train_raw = require_products(train_raw, products, "holdout train")
            val_raw = require_products(val_raw, products, "holdout val")
            t0 = time.perf_counter()
            hp, study = self.search(train_raw, spec, out_dir / "holdout", products)
            metrics, elast, cells, n_parameters, used_gpu = self.fit(
                train_raw, val_raw, spec, hp, products, verbose=True
            )
            compute = compute_row(n_parameters, time.perf_counter() - t0, used_gpu=used_gpu, study=study)
        except UniverseError as e:
            print("  SKIP holdout", e)
            return pd.DataFrame()

        print_holdout_banner(name, train_raw, val_raw)
        self.print_fit(metrics, elast)
        print("  compute", compute)
        holdout = tag_series(summary_series(elast), name, self.model)
        save_table(holdout, out_dir, "holdout_elasticities.csv")
        save_table(attach_pred(val_cells(val_raw, name, "holdout"), cells), out_dir, "holdout_pred_cells.csv")

        periods = sorted(train_raw["week_id"].unique())
        sampler = block_sampler()
        rows, replicates = [], []
        for b in range(self.n_boot):
            print(f"  boot {b + 1}/{self.n_boot}")
            train_b = sampler.sample(train_raw, periods)
            try:
                train_b = require_products(train_b, products, f"boot{b} train")
                t0 = time.perf_counter()
                metrics_b, elast_b, _, n_parameters_b, used_gpu_b = self.fit(
                    train_b, val_raw, spec, hp, products, verbose=False
                )
                compute_b = compute_row(n_parameters_b, time.perf_counter() - t0, used_gpu=used_gpu_b)
            except UniverseError as e:
                print("  SKIP boot", b + 1, e)
                continue
            rows.append(self.summarize(metrics_b, elast_b, dataset=name, boot=b, **compute_b))
            replicates.append(tag_series(summary_series(elast_b), name, self.model, bootstrap_id=b))
            save_table(pd.DataFrame(rows), out_dir, "bootstrap.csv")
            save_table(pd.concat(replicates, ignore_index=True), out_dir, "bootstrap_replicates.csv")
        if not replicates:
            return pd.DataFrame()
        return save_bootstrap_report(out_dir, rows, replicates, holdout)

    def run_all(self):
        return run_all_datasets(self.datasets, self.run_kfold, self.run_bootstrap)
