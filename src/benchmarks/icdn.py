"""ICDN experiment: nested Optuna, frozen SKU universe, causal prices, non-overlapping block bootstrap.

`patch_panel_builder()` replaces ICDN's price fill (which could bfill) with
CausalPriceFill. Training fill is fit on the internal train slice; inference
fill and `_train_tail` use the full outer train (including early-stop).
`patch_icdn_smooth` resets warmup smoothing on `bootstrap_block_id`.
`patch_icdn_universe` forces the same product order in every split.
Elasticities come from `model.elasticities(..., aggregate=True)`.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from icdn import ICDNConfig, ICDNModel
from icdn.data.splits import TemporalSplitter

from src.benchmarks.constants import (
    BLOCK_SIZE,
    HIDDEN_CHOICES,
    MIN_INNER_FRAC,
    N_BOOT_ICDN,
    N_FOLDS,
    N_INNER_FOLDS,
    N_TRIALS_ICDN,
    PERIOD_COL,
    PRUNER_STARTUP_ICDN,
    SEED,
    MLP_MAX_EPOCHS,
)
from src.benchmarks.bootstrap import BLOCK_ID_COL, BootstrapError, BootstrapPlan, save_bootstrap_manifest
from src.benchmarks.features import frozen_calendar, patch_feature_builder_two_clocks
from src.benchmarks.predict import (
    compute_row,
    n_torch_params,
    native_metrics_from_cells,
    summary_series,
    tag_series,
)
from src.benchmarks.prices import CausalPriceFill, patch_icdn_smooth, patch_panel_builder, set_active
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
from src.benchmarks.reporting import (
    append_run_manifest,
    append_spline_row,
    record_failure,
    run_manifest_row,
    spline_support_row,
)
from src.benchmarks.search import dump_best, make_study, report_and_maybe_prune
from src.benchmarks.universe import (
    UniverseError,
    allow_missing_validation_products,
    assert_layout,
    freeze_products,
    patch_icdn_universe,
    require_training_products,
)

# Must run before any ICDNModel.fit so PanelBuilder never backfills prices,
# FeatureBuilder keeps calendar on source_week_id during bootstrap, and
# warmup smoothing does not cross bootstrap_block_id.
patch_panel_builder()
patch_feature_builder_two_clocks()
patch_icdn_smooth()

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
    "dominick": {
        "own_elasticity_bounds": (-5.0, 0.0),
        "cross_elasticity_bounds": (-1.0, 1.0),
        "beta_prior": -2.0,
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
            warmup_epochs=MLP_MAX_EPOCHS,
            epochs=MLP_MAX_EPOCHS,
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

    def fit_slice(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Weeks ICDN actually fits on. The tail is internal early stopping, not a universe gate."""
        fit_raw, _ = TemporalSplitter(period_col=PERIOD_COL).single_split(
            panel, train_frac=self.fit_frac
        )
        return fit_raw

    @property
    def fit_frac(self) -> float:
        """Must match ICDNModel.fit's internal split (1 - validation_fraction)."""
        return 1.0 - ICDNConfig().validation_fraction

    def set_inference_tail(self, model: ICDNModel, panel: pd.DataFrame) -> None:
        """Use the tail of the full train window (including early-stop) at inference."""
        cfg = model.config
        n_tail = max(list(cfg.lags) + list(cfg.rolling_windows) + [cfg.smoothing_window])
        store, product, period = cfg.schema.store, cfg.schema.product, cfg.schema.period
        keys = [store, product]
        if BLOCK_ID_COL in panel.columns:
            keys = [store, product, BLOCK_ID_COL]
        model._train_tail = (
            panel.sort_values(keys + [period])
            .groupby(keys, group_keys=False)
            .tail(n_tail)
        )

    def fit_model(self, train_raw, spec, searched, products, verbose=False) -> ICDNModel:
        """Fit on the internal split; inference tail and price fill use full `train_raw`.

        On success the inference fill stays installed for `evaluate` / `elasticities`.
        On any exception the global fill is cleared so a later fold cannot inherit it.
        """
        fit_raw = self.fit_slice(train_raw)
        set_active(CausalPriceFill().fit(fit_raw))
        ok = False
        try:
            model = ICDNModel(self.make_config(spec, len(products), searched, verbose=verbose))
            model.fit(train_raw)
            assert_layout(model.products, products, "icdn")
            self.set_inference_tail(model, train_raw)
            set_active(CausalPriceFill().fit(train_raw))
            ok = True
            return model
        finally:
            if not ok:
                set_active(None)

    def mae_one(self, train_raw, val_raw, spec, searched, products):
        require_training_products(self.fit_slice(train_raw), products, "icdn inner train")
        val_raw = allow_missing_validation_products(val_raw, products)
        try:
            model = self.fit_model(train_raw, spec, searched, products, verbose=False)
            return float(model.evaluate(val_raw)["mae"])
        finally:
            set_active(None)

    def search(self, train_raw, spec, out_dir, products):
        """Inner expanding folds of outer train.

        Training fill is fit on the internal train slice. After the checkpoint,
        inference tail and fill use the full inner train (including early-stop).
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
            require_training_products(self.fit_slice(tr), products, f"inner{i} train")
            va = allow_missing_validation_products(va, products)
            prepared.append((tr, va))
        if len(prepared) != N_INNER_FOLDS:
            raise UniverseError(
                f"inner folds {len(prepared)}/{N_INNER_FOLDS} for frozen product universe"
            )
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

        try:
            study.optimize(objective, n_trials=self.n_trials, gc_after_trial=True)
            return dump_best(study, out_dir)
        finally:
            set_active(None)

    def fit(self, train_raw, val_raw, spec, hp, products, verbose=True):
        """Fit ICDN on train with causal prices; score val cells and week-level elasticities."""
        require_training_products(self.fit_slice(train_raw), products, "icdn train")
        val_raw = allow_missing_validation_products(val_raw, products)
        try:
            model = self.fit_model(train_raw, spec, hp, products, verbose=verbose)
            cells = self.cells(model, val_raw)
            long = model.elasticities(val_raw, aggregate=False).rename(columns={
                "product_code": "product_i",
                "competitor": "product_j",
            })
            long["observed_i"] = True
            long["observed_j"] = True
            # parquet store_code is categorical; pandas 2.3 as_index=False needs observed=True
            elast = (
                long.groupby(
                    ["store_code", "product_i", "product_j", "kind"],
                    observed=True,
                    as_index=False,
                )
                .agg(elasticity=("elasticity", "mean"), n_val=("elasticity", "size"))
            )
            n_parameters, used_gpu = self.compute_stats(model)
            spline = spline_support_row(model, val_raw)
            return elast, long, cells, n_parameters, used_gpu, spline
        finally:
            set_active(None)

    def summarize(self, metrics, elast, **extra):
        return summarize_kind(metrics, elast, self.model, n_cells_required=True, **extra)

    def print_fit(self, metrics, elast) -> None:
        own = elast[elast.kind == "own"]
        cross = elast[elast.kind == "cross"]
        print(
            "  mae/rmse/r2", metrics["mae_val"], metrics["rmse_val"], metrics["r2_val"],
            "n_cells", metrics["n_cells"],
            "coverage", metrics.get("prediction_coverage"),
            "n_val", metrics.get("n_val_cells"),
        )
        if len(own):
            print("  own  mean/min/max", own.elasticity.mean(), own.elasticity.min(), own.elasticity.max())
        if len(cross):
            print("  cross mean/min/max", cross.elasticity.mean(), cross.elasticity.min(), cross.elasticity.max())

    def run_kfold(self, name, spec, plan: SplitPlan | None = None):
        """Outer expanding CV. Search and fit stay inside outer train; val is evaluation only."""
        panel = load_panel(spec)
        products = freeze_products(panel)
        patch_icdn_universe(products)
        print("  frozen products", len(products), products)
        if plan is None:
            plan = dataset_split_plan(panel, spec, name, products=products)
        folds = plan.materialize_folds(panel)
        n_folds = len(folds)
        require_training_products(
            self.fit_slice(first_inner_train(folds[0][0])),
            products,
            "shortest nested fit",
        )
        rows, series = [], []
        n_folds_attempted = 0
        n_folds_ok = 0
        extras = {k: spec[k] for k in (
            "own_elasticity_bounds", "cross_elasticity_bounds", "beta_prior", "same_category_first",
        ) if k in spec}
        for k, (train_raw, val_raw) in enumerate(folds, 1):
            print_fold_banner(name, k, n_folds, train_raw, val_raw)
            n_folds_attempted += 1
            try:
                require_training_products(self.fit_slice(train_raw), products, "fold train")
                t_search = time.perf_counter()
                hp, study = self.search(train_raw, spec, spec["out"] / f"fold{k}", products)
                tuning_seconds = time.perf_counter() - t_search
                t_fit = time.perf_counter()
                elast, long, cells, n_parameters, used_gpu, spline = self.fit(
                    train_raw, val_raw, spec, hp, products, verbose=True
                )
                fit_seconds = time.perf_counter() - t_fit
                compute = compute_row(
                    n_parameters, tuning_seconds + fit_seconds, used_gpu=used_gpu, study=study,
                    tuning_seconds=tuning_seconds, fit_seconds=fit_seconds,
                )
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
            metrics = save_pred_grid(val_raw, cells, name, k, spec["out"])
            save_elasticities_long(long, name, self.model, k, spec["out"])
            append_spline_row(spec["out"].parent, {
                "dataset": name, "model": self.model, "fold_or_boot_id": k, **spline,
            })
            _, es_raw = TemporalSplitter(period_col=PERIOD_COL).single_split(
                train_raw, train_frac=self.fit_frac
            )
            append_run_manifest(
                spec["out"].parent,
                run_manifest_row(
                    dataset=name, model=self.model, stage="kfold", outer_fold=k,
                    train_raw=train_raw, val_raw=val_raw, products=products,
                    early_stop=es_raw, extras=extras,
                ),
            )
            self.print_fit(metrics, elast)
            print("  compute", compute)
            rows.append(self.summarize(metrics, elast, dataset=name, fold=k, **compute))
            series.append(tag_series(summary_series(elast), name, self.model, outer_fold=k))
            n_folds_ok += 1
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
        patch_icdn_universe(products)
        print("  frozen products", len(products), products)
        if plan is None:
            plan = dataset_split_plan(panel, spec, name, products=products)
        train_raw, val_raw = plan.materialize_holdout(panel)
        if boot_plan is None:
            boot_plan = dataset_bootstrap_plan(train_raw, val_raw, spec, name)
        extras = {k: spec[k] for k in (
            "own_elasticity_bounds", "cross_elasticity_bounds", "beta_prior", "same_category_first",
        ) if k in spec}
        try:
            require_training_products(self.fit_slice(train_raw), products, "holdout train")
            require_training_products(
                self.fit_slice(first_inner_train(train_raw)),
                products,
                "shortest nested holdout fit",
            )
            t_search = time.perf_counter()
            hp, study = self.search(train_raw, spec, out_dir / "holdout", products)
            tuning_seconds = time.perf_counter() - t_search
            t_fit = time.perf_counter()
            with frozen_calendar(boot_plan.calendar):
                elast, long, cells, n_parameters, used_gpu, spline = self.fit(
                    train_raw, val_raw, spec, hp, products, verbose=True
                )
            fit_seconds = time.perf_counter() - t_fit
            compute = compute_row(
                n_parameters, tuning_seconds + fit_seconds, used_gpu=used_gpu, study=study,
                tuning_seconds=tuning_seconds, fit_seconds=fit_seconds,
            )
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
        metrics = save_pred_grid(val_raw, cells, name, "holdout", out_dir)
        append_spline_row(out_dir.parent, {
            "dataset": name, "model": self.model, "fold_or_boot_id": "holdout", **spline,
        })
        _, es_raw = TemporalSplitter(period_col=PERIOD_COL).single_split(
            train_raw, train_frac=self.fit_frac
        )
        append_run_manifest(
            out_dir.parent,
            run_manifest_row(
                dataset=name, model=self.model, stage="holdout", outer_fold="holdout",
                train_raw=train_raw, val_raw=val_raw, products=products,
                early_stop=es_raw, extras=extras,
            ),
        )

        print_holdout_banner(name, train_raw, val_raw)
        self.print_fit(metrics, elast)
        print("  compute", compute)
        holdout = tag_series(summary_series(elast), name, self.model)
        save_table(holdout, out_dir, "holdout_elasticities.csv")

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
            train_b, val_b = draw.train, draw.val
            try:
                require_training_products(self.fit_slice(train_b), products, f"boot{b} train")
                t0 = time.perf_counter()
                with frozen_calendar(boot_plan.calendar):
                    elast_b, _, cells_b, n_parameters_b, used_gpu_b, spline_b = self.fit(
                        train_b, val_b, spec, hp, products, verbose=False
                    )
                elapsed_b = time.perf_counter() - t0
                metrics_b = native_metrics_from_cells(val_b, cells_b, name, b)
                compute_b = compute_row(
                    n_parameters_b, elapsed_b, used_gpu=used_gpu_b,
                    tuning_seconds=0.0, fit_seconds=elapsed_b,
                    n_attempted=len(manifests) + 1, n_successful=len(replicates) + 1,
                )
            except (UniverseError, BootstrapError) as e:
                manifests.append({"bootstrap_id": b, "accepted": False, "skip": str(e), **draw.manifest})
                save_bootstrap_manifest(out_dir, manifests, block_size=BLOCK_SIZE, seed=SEED)
                print("  SKIP boot", b + 1, e)
                record_failure(
                    out_dir.parent,
                    dataset=name, model=self.model, stage="bootstrap", fold_or_boot_id=b,
                    error_type=type(e).__name__, error_message=str(e),
                    n_attempted=len(manifests), n_successful=len(replicates),
                )
                continue
            manifests.append({"bootstrap_id": b, "accepted": True, **draw.manifest})
            append_spline_row(out_dir.parent, {
                "dataset": name, "model": self.model, "fold_or_boot_id": f"boot{b}", **spline_b,
            })
            rows.append(self.summarize(metrics_b, elast_b, dataset=name, boot=b, **compute_b))
            replicates.append(tag_series(summary_series(elast_b), name, self.model, bootstrap_id=b))
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
