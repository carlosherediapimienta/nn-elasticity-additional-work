"""OLS / Ridge experiment: pairwise equations, expanding CV, non-overlapping block bootstrap.

Own and cross elasticities are both taken from `run_cross`. Cell MAE is the
mean of pairwise ŷ_ijst over j, merged onto the common (store, product, week)
validation grid. Bootstrap resamples raw train weeks, then rebuilds features.
Ridge selects α once on holdout train (expanding MAE) and reuses those values.
"""

from __future__ import annotations

import time

import pandas as pd

from src.benchmarks.constants import (
    BLOCK_SIZE,
    LINEAR_BOOT_CHECKPOINT,
    N_BOOT_LINEAR,
    N_FOLDS,
    SEED,
    SHORT,
)
from src.benchmarks.pairs import PairDatasetBuilder
from src.benchmarks.pairwise_ols import PairwiseOLS
from src.benchmarks.pairwise_ridge import PairwiseRidge, freeze_alphas
from src.benchmarks.predict import (
    compute_row,
    native_metrics_from_cells,
    own_cross_series,
    pair_elasticities,
    pairwise_to_cells,
    tag_series,
)
from src.benchmarks.protocol import (
    SplitPlan,
    dataset_bootstrap_plan,
    dataset_split_plan,
    featurize,
    load_panel,
    model_datasets,
    print_fold_banner,
    print_holdout_banner,
    project_root,
    run_all_datasets,
    save_bootstrap_report,
    save_kfold_tables,
    save_pred_grid,
    save_pred_ij,
    save_table,
    summarize_pairwise,
)
from src.benchmarks.bootstrap import BootstrapError, BootstrapPlan, save_bootstrap_manifest
from src.benchmarks.features import frozen_calendar


class PairwiseExperiment:
    """One class for OLS and Ridge; only the estimator and VIF printout differ."""

    def __init__(self, model: str, estimator_cls, *, print_vif: bool = False, n_boot: int = N_BOOT_LINEAR):
        self.model = model
        self.estimator_cls = estimator_cls
        self.print_vif = print_vif
        self.n_boot = n_boot
        self.n_folds = N_FOLDS
        self.controls = list(SHORT)
        self.root = project_root()
        self.datasets = model_datasets(self.root, model)

    @classmethod
    def ols(cls) -> "PairwiseExperiment":
        return cls("ols", PairwiseOLS, print_vif=True)

    @classmethod
    def ridge(cls) -> "PairwiseExperiment":
        return cls("ridge", PairwiseRidge, print_vif=False)

    def fit(self, train, val, selected_alphas=None):
        """Pairwise cross equations → own/cross elasticities, cell ŷ, MAE, parameter count."""
        builder = PairDatasetBuilder(self.controls)
        est = self.estimator_cls(self.controls)
        train_pairs, val_pairs = builder.build(train), builder.build(val)
        if selected_alphas is None:
            cross, pred_ij = est.run_cross(train_pairs, val_pairs)
        else:
            cross, pred_ij = est.run_cross(train_pairs, val_pairs, selected_alphas=selected_alphas)
        n_parameters = int(cross["n_params"].sum()) if len(cross) else 0
        own, cross = pair_elasticities(cross)
        cells = pairwise_to_cells(pred_ij)
        return own, cross, pred_ij, cells, n_parameters

    def print_fit(self, own, cross, metrics) -> None:
        print(f"  own n={len(own)}  cross n={len(cross)}")
        print(
            "  mae/rmse/r2", metrics["mae_val"], metrics["rmse_val"], metrics["r2_val"],
            "n_cells", metrics["n_cells"],
            "coverage", metrics.get("prediction_coverage"),
            "n_val", metrics.get("n_val_cells"),
        )
        if len(own):
            print(
                "  own  mean/min/max",
                own.own_elasticity.mean(), own.own_elasticity.min(), own.own_elasticity.max(),
            )
            print("  own  mean n_partners", own.n_partners.mean())
            if "partner_set_id" in own.columns:
                print("  own  n_partner_sets", own.partner_set_id.nunique())
        if len(cross):
            print(
                "  cross mean/min/max",
                cross.cross_elasticity.mean(), cross.cross_elasticity.min(), cross.cross_elasticity.max(),
            )
            if self.print_vif:
                print(
                    "  median VIF log_p_i/log_p_j",
                    cross.vif_log_p_i.median(), cross.vif_log_p_j.median(),
                )

    def _row(self, own, cross, metrics, **extra):
        return summarize_pairwise(own, cross, metrics, self.model, **extra)

    def run_kfold(self, name, spec, plan: SplitPlan | None = None):
        """Expanding outer folds. Features fit on train only. Outer val is evaluation only."""
        panel = load_panel(spec)
        if plan is None:
            plan = dataset_split_plan(panel, spec, name)
        folds = plan.materialize_folds(panel)
        n_folds = len(folds)
        rows, series = [], []
        for k, (train_raw, val_raw) in enumerate(folds, 1):
            print_fold_banner(name, k, n_folds, train_raw, val_raw)
            train, val = featurize(spec, train_raw, val_raw)
            t0 = time.perf_counter()
            own, cross, pred_ij, cells, n_parameters = self.fit(train, val)
            elapsed = time.perf_counter() - t0
            compute = compute_row(n_parameters, elapsed, tuning_seconds=0.0, fit_seconds=elapsed)
            metrics = save_pred_grid(val, cells, name, k, spec["out"])
            self.print_fit(own, cross, metrics)
            save_pred_ij(pred_ij, name, k, spec["out"])
            rows.append(self._row(own, cross, metrics, dataset=name, fold=k, **compute))
            series.append(tag_series(own_cross_series(own, cross), name, self.model, outer_fold=k))
        return save_kfold_tables(spec["out"], rows, series, allow_empty=False)

    def run_bootstrap(
        self,
        name,
        spec,
        plan: SplitPlan | None = None,
        boot_plan: BootstrapPlan | None = None,
    ):
        """Holdout point estimate, then non-overlapping block bootstrap of train (val weeks fixed)."""
        out_dir = spec["out"]
        panel = load_panel(spec)
        if plan is None:
            plan = dataset_split_plan(panel, spec, name)
        train_raw, val_raw = plan.materialize_holdout(panel)
        if boot_plan is None:
            boot_plan = dataset_bootstrap_plan(train_raw, val_raw, spec, name)
        with frozen_calendar(boot_plan.calendar):
            train, val = featurize(spec, train_raw, val_raw)

        print_holdout_banner(name, train_raw, val_raw)
        own, cross, pred_ij, cells, _ = self.fit(train, val)
        metrics = save_pred_grid(val, cells, name, "holdout", out_dir)
        self.print_fit(own, cross, metrics)
        holdout = tag_series(own_cross_series(own, cross), name, self.model)
        save_table(holdout, out_dir, "holdout_elasticities.csv")
        save_pred_ij(pred_ij, name, "holdout", out_dir)
        selected_alphas = freeze_alphas(cross) if self.model == "ridge" else None
        if selected_alphas:
            save_table(
                cross[["store_code", "product_i", "product_j", "alpha_selected"]],
                out_dir,
                "holdout_alphas.csv",
            )

        n_boot = min(self.n_boot, boot_plan.n_boot)
        rows, replicates, manifests = [], [], []
        for b in range(n_boot):
            try:
                draw = boot_plan.draw(train_raw, val_raw, b)
                with frozen_calendar(boot_plan.calendar):
                    train_b, val_b = featurize(spec, draw.train, draw.val)
            except BootstrapError as e:
                rec = dict(boot_plan.replicates[b]) if b < len(boot_plan.replicates) else {}
                manifests.append({"bootstrap_id": b, "accepted": False, "skip": str(e), **rec})
                save_bootstrap_manifest(out_dir, manifests, block_size=BLOCK_SIZE, seed=SEED)
                print("  SKIP boot", b + 1, e)
                continue
            manifests.append({"bootstrap_id": b, "accepted": True, **draw.manifest})
            t0 = time.perf_counter()
            own_b, cross_b, _, cells_b, n_parameters_b = self.fit(
                train_b, val_b, selected_alphas=selected_alphas
            )
            elapsed_b = time.perf_counter() - t0
            metrics_b = native_metrics_from_cells(val_b, cells_b, name, b)
            compute_b = compute_row(
                n_parameters_b, elapsed_b, tuning_seconds=0.0, fit_seconds=elapsed_b,
            )
            rows.append(self._row(own_b, cross_b, metrics_b, dataset=name, boot=b, **compute_b))
            replicates.append(tag_series(own_cross_series(own_b, cross_b), name, self.model, bootstrap_id=b))
            if (b + 1) % LINEAR_BOOT_CHECKPOINT == 0 or b == 0:
                print(f"  boot {b + 1}/{n_boot}")
                save_table(pd.DataFrame(rows), out_dir, "bootstrap.csv")
                save_table(pd.concat(replicates, ignore_index=True), out_dir, "bootstrap_replicates.csv")
                save_bootstrap_manifest(out_dir, manifests, block_size=BLOCK_SIZE, seed=SEED)
        save_bootstrap_manifest(out_dir, manifests, block_size=BLOCK_SIZE, seed=SEED)
        if not replicates:
            return pd.DataFrame()
        return save_bootstrap_report(out_dir, rows, replicates, holdout)

    def run_all(self):
        """Walmart then 1C: kfold followed by bootstrap."""
        controls = self.controls

        def extra():
            print("controls:", len(controls), controls)

        return run_all_datasets(self.datasets, self.run_kfold, self.run_bootstrap, extra_print=extra)
