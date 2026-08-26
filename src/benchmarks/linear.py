"""OLS / Ridge experiment: pairwise equations, expanding CV, raw-block bootstrap.

Own and cross elasticities are both taken from `run_cross`. Cell MAE is the
mean of pairwise ŷ_ijst over j, merged onto the common (store, product, week)
validation grid. Bootstrap resamples raw train weeks, then rebuilds features.
"""

from __future__ import annotations

import time

import pandas as pd

from src.benchmarks.constants import (
    LINEAR_BOOT_CHECKPOINT,
    N_BOOT_LINEAR,
    N_FOLDS,
    SHORT,
)
from src.benchmarks.pairs import PairDatasetBuilder
from src.benchmarks.pairwise_ols import PairwiseOLS
from src.benchmarks.pairwise_ridge import PairwiseRidge
from src.benchmarks.predict import (
    attach_pred,
    compute_row,
    metrics_from_cells,
    own_cross_series,
    pair_elasticities,
    pairwise_to_cells,
    tag_series,
    val_cells,
)
from src.benchmarks.protocol import (
    block_sampler,
    expanding_folds,
    featurize,
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
    summarize_pairwise,
)


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

    def fit(self, train, val):
        """Pairwise cross equations → own/cross elasticities, cell ŷ, MAE, parameter count."""
        builder = PairDatasetBuilder(self.controls)
        est = self.estimator_cls(self.controls)
        cross, pred_ij = est.run_cross(builder.build(train), builder.build(val))
        n_parameters = int(cross["n_params"].sum()) if len(cross) else 0
        own, cross = pair_elasticities(cross)
        cells = pairwise_to_cells(pred_ij)
        metrics = metrics_from_cells(cells)
        return own, cross, pred_ij, cells, metrics, n_parameters

    def print_fit(self, own, cross, metrics) -> None:
        print(f"  own n={len(own)}  cross n={len(cross)}")
        print(
            "  mae/rmse/r2", metrics["mae_val"], metrics["rmse_val"], metrics["r2_val"],
            "n_cells", metrics["n_cells"],
        )
        if len(own):
            print(
                "  own  mean/min/max",
                own.own_elasticity.mean(), own.own_elasticity.min(), own.own_elasticity.max(),
            )
            print("  own  mean n_partners", own.n_partners.mean())
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

    def run_kfold(self, name, spec):
        """Expanding outer folds. Features fit on train only. Outer val is evaluation only."""
        panel = load_panel(spec)
        folds = expanding_folds(panel, n_folds=self.n_folds)
        rows, series = [], []
        for k, (train_raw, val_raw) in enumerate(folds, 1):
            print_fold_banner(name, k, self.n_folds, train_raw, val_raw)
            train, val = featurize(spec, train_raw, val_raw)
            t0 = time.perf_counter()
            own, cross, pred_ij, cells, metrics, n_parameters = self.fit(train, val)
            compute = compute_row(n_parameters, time.perf_counter() - t0)
            self.print_fit(own, cross, metrics)
            save_table(pred_ij, spec["out"], f"fold{k}_pred_ij.csv")
            save_table(attach_pred(val_cells(val, name, k), cells), spec["out"], f"fold{k}_pred_cells.csv")
            rows.append(self._row(own, cross, metrics, dataset=name, fold=k, **compute))
            series.append(tag_series(own_cross_series(own, cross), name, self.model, outer_fold=k))
        return save_kfold_tables(spec["out"], rows, series, allow_empty=False)

    def run_bootstrap(self, name, spec):
        """Holdout point estimate, then raw-block bootstrap of train (val weeks fixed)."""
        out_dir = spec["out"]
        panel = load_panel(spec)
        train_raw, val_raw = holdout_split(panel)
        train, val = featurize(spec, train_raw, val_raw)

        print_holdout_banner(name, train_raw, val_raw)
        own, cross, pred_ij, cells, metrics, _ = self.fit(train, val)
        self.print_fit(own, cross, metrics)
        holdout = tag_series(own_cross_series(own, cross), name, self.model)
        save_table(holdout, out_dir, "holdout_elasticities.csv")
        save_table(pred_ij, out_dir, "holdout_pred_ij.csv")
        save_table(attach_pred(val_cells(val, name, "holdout"), cells), out_dir, "holdout_pred_cells.csv")

        periods = sorted(train_raw["week_id"].unique())
        sampler = block_sampler()
        rows, replicates = [], []
        for b in range(self.n_boot):
            train_b_raw = sampler.sample(train_raw, periods)
            train_b, val_b = featurize(spec, train_b_raw, val_raw)
            t0 = time.perf_counter()
            own_b, cross_b, _, _, metrics_b, n_parameters_b = self.fit(train_b, val_b)
            compute_b = compute_row(n_parameters_b, time.perf_counter() - t0)
            rows.append(self._row(own_b, cross_b, metrics_b, dataset=name, boot=b, **compute_b))
            replicates.append(tag_series(own_cross_series(own_b, cross_b), name, self.model, bootstrap_id=b))
            if (b + 1) % LINEAR_BOOT_CHECKPOINT == 0 or b == 0:
                print(f"  boot {b + 1}/{self.n_boot}")
                save_table(pd.DataFrame(rows), out_dir, "bootstrap.csv")
                save_table(pd.concat(replicates, ignore_index=True), out_dir, "bootstrap_replicates.csv")
        return save_bootstrap_report(out_dir, rows, replicates, holdout)

    def run_all(self):
        """Walmart then 1C: kfold followed by bootstrap."""
        controls = self.controls

        def extra():
            print("controls:", len(controls), controls)

        return run_all_datasets(self.datasets, self.run_kfold, self.run_bootstrap, extra_print=extra)
