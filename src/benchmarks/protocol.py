"""Shared evaluation protocol: panels, splits, tables, and summaries.

Every benchmark notebook used to copy these helpers. The numerical protocol
is unchanged: positive price/units filter, expanding outer folds, raw-block
bootstrap on train only, cell-level MAE, and series-level bootstrap CIs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from icdn import PanelSchema
from icdn.data.splits import BlockBootstrapSampler, TemporalSplitter

from src.benchmarks.constants import (
    BLOCK_SIZE,
    HOLDOUT_TRAIN_FRAC,
    MIN_TRAIN_FRAC,
    N_FOLDS,
    PERIOD_COL,
    SEED,
)
from src.benchmarks.features import ICDNFeaturePipeline
from src.benchmarks.predict import (
    boot_fold_ratio,
    bootstrap_series_report,
    fold_series_stats,
    matched_global,
    point_in_boot_ci,
)


def project_root(cwd: Path | None = None) -> Path:
    """Repo root whether the kernel was started in the repo or in notebooks/."""
    cwd = Path.cwd() if cwd is None else cwd
    return cwd if cwd.name != "notebooks" else cwd.parent


def model_datasets(root: Path, model: str, extras: dict | None = None) -> dict:
    """Walmart M5 and 1C panels, with per-model output directory `panel/<model>/`.

    `extras` is merged per dataset (ICDN bounds, category rule).
    """
    extras = extras or {}
    specs = {
        "walmart": {
            "path": root / "data" / "M5-walmart" / "panel" / "m5_icdn_panel.parquet",
            "out": root / "data" / "M5-walmart" / "panel" / model,
            "schema": PanelSchema(category="category"),
        },
        "one_c": {
            "path": root / "data" / "predict-future-sales-1c" / "panel" / "1c_icdn_panel.parquet",
            "out": root / "data" / "predict-future-sales-1c" / "panel" / model,
            "schema": PanelSchema(category="category"),
        },
    }
    for name, extra in extras.items():
        specs[name] = {**specs[name], **extra}
    return specs


def load_panel(spec: dict) -> pd.DataFrame:
    """Drop non-positive prices or units. ICDN and the linear models both need logs."""
    panel = pd.read_parquet(spec["path"])
    return panel[(panel["price"] > 0) & (panel["units"] > 0)].copy()


def save_table(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    """Write a CSV and echo the path so notebook logs stay searchable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df.to_csv(path, index=False)
    print("  wrote", path)


def temporal_splitter() -> TemporalSplitter:
    return TemporalSplitter(period_col=PERIOD_COL)


def expanding_folds(panel: pd.DataFrame, n_folds: int = N_FOLDS, min_train_frac: float = MIN_TRAIN_FRAC):
    return temporal_splitter().expanding_splits(panel, n_folds=n_folds, min_train_frac=min_train_frac)


def holdout_split(panel: pd.DataFrame, train_frac: float = HOLDOUT_TRAIN_FRAC):
    return temporal_splitter().single_split(panel, train_frac=train_frac)


def block_sampler(seed: int = SEED, block_size: int = BLOCK_SIZE) -> BlockBootstrapSampler:
    return BlockBootstrapSampler(
        period_col=PERIOD_COL,
        block_size=block_size,
        rng=np.random.default_rng(seed),
    )


def featurize(spec: dict, train_raw: pd.DataFrame, val_raw: pd.DataFrame):
    """Fit ICDN features on train only; transform val with the train tail (no leakage)."""
    feats = ICDNFeaturePipeline(schema=spec["schema"])
    train = feats.fit(train_raw).transform(train_raw)
    val = feats.transform_val(val_raw)
    return train, val


def print_dataset_banner(name: str, panel: pd.DataFrame) -> None:
    print(
        f"\n=== {name} === {panel.shape}  "
        f"products={panel['product_code'].nunique()}  "
        f"stores={panel['store_code'].nunique()}"
    )


def print_fold_banner(name: str, k: int, n_folds: int, train_raw: pd.DataFrame, val_raw: pd.DataFrame) -> None:
    print(
        f"--- {name} fold {k}/{n_folds}  "
        f"weeks {train_raw.week_id.min()}–{train_raw.week_id.max()} | "
        f"{val_raw.week_id.min()}–{val_raw.week_id.max()}"
    )


def print_holdout_banner(name: str, train_raw: pd.DataFrame, val_raw: pd.DataFrame) -> None:
    print(
        f"--- {name} holdout  train weeks {train_raw.week_id.min()}–{train_raw.week_id.max()} | "
        f"val {val_raw.week_id.min()}–{val_raw.week_id.max()}"
    )


def summarize_pairwise(own: pd.DataFrame, cross: pd.DataFrame, metrics: dict, model: str, **extra) -> dict:
    """One row of fold/bootstrap metrics from pairwise own/cross tables."""
    row = dict(extra)
    row["model"] = model
    row["n_own"] = len(own)
    row["n_cross"] = len(cross)
    row["own_mean"] = float(own.own_elasticity.mean()) if len(own) else np.nan
    row["cross_mean"] = float(cross.cross_elasticity.mean()) if len(cross) else np.nan
    row["mae_val"] = float(metrics["mae_val"])
    row["rmse_val"] = float(metrics["rmse_val"])
    row["n_cells"] = int(metrics["n_cells"])
    return row


def summarize_kind(metrics: dict, table: pd.DataFrame, model: str, *, n_cells_required: bool = True, **extra) -> dict:
    """One row of fold/bootstrap metrics from a long table with a `kind` column.

    MLP uses `metrics.get("n_cells", 0)`; ICDN indexes `metrics["n_cells"]`.
    `n_cells_required` preserves that difference.
    """
    own = table[table.kind == "own"]
    cross = table[table.kind == "cross"]
    row = dict(extra)
    row["model"] = model
    row["n_own"] = len(own)
    row["n_cross"] = len(cross)
    row["own_mean"] = float(own.elasticity.mean()) if len(own) else np.nan
    row["cross_mean"] = float(cross.elasticity.mean()) if len(cross) else np.nan
    row["mae_val"] = float(metrics["mae_val"])
    row["rmse_val"] = float(metrics["rmse_val"])
    row["n_cells"] = int(metrics["n_cells"]) if n_cells_required else int(metrics.get("n_cells", 0))
    return row


def save_kfold_tables(out_dir: Path, rows: list, series: list, *, allow_empty: bool = False) -> pd.DataFrame:
    """Write kfold.csv, kfold_series.csv, and (when appropriate) kfold_series_stats.csv.

    Linear models always concatenate `series` (empty list raises). Neural models
    allow an empty skip-all-folds run and only write stats if any series exist.
    """
    out = pd.DataFrame(rows)
    if allow_empty:
        fold_long = pd.concat(series, ignore_index=True) if series else pd.DataFrame()
    else:
        fold_long = pd.concat(series, ignore_index=True)
    metric_cols = ["own_mean", "cross_mean", "mae_val", "rmse_val"]
    if len(out) and all(c in out.columns for c in metric_cols):
        print(out[metric_cols].agg(["mean", "std"]))
    else:
        print("  no completed folds")
    save_table(out, out_dir, "kfold.csv")
    save_table(fold_long, out_dir, "kfold_series.csv")
    if (not allow_empty) or len(fold_long):
        save_table(fold_series_stats(fold_long), out_dir, "kfold_series_stats.csv")
    return out


def save_bootstrap_report(out_dir: Path, rows: list, replicates: list, holdout: pd.DataFrame) -> pd.DataFrame:
    """Matched series CIs, holdout-in-CI, and boot/fold SD ratio if kfold stats exist."""
    boots = pd.DataFrame(rows)
    boot_long = pd.concat(replicates, ignore_index=True)
    boot_ci = bootstrap_series_report(boot_long, universe=holdout)
    matched = boot_ci[boot_ci["matched"]]
    save_table(boots, out_dir, "bootstrap.csv")
    save_table(boot_long, out_dir, "bootstrap_replicates.csv")
    save_table(boot_ci, out_dir, "bootstrap_series_ci.csv")
    save_table(matched, out_dir, "bootstrap_matched.csv")
    save_table(matched_global(boot_ci), out_dir, "bootstrap_matched_global.csv")
    save_table(point_in_boot_ci(holdout, matched), out_dir, "holdout_in_boot_ci.csv")
    fold_stats_path = out_dir / "kfold_series_stats.csv"
    if fold_stats_path.exists():
        fold_stats = pd.read_csv(fold_stats_path)
        save_table(boot_fold_ratio(matched, fold_stats), out_dir, "boot_fold_ratio.csv")
    print(
        "  series", len(boot_ci),
        "matched", int(boot_ci["matched"].sum()),
        "mean freq", float(boot_ci["freq"].mean()),
    )
    print(matched_global(boot_ci))
    return boot_long


def run_all_datasets(datasets: dict, run_kfold, run_bootstrap, extra_print=None):
    """Cell-2 loop: banner, optional extra line, kfold, then bootstrap."""
    kfold_tables, boot_tables = {}, {}
    for name, spec in datasets.items():
        panel = load_panel(spec)
        print_dataset_banner(name, panel)
        if extra_print is not None:
            extra_print()
        print("\n Start kfold")
        kfold_tables[name] = run_kfold(name, spec)
        print("\n Start bootstrap")
        boot_tables[name] = run_bootstrap(name, spec)
    return kfold_tables, boot_tables
