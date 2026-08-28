"""Paired ICDN–MLP comparisons and expanded elasticity diagnostics.

Bootstrap stability uses the intersection of replicate IDs for each series.
Predictive inference uses a week-block bootstrap of already-fitted outer
validation predictions (no retraining). Elasticity tables split own-price
and cross-price summaries; figure clips are audited, not silent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmarks.constants import (
    BLOCK_SIZE,
    N_BOOT_PRED_INFER,
    PRED_NI_MARGIN,
    SEED,
)
from src.benchmarks.predict import (
    CELL_KEYS,
    normalize_series_keys,
    regression_metrics,
)

PAIR_ID_COLS = ["store_code", "product_i", "product_j", "kind"]
ELAST_CLIP = {"own": (-5.0, 5.0), "cross": (-2.0, 2.0), "own_eq": (-5.0, 5.0)}
ELAST_CLIP_BY_DATASET = {
    "walmart": ELAST_CLIP,
    "one_c": ELAST_CLIP,
    "dominick": {"own": (-5.0, 0.0), "cross": (-1.0, 1.0), "own_eq": (-5.0, 0.0)},
}


def clips_for(dataset: str) -> dict[str, tuple[float, float]]:
    return ELAST_CLIP_BY_DATASET.get(str(dataset), ELAST_CLIP)


def _numeric(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _moments(e: pd.Series) -> dict:
    v = _numeric(e).dropna()
    n = int(len(v))
    if n == 0:
        return {"sd": np.nan, "ci_width": np.nan, "mean": np.nan}
    if n < 2:
        return {"sd": np.nan, "ci_width": np.nan, "mean": float(v.mean())}
    return {
        "sd": float(v.std(ddof=1)),
        "ci_width": float(v.quantile(0.975) - v.quantile(0.025)),
        "mean": float(v.mean()),
    }


def paired_bootstrap_from_replicates(icdn: pd.DataFrame, mlp: pd.DataFrame) -> pd.DataFrame:
    """One row per series on the intersection of ICDN and MLP bootstrap IDs.

    SD and percentile width are computed only on those common IDs. ICDN
    cross-price `edge_selection_frequency` is the share of ICDN replicates
    in which the edge appears, and is not the same object as the SD of the
    elasticity when the edge is present.
    """
    if icdn is None or mlp is None or icdn.empty or mlp.empty:
        return pd.DataFrame()
    a = normalize_series_keys(icdn)
    b = normalize_series_keys(mlp)
    need = PAIR_ID_COLS + ["bootstrap_id", "elasticity"]
    if any(c not in a.columns or c not in b.columns for c in need):
        return pd.DataFrame()
    a = a[need + (["dataset"] if "dataset" in a.columns else [])].drop_duplicates(
        PAIR_ID_COLS + ["bootstrap_id"]
    )
    b = b[need].drop_duplicates(PAIR_ID_COLS + ["bootstrap_id"])
    a_ids = set(pd.to_numeric(a["bootstrap_id"], errors="coerce").dropna().astype(int))
    b_ids = set(pd.to_numeric(b["bootstrap_id"], errors="coerce").dropna().astype(int))
    plan_ids = a_ids & b_ids
    n_plan = int(len(plan_ids))
    n_icdn_global = int(len(a_ids))
    n_mlp_global = int(len(b_ids))
    if n_plan == 0:
        return pd.DataFrame()

    left = a.rename(columns={"elasticity": "e_icdn"})
    right = b.rename(columns={"elasticity": "e_mlp"})
    common = left.merge(right, on=PAIR_ID_COLS + ["bootstrap_id"], how="inner")
    common = common[pd.to_numeric(common["bootstrap_id"], errors="coerce").isin(plan_ids)]
    if common.empty:
        return pd.DataFrame()

    n_icdn = (
        a.groupby(PAIR_ID_COLS, as_index=False)["bootstrap_id"]
        .nunique()
        .rename(columns={"bootstrap_id": "n_icdn_bootstrap_ids"})
    )
    n_mlp = (
        b.groupby(PAIR_ID_COLS, as_index=False)["bootstrap_id"]
        .nunique()
        .rename(columns={"bootstrap_id": "n_mlp_bootstrap_ids"})
    )

    rows = []
    grouped = common.groupby(PAIR_ID_COLS, sort=False)
    for key, part in grouped:
        rec = dict(zip(PAIR_ID_COLS, key if isinstance(key, tuple) else (key,)))
        if "dataset" in part.columns:
            rec["dataset"] = part["dataset"].iloc[0]
        icdn_m = _moments(part["e_icdn"])
        mlp_m = _moments(part["e_mlp"])
        n_common = int(part["bootstrap_id"].nunique())
        rec.update({
            "n_common_bootstrap_ids": n_common,
            "n_bootstrap_ids": n_plan,
            "common_bootstrap_frequency": (n_common / n_plan) if n_plan else np.nan,
            "sd_icdn": icdn_m["sd"],
            "sd_mlp": mlp_m["sd"],
            "width_icdn": icdn_m["ci_width"],
            "width_mlp": mlp_m["ci_width"],
            "mean_icdn": icdn_m["mean"],
            "mean_mlp": mlp_m["mean"],
            "sd_diff": icdn_m["sd"] - mlp_m["sd"] if np.isfinite(icdn_m["sd"]) and np.isfinite(mlp_m["sd"]) else np.nan,
            "width_diff": (
                icdn_m["ci_width"] - mlp_m["ci_width"]
                if np.isfinite(icdn_m["ci_width"]) and np.isfinite(mlp_m["ci_width"])
                else np.nan
            ),
        })
        rows.append(rec)
    out = pd.DataFrame(rows)
    out = out.merge(n_icdn, on=PAIR_ID_COLS, how="left").merge(n_mlp, on=PAIR_ID_COLS, how="left")
    out["edge_selection_frequency"] = np.where(
        n_icdn_global, out["n_icdn_bootstrap_ids"] / n_icdn_global, np.nan
    )
    out["n_icdn_global_boots"] = n_icdn_global
    out["n_mlp_global_boots"] = n_mlp_global
    cols = [
        "dataset", *PAIR_ID_COLS,
        "n_common_bootstrap_ids", "n_bootstrap_ids", "common_bootstrap_frequency",
        "n_icdn_bootstrap_ids", "n_mlp_bootstrap_ids",
        "n_icdn_global_boots", "n_mlp_global_boots", "edge_selection_frequency",
        "sd_icdn", "sd_mlp", "sd_diff",
        "width_icdn", "width_mlp", "width_diff",
        "mean_icdn", "mean_mlp",
    ]
    return out[[c for c in cols if c in out.columns]]


def summarize_paired_bootstrap(series: pd.DataFrame) -> pd.DataFrame:
    """P(SD_ICDN < SD_MLP), P(width_ICDN < width_MLP), and median SD difference."""
    if series is None or series.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, kind), part in series.groupby(["dataset", "kind"]):
        sd_ok = part.dropna(subset=["sd_icdn", "sd_mlp"])
        w_ok = part.dropna(subset=["width_icdn", "width_mlp"])
        rec = {
            "dataset": dataset,
            "kind": kind,
            "n_series": int(len(part)),
            "n_series_sd_defined": int(len(sd_ok)),
            "p_sd_icdn_lt_mlp": float((sd_ok["sd_icdn"] < sd_ok["sd_mlp"]).mean()) if len(sd_ok) else np.nan,
            "p_width_icdn_lt_mlp": float((w_ok["width_icdn"] < w_ok["width_mlp"]).mean()) if len(w_ok) else np.nan,
            "median_sd_diff": float((sd_ok["sd_icdn"] - sd_ok["sd_mlp"]).median()) if len(sd_ok) else np.nan,
            "median_width_diff": float((w_ok["width_icdn"] - w_ok["width_mlp"]).median()) if len(w_ok) else np.nan,
            "median_n_common_bootstrap_ids": float(part["n_common_bootstrap_ids"].median()),
            "mean_common_bootstrap_frequency": float(part["common_bootstrap_frequency"].mean()),
        }
        if str(kind) == "cross":
            rec["mean_edge_selection_frequency"] = float(part["edge_selection_frequency"].mean())
            rec["median_edge_selection_frequency"] = float(part["edge_selection_frequency"].median())
        else:
            rec["mean_edge_selection_frequency"] = np.nan
            rec["median_edge_selection_frequency"] = np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values(["dataset", "kind"]) if len(out) else out


def _prep_pred_cells(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["store_code"] = out["store_code"].astype(str)
    out["product_code"] = out["product_code"].astype(str)
    out["week_id"] = pd.to_numeric(out["week_id"], errors="coerce")
    out = out.dropna(subset=["week_id"])
    out["week_id"] = out["week_id"].astype(int)
    return out


def matched_icdn_mlp_cells(icdn: pd.DataFrame, mlp: pd.DataFrame) -> pd.DataFrame:
    """Inner join of finite ICDN and MLP predictions on (store, product, week)."""
    if icdn is None or mlp is None or icdn.empty or mlp.empty:
        return pd.DataFrame()
    a = _prep_pred_cells(icdn)
    b = _prep_pred_cells(mlp)
    keep = CELL_KEYS + ["y_true", "y_pred"]
    extra = [c for c in ("dataset", "outer_fold") if c in a.columns]
    a = a[keep + extra].rename(columns={"y_pred": "y_pred_icdn", "y_true": "y_true_icdn"})
    b = b[keep].rename(columns={"y_pred": "y_pred_mlp", "y_true": "y_true_mlp"})
    out = a.merge(b, on=CELL_KEYS, how="inner")
    y = _numeric(out["y_true_icdn"])
    if y.isna().all():
        y = _numeric(out["y_true_mlp"])
    out["y_true"] = y
    out["y_pred_icdn"] = _numeric(out["y_pred_icdn"])
    out["y_pred_mlp"] = _numeric(out["y_pred_mlp"])
    ok = np.isfinite(out["y_true"]) & np.isfinite(out["y_pred_icdn"]) & np.isfinite(out["y_pred_mlp"])
    out = out.loc[ok].copy()
    out["abs_err_icdn"] = (out["y_true"] - out["y_pred_icdn"]).abs()
    out["abs_err_mlp"] = (out["y_true"] - out["y_pred_mlp"]).abs()
    out["d_abs"] = out["abs_err_icdn"] - out["abs_err_mlp"]
    return out


def predictive_delta_metrics(y_true, y_icdn, y_mlp) -> dict:
    """ΔMAE, ΔRMSE, ΔR² on a matched cell sample (ICDN − MLP). Negative favours ICDN."""
    a = regression_metrics(y_true, y_icdn)
    b = regression_metrics(y_true, y_mlp)
    d = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_icdn, dtype=float))
    d = d - np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_mlp, dtype=float))
    d = d[np.isfinite(d)]
    return {
        "n_cells": int(a["n_cells"]),
        "mae_icdn": a["mae_val"],
        "mae_mlp": b["mae_val"],
        "rmse_icdn": a["rmse_val"],
        "rmse_mlp": b["rmse_val"],
        "r2_icdn": a["r2_val"],
        "r2_mlp": b["r2_val"],
        "delta_mae": a["mae_val"] - b["mae_val"],
        "delta_rmse": a["rmse_val"] - b["rmse_val"],
        "delta_r2": a["r2_val"] - b["r2_val"] if np.isfinite(a["r2_val"]) and np.isfinite(b["r2_val"]) else np.nan,
        "mean_d_abs": float(d.mean()) if d.size else np.nan,
    }


def store_product_mae_comparison(cells: pd.DataFrame) -> pd.DataFrame:
    """One row per store–product series on the matched cell set."""
    if cells is None or cells.empty:
        return pd.DataFrame()
    g = cells.groupby(["store_code", "product_code"], as_index=False).agg(
        n_cells=("d_abs", "size"),
        mae_icdn=("abs_err_icdn", "mean"),
        mae_mlp=("abs_err_mlp", "mean"),
        mean_d_abs=("d_abs", "mean"),
    )
    if "dataset" in cells.columns:
        ds = cells.groupby(["store_code", "product_code"], as_index=False)["dataset"].first()
        g = g.merge(ds, on=["store_code", "product_code"], how="left")
    g["delta_mae"] = g["mae_icdn"] - g["mae_mlp"]
    g["icdn_better"] = g["mae_icdn"] < g["mae_mlp"]
    return g


def _week_blocks(weeks: np.ndarray, block_size: int) -> list[np.ndarray]:
    weeks = np.sort(np.unique(np.asarray(weeks, dtype=int)))
    if weeks.size == 0:
        return []
    return [weeks[i:i + block_size] for i in range(0, weeks.size, int(block_size))]


def week_block_bootstrap_pred(
    cells: pd.DataFrame,
    n_boot: int = N_BOOT_PRED_INFER,
    block_size: int = BLOCK_SIZE,
    seed: int = SEED,
) -> pd.DataFrame:
    """Resample non-overlapping week blocks; recompute ΔMAE / ΔRMSE / ΔR².

    Week-level sufficient statistics are aggregated first, then each replicate
    is a multiplicity-weighted sum. That matches resampling cells by week block
    (with replacement) without rebuilding the cell index 999 times.
    """
    if cells is None or cells.empty:
        return pd.DataFrame()
    weeks = cells["week_id"].to_numpy(dtype=int)
    y = cells["y_true"].to_numpy(dtype=float)
    ya = cells["y_pred_icdn"].to_numpy(dtype=float)
    yb = cells["y_pred_mlp"].to_numpy(dtype=float)
    uniq, inv = np.unique(weeks, return_inverse=True)
    n_w = np.bincount(inv).astype(float)
    sum_abs_a = np.bincount(inv, weights=np.abs(y - ya))
    sum_abs_b = np.bincount(inv, weights=np.abs(y - yb))
    sse_a = np.bincount(inv, weights=(y - ya) ** 2)
    sse_b = np.bincount(inv, weights=(y - yb) ** 2)
    sum_y = np.bincount(inv, weights=y)
    sum_y2 = np.bincount(inv, weights=y ** 2)

    blocks = _week_blocks(uniq, block_size)
    n_blocks = len(blocks)
    if n_blocks == 0:
        return pd.DataFrame()
    block_idx = [np.searchsorted(uniq, blk) for blk in blocks]
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, n_blocks, size=(int(n_boot), n_blocks))
    n_weeks = int(uniq.size)
    mult = np.zeros((int(n_boot), n_weeks), dtype=np.int32)
    for b in range(int(n_boot)):
        for j in draws[b]:
            ix = block_idx[int(j)]
            mult[b, ix] += 1
    n = mult @ n_w
    mae_a = (mult @ sum_abs_a) / n
    mae_b = (mult @ sum_abs_b) / n
    rmse_a = np.sqrt((mult @ sse_a) / n)
    rmse_b = np.sqrt((mult @ sse_b) / n)
    tot_y = mult @ sum_y
    tot_y2 = mult @ sum_y2
    ss_tot = tot_y2 - (tot_y ** 2) / n
    r2_a = np.where(ss_tot > 0, 1.0 - (mult @ sse_a) / ss_tot, np.nan)
    r2_b = np.where(ss_tot > 0, 1.0 - (mult @ sse_b) / ss_tot, np.nan)
    n_weeks_b = mult @ np.ones(n_weeks, dtype=float)
    return pd.DataFrame({
        "boot": np.arange(int(n_boot)),
        "n_cells": n,
        "n_weeks": n_weeks_b,
        "n_blocks": n_blocks,
        "mae_icdn": mae_a,
        "mae_mlp": mae_b,
        "rmse_icdn": rmse_a,
        "rmse_mlp": rmse_b,
        "r2_icdn": r2_a,
        "r2_mlp": r2_b,
        "delta_mae": mae_a - mae_b,
        "delta_rmse": rmse_a - rmse_b,
        "delta_r2": r2_a - r2_b,
        "mean_d_abs": mae_a - mae_b,
    })


def pred_delta_conclusion(ci_lo: float, ci_hi: float, margin: float = PRED_NI_MARGIN) -> str:
    """Classify the ΔMAE CI: superior, inferior, non-inferior, or indistinguishable."""
    if not np.isfinite(ci_lo) or not np.isfinite(ci_hi):
        return "undefined"
    if ci_hi < 0:
        return "icdn_superior"
    if ci_lo > 0:
        return "mlp_superior"
    if ci_hi < margin:
        return "icdn_noninferior"
    return "indistinguishable"


def paired_predictive_inference(
    cells: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    n_boot: int = N_BOOT_PRED_INFER,
    block_size: int = BLOCK_SIZE,
    seed: int = SEED,
    ni_margin: float = PRED_NI_MARGIN,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Point ΔMAE plus week-block CI, store–product series table, and boot draws."""
    empty = pd.DataFrame()
    if cells is None or cells.empty:
        return {}, empty, empty
    point = predictive_delta_metrics(cells["y_true"], cells["y_pred_icdn"], cells["y_pred_mlp"])
    series = store_product_mae_comparison(cells)
    boots = week_block_bootstrap_pred(cells, n_boot=n_boot, block_size=block_size, seed=seed)
    if len(series):
        n_series = int(len(series))
        share = float(series["icdn_better"].mean())
        w = pd.to_numeric(series["n_cells"], errors="coerce")
        share_w = float(np.average(series["icdn_better"].astype(float), weights=w)) if w.sum() else np.nan
    else:
        n_series = 0
        share = share_w = np.nan
    if boots.empty:
        lo = hi = np.nan
        lo_rmse = hi_rmse = lo_r2 = hi_r2 = np.nan
    else:
        lo, hi = (float(boots["delta_mae"].quantile(q)) for q in (0.025, 0.975))
        lo_rmse, hi_rmse = (float(boots["delta_rmse"].quantile(q)) for q in (0.025, 0.975))
        lo_r2, hi_r2 = (float(boots["delta_r2"].quantile(q)) for q in (0.025, 0.975))
    rec = {
        "dataset": dataset,
        "split": split,
        "n_cells": point["n_cells"],
        "n_series": n_series,
        "n_weeks": int(cells["week_id"].nunique()),
        "n_blocks": int(np.ceil(cells["week_id"].nunique() / max(int(block_size), 1))),
        "block_size": int(block_size),
        "n_boot": int(n_boot),
        "ni_margin": float(ni_margin),
        "mae_icdn": point["mae_icdn"],
        "mae_mlp": point["mae_mlp"],
        "rmse_icdn": point["rmse_icdn"],
        "rmse_mlp": point["rmse_mlp"],
        "r2_icdn": point["r2_icdn"],
        "r2_mlp": point["r2_mlp"],
        "delta_mae": point["delta_mae"],
        "delta_rmse": point["delta_rmse"],
        "delta_r2": point["delta_r2"],
        "delta_mae_ci_lo": lo,
        "delta_mae_ci_hi": hi,
        "delta_rmse_ci_lo": lo_rmse,
        "delta_rmse_ci_hi": hi_rmse,
        "delta_r2_ci_lo": lo_r2,
        "delta_r2_ci_hi": hi_r2,
        "share_series_icdn_better": share,
        "share_series_icdn_better_weighted": share_w,
        "share_cells_icdn_better": float((cells["abs_err_icdn"] < cells["abs_err_mlp"]).mean()),
        "conclusion": pred_delta_conclusion(lo, hi, ni_margin),
    }
    if len(series):
        series = series.copy()
        series["dataset"] = dataset
        series["split"] = split
        lead = ["dataset", "split", "store_code", "product_code"]
        series = series[[c for c in lead if c in series.columns] + [c for c in series.columns if c not in lead]]
    if len(boots):
        boots = boots.copy()
        boots["dataset"] = dataset
        boots["split"] = split
        lead = ["dataset", "split", "boot"]
        boots = boots[[c for c in lead if c in boots.columns] + [c for c in boots.columns if c not in lead]]
    return rec, series, boots


def _iqr(v: pd.Series) -> float:
    return float(v.quantile(0.75) - v.quantile(0.25))


def elasticity_kind_stats(e: pd.Series, kind: str) -> dict:
    v = _numeric(e).dropna()
    n = int(len(v))
    if n == 0:
        return {"n": 0}
    rec = {
        "n": n,
        "mean": float(v.mean()),
        "median": float(v.median()),
        "std": float(v.std(ddof=1)) if n >= 2 else np.nan,
        "iqr": _iqr(v),
        "min": float(v.min()),
        "max": float(v.max()),
        "q025": float(v.quantile(0.025)),
        "q975": float(v.quantile(0.975)),
        "p05": float(v.quantile(0.05)),
        "p95": float(v.quantile(0.95)),
    }
    kind = str(kind)
    if kind in {"own", "own_eq"}:
        rec.update({
            "share_in_minus5_0": float(((v >= -5) & (v <= 0)).mean()),
            "share_positive": float((v > 0).mean()),
            "share_abs_gt5": float((v.abs() > 5).mean()),
            "share_negative": float((v < 0).mean()),
            "share_abs_le1": float((v.abs() <= 1).mean()),
            "share_abs_gt1": float((v.abs() > 1).mean()),
            "share_abs_gt3": float((v.abs() > 3).mean()),
        })
    else:
        rec.update({
            "share_positive": float((v > 0).mean()),
            "share_negative": float((v < 0).mean()),
            "share_abs_le1": float((v.abs() <= 1).mean()),
            "share_abs_gt1": float((v.abs() > 1).mean()),
            "share_abs_gt3": float((v.abs() > 3).mean()),
            "share_in_minus5_0": float(((v >= -5) & (v <= 0)).mean()),
            "share_abs_gt5": float((v.abs() > 5).mean()),
        })
    return rec


def elasticity_distribution(series: pd.DataFrame) -> pd.DataFrame:
    """Own- and cross-price summaries by dataset, model, and outer fold."""
    if series is None or series.empty:
        return pd.DataFrame()
    df = series.copy()
    if "outer_fold" not in df.columns and "fold" in df.columns:
        df["outer_fold"] = df["fold"]
    rows = []
    keys = [c for c in ("dataset", "model", "outer_fold", "kind") if c in df.columns]
    for key, part in df.groupby(keys):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        rec.update(elasticity_kind_stats(part["elasticity"], rec.get("kind", "")))
        rows.append(rec)
    out = pd.DataFrame(rows)
    if "outer_fold" in out.columns:
        out = out.rename(columns={"outer_fold": "fold"})
    return out


def elasticity_clip_diagnostics(
    series: pd.DataFrame,
    clips: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """How many series the boxplots drop, plus the unclipped min/max."""
    clips = clips or ELAST_CLIP
    if series is None or series.empty:
        return pd.DataFrame()
    df = series.copy()
    rows = []
    keys = [c for c in ("dataset", "model", "kind") if c in df.columns]
    for key, part in df.groupby(keys):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        kind = str(rec.get("kind", ""))
        clip = clips.get(kind)
        e = _numeric(part["elasticity"]).dropna()
        rec["n"] = int(len(e))
        rec["min_full"] = float(e.min()) if len(e) else np.nan
        rec["max_full"] = float(e.max()) if len(e) else np.nan
        if clip is None or e.empty:
            rec["clip_lo"] = rec["clip_hi"] = np.nan
            rec["n_clipped"] = 0
            rec["share_clipped"] = np.nan
        else:
            lo, hi = clip
            mask = (e < lo) | (e > hi)
            rec["clip_lo"] = lo
            rec["clip_hi"] = hi
            rec["n_clipped"] = int(mask.sum())
            rec["share_clipped"] = float(mask.mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def share_obs_both(part: pd.DataFrame) -> float:
    """P(observed_i ∧ observed_j), not the product of the marginals."""
    if "observed_i" not in part.columns or "observed_j" not in part.columns:
        return np.nan
    return float((part["observed_i"].astype(bool) & part["observed_j"].astype(bool)).mean())


def best_complete_trial(part: pd.DataFrame) -> pd.Series | None:
    """Optuna best among COMPLETE trials only (pruned intermediates are ignored)."""
    if part is None or part.empty or "value" not in part.columns:
        return None
    complete = part[part["state"].astype(str) == "COMPLETE"]
    complete = complete[pd.to_numeric(complete["value"], errors="coerce").notna()]
    if complete.empty:
        return None
    return complete.loc[pd.to_numeric(complete["value"], errors="coerce").idxmin()]


def load_outer_pred_cells(panel_dir: Path, model: str) -> pd.DataFrame:
    parts = []
    for path in sorted((Path(panel_dir) / model).glob("fold*_pred_cells.csv")):
        if "holdout" in path.name:
            continue
        df = pd.read_csv(path)
        if len(df):
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_holdout_pred_cells(panel_dir: Path, model: str) -> pd.DataFrame:
    path = Path(panel_dir) / model / "holdout_pred_cells.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_paired_tables(panel_dir: Path) -> None:
    """Write paired ICDN–MLP bootstrap and predictive tables for one panel."""
    from src.benchmarks.protocol import save_table

    panel_dir = Path(panel_dir)
    icdn_boot = _read_csv(panel_dir / "icdn" / "bootstrap_replicates.csv")
    mlp_boot = _read_csv(panel_dir / "mlp" / "bootstrap_replicates.csv")
    stability = paired_bootstrap_from_replicates(icdn_boot, mlp_boot)
    if len(stability):
        save_table(stability, panel_dir, "paired_bootstrap_stability.csv")
        summary = summarize_paired_bootstrap(stability)
        if len(summary):
            save_table(summary, panel_dir, "paired_bootstrap_summary.csv")

    inf_rows, series_parts = [], []
    for split, loader in (("outer", load_outer_pred_cells), ("holdout", load_holdout_pred_cells)):
        icdn = loader(panel_dir, "icdn")
        mlp = loader(panel_dir, "mlp")
        cells = matched_icdn_mlp_cells(icdn, mlp)
        if cells.empty:
            continue
        dataset = (
            str(cells["dataset"].iloc[0]) if "dataset" in cells.columns else panel_dir.parent.name
        )
        rec, series, _boots = paired_predictive_inference(cells, dataset=dataset, split=split)
        if rec:
            inf_rows.append(rec)
        if len(series):
            series_parts.append(series)
    if inf_rows:
        save_table(pd.DataFrame(inf_rows), panel_dir, "paired_predictive_inference.csv")
    if series_parts:
        save_table(pd.concat(series_parts, ignore_index=True), panel_dir, "paired_predictive_series.csv")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
