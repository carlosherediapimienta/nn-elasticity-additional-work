"""Metrics, cell alignment, series-level bootstrap CIs, and compute accounting.

Cell MAE is comparable across models: ŷ_ist for pairwise models is the mean
of ŷ_ijst over partners j. Elasticity uncertainty is series-level (store, i, j,
kind), not a CI around the global mean. A series is "matched" if it appears
in at least MIN_BOOT_FREQ of bootstrap replicates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from optuna.trial import TrialState

SERIES_KEYS = ["dataset", "model", "store_code", "product_i", "product_j", "kind"]
SERIES_ID_COLS = ["store_code", "product_i", "product_j"]
MIN_BOOT_FREQ = 0.8


def normalize_series_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Cast series IDs to str so in-memory tables merge with CSVs (pandas infers int)."""
    out = df.copy()
    for col in SERIES_ID_COLS:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def val_cells(val: pd.DataFrame, dataset: str, outer_fold) -> pd.DataFrame:
    """Common (store, product, week) keys on the validation window, with y_true."""
    keys = ["store_code", "product_code", "week_id"]
    cols = keys + (["log_demand"] if "log_demand" in val.columns else ["units"])
    out = val[cols].drop_duplicates(keys).copy()
    if "log_demand" in out.columns:
        out = out.rename(columns={"log_demand": "y_true"})
    else:
        out["y_true"] = np.log(out.pop("units").astype(float))
    out.insert(0, "dataset", dataset)
    out.insert(1, "outer_fold", outer_fold)
    return out


def pairwise_to_cells(pred_ij: pd.DataFrame) -> pd.DataFrame:
    """ŷ_ist = mean_j ŷ_ijst from pairwise equations only."""
    if pred_ij is None or len(pred_ij) == 0:
        return pd.DataFrame(columns=["store_code", "product_code", "week_id", "y_true", "y_pred", "n_eq"])
    return (
        pred_ij.groupby(["store_code", "product_i", "week_id"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", "mean"),
            n_eq=("product_j", "size"),
        )
        .rename(columns={"product_i": "product_code"})
    )


def pair_elasticities(cross: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Own and cross from the same pairwise system.

    own(store, i) = mean_j β_ij^own  (log p_i in the (i, j) equation)
    cross(store, i, j) = β_ij^cross (log p_j)
    """
    if cross is None or len(cross) == 0:
        return pd.DataFrame(), pd.DataFrame()
    own = (
        cross.groupby(["store_code", "product_i"], as_index=False)
        .agg(
            own_elasticity=("own_elasticity", "mean"),
            n_partners=("product_j", "size"),
            n_train=("n_train", "mean"),
            n_val=("n_val", "mean"),
        )
        .rename(columns={"product_i": "product_code"})
    )
    return own, cross


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if y_true.size == 0:
        return dict(mae_val=np.nan, rmse_val=np.nan, r2_val=np.nan, n_cells=0)
    resid = y_true - y_pred
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return dict(
        mae_val=float(np.mean(np.abs(resid))),
        rmse_val=float(np.sqrt(np.mean(resid ** 2))),
        r2_val=np.nan if ss_tot == 0 else float(1.0 - np.sum(resid ** 2) / ss_tot),
        n_cells=int(y_true.size),
    )


def metrics_from_cells(cells: pd.DataFrame) -> dict:
    if cells is None or len(cells) == 0:
        return dict(mae_val=np.nan, rmse_val=np.nan, r2_val=np.nan, n_cells=0)
    return regression_metrics(cells["y_true"], cells["y_pred"])


def attach_pred(keys: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Left-join predictions onto the common validation grid (missing ŷ → NaN)."""
    if cells is None or len(cells) == 0:
        out = keys.copy()
        out["y_pred"] = np.nan
        return out
    pred = cells[["store_code", "product_code", "week_id", "y_pred"]].drop_duplicates(
        ["store_code", "product_code", "week_id"]
    )
    return keys.merge(pred, on=["store_code", "product_code", "week_id"], how="left")

def own_cross_series(own: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    """Long (store, i, j, kind, elasticity) from pairwise own/cross tables."""
    frames = []
    if own is not None and len(own):
        pi = own["product_code"]
        frames.append(pd.DataFrame({
            "store_code": own["store_code"].to_numpy(),
            "product_i": pi.to_numpy(),
            "product_j": pi.to_numpy(),
            "kind": "own",
            "elasticity": own["own_elasticity"].to_numpy(),
        }))
    if cross is not None and len(cross):
        frames.append(pd.DataFrame({
            "store_code": cross["store_code"].to_numpy(),
            "product_i": cross["product_i"].to_numpy(),
            "product_j": cross["product_j"].to_numpy(),
            "kind": "cross",
            "elasticity": cross["cross_elasticity"].to_numpy(),
        }))
    if not frames:
        return pd.DataFrame(columns=["store_code", "product_i", "product_j", "kind", "elasticity"])
    return normalize_series_keys(pd.concat(frames, ignore_index=True))


def summary_series(summary: pd.DataFrame) -> pd.DataFrame:
    """Keep the series keys from an already-long elasticity table (MLP / ICDN)."""
    cols = ["store_code", "product_i", "product_j", "kind", "elasticity"]
    return normalize_series_keys(summary[cols])


def tag_series(df: pd.DataFrame, dataset: str, model: str, **ids) -> pd.DataFrame:
    """Prefix dataset/model and optional ids (outer_fold, bootstrap_id)."""
    out = normalize_series_keys(df)
    out.insert(0, "dataset", dataset)
    out.insert(1, "model", model)
    for key, value in ids.items():
        out[key] = value
    return out

def bootstrap_series_ci(boot_long: pd.DataFrame) -> pd.DataFrame:
    """Conditional on the pair being estimated in that replica."""
    grouped = boot_long.groupby(SERIES_KEYS, as_index=False)["elasticity"]
    return grouped.agg(
        n_present="size",
        mean="mean",
        sd="std",
        q025=lambda s: s.quantile(0.025),
        q50=lambda s: s.quantile(0.5),
        q975=lambda s: s.quantile(0.975),
    )

def bootstrap_series_report(
    boot_long: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    min_freq: float = MIN_BOOT_FREQ,
) -> pd.DataFrame:
    """CI per series, plus frequency in the holdout universe and a matched flag."""
    n_replicates = int(boot_long["bootstrap_id"].nunique())
    ci = normalize_series_keys(bootstrap_series_ci(boot_long))
    if universe is not None and len(universe):
        keys = normalize_series_keys(universe[SERIES_KEYS].drop_duplicates())
        ci = keys.merge(ci, on=SERIES_KEYS, how="left")
        ci["n_present"] = ci["n_present"].fillna(0).astype(int)
    ci["n_replicates"] = n_replicates
    ci["freq"] = ci["n_present"] / n_replicates if n_replicates else np.nan
    ci["matched"] = ci["freq"] >= min_freq
    return ci
    
def matched_global(boot_ci: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic: mean of matched series means (not a CI for a global parameter)."""
    rows = []
    keep = boot_ci[boot_ci["matched"] == True]
    for kind, part in keep.groupby("kind"):
        rows.append({
            "kind": kind,
            "n_series": len(part),
            "mean_freq": float(part["freq"].mean()),
            "mean_conditional": float(part["mean"].mean()),
            "median_width": float((part["q975"] - part["q025"]).median()),
        })
    return pd.DataFrame(rows)


def _sign_stable(s: pd.Series) -> float:
    v = s.to_numpy(dtype=float)
    v = v[np.isfinite(v) & (v != 0)]
    if v.size == 0:
        return np.nan
    return float(np.all(np.sign(v) == np.sign(v[0])))


def fold_series_stats(fold_long: pd.DataFrame) -> pd.DataFrame:
    """Across outer folds: mean, SD, and whether the sign is stable."""
    grouped = normalize_series_keys(fold_long).groupby(SERIES_KEYS, as_index=False)
    return grouped.agg(
        n_folds=("elasticity", "size"),
        fold_mean=("elasticity", "mean"),
        fold_sd=("elasticity", "std"),
        sign_stable=("elasticity", _sign_stable),
    )


def point_in_boot_ci(point: pd.DataFrame, boot_ci: pd.DataFrame) -> pd.DataFrame:
    """Whether the holdout point estimate sits inside the bootstrap interval."""
    out = normalize_series_keys(point).merge(
        normalize_series_keys(boot_ci), on=SERIES_KEYS, how="inner"
    )
    out["in_boot_ci"] = (out["elasticity"] >= out["q025"]) & (out["elasticity"] <= out["q975"])
    out["boot_width"] = out["q975"] - out["q025"]
    return out


def boot_fold_ratio(boot_ci: pd.DataFrame, fold_stats: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap SD relative to outer-fold SD for the same series."""
    out = normalize_series_keys(boot_ci).merge(
        normalize_series_keys(fold_stats), on=SERIES_KEYS, how="inner"
    )
    out["sd_ratio_boot_fold"] = out["sd"] / out["fold_sd"]
    return out

def trial_counts(study=None) -> dict:
    """Optuna trial states. Linear models pass study=None → zeros."""
    if study is None:
        return dict(completed_trials=0, pruned_trials=0, failed_trials=0)
    states = [t.state for t in study.trials]
    return dict(
        completed_trials=int(sum(s == TrialState.COMPLETE for s in states)),
        pruned_trials=int(sum(s == TrialState.PRUNED for s in states)),
        failed_trials=int(sum(s == TrialState.FAIL for s in states)),
    )
    
def gpu_hours(seconds: float, used_gpu: bool) -> float:
    """Wall time in hours if a GPU was used; else 0 (CPU does not count as GPU-hours)."""
    if not used_gpu:
        return 0.0
    return float(seconds) / 3600.0
    
def n_torch_params(module) -> int:
    """Total `numel` over parameters (includes embeddings)."""
    return int(sum(p.numel() for p in module.parameters()))

def compute_row(n_parameters, seconds, used_gpu=False, study=None) -> dict:
    """Columns appended to kfold.csv / bootstrap.csv."""
    return dict(
        n_parameters=int(n_parameters) if n_parameters is not None else np.nan,
        training_seconds=float(seconds),
        gpu_hours=gpu_hours(seconds, used_gpu),
        **trial_counts(study),
    )