"""Metrics, cell alignment, series-level bootstrap CIs, and compute accounting.

Native cell MAE is each model on the cells it can predict, with
`prediction_coverage` against the full validation grid. Direct model
comparison uses the intersection of predicted cells (see
`matched_eval_rows`). Cross-elasticities are compared on matched directed
edges (`matched_edge_rows`): ICDN∩MLP, then ICDN∩Ridge and ICDN∩OLS.
Pairwise own-price (OLS/Ridge) is stored both as the product mean over J_i
(`kind=own`, with n_partners / partner_set_id) and as the equation-level
β_ij^own (`kind=own_eq`). Stability comparisons use common partners.
Cross `fold_sd` is the complete-fold set: the edge is in all `N_FOLDS`
outer folds, and a matched comparison uses the same fold IDs in both
models. Presence frequency and partial-fold counts are a separate coverage
table, not the main stability result.
Elasticity uncertainty is series-level (store, i, j, kind), not a CI around
the global mean. A series is "matched" if it appears in at least
MIN_BOOT_FREQ of bootstrap replicates.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from optuna.trial import TrialState

from src.benchmarks.constants import N_FOLDS

SERIES_KEYS = ["dataset", "model", "store_code", "product_i", "product_j", "kind"]
SERIES_ID_COLS = ["store_code", "product_i", "product_j"]
CELL_KEYS = ["store_code", "product_code", "week_id"]
EDGE_KEYS = ["store_code", "product_i", "product_j"]
COMPARE_MODELS = ("ols", "ridge", "mlp", "icdn")
LINEAR_MODELS = ("ols", "ridge")
NEURAL_MODELS = ("mlp", "icdn")
EDGE_COMPARISONS = (
    ("matched_icdn_mlp", "icdn", "mlp"),
    ("matched_icdn_ridge", "icdn", "ridge"),
    ("matched_icdn_ols", "icdn", "ols"),
)
OWN_EQ_KIND = "own_eq"
OWN_META_COLS = ("n_partners", "partner_set_id", "n_train", "n_val")
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


def partner_set_id(partners) -> str:
    """Stable id of the competitor set J_i used in a product-level own mean."""
    toks = sorted({str(p) for p in partners if p is not None and str(p) != "nan"})
    return hashlib.sha1("|".join(toks).encode("utf-8")).hexdigest()[:12]


def pair_elasticities(cross: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Own and cross from the same pairwise system.

    own(store, i) = mean_j β_ij^own  (log p_i in the (i, j) equation)
    cross(store, i, j) = β_ij^cross (log p_j)

    The product-level own mean also stores n_partners, partner_set_id, n_train,
    n_val so fold/bootstrap SD is not silently mixing a change in J_i.
    """
    if cross is None or len(cross) == 0:
        return pd.DataFrame(), pd.DataFrame()
    cross = normalize_series_keys(cross.copy())
    ids = (
        cross.groupby(["store_code", "product_i"], sort=False)["product_j"]
        .agg(lambda s: partner_set_id(s.tolist()))
        .rename("partner_set_id")
        .reset_index()
    )
    agg = {
        "own_elasticity": ("own_elasticity", "mean"),
        "n_partners": ("product_j", "size"),
    }
    if "n_train" in cross.columns:
        agg["n_train"] = ("n_train", "mean")
    if "n_val" in cross.columns:
        agg["n_val"] = ("n_val", "mean")
    own = (
        cross.groupby(["store_code", "product_i"], as_index=False)
        .agg(**agg)
        .merge(ids, on=["store_code", "product_i"], how="left")
        .rename(columns={"product_i": "product_code"})
    )
    cross = cross.merge(ids, on=["store_code", "product_i"], how="left")
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


def normalize_cell_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CELL_KEYS:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def native_metrics(grid: pd.DataFrame) -> dict:
    """MAE/RMSE/R² on cells with a finite ŷ, plus coverage vs the full val grid."""
    n_val = 0 if grid is None else int(len(grid))
    m = metrics_from_cells(grid)
    n_hat = int(m["n_cells"])
    coverage = np.nan if n_val == 0 else n_hat / n_val
    return {
        "mae_val": m["mae_val"],
        "rmse_val": m["rmse_val"],
        "r2_val": m["r2_val"],
        "mae_native": m["mae_val"],
        "rmse_native": m["rmse_val"],
        "r2_native": m["r2_val"],
        "n_cells": n_hat,
        "n_val_cells": n_val,
        "prediction_coverage": float(coverage) if n_val else np.nan,
    }


def native_metrics_from_cells(val: pd.DataFrame, cells: pd.DataFrame, dataset: str, fold) -> dict:
    """Native metrics after joining predictions onto the validation grid."""
    return native_metrics(attach_pred(val_cells(val, dataset, fold), cells))


def predicted_cell_keys(grid: pd.DataFrame) -> pd.DataFrame:
    g = normalize_cell_keys(grid)
    ok = np.isfinite(g["y_pred"].to_numpy(dtype=float)) & np.isfinite(g["y_true"].to_numpy(dtype=float))
    return g.loc[ok, CELL_KEYS].drop_duplicates()


def intersect_predicted_keys(grids: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = None
    for grid in grids.values():
        part = predicted_cell_keys(grid)
        keys = part if keys is None else keys.merge(part, on=CELL_KEYS, how="inner")
    if keys is None or keys.empty:
        return pd.DataFrame(columns=CELL_KEYS)
    return keys


def metrics_on_keys(grid: pd.DataFrame, keys: pd.DataFrame) -> dict:
    n = 0 if keys is None else int(len(keys))
    if keys is None or keys.empty:
        out = native_metrics(pd.DataFrame({"y_true": [], "y_pred": []}))
        out["n_val_cells"] = 0
        return out
    cols = [c for c in CELL_KEYS + ["y_true", "y_pred"] if c in grid.columns]
    sub = keys.merge(normalize_cell_keys(grid)[cols], on=CELL_KEYS, how="inner")
    out = native_metrics(sub)
    out["n_val_cells"] = n
    out["prediction_coverage"] = (out["n_cells"] / n) if n else np.nan
    return out


def _eval_row(dataset, outer_fold, eval_name: str, model: str, metrics: dict, *, models: list[str]) -> dict:
    return {
        "dataset": dataset,
        "outer_fold": outer_fold,
        "eval": eval_name,
        "model": model,
        "models": ",".join(sorted(models)),
        "n_models": len(models),
        "mae": metrics["mae_native"],
        "rmse": metrics["rmse_native"],
        "r2": metrics["r2_native"],
        "n_cells": metrics["n_cells"],
        "n_val_cells": metrics["n_val_cells"],
        "prediction_coverage": metrics["prediction_coverage"],
    }


def matched_eval_rows(grids: dict[str, pd.DataFrame], outer_fold) -> list[dict]:
    """Native rows plus matched_all and matched_icdn_mlp when those sets exist."""
    rows = []
    if not grids:
        return rows
    dataset = None
    for grid in grids.values():
        if grid is not None and len(grid) and "dataset" in grid.columns:
            dataset = grid["dataset"].iloc[0]
            break
    models = list(grids)
    for model, grid in grids.items():
        rows.append(_eval_row(dataset, outer_fold, "native", model, native_metrics(grid), models=models))
    if len(grids) >= 2:
        keys = intersect_predicted_keys(grids)
        for model, grid in grids.items():
            rows.append(
                _eval_row(dataset, outer_fold, "matched_all", model, metrics_on_keys(grid, keys), models=models)
            )
    neural = {m: grids[m] for m in NEURAL_MODELS if m in grids}
    if len(neural) == 2:
        keys = intersect_predicted_keys(neural)
        for model, grid in neural.items():
            rows.append(
                _eval_row(
                    dataset, outer_fold, "matched_icdn_mlp", model,
                    metrics_on_keys(grid, keys), models=list(neural),
                )
            )
    return rows


def kind_only(series: pd.DataFrame, kind: str) -> pd.DataFrame:
    out = normalize_series_keys(series)
    if out.empty or "kind" not in out.columns:
        return out.iloc[0:0]
    return out.loc[out["kind"].astype(str) == kind].copy()


def share_abs_le_1(values) -> float:
    e = np.asarray(values, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return np.nan
    return float(np.mean(np.abs(e) <= 1.0))


def cross_only(series: pd.DataFrame) -> pd.DataFrame:
    return kind_only(series, "cross")


def own_eq_only(series: pd.DataFrame) -> pd.DataFrame:
    return kind_only(series, OWN_EQ_KIND)


def intersect_kind_keys(left: pd.DataFrame, right: pd.DataFrame, kind: str) -> pd.DataFrame:
    a = kind_only(left, kind)
    b = kind_only(right, kind)
    if a.empty or b.empty or any(c not in a.columns or c not in b.columns for c in EDGE_KEYS):
        return pd.DataFrame(columns=EDGE_KEYS)
    return a[EDGE_KEYS].drop_duplicates().merge(b[EDGE_KEYS].drop_duplicates(), on=EDGE_KEYS, how="inner")


def intersect_cross_keys(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return intersect_kind_keys(left, right, "cross")


def cross_edge_stats(cross: pd.DataFrame) -> dict:
    if cross is None or cross.empty or "elasticity" not in cross.columns:
        e = np.array([])
    else:
        edges = cross.drop_duplicates(EDGE_KEYS) if all(c in cross.columns for c in EDGE_KEYS) else cross
        e = pd.to_numeric(edges["elasticity"], errors="coerce").to_numpy(dtype=float)
        e = e[np.isfinite(e)]
    n = int(e.size)
    return {
        "n_cross": n,
        "cross_mean": float(e.mean()) if n else np.nan,
        "share_abs_le_1": share_abs_le_1(e),
    }


def _edge_eval_row(dataset, outer_fold, eval_name, model, cross, *, models, n_native: int) -> dict:
    stats = cross_edge_stats(cross)
    n = stats["n_cross"]
    return {
        "dataset": dataset,
        "outer_fold": outer_fold,
        "eval": eval_name,
        "model": model,
        "models": ",".join(sorted(models)),
        "n_models": len(models),
        "n_cross": n,
        "n_cross_native": int(n_native),
        "cross_coverage": (n / n_native) if n_native else np.nan,
        "cross_mean": stats["cross_mean"],
        "share_abs_le_1": stats["share_abs_le_1"],
    }


def matched_edge_rows(series_by_model: dict[str, pd.DataFrame], outer_fold) -> tuple[list[dict], list[pd.DataFrame]]:
    """Native cross stats plus ICDN∩MLP / ICDN∩Ridge / ICDN∩OLS on the same edges."""
    rows: list[dict] = []
    long_parts: list[pd.DataFrame] = []
    if not series_by_model:
        return rows, long_parts
    dataset = None
    for df in series_by_model.values():
        if df is not None and len(df) and "dataset" in df.columns:
            dataset = df["dataset"].iloc[0]
            break
    models = list(series_by_model)
    native_n = {}
    for model, df in series_by_model.items():
        cross = cross_only(df)
        native_n[model] = int(len(cross[EDGE_KEYS].drop_duplicates())) if len(cross) else 0
        rows.append(
            _edge_eval_row(
                dataset, outer_fold, "native", model, cross,
                models=models, n_native=native_n[model],
            )
        )
    for eval_name, left, right in EDGE_COMPARISONS:
        if left not in series_by_model or right not in series_by_model:
            continue
        keys = intersect_cross_keys(series_by_model[left], series_by_model[right])
        pair = [left, right]
        for model in pair:
            cross = cross_only(series_by_model[model])
            matched = keys.merge(cross, on=EDGE_KEYS, how="inner") if len(keys) else cross.iloc[0:0]
            if len(matched):
                matched = matched.drop_duplicates(EDGE_KEYS)
            rows.append(
                _edge_eval_row(
                    dataset, outer_fold, eval_name, model, matched,
                    models=pair, n_native=native_n[model],
                )
            )
            if len(matched):
                part = matched.copy()
                part["eval"] = eval_name
                if "outer_fold" not in part.columns:
                    part["outer_fold"] = outer_fold
                long_parts.append(part)
    return rows, long_parts


def series_fold_sd_summary(
    stats: pd.DataFrame,
    kind: str,
    *,
    mean_name: str,
    sd_name: str,
) -> pd.DataFrame:
    """Mean of per-series fold SDs for one `kind`, grouped by dataset/eval/model."""
    if stats is None or stats.empty:
        return pd.DataFrame()
    keep = stats[stats["kind"].astype(str) == kind].copy()
    if keep.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [c for c in ("dataset", "eval", "model") if c in keep.columns]
    if not group_cols:
        group_cols = [c for c in ("eval", "model") if c in keep.columns]
    for key, part in keep.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(group_cols, key))
        sd = pd.to_numeric(part["fold_sd"], errors="coerce") if "fold_sd" in part.columns else pd.Series(dtype=float)
        finite = sd.notna()
        multi = part.loc[finite] if finite.any() else part.iloc[0:0]
        rec.update({
            "kind": kind,
            "n_series": int(len(part)),
            "n_series_sd": int(len(multi)),
            "n_folds": int(part["n_folds"].max()) if "n_folds" in part.columns else np.nan,
            mean_name: float(part["fold_mean"].mean()),
            sd_name: float(sd[finite].mean()) if finite.any() else np.nan,
            "share_sign_stable": float(multi["sign_stable"].mean()) if len(multi) and "sign_stable" in multi.columns else np.nan,
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def _eval_pair(eval_name) -> tuple[str, str] | None:
    name = str(eval_name)
    for label, left, right in EDGE_COMPARISONS:
        if name == label:
            return left, right
    return None


def complete_edge_stats(stats: pd.DataFrame, n_protocol_folds: int = N_FOLDS) -> pd.DataFrame:
    """Edges in every protocol fold; matched evals on the same keys and fold IDs.

    Native rows are filtered independently (`n_folds == N_FOLDS`). ICDN∩MLP /
    ICDN∩Ridge / ICDN∩OLS keep an edge only if both models have it in all
    `n_protocol_folds` folds and the fold-ID sets are identical.
    """
    if stats is None or stats.empty or "n_folds" not in stats.columns:
        return pd.DataFrame()
    full = stats.loc[pd.to_numeric(stats["n_folds"], errors="coerce") == n_protocol_folds].copy()
    if full.empty:
        return full
    if "eval" not in full.columns:
        return full
    parts = []
    for eval_name, part in full.groupby("eval", dropna=False):
        pair = _eval_pair(eval_name)
        if pair is None:
            parts.append(part)
            continue
        left, right = pair
        a = part[part["model"].astype(str) == left]
        b = part[part["model"].astype(str) == right]
        key_cols = [c for c in ("dataset", *EDGE_KEYS, "kind", "fold_ids") if c in part.columns]
        if a.empty or b.empty or not key_cols:
            continue
        common = (
            a[key_cols].drop_duplicates()
            .merge(b[key_cols].drop_duplicates(), on=key_cols, how="inner")
        )
        if common.empty:
            continue
        parts.append(pd.concat(
            [a.merge(common, on=key_cols, how="inner"), b.merge(common, on=key_cols, how="inner")],
            ignore_index=True,
        ))
    return pd.concat(parts, ignore_index=True) if parts else full.iloc[0:0]


def edge_presence_coverage(stats: pd.DataFrame, n_protocol_folds: int = N_FOLDS) -> pd.DataFrame:
    """How often a cross edge appears across outer folds (not the stability set)."""
    if stats is None or stats.empty:
        return pd.DataFrame()
    keep = stats[stats["kind"].astype(str) == "cross"].copy() if "kind" in stats.columns else stats.copy()
    if keep.empty:
        return pd.DataFrame()
    complete = complete_edge_stats(keep, n_protocol_folds=n_protocol_folds)
    group_cols = [c for c in ("dataset", "eval", "model") if c in keep.columns]
    if not group_cols:
        group_cols = [c for c in ("eval", "model") if c in keep.columns]
    rows = []
    for key, part in keep.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(group_cols, key))
        n = pd.to_numeric(part["n_folds"], errors="coerce") if "n_folds" in part.columns else pd.Series(dtype=float)
        freq = pd.to_numeric(part["presence_freq"], errors="coerce") if "presence_freq" in part.columns else n / n_protocol_folds
        rec.update({
            "kind": "cross",
            "n_protocol_folds": int(n_protocol_folds),
            "n_edges": int(len(part)),
            "n_edges_all_folds": int((n == n_protocol_folds).sum()) if len(n) else 0,
            "n_edges_at_least_2": int((n >= 2).sum()) if len(n) else 0,
            "mean_presence_freq": float(freq.mean()) if len(freq) else np.nan,
            "median_n_folds": float(n.median()) if len(n) else np.nan,
            "n_edges_stable": 0,
        })
        if len(complete):
            mask = pd.Series(True, index=complete.index)
            for col, val in rec.items():
                if col in group_cols and col in complete.columns:
                    mask &= complete[col].astype(str) == str(val)
            rec["n_edges_stable"] = int(mask.sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def edge_fold_sd_summary(stats: pd.DataFrame, n_protocol_folds: int = N_FOLDS) -> pd.DataFrame:
    """Mean of per-edge fold SDs on the complete matched (or native) cross set."""
    complete = complete_edge_stats(stats, n_protocol_folds=n_protocol_folds)
    out = series_fold_sd_summary(complete, "cross", mean_name="cross_mean", sd_name="cross_fold_sd")
    if len(out):
        out["n_protocol_folds"] = int(n_protocol_folds)
        out["stability_set"] = "all_folds"
    return out


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

def _series_meta(df: pd.DataFrame) -> dict:
    return {c: df[c].to_numpy() for c in OWN_META_COLS if c in df.columns}


def own_cross_series(own: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    """Long series from pairwise own/cross tables.

    kind=own: product-level mean_j β_ij^own, with n_partners / partner_set_id /
    n_train / n_val when present. kind=own_eq: the same β_ij^own at (store, i, j).
    kind=cross: β_ij^cross.
    """
    frames = []
    if own is not None and len(own):
        pi = own["product_code"] if "product_code" in own.columns else own["product_i"]
        rec = {
            "store_code": own["store_code"].to_numpy(),
            "product_i": pi.to_numpy(),
            "product_j": pi.to_numpy(),
            "kind": "own",
            "elasticity": own["own_elasticity"].to_numpy(),
        }
        rec.update(_series_meta(own))
        frames.append(pd.DataFrame(rec))
    if cross is not None and len(cross):
        base = {
            "store_code": cross["store_code"].to_numpy(),
            "product_i": cross["product_i"].to_numpy(),
            "product_j": cross["product_j"].to_numpy(),
        }
        meta = _series_meta(cross)
        if "own_elasticity" in cross.columns:
            eq = dict(base)
            eq["kind"] = OWN_EQ_KIND
            eq["elasticity"] = cross["own_elasticity"].to_numpy()
            eq.update(meta)
            frames.append(pd.DataFrame(eq))
        if "cross_elasticity" in cross.columns:
            cr = dict(base)
            cr["kind"] = "cross"
            cr["elasticity"] = cross["cross_elasticity"].to_numpy()
            cr.update(meta)
            frames.append(pd.DataFrame(cr))
    if not frames:
        return pd.DataFrame(columns=["store_code", "product_i", "product_j", "kind", "elasticity"])
    return normalize_series_keys(pd.concat(frames, ignore_index=True))


def own_product_diagnostics(fold_long: pd.DataFrame) -> pd.DataFrame:
    """Product-level own fold stats plus whether J_i was stable across folds."""
    own = kind_only(fold_long, "own")
    if own.empty:
        return pd.DataFrame()
    keys = [k for k in SERIES_KEYS if k in own.columns]
    grouped = own.groupby(keys, as_index=False)
    agg = dict(
        n_folds=("elasticity", "size"),
        fold_mean=("elasticity", "mean"),
        fold_sd=("elasticity", "std"),
        sign_stable=("elasticity", _sign_stable),
    )
    if "partner_set_id" in own.columns:
        agg["n_partner_sets"] = ("partner_set_id", "nunique")
    if "n_partners" in own.columns:
        agg["n_partners_mean"] = ("n_partners", "mean")
        agg["n_partners_min"] = ("n_partners", "min")
        agg["n_partners_max"] = ("n_partners", "max")
    if "n_train" in own.columns:
        agg["n_train_mean"] = ("n_train", "mean")
    if "n_val" in own.columns:
        agg["n_val_mean"] = ("n_val", "mean")
    out = grouped.agg(**agg)
    if "n_partner_sets" in out.columns:
        out["partner_set_stable"] = out["n_partner_sets"] <= 1
    return out


def fixed_partner_product_own(fold_long: pd.DataFrame) -> pd.DataFrame:
    """Product-level own using only (i, j) equations present in every outer fold."""
    eq = own_eq_only(fold_long)
    if eq.empty or "outer_fold" not in eq.columns:
        return pd.DataFrame()
    n_folds = eq["outer_fold"].nunique()
    counts = eq.groupby(EDGE_KEYS, as_index=False)["outer_fold"].nunique()
    common = counts.loc[counts["outer_fold"] == n_folds, EDGE_KEYS]
    if common.empty:
        return pd.DataFrame()
    kept = common.merge(eq, on=EDGE_KEYS, how="inner")
    gcols = [c for c in ("dataset", "model", "store_code", "product_i", "outer_fold") if c in kept.columns]
    agg = dict(
        elasticity=("elasticity", "mean"),
        n_partners=("product_j", "nunique"),
    )
    if "n_train" in kept.columns:
        agg["n_train"] = ("n_train", "mean")
    if "n_val" in kept.columns:
        agg["n_val"] = ("n_val", "mean")
    out = kept.groupby(gcols, as_index=False).agg(**agg)
    set_ids = (
        kept.groupby(gcols, as_index=False)["product_j"]
        .agg(lambda s: partner_set_id(s))
        .rename(columns={"product_j": "partner_set_id"})
    )
    out = out.merge(set_ids, on=gcols, how="left")
    out["product_j"] = out["product_i"]
    out["kind"] = "own_fixed"
    return normalize_series_keys(out)


def product_own_on_keys(eq: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Mean of equation-level own over a fixed (store, i, j) key set, per product."""
    if keys is None or keys.empty:
        return pd.DataFrame()
    sub = keys.merge(own_eq_only(eq), on=EDGE_KEYS, how="inner")
    if sub.empty:
        return sub
    gcols = [c for c in ("dataset", "model", "store_code", "product_i", "outer_fold") if c in sub.columns]
    agg = dict(
        elasticity=("elasticity", "mean"),
        n_partners=("product_j", "nunique"),
    )
    if "n_train" in sub.columns:
        agg["n_train"] = ("n_train", "mean")
    if "n_val" in sub.columns:
        agg["n_val"] = ("n_val", "mean")
    out = sub.groupby(gcols, as_index=False).agg(**agg)
    out["product_j"] = out["product_i"]
    out["kind"] = "own"
    return normalize_series_keys(out)


def _own_eval_row(dataset, outer_fold, eval_name, model, values, *, models, n_native, n_partners=None) -> dict:
    e = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    e = e[np.isfinite(e)]
    n = int(e.size)
    return {
        "dataset": dataset,
        "outer_fold": outer_fold,
        "eval": eval_name,
        "model": model,
        "models": ",".join(sorted(models)),
        "n_models": len(models),
        "n_own": n,
        "n_own_native": int(n_native),
        "own_coverage": (n / n_native) if n_native else np.nan,
        "own_mean": float(e.mean()) if n else np.nan,
        "n_partners_mean": (
            float(pd.to_numeric(pd.Series(n_partners), errors="coerce").mean())
            if n_partners is not None and len(pd.Series(n_partners))
            else np.nan
        ),
    }


def matched_own_rows(series_by_model: dict[str, pd.DataFrame], outer_fold) -> tuple[list[dict], list[pd.DataFrame]]:
    """Native product own / own_eq, plus OLS∩Ridge on the same equation-level partners."""
    rows: list[dict] = []
    long_parts: list[pd.DataFrame] = []
    if not series_by_model:
        return rows, long_parts
    dataset = None
    for df in series_by_model.values():
        if df is not None and len(df) and "dataset" in df.columns:
            dataset = df["dataset"].iloc[0]
            break
    models = list(series_by_model)
    native_own_n, native_eq_n = {}, {}
    for model, df in series_by_model.items():
        own = kind_only(df, "own")
        eq = own_eq_only(df)
        native_own_n[model] = int(len(own.drop_duplicates(["store_code", "product_i"]))) if len(own) else 0
        native_eq_n[model] = int(len(eq[EDGE_KEYS].drop_duplicates())) if len(eq) else 0
        rows.append(_own_eval_row(
            dataset, outer_fold, "native_own", model,
            own["elasticity"] if len(own) else [],
            models=models, n_native=native_own_n[model],
            n_partners=own["n_partners"] if "n_partners" in own.columns and len(own) else None,
        ))
        if native_eq_n[model]:
            rows.append(_own_eval_row(
                dataset, outer_fold, "native_own_eq", model,
                eq["elasticity"] if len(eq) else [],
                models=models, n_native=native_eq_n[model],
            ))
    if "ols" in series_by_model and "ridge" in series_by_model:
        keys = intersect_kind_keys(series_by_model["ols"], series_by_model["ridge"], OWN_EQ_KIND)
        pair = ["ols", "ridge"]
        if len(keys):
            for model in pair:
                eq = own_eq_only(series_by_model[model])
                matched = keys.merge(eq, on=EDGE_KEYS, how="inner")
                if len(matched):
                    matched = matched.drop_duplicates(EDGE_KEYS)
                rows.append(_own_eval_row(
                    dataset, outer_fold, "matched_own_eq_ols_ridge", model,
                    matched["elasticity"] if len(matched) else [],
                    models=pair, n_native=native_eq_n[model],
                ))
                if len(matched):
                    part = matched.copy()
                    part["eval"] = "matched_own_eq_ols_ridge"
                    if "outer_fold" not in part.columns:
                        part["outer_fold"] = outer_fold
                    long_parts.append(part)
                prod = product_own_on_keys(series_by_model[model], keys)
                rows.append(_own_eval_row(
                    dataset, outer_fold, "matched_own_product_ols_ridge", model,
                    prod["elasticity"] if len(prod) else [],
                    models=pair, n_native=native_own_n[model],
                    n_partners=prod["n_partners"] if len(prod) and "n_partners" in prod.columns else None,
                ))
                if len(prod):
                    part = prod.copy()
                    part["eval"] = "matched_own_product_ols_ridge"
                    if "outer_fold" not in part.columns:
                        part["outer_fold"] = outer_fold
                    long_parts.append(part)
    return rows, long_parts


def elasticity_long(df: pd.DataFrame) -> pd.DataFrame:
    """Week-level own/cross rows with observed_i / observed_j flags."""
    out = df.copy()
    rename = {}
    if "product_code" in out.columns and "product_i" not in out.columns:
        rename["product_code"] = "product_i"
    if "competitor" in out.columns and "product_j" not in out.columns:
        rename["competitor"] = "product_j"
    if "week_id" not in out.columns:
        for cand in ("period",):
            if cand in out.columns:
                rename[cand] = "week_id"
    if rename:
        out = out.rename(columns=rename)
    keep = ["store_code", "week_id", "product_i", "product_j", "kind", "elasticity"]
    extra = [c for c in ("observed_i", "observed_j") if c in out.columns]
    out = out[keep + extra].copy()
    if "observed_i" not in out.columns:
        out["observed_i"] = True
    if "observed_j" not in out.columns:
        out["observed_j"] = True
    return normalize_series_keys(out)


def summary_series(summary: pd.DataFrame) -> pd.DataFrame:
    """Keep the series keys from an already-long elasticity table (MLP / ICDN)."""
    cols = ["store_code", "product_i", "product_j", "kind", "elasticity"]
    extra = [c for c in ("n_val", "n_train", "n_partners", "n_obs") if c in summary.columns]
    out = summary[cols + extra].copy()
    if "n_obs" in out.columns and "n_val" not in out.columns:
        out = out.rename(columns={"n_obs": "n_val"})
    return normalize_series_keys(out)


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
    out = grouped.agg(
        n_present="size",
        mean="mean",
        sd="std",
        q025=lambda s: s.quantile(0.025),
        q50=lambda s: s.quantile(0.5),
        q975=lambda s: s.quantile(0.975),
        share_positive=lambda s: float(np.mean(np.asarray(s, dtype=float) > 0)),
        share_negative=lambda s: float(np.mean(np.asarray(s, dtype=float) < 0)),
    )
    out["ci_width"] = out["q975"] - out["q025"]
    return out

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
            "median_sd": float(part["sd"].median()) if "sd" in part.columns else np.nan,
        })
    return pd.DataFrame(rows)


def _sign_stable(s: pd.Series) -> float:
    v = s.to_numpy(dtype=float)
    v = v[np.isfinite(v) & (v != 0)]
    if v.size == 0:
        return np.nan
    return float(np.all(np.sign(v) == np.sign(v[0])))


def _dominant_sign_share(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v) & (v != 0)]
    if v.size == 0:
        return np.nan
    pos = float(np.mean(np.sign(v) == 1))
    return max(pos, 1.0 - pos)


def _fold_range(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(v.max() - v.min())


def _join_fold_ids(s: pd.Series) -> str:
    toks = {str(x) for x in s.dropna()}
    return ",".join(sorted(toks, key=lambda x: (not str(x).isdigit(), int(x) if str(x).isdigit() else str(x))))


def fold_series_stats(fold_long: pd.DataFrame, extra_keys: tuple[str, ...] = ()) -> pd.DataFrame:
    """Across outer folds: mean, SD, range, and sign stability.

    `n_folds` is unique `outer_fold` IDs when that column exists. `fold_ids` is
    the sorted ID set so a matched comparison can require the same temporal
    base. `presence_freq` is n_folds / N_FOLDS.
    """
    data = normalize_series_keys(fold_long)
    if data.empty:
        return pd.DataFrame()
    keys = [k for k in list(SERIES_KEYS) + list(extra_keys) if k in data.columns]
    if "outer_fold" in data.columns:
        data = data.drop_duplicates([*keys, "outer_fold"])
    grouped = data.groupby(keys, as_index=False)
    agg = dict(
        n_folds=("elasticity", "size"),
        fold_mean=("elasticity", "mean"),
        fold_sd=("elasticity", "std"),
        fold_range=("elasticity", _fold_range),
        sign_stable=("elasticity", _sign_stable),
        dominant_sign_share=("elasticity", _dominant_sign_share),
    )
    if "outer_fold" in data.columns:
        agg["fold_ids"] = ("outer_fold", _join_fold_ids)
    out = grouped.agg(**agg)
    out["presence_freq"] = pd.to_numeric(out["n_folds"], errors="coerce") / float(N_FOLDS)
    return out


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
    out["zero_fold_sd_flag"] = (pd.to_numeric(out["fold_sd"], errors="coerce") == 0) | (
        pd.to_numeric(out["fold_sd"], errors="coerce").isna()
    )
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

def compute_row(
    n_parameters,
    seconds,
    used_gpu=False,
    study=None,
    *,
    tuning_seconds=None,
    fit_seconds=None,
    n_attempted=None,
    n_successful=None,
) -> dict:
    """Columns appended to kfold.csv / bootstrap.csv."""
    total = float(seconds)
    tun = 0.0 if tuning_seconds is None else float(tuning_seconds)
    fit = total if fit_seconds is None else float(fit_seconds)
    row = dict(
        n_parameters=int(n_parameters) if n_parameters is not None else np.nan,
        training_seconds=total,
        tuning_seconds=tun,
        fit_seconds=fit,
        gpu_hours=gpu_hours(seconds, used_gpu),
        **trial_counts(study),
    )
    if n_attempted is not None:
        row["n_attempted"] = int(n_attempted)
    if n_successful is not None:
        row["n_successful"] = int(n_successful)
    return row