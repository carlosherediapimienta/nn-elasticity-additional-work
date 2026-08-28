"""Derived paper tables: matched comparisons, bootstrap audit, and CSV backfill.

Writers in the experiment loop emit native files. This module adds fields that
can be computed from those files and builds the cross-model tables used in
`notebooks/analysis.ipynb`. A full re-run still fills week-level ICDN/MLP
elasticities, the tuning/fit split, spline support, and skipped-fold rows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmarks.constants import (
    INNER_TRAIN_FRAC,
    SEED,
)
from src.benchmarks.predict import (
    CELL_KEYS,
    COMPARE_MODELS,
    LINEAR_MODELS,
    NEURAL_MODELS,
    SERIES_KEYS,
    boot_fold_ratio,
    fold_series_stats,
    metrics_on_keys,
    native_metrics,
    normalize_cell_keys,
    normalize_series_keys,
    predicted_cell_keys,
)
from src.benchmarks.protocol import n_fit_periods, save_table

PAIRS = (
    ("icdn", "mlp"),
    ("icdn", "ridge"),
    ("icdn", "ols"),
    ("mlp", "ridge"),
    ("mlp", "ols"),
    ("ridge", "ols"),
)
EDGE_PAIRS = (("icdn", "mlp"), ("icdn", "ridge"), ("icdn", "ols"))

KFOLD_LEAD = [
    "dataset", "model", "fold",
    "mae_val", "rmse_val", "r2_val",
    "n_cells", "n_validation_cells", "n_val_cells", "prediction_coverage",
    "n_own", "n_cross", "own_mean", "cross_mean",
    "n_parameters", "completed_trials", "pruned_trials", "failed_trials",
    "tuning_seconds", "fit_seconds", "training_seconds", "gpu_hours",
]
BOOT_LEAD = [
    "dataset", "model", "boot",
    "mae_val", "rmse_val", "r2_val",
    "n_cells", "n_own", "n_cross", "own_mean", "cross_mean",
    "n_parameters", "fit_seconds", "training_seconds",
    "n_attempted", "n_successful",
]
RUN_MANIFEST_COLS = [
    "dataset", "model", "stage", "outer_fold",
    "train_week_min", "train_week_max",
    "early_stop_week_min", "early_stop_week_max",
    "validation_week_min", "validation_week_max",
    "n_train_periods", "n_val_periods",
    "frozen_products_hash", "config_hash", "seed",
]
BOOT_MANIFEST_COLS = [
    "dataset", "model", "bootstrap_id", "source_block_id",
    "source_week_start", "source_week_end",
    "bootstrap_week_start", "bootstrap_week_end",
    "validation_week_min", "seed", "n_sampled_periods", "period_overlap_flag",
]
FAILURE_COLS = [
    "dataset", "model", "stage", "fold_or_boot_id",
    "error_type", "error_message", "n_attempted", "n_successful",
]
SPLINE_COLS = [
    "dataset", "model", "fold_or_boot_id",
    "n_observed_cells", "n_below_support", "n_above_support", "share_outside_support",
]


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    #print("  wrote", path)


def _reorder(df: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def _ensure_cols(df: pd.DataFrame, cols: list[str], fill=np.nan) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = fill
    return out


def config_hash(model: str, extras: dict | None = None) -> str:
    from src.benchmarks import constants as C

    payload = {
        "model": model,
        "N_FOLDS": C.N_FOLDS,
        "MIN_TRAIN_FRAC": C.MIN_TRAIN_FRAC,
        "N_INNER_FOLDS": C.N_INNER_FOLDS,
        "MIN_INNER_FRAC": C.MIN_INNER_FRAC,
        "INNER_TRAIN_FRAC": C.INNER_TRAIN_FRAC,
        "HOLDOUT_TRAIN_FRAC": C.HOLDOUT_TRAIN_FRAC,
        "BLOCK_SIZE": C.BLOCK_SIZE,
        "SEED": C.SEED,
        "N_TRIALS_MLP": C.N_TRIALS_MLP,
        "N_TRIALS_ICDN": C.N_TRIALS_ICDN,
        "N_BOOT_LINEAR": C.N_BOOT_LINEAR,
        "N_BOOT_MLP": C.N_BOOT_MLP,
        "N_BOOT_ICDN": C.N_BOOT_ICDN,
        "extras": extras or {},
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def products_hash(products) -> str:
    toks = sorted({str(p) for p in products})
    return hashlib.sha1("|".join(toks).encode()).hexdigest()[:12]


def _parquet_frame(panel_dir: Path, columns: list[str]) -> pd.DataFrame | None:
    for name in ("m5_icdn_panel.parquet", "1c_icdn_panel.parquet", "dominick_icdn_panel.parquet"):
        path = panel_dir / name
        if path.exists():
            return pd.read_parquet(path, columns=columns)
    return None


def _frozen_hash(panel_dir: Path) -> str:
    frame = _parquet_frame(panel_dir, ["product_code"])
    if frame is None:
        return ""
    return products_hash(frame["product_code"].astype(str).unique())


def _panel_weeks(panel_dir: Path) -> np.ndarray:
    frame = _parquet_frame(panel_dir, ["week_id"])
    if frame is None:
        return np.array([], dtype=int)
    return np.sort(pd.to_numeric(frame["week_id"], errors="coerce").dropna().unique().astype(int))


def record_failure(panel_dir: Path, **row) -> None:
    path = Path(panel_dir) / "run_failures.csv"
    rec = {c: row.get(c, np.nan) for c in FAILURE_COLS}
    rec.update({k: v for k, v in row.items() if k not in rec})
    extra = pd.DataFrame([rec])
    if path.exists():
        extra = pd.concat([pd.read_csv(path), extra], ignore_index=True)
        extra = extra.drop_duplicates(
            subset=["dataset", "model", "stage", "fold_or_boot_id", "error_type", "error_message"],
            keep="last",
        )
    _write(_reorder(extra, FAILURE_COLS), path)


def append_run_manifest(panel_dir: Path, row: dict) -> None:
    path = Path(panel_dir) / "run_manifest.csv"
    rec = {c: row.get(c, np.nan) for c in RUN_MANIFEST_COLS}
    extra = pd.DataFrame([rec])
    if path.exists():
        extra = pd.concat([pd.read_csv(path), extra], ignore_index=True)
        extra = extra.drop_duplicates(subset=["dataset", "model", "stage", "outer_fold"], keep="last")
    _write(_reorder(extra, RUN_MANIFEST_COLS), path)


def run_manifest_row(
    *,
    dataset: str,
    model: str,
    stage: str,
    outer_fold,
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    products,
    early_stop: pd.DataFrame | None = None,
    extras: dict | None = None,
    seed: int = SEED,
) -> dict:
    train_weeks = pd.to_numeric(train_raw["week_id"], errors="coerce").dropna().astype(int)
    val_weeks = pd.to_numeric(val_raw["week_id"], errors="coerce").dropna().astype(int)
    es_min = es_max = np.nan
    if early_stop is not None and len(early_stop):
        es = pd.to_numeric(early_stop["week_id"], errors="coerce").dropna().astype(int)
        if len(es):
            es_min, es_max = int(es.min()), int(es.max())
    return {
        "dataset": dataset,
        "model": model,
        "stage": stage,
        "outer_fold": outer_fold,
        "train_week_min": int(train_weeks.min()) if len(train_weeks) else np.nan,
        "train_week_max": int(train_weeks.max()) if len(train_weeks) else np.nan,
        "early_stop_week_min": es_min,
        "early_stop_week_max": es_max,
        "validation_week_min": int(val_weeks.min()) if len(val_weeks) else np.nan,
        "validation_week_max": int(val_weeks.max()) if len(val_weeks) else np.nan,
        "n_train_periods": int(train_weeks.nunique()) if len(train_weeks) else np.nan,
        "n_val_periods": int(val_weeks.nunique()) if len(val_weeks) else np.nan,
        "frozen_products_hash": products_hash(products),
        "config_hash": config_hash(model, extras),
        "seed": int(seed),
    }


def append_spline_row(panel_dir: Path, row: dict) -> None:
    path = Path(panel_dir) / "spline_support_diagnostics.csv"
    rec = {c: row.get(c, np.nan) for c in SPLINE_COLS}
    extra = pd.DataFrame([rec])
    if path.exists():
        extra = pd.concat([pd.read_csv(path), extra], ignore_index=True)
        extra = extra.drop_duplicates(subset=["dataset", "model", "fold_or_boot_id"], keep="last")
    _write(_reorder(extra, SPLINE_COLS), path)


def spline_support_row(model, panel: pd.DataFrame) -> dict:
    """Count observed log-prices outside the ICDN cubic spline knot range."""
    wide, _ = model._prepare(panel)
    splines = model._model.price_splines
    n = model.layout.n_products
    lo = (splines.knots.min(dim=1).values * splines.scale + splines.shift).detach().cpu().numpy()
    hi = (splines.knots.max(dim=1).values * splines.scale + splines.shift).detach().cpu().numpy()
    x = wide[[f"log_price_{i}" for i in range(n)]].to_numpy(dtype=np.float64)
    mask = wide[[f"obs_mask_{i}" for i in range(n)]].to_numpy(dtype=np.float64) > 0
    n_obs = int(mask.sum())
    n_below = int((mask & (x < lo)).sum())
    n_above = int((mask & (x > hi)).sum())
    share = np.nan if n_obs == 0 else (n_below + n_above) / n_obs
    return {
        "n_observed_cells": n_obs,
        "n_below_support": n_below,
        "n_above_support": n_above,
        "share_outside_support": float(share) if n_obs else np.nan,
    }


def enrich_kfold_csv(model_dir: Path) -> None:
    path = model_dir / "kfold.csv"
    kfold = _read(path)
    if kfold is None or kfold.empty:
        return
    rows = []
    model = model_dir.name
    for _, row in kfold.iterrows():
        rec = row.to_dict()
        rec["model"] = rec.get("model", model)
        fold = rec.get("fold")
        grid = _read(model_dir / f"fold{fold}_pred_cells.csv")
        if grid is not None and len(grid):
            m = native_metrics(grid)
            rec["mae_val"] = m["mae_val"]
            rec["rmse_val"] = m["rmse_val"]
            rec["r2_val"] = m["r2_val"]
            rec["n_cells"] = m["n_cells"]
            rec["n_val_cells"] = m["n_val_cells"]
            rec["n_validation_cells"] = m["n_val_cells"]
            rec["prediction_coverage"] = m["prediction_coverage"]
        else:
            rec.setdefault("r2_val", rec.get("r2_native", np.nan))
            rec.setdefault("n_val_cells", rec.get("n_cells", np.nan))
            rec.setdefault("n_validation_cells", rec.get("n_val_cells", np.nan))
            rec.setdefault("prediction_coverage", np.nan)
        if "tuning_seconds" not in rec or pd.isna(rec.get("tuning_seconds")):
            rec["tuning_seconds"] = 0.0 if model in LINEAR_MODELS else rec.get("tuning_seconds", np.nan)
        if "fit_seconds" not in rec or pd.isna(rec.get("fit_seconds")):
            rec["fit_seconds"] = rec.get("training_seconds", np.nan)
        rows.append(rec)
    _write(_reorder(pd.DataFrame(rows), KFOLD_LEAD), path)


def enrich_pred_ij(model_dir: Path) -> None:
    kfold = _read(model_dir / "kfold.csv")
    dataset = kfold["dataset"].iloc[0] if kfold is not None and len(kfold) else ""
    ok_folds = set(kfold["fold"].astype(str)) if kfold is not None and len(kfold) and "fold" in kfold.columns else None
    paths = list(model_dir.glob("fold*_pred_ij.csv"))
    holdout = model_dir / "holdout_pred_ij.csv"
    if holdout.exists():
        paths.append(holdout)
    for path in paths:
        df = _read(path)
        if df is None or df.empty:
            continue
        if "holdout" in path.name:
            fold = "holdout"
        else:
            fold = path.name.removeprefix("fold").removesuffix("_pred_ij.csv")
            if ok_folds is not None and str(fold) not in ok_folds:
                continue
        if "dataset" not in df.columns:
            df.insert(0, "dataset", dataset)
        if "outer_fold" not in df.columns:
            df.insert(1, "outer_fold", fold)
        _write(df, path)


def enrich_holdout_and_replicates(model_dir: Path) -> None:
    for name in ("holdout_elasticities.csv", "bootstrap_replicates.csv", "kfold_series.csv"):
        path = model_dir / name
        df = _read(path)
        if df is None or df.empty:
            continue
        changed = False
        for col in ("n_val", "n_train", "n_partners"):
            if col not in df.columns:
                df[col] = np.nan
                changed = True
        if changed:
            _write(df, path)


def enrich_bootstrap_csv(model_dir: Path) -> None:
    path = model_dir / "bootstrap.csv"
    boots = _read(path)
    if boots is None or boots.empty:
        return
    if "r2_val" not in boots.columns:
        boots["r2_val"] = boots["r2_native"] if "r2_native" in boots.columns else np.nan
    n_ok = int(boots["boot"].nunique()) if "boot" in boots.columns else len(boots)
    payload = _read_bootstrap_json(model_dir)
    n_att = n_ok
    if payload is not None:
        n_att = max(n_ok, int(len(payload.get("replicates") or [])))
    if "n_successful" not in boots.columns:
        boots["n_successful"] = n_ok
    if "n_attempted" not in boots.columns:
        boots["n_attempted"] = n_att
    if "tuning_seconds" not in boots.columns:
        boots["tuning_seconds"] = 0.0 if model_dir.name in LINEAR_MODELS else np.nan
    if "fit_seconds" not in boots.columns:
        boots["fit_seconds"] = boots["training_seconds"] if "training_seconds" in boots.columns else np.nan
    _write(_reorder(boots, BOOT_LEAD), path)


def _read_bootstrap_json(model_dir: Path) -> dict | None:
    path = model_dir / "bootstrap_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def enrich_bootstrap_ci(model_dir: Path) -> None:
    ci_path = model_dir / "bootstrap_series_ci.csv"
    ci = _read(ci_path)
    reps = _read(model_dir / "bootstrap_replicates.csv")
    if ci is None or ci.empty:
        return
    ci = normalize_series_keys(ci)
    if "ci_width" not in ci.columns and {"q025", "q975"} <= set(ci.columns):
        ci["ci_width"] = ci["q975"] - ci["q025"]
    if reps is not None and len(reps) and (
        "share_positive" not in ci.columns or ci["share_positive"].isna().all()
    ):
        signs = (
            normalize_series_keys(reps)
            .groupby([c for c in SERIES_KEYS if c in reps.columns], as_index=False)["elasticity"]
            .agg(
                share_positive=lambda s: float(np.mean(np.asarray(s, dtype=float) > 0)),
                share_negative=lambda s: float(np.mean(np.asarray(s, dtype=float) < 0)),
            )
        )
        drop = [c for c in ("share_positive", "share_negative") if c in ci.columns]
        ci = ci.drop(columns=drop).merge(signs, on=[c for c in SERIES_KEYS if c in ci.columns and c in signs.columns], how="left")
    _write(ci, ci_path)
    matched = ci[ci["matched"] == True] if "matched" in ci.columns else ci
    _write(matched, model_dir / "bootstrap_matched.csv")
    if matched is None or matched.empty:
        return
    agg = {
        "n_series": ("kind", "size"),
        "mean_freq": ("freq", "mean"),
        "mean_conditional": ("mean", "mean"),
        "median_sd": ("sd", "median"),
    }
    if "ci_width" in matched.columns:
        agg["median_width"] = ("ci_width", "median")
    extra = matched.groupby("kind", as_index=False).agg(**agg)
    glob = _read(model_dir / "bootstrap_matched_global.csv")
    if glob is None or glob.empty:
        glob = extra
    else:
        glob = glob.drop(columns=["median_sd"], errors="ignore").merge(
            extra[["kind", "median_sd"]], on="kind", how="left"
        )
        if "median_width" in extra.columns and "median_width" not in glob.columns:
            glob = glob.merge(extra[["kind", "median_width"]], on="kind", how="left")
    _write(glob, model_dir / "bootstrap_matched_global.csv")


def enrich_series_stats(model_dir: Path) -> None:
    series = _read(model_dir / "kfold_series.csv")
    if series is None or series.empty:
        return
    stats = fold_series_stats(series)
    _write(stats, model_dir / "kfold_series_stats.csv")


def enrich_boot_fold_ratio(model_dir: Path) -> None:
    matched = _read(model_dir / "bootstrap_matched.csv")
    stats = _read(model_dir / "kfold_series_stats.csv")
    if matched is None or stats is None:
        return
    ratio = boot_fold_ratio(matched, stats)
    _write(ratio, model_dir / "boot_fold_ratio.csv")


def matched_prediction_metrics(panel_dir: Path) -> pd.DataFrame:
    rows = []
    folds: set[str] = set()
    for model in COMPARE_MODELS:
        for path in (panel_dir / model).glob("fold*_pred_cells.csv"):
            folds.add(path.name.removeprefix("fold").removesuffix("_pred_cells.csv"))
        if (panel_dir / model / "holdout_pred_cells.csv").exists():
            folds.add("holdout")
    for tag in sorted(folds, key=lambda x: int(x) if str(x).isdigit() else 10**9):
        grids = {}
        for model in COMPARE_MODELS:
            name = "holdout_pred_cells.csv" if tag == "holdout" else f"fold{tag}_pred_cells.csv"
            df = _read(panel_dir / model / name)
            if df is not None and len(df):
                grids[model] = normalize_cell_keys(df)
        dataset = next(
            (g["dataset"].iloc[0] for g in grids.values() if "dataset" in g.columns),
            panel_dir.parent.name,
        )
        for a, b in PAIRS:
            if a not in grids or b not in grids:
                continue
            keys = predicted_cell_keys(grids[a]).merge(predicted_cell_keys(grids[b]), on=CELL_KEYS, how="inner")
            ma = metrics_on_keys(grids[a], keys)
            mb = metrics_on_keys(grids[b], keys)
            rows.append({
                "dataset": dataset,
                "fold": tag,
                "model_a": a,
                "model_b": b,
                "n_common_cells": int(len(keys)),
                "mae_a": ma["mae_native"],
                "mae_b": mb["mae_native"],
                "rmse_a": ma["rmse_native"],
                "rmse_b": mb["rmse_native"],
                "r2_a": ma["r2_native"],
                "r2_b": mb["r2_native"],
                "delta_mae": ma["mae_native"] - mb["mae_native"],
                "delta_rmse": ma["rmse_native"] - mb["rmse_native"],
                "delta_r2": ma["r2_native"] - mb["r2_native"],
            })
    out = pd.DataFrame(rows)
    if len(out):
        _write(out, panel_dir / "matched_prediction_metrics.csv")
    return out


def matched_elasticity_series(panel_dir: Path) -> pd.DataFrame:
    rows = []
    for filename, fold_col, default_fold in (
        ("kfold_series.csv", "outer_fold", None),
        ("holdout_elasticities.csv", None, "holdout"),
    ):
        series = {}
        for model in COMPARE_MODELS:
            df = _read(panel_dir / model / filename)
            if df is not None and len(df):
                series[model] = normalize_series_keys(df)
        if len(series) < 2:
            continue
        if fold_col is None:
            folds = [default_fold]
        else:
            folds = sorted(
                {str(x) for df in series.values() if fold_col in df.columns for x in df[fold_col].unique()},
                key=lambda x: int(x) if x.isdigit() else 10**9,
            )
        for tag in folds:
            by_model = {}
            for model, df in series.items():
                part = df if fold_col is None else df[df[fold_col].astype(str) == str(tag)]
                if len(part):
                    by_model[model] = part
            for a, b in EDGE_PAIRS:
                if a not in by_model or b not in by_model:
                    continue
                keys = ["store_code", "product_i", "product_j", "kind"]
                left = by_model[a][keys + ["elasticity"] + (["dataset"] if "dataset" in by_model[a].columns else [])]
                left = left.rename(columns={"elasticity": "elasticity_a"})
                right = by_model[b][keys + ["elasticity"]].rename(columns={"elasticity": "elasticity_b"})
                merged = left.merge(right, on=keys, how="inner")
                if merged.empty:
                    continue
                if "dataset" not in merged.columns:
                    merged["dataset"] = panel_dir.parent.name
                merged["fold"] = tag
                merged["model_a"] = a
                merged["model_b"] = b
                merged["absolute_difference"] = (merged["elasticity_a"] - merged["elasticity_b"]).abs()
                sa = np.sign(pd.to_numeric(merged["elasticity_a"], errors="coerce"))
                sb = np.sign(pd.to_numeric(merged["elasticity_b"], errors="coerce"))
                merged["same_sign"] = (sa == sb) & (sa != 0)
                rows.append(merged)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        cols = [
            "dataset", "fold", "model_a", "model_b",
            "store_code", "product_i", "product_j", "kind",
            "elasticity_a", "elasticity_b", "absolute_difference", "same_sign",
        ]
        _write(out[[c for c in cols if c in out.columns]], panel_dir / "matched_elasticity_series.csv")
    return out


def fold_in_boot_ci(model_dir: Path) -> None:
    series = _read(model_dir / "kfold_series.csv")
    ci = _read(model_dir / "bootstrap_series_ci.csv")
    if series is None or ci is None or series.empty or ci.empty:
        return
    keys = [c for c in SERIES_KEYS if c in series.columns and c in ci.columns]
    keep_ci = keys + [c for c in ("q025", "q975", "freq", "matched") if c in ci.columns]
    out = normalize_series_keys(series).merge(normalize_series_keys(ci)[keep_ci], on=keys, how="inner")
    out = out.rename(columns={"elasticity": "fold_elasticity"})
    out["inside_boot_ci"] = (out["fold_elasticity"] >= out["q025"]) & (out["fold_elasticity"] <= out["q975"])
    keep = [c for c in keys + ["outer_fold", "fold_elasticity", "q025", "q975", "inside_boot_ci"] if c in out.columns]
    _write(out[keep], model_dir / "fold_in_boot_ci.csv")


def flatten_bootstrap_manifest(model_dir: Path, dataset: str) -> pd.DataFrame:
    payload = _read_bootstrap_json(model_dir)
    if payload is None:
        return pd.DataFrame(columns=BOOT_MANIFEST_COLS)
    rows = []
    seed = payload.get("seed", SEED)
    for rec in payload.get("replicates", []):
        bid = rec.get("bootstrap_id")
        val = rec.get("val") or {}
        src_val = val.get("source_weeks") or []
        val_min = min(src_val) if src_val else np.nan
        n_sampled = rec.get("n_sampled_periods")
        seen: set[int] = set()
        overlap = False
        blocks = rec.get("blocks") or []
        for block in blocks:
            src = [int(w) for w in block.get("source_weeks") or []]
            if seen.intersection(src):
                overlap = True
            seen.update(src)
        for i, block in enumerate(blocks):
            src = [int(w) for w in block.get("source_weeks") or []]
            boot = [int(w) for w in block.get("bootstrap_order") or []]
            if not src:
                continue
            rows.append({
                "dataset": dataset,
                "model": model_dir.name,
                "bootstrap_id": bid,
                "source_block_id": int(block.get("block_id", i)),
                "source_week_start": min(src),
                "source_week_end": max(src),
                "bootstrap_week_start": min(boot) if boot else np.nan,
                "bootstrap_week_end": max(boot) if boot else np.nan,
                "validation_week_min": val_min,
                "seed": seed,
                "n_sampled_periods": n_sampled,
                "period_overlap_flag": overlap,
            })
    return pd.DataFrame(rows)


def rebuild_run_manifest(panel_dir: Path) -> pd.DataFrame:
    weeks = _panel_weeks(panel_dir)
    frozen = _frozen_hash(panel_dir)
    rows = []
    for model in COMPARE_MODELS:
        mdir = panel_dir / model
        files = sorted(mdir.glob("fold*_pred_cells.csv")) + sorted(mdir.glob("holdout_pred_cells.csv"))
        kfold = _read(mdir / "kfold.csv")
        ok_folds = set()
        if kfold is not None and len(kfold) and "fold" in kfold.columns:
            ok_folds = set(kfold["fold"].astype(str))
        for path in files:
            df = _read(path)
            if df is None or df.empty:
                continue
            tag = "holdout" if "holdout" in path.name else path.name.removeprefix("fold").removesuffix("_pred_cells.csv")
            if tag != "holdout" and ok_folds and str(tag) not in ok_folds:
                continue
            val = pd.to_numeric(df["week_id"], errors="coerce").dropna().astype(int)
            val_min, val_max = int(val.min()), int(val.max())
            if len(weeks):
                train = weeks[weeks < val_min]
            else:
                train = np.arange(1, val_min) if val_min > 1 else np.array([], dtype=int)
            es_min = es_max = np.nan
            if model in NEURAL_MODELS and len(train):
                n_fit = n_fit_periods(int(len(train)), INNER_TRAIN_FRAC)
                es = train[n_fit:]
                if len(es):
                    es_min, es_max = int(es.min()), int(es.max())
            dataset = df["dataset"].iloc[0] if "dataset" in df.columns else panel_dir.parent.name
            rows.append({
                "dataset": dataset,
                "model": model,
                "stage": "holdout" if tag == "holdout" else "kfold",
                "outer_fold": tag,
                "train_week_min": int(train.min()) if len(train) else np.nan,
                "train_week_max": int(train.max()) if len(train) else np.nan,
                "early_stop_week_min": es_min,
                "early_stop_week_max": es_max,
                "validation_week_min": val_min,
                "validation_week_max": val_max,
                "n_train_periods": int(len(train)) if len(train) else np.nan,
                "n_val_periods": int(val.nunique()),
                "frozen_products_hash": frozen,
                "config_hash": config_hash(model),
                "seed": SEED,
            })
    out = pd.DataFrame(rows)
    path = panel_dir / "run_manifest.csv"
    if path.exists() and path.stat().st_size > 0 and len(out):
        old = pd.read_csv(path)
        if len(old):
            keys = ["dataset", "model", "stage", "outer_fold"]
            old = old.copy()
            recon = out.copy()
            old["_key"] = old[keys].astype(str).agg("|".join, axis=1)
            recon["_key"] = recon[keys].astype(str).agg("|".join, axis=1)
            prefer = old[old["_key"].isin(set(recon["_key"]))]
            missing = recon[~recon["_key"].isin(set(prefer["_key"]))]
            out = pd.concat(
                [prefer.drop(columns="_key"), missing.drop(columns="_key")],
                ignore_index=True,
            )
    if len(out):
        _write(_reorder(out, RUN_MANIFEST_COLS), path)
    return out


def rebuild_bootstrap_manifest_csv(panel_dir: Path) -> pd.DataFrame:
    parts = []
    for model in COMPARE_MODELS:
        kfold = _read(panel_dir / model / "kfold.csv")
        dataset = kfold["dataset"].iloc[0] if kfold is not None and len(kfold) else panel_dir.parent.name
        part = flatten_bootstrap_manifest(panel_dir / model, dataset)
        if len(part):
            parts.append(part)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=BOOT_MANIFEST_COLS)
    if len(out):
        _write(_reorder(out, BOOT_MANIFEST_COLS), panel_dir / "bootstrap_manifest.csv")
    elif not (panel_dir / "bootstrap_manifest.csv").exists():
        _write(pd.DataFrame(columns=BOOT_MANIFEST_COLS), panel_dir / "bootstrap_manifest.csv")
    return out


def rebuild_failures_from_manifests(panel_dir: Path) -> None:
    rows = []
    for model in COMPARE_MODELS:
        payload = _read_bootstrap_json(panel_dir / model)
        kfold = _read(panel_dir / model / "kfold.csv")
        dataset = kfold["dataset"].iloc[0] if kfold is not None and len(kfold) else panel_dir.parent.name
        if payload is None:
            continue
        recs = payload.get("replicates") or []
        n_att = len(recs)
        n_ok = sum(1 for r in recs if r.get("accepted", True))
        for r in recs:
            if r.get("accepted", True):
                continue
            rows.append({
                "dataset": dataset,
                "model": model,
                "stage": "bootstrap",
                "fold_or_boot_id": r.get("bootstrap_id"),
                "error_type": "UniverseError",
                "error_message": r.get("skip", ""),
                "n_attempted": n_att,
                "n_successful": n_ok,
            })
    path = panel_dir / "run_failures.csv"
    extra = pd.DataFrame(rows, columns=FAILURE_COLS) if rows else pd.DataFrame(columns=FAILURE_COLS)
    if path.exists() and len(pd.read_csv(path)):
        extra = pd.concat([pd.read_csv(path), extra], ignore_index=True)
        extra = extra.drop_duplicates(
            subset=["dataset", "model", "stage", "fold_or_boot_id", "error_type", "error_message"],
            keep="last",
        )
    _write(_reorder(extra, FAILURE_COLS), path)


def ensure_spline_placeholder(panel_dir: Path) -> None:
    path = panel_dir / "spline_support_diagnostics.csv"
    if not path.exists():
        _write(pd.DataFrame(columns=SPLINE_COLS), path)


def concat_model_csv(panel_dir: Path, name: str) -> pd.DataFrame:
    parts = []
    for model in COMPARE_MODELS:
        df = _read(panel_dir / model / name)
        if df is not None and len(df):
            if "model" not in df.columns:
                df = df.copy()
                df["model"] = model
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def panel_dirs(root: Path | None = None) -> list[Path]:
    from src.benchmarks.protocol import project_root

    root = Path(root) if root is not None else project_root()
    out = []
    for panel in (
        root / "data" / "M5-walmart" / "panel",
        root / "data" / "predict-future-sales-1c" / "panel",
        root / "data" / "Dominick" / "panel",
    ):
        if panel.exists():
            out.append(panel)
    return out


def rebuild_panel_tables(panel_dir: Path) -> None:
    """Backfill fields and write cross-model paper tables from whatever CSVs exist."""
    panel_dir = Path(panel_dir)
    for model in COMPARE_MODELS:
        mdir = panel_dir / model
        if not mdir.exists():
            continue
        enrich_kfold_csv(mdir)
        enrich_pred_ij(mdir)
        enrich_holdout_and_replicates(mdir)
        enrich_bootstrap_csv(mdir)
        enrich_bootstrap_ci(mdir)
        enrich_series_stats(mdir)
        enrich_boot_fold_ratio(mdir)
        fold_in_boot_ci(mdir)
    matched_prediction_metrics(panel_dir)
    matched_elasticity_series(panel_dir)
    from src.benchmarks.protocol import refresh_matched_edges

    refresh_matched_edges(panel_dir)
    rebuild_run_manifest(panel_dir)
    rebuild_bootstrap_manifest_csv(panel_dir)
    rebuild_failures_from_manifests(panel_dir)
    ensure_spline_placeholder(panel_dir)
    from src.benchmarks.paired import write_paired_tables

    write_paired_tables(panel_dir)


def rebuild_all(root: Path | None = None) -> None:
    from src.benchmarks.protocol import project_root

    root = Path(root) if root is not None else project_root()
    for panel in panel_dirs(root):
        print("\n=== rebuild", panel)
        rebuild_panel_tables(panel)
