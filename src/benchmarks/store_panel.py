"""Compact store x week x SKU arrays for lazy pairwise equations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np

PAIR_COLS = {"week_id", "log_v_i", "log_p_i", "log_p_j"}

@dataclass(frozen=True)
class StorePanel:
    weeks: np.ndarray
    products: np.ndarray
    demand: np.ndarray
    price: np.ndarray
    present: np.ndarray
    controls: dict[str, np.ndarray]
    control_cols: tuple[str, ...]
    product_index: pd.Index

def build_store_panel(df: pd.DataFrame, control_cols: list[str]) -> StorePanel:
    control_cols = list(control_cols)
    if df.empty:
        empty = np.empty((0, 0), dtype=np.float64)
        products = np.array([], dtype=object)
        return StorePanel(
            weeks=np.array([], dtype=np.int64),
            products=products,
            demand=empty,
            price=empty,
            present=np.empty((0, 0), dtype=bool),
            controls={c: empty for c in control_cols},
            control_cols=tuple(control_cols),
            product_index=pd.Index(products),
        )
    
    if df.duplicated(subset=["week_id", "store_code", "product_code"]).any():
        raise ValueError("store panel is not unique on (week_id, store_code, product_code)")

    weeks = np.unique(df["week_id"].to_numpy())
    products = np.unique(df["product_code"].to_numpy())
    week_index = pd.Index(weeks)
    product_index = pd.Index(products)
    t = week_index.get_indexer(df["week_id"])
    k = product_index.get_indexer(df["product_code"])
    n_t, n_p = len(weeks), len(products)

    demand = np.full((n_t, n_p), np.nan, dtype=np.float64)
    price = np.full((n_t, n_p), np.nan, dtype=np.float64)
    present = np.zeros((n_t, n_p), dtype=bool)
    demand[t, k] = df["log_demand"].to_numpy(np.float64, copy=False)
    price[t, k] = df["log_price"].to_numpy(np.float64, copy=False)
    present[t, k] = True
    
    controls = {}
    for c in control_cols:
        mat = np.full((n_t, n_p), np.nan, dtype=np.float64)
        mat[t, k] = df[c].to_numpy(np.float64, copy=False)
        controls[c] = mat
    
    return StorePanel(
        weeks=weeks,
        products=products,
        demand=demand,
        price=price,
        present=present,
        controls=controls,
        control_cols=tuple(control_cols),
        product_index=product_index,
    )

def _empty_pair_frame(control_cols: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=[*PAIR_COLS, *control_cols])

def _pair_frame(cache: StorePanel, i: int, j: int) -> pd.DataFrame:
    ii = cache.product_index.get_loc(i) if i in cache.product_index else -1
    jj = cache.product_index.get_loc(j) if j in cache.product_index else -1
    if ii == -1 or jj == -1:
        return _empty_pair_frame(cache.control_cols)

    ok = cache.present[:, ii] & cache.present[:, jj]
    data = {
        "week_id": cache.weeks[ok],
        "log_v_i": cache.demand[ok, ii],
        "log_p_i": cache.price[ok, ii],
        "log_p_j": cache.price[ok, jj],
    }
    for c in cache.control_cols:
        data[c] = cache.controls[c][ok, ii]
    return pd.DataFrame(data)

def extract_pair(
    cache_tr: StorePanel,
    cache_va: StorePanel,
    i: int,
    j: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/val equation frames for directed (i,j). Inner-merge weeks: both SKUs present."""
    return _pair_frame(cache_tr, i, j), _pair_frame(cache_va, i, j)