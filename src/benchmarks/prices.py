"""Causal log-price fill: never use a future price.

Observed price, else last past store-product price (ffill in time), else the
train SKU mean. When `bootstrap_block_id` is present, ffill is grouped by
store × block so concatenated blocks do not inherit each other's prices.
ICDN's default PanelBuilder can bfill; `patch_panel_builder` replaces that
path when `_ACTIVE` is set.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd

from src.benchmarks.bootstrap import BLOCK_ID_COL

_ACTIVE: "CausalPriceFill | None" = None


class CausalPriceFill:
    """Observed → last past store-product price → train mean. Never bfill."""

    def fit(
        self,
        train: pd.DataFrame,
        store: str = "store_code",
        product: str = "product_code",
        period: str = "week_id",
        log_col: str = "log_price",
        price_col: str = "price",
    ) -> "CausalPriceFill":
        t = train.copy()
        t[product] = t[product].astype(str)
        if log_col not in t.columns:
            t[log_col] = np.log(t[price_col].astype(float))
        self.store = store
        self.product = product
        self.period = period
        self.mean_ = t.groupby(product)[log_col].mean()
        self.hist_wide_ = t.pivot_table(
            index=[store, period],
            columns=product,
            values=log_col,
            aggfunc="mean",
        )
        self.hist_wide_.columns = self.hist_wide_.columns.astype(str)
        if BLOCK_ID_COL in t.columns:
            self.block_map_ = t.groupby([store, period])[BLOCK_ID_COL].first()
        else:
            self.block_map_ = None
        return self

    def fill_wide(self, obs_wide: pd.DataFrame, panel: pd.DataFrame | None = None) -> pd.DataFrame:
        cols = [str(c) for c in obs_wide.columns]
        obs = obs_wide.copy()
        obs.columns = cols
        hist = self.hist_wide_.reindex(columns=cols)
        combined = pd.concat([hist, obs])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        block_map = None if getattr(self, "block_map_", None) is None else self.block_map_.copy()
        if panel is not None and BLOCK_ID_COL in panel.columns:
            extra = panel.groupby([self.store, self.period])[BLOCK_ID_COL].first()
            block_map = extra if block_map is None else pd.concat([block_map, extra])
            block_map = block_map[~block_map.index.duplicated(keep="last")]
        if block_map is not None:
            tmp = combined.reset_index()
            on = [self.store, self.period]
            bid = block_map.rename(BLOCK_ID_COL).reset_index()
            tmp = tmp.merge(bid, on=on, how="left")
            tmp[BLOCK_ID_COL] = tmp[BLOCK_ID_COL].fillna(-1)
            tmp = tmp.sort_values([self.store, BLOCK_ID_COL, self.period])
            tmp[cols] = tmp.groupby([self.store, BLOCK_ID_COL], sort=False)[cols].ffill()
            combined = tmp.drop(columns=[BLOCK_ID_COL]).set_index(on)
        else:
            combined = combined.groupby(level=0, group_keys=False).ffill()
        combined = combined.fillna(self.mean_.reindex(cols))
        return combined.reindex(obs.index)


def set_active(fill: CausalPriceFill | None) -> None:
    """Install (or clear) the fill used by the patched PanelBuilder.transform.

    `_ACTIVE` is process-global. Always pair an install with `set_active(None)`
    in a `finally` (see `price_fill_guard`); otherwise a failed fold leaves its
    fill in place for the next call.
    """
    global _ACTIVE
    _ACTIVE = fill


def get_active() -> CausalPriceFill | None:
    return _ACTIVE


@contextmanager
def price_fill_guard():
    """Clear the global causal fill on the way out, including after exceptions."""
    try:
        yield
    finally:
        set_active(None)


def patch_panel_builder() -> None:
    """Monkey-patch ICDN PanelBuilder.transform to use CausalPriceFill when active.

    If `_ACTIVE` is None the original ffill-then-mean fallback is kept, except we
    still do not bfill when a fill object is installed.
    """
    from icdn.data.features import LOG_DEMAND, LOG_PRICE
    from icdn.data.panel import STORE_INDEX, PanelBuilder

    def transform(self, df):
        if self.layout is None:
            raise RuntimeError("PanelBuilder.transform called before fit")

        layout = self.layout
        store, product, period = self.schema.store, self.schema.product, self.schema.period
        products = layout.products
        n = len(products)

        df = df[df[product].isin(products)].copy()
        if df.empty:
            raise ValueError("none of the modelled products appear in this panel")

        if self.config.min_products is not None:
            counts = df.groupby([store, period])[product].nunique()
            dense = counts[counts >= self.config.min_products].index
            df = (
                df.set_index([store, period])
                .loc[lambda d: d.index.isin(dense)]
                .reset_index()
            )
            if df.empty:
                raise ValueError(
                    "no store-period has at least min_products modelled products"
                )

        price = self._pivot(df, LOG_PRICE, products)
        if _ACTIVE is not None:
            price = _ACTIVE.fill_wide(price, panel=df)
        else:
            price = price.groupby(level=0, group_keys=False).apply(lambda g: g.ffill())
            fallback = (
                self._price_fallback_mean
                if self._price_fallback_mean is not None
                else price.mean()
            )
            price = price.fillna(fallback)
        if price.isna().any().any():
            missing = price.columns[price.isna().all()].tolist()
            raise ValueError(
                f"no finite fallback price for products {missing}. "
                "Fit the model on data that includes them, or pass those products in the panel."
            )
        price.columns = [f"log_price_{i}" for i in range(n)]

        demand = self._pivot(df, LOG_DEMAND, products)
        obs_mask = demand.notna().astype(float)
        obs_mask.columns = [f"obs_mask_{i}" for i in range(n)]
        demand = demand.fillna(0.0)
        demand.columns = [f"log_demand_{i}" for i in range(n)]

        shared = df.groupby([store, period], observed=True)[layout.shared_features].first()

        blocks = [demand, obs_mask, shared]
        for feature in layout.product_features:
            block = self._pivot(df, feature, products).fillna(0.0)
            block.columns = [f"{feature}_{i}" for i in range(n)]
            blocks.append(block)

        wide = price.join(blocks, how="left").reset_index()
        if BLOCK_ID_COL in df.columns:
            bid = df.groupby([store, period])[BLOCK_ID_COL].first().rename(BLOCK_ID_COL)
            wide = wide.merge(bid.reset_index(), on=[store, period], how="left")
        wide[STORE_INDEX] = layout.store_encoder.transform(wide[store])
        return wide.sort_values([store, period]).reset_index(drop=True)

    PanelBuilder.transform = transform


def patch_icdn_smooth() -> None:
    """ICDN warmup smoothing does not cross bootstrap_block_id. Idempotent."""
    from icdn.api import ICDNModel

    if getattr(ICDNModel, "_block_smooth_patched", False):
        return
    _orig = ICDNModel._smooth

    def _smooth(self, wide: pd.DataFrame) -> pd.DataFrame:
        if BLOCK_ID_COL not in wide.columns:
            return _orig(self, wide)
        cfg = self.config
        store, period = cfg.schema.store, cfg.schema.period
        window = cfg.smoothing_window
        smoothed = wide.sort_values([store, BLOCK_ID_COL, period]).copy()
        for i in range(self.layout.n_products):
            demand_col = f"log_demand_{i}"
            mask_col = f"obs_mask_{i}"
            rolled = pd.Series(index=smoothed.index, dtype=np.float64)
            for _, g in smoothed.groupby([store, BLOCK_ID_COL], sort=False):
                series = g[demand_col].where(g[mask_col] > 0)
                series.index = g[period].astype(int)
                start, end = int(series.index.min()), int(series.index.max())
                full = series.reindex(range(start, end + 1))
                calendar = full.rolling(window, min_periods=1).mean()
                rolled.loc[g.index] = calendar.reindex(series.index).to_numpy()
            smoothed[demand_col] = rolled.fillna(0.0)
        return smoothed

    ICDNModel._smooth = _smooth
    ICDNModel._block_smooth_patched = True