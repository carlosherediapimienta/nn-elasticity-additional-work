"""ICDN feature builder reused by OLS, Ridge, and the MLP.

Fit on train only. Validation (and bootstrap val) is transformed with a
past history window so lags/rolling windows do not peek into the future.
When train is split for early stopping, that history is the full train
(fit + early-stop), not the fit slice alone. Shared features are store-week;
product features are SKU-week.

When a bootstrap draw has `source_week_id`, calendar and lifecycle use that
column with the *frozen* holdout-train origin (not the replicate min).
Lags and rollings stay on `schema.period` (`week_id` = bootstrap_order) and
are grouped by `bootstrap_block_id` so concatenated blocks do not glue.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd
from icdn import ICDNConfig, PanelSchema
from icdn.data.features import LOG_DEMAND, PERIOD_RANK, FeatureBuilder

from src.benchmarks.bootstrap import BLOCK_ID_COL, SOURCE_WEEK_COL, FrozenCalendar

_FROZEN_CALENDAR: FrozenCalendar | None = None


def get_frozen_calendar() -> FrozenCalendar | None:
    return _FROZEN_CALENDAR


@contextmanager
def frozen_calendar(calendar: FrozenCalendar | None):
    """Install holdout-train calendar ranks for one bootstrap replicate."""
    global _FROZEN_CALENDAR
    prev = _FROZEN_CALENDAR
    _FROZEN_CALENDAR = calendar
    try:
        yield calendar
    finally:
        _FROZEN_CALENDAR = prev


def _calendar_col(df: pd.DataFrame, period: str) -> str:
    return SOURCE_WEEK_COL if SOURCE_WEEK_COL in df.columns else period


def _keys_like(mapping: dict, series: pd.Series) -> dict:
    if series.empty:
        return {str(k): int(v) for k, v in mapping.items()}
    if pd.api.types.is_numeric_dtype(series):
        return {int(k): int(v) for k, v in mapping.items()}
    return {str(k): int(v) for k, v in mapping.items()}


def _apply_frozen_calendar(builder: FeatureBuilder, panel: pd.DataFrame, frozen: FrozenCalendar) -> None:
    schema = builder.schema
    builder._calendar_origin = int(frozen.origin)
    builder._max_train_rank = int(frozen.max_train_rank)
    builder._product_first_rank = _keys_like(frozen.product_first_rank, panel[schema.product])
    stores, products, ranks = [], [], []
    for s, p, r in frozen.store_product_first_rank:
        stores.append(s)
        products.append(p)
        ranks.append(int(r))
    store_vals = pd.Series(stores)
    prod_vals = pd.Series(products)
    if pd.api.types.is_numeric_dtype(panel[schema.store]):
        store_vals = pd.to_numeric(store_vals, errors="coerce")
    else:
        store_vals = store_vals.astype(str)
    if pd.api.types.is_numeric_dtype(panel[schema.product]):
        prod_vals = pd.to_numeric(prod_vals, errors="coerce")
    else:
        prod_vals = prod_vals.astype(str)
    idx = pd.MultiIndex.from_arrays([store_vals, prod_vals], names=[schema.store, schema.product])
    builder._store_product_first_rank = pd.Series(ranks, index=idx, dtype=float)


def patch_feature_builder_two_clocks() -> None:
    """Calendar from source_week_id; lags from schema.period, isolated by block. Idempotent."""
    if getattr(FeatureBuilder, "_two_clocks_patched", False):
        return

    _orig_fit = FeatureBuilder.fit
    _orig_calendar = FeatureBuilder._add_calendar

    def fit(self, panel: pd.DataFrame) -> FeatureBuilder:
        _orig_fit(self, panel)
        schema = self.schema
        frozen = get_frozen_calendar()
        if frozen is not None:
            _apply_frozen_calendar(self, panel, frozen)
            return self
        cal = _calendar_col(panel, schema.period)
        self._calendar_origin = int(panel[cal].dropna().min())
        if cal == schema.period:
            return self
        temp = panel[[schema.store, schema.product]].copy()
        temp[PERIOD_RANK] = panel[cal].astype(int) - self._calendar_origin + 1
        self._product_first_rank = temp.groupby(schema.product)[PERIOD_RANK].min().to_dict()
        self._store_product_first_rank = temp.groupby(
            [schema.store, schema.product]
        )[PERIOD_RANK].min()
        self._max_train_rank = int(panel[cal].max()) - self._calendar_origin + 1
        return self

    def _add_calendar(self, df: pd.DataFrame) -> pd.DataFrame:
        cal = _calendar_col(df, self.schema.period)
        origin = getattr(self, "_calendar_origin", None)
        if get_frozen_calendar() is not None or cal != self.schema.period:
            if origin is None:
                origin = int(df[cal].min())
            df[PERIOD_RANK] = df[cal].astype(int) - int(origin) + 1
            self.shared_features.append(PERIOD_RANK)
            for p in self.config.seasonal_periods:
                df[f"sin_{p}"] = np.sin(2 * np.pi * df[PERIOD_RANK] / p)
                df[f"cos_{p}"] = np.cos(2 * np.pi * df[PERIOD_RANK] / p)
                self.shared_features += [f"sin_{p}", f"cos_{p}"]
            return df
        return _orig_calendar(self, df)

    def _join_calendar_feature(self, df, grid, col, miss):
        period = self.schema.period
        names = [n for n in list(grid.index.names) if n is not None] + [period]
        stacked = (
            grid.stack(future_stack=True)
            .rename(col)
            .rename_axis(names)
            .reset_index(drop=False)
        )
        on = [c for c in names if c in df.columns and c in stacked.columns]
        df = df.merge(stacked, on=on, how="left")
        df[miss] = (~np.isfinite(df[col])).astype(float)
        df[col] = df[col].fillna(0.0)
        self.product_features += [col, miss]
        return df

    def _add_lags_and_rollings(self, df: pd.DataFrame) -> pd.DataFrame:
        store, product, period = self.schema.store, self.schema.product, self.schema.period
        idx = [store, product]
        if BLOCK_ID_COL in df.columns:
            idx = [store, product, BLOCK_ID_COL]
        grid = self._week_grid(df)
        demand = (
            df.pivot_table(
                index=idx,
                columns=period,
                values=LOG_DEMAND,
                aggfunc="mean",
            )
            .reindex(columns=grid)
        )
        for k in self.config.lags:
            col, miss = f"lag_{k}", f"miss_lag_{k}"
            df = self._join_calendar_feature(df, demand.shift(k, axis=1), col, miss)
        for window in self.config.rolling_windows:
            col, miss = f"roll_{window}", f"miss_roll_{window}"
            rolled = demand.shift(1, axis=1).T.rolling(window, min_periods=1).mean().T
            df = self._join_calendar_feature(df, rolled, col, miss)
        return df

    FeatureBuilder.fit = fit
    FeatureBuilder._add_calendar = _add_calendar
    FeatureBuilder._join_calendar_feature = _join_calendar_feature
    FeatureBuilder._add_lags_and_rollings = _add_lags_and_rollings
    FeatureBuilder._two_clocks_patched = True


patch_feature_builder_two_clocks()


class ICDNFeaturePipeline:
    """Same features ICDN builds internally; schema is the only dataset-specific bit."""

    def __init__(self, schema: PanelSchema | None = None, config: ICDNConfig | None = None):
        self.config = config or ICDNConfig(schema=schema or PanelSchema())
        if schema is not None:
            self.config.schema = schema
        self._builder: FeatureBuilder | None = None
        self._train_tail: pd.DataFrame | None = None

    @property
    def shared_cols(self) -> list[str]:
        if self._builder is None:
            raise RuntimeError("fit() first")
        return list(self._builder.shared_features)

    @property
    def product_cols(self) -> list[str]:
        if self._builder is None:
            raise RuntimeError("fit() first")
        return list(self._builder.product_features)

    def fit(self, panel: pd.DataFrame) -> "ICDNFeaturePipeline":
        schema = self.config.schema
        n_tail = max(list(self.config.lags) + list(self.config.rolling_windows))
        keys = [schema.store, schema.product]
        if BLOCK_ID_COL in panel.columns:
            keys = [schema.store, schema.product, BLOCK_ID_COL]
        self._train_tail = (
            panel.sort_values(keys + [schema.period])
            .groupby(keys, group_keys=False)
            .tail(n_tail)
        )
        self._builder = FeatureBuilder(self.config)
        self._builder.fit(panel)
        return self

    def transform(self, panel: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        if self._builder is None:
            raise RuntimeError("fit() first")
        period = self.config.schema.period
        if history is not None and not history.empty:
            first = panel[period].min()
            tail = history[history[period] < first]
            extended = pd.concat([tail, panel], ignore_index=True) if not tail.empty else panel
            out = self._builder.transform(extended)
            keep = set(panel[period].unique())
            return out[out[period].isin(keep)].reset_index(drop=True)
        return self._builder.transform(panel)

    def transform_val(self, val: pd.DataFrame) -> pd.DataFrame:
        return self.transform(val, history=self._train_tail)
