"""ICDN feature builder reused by OLS, Ridge, and the MLP.

Fit on train only. Validation (and bootstrap val) is transformed with the
train tail so lags/rolling windows do not peek into the future. Shared
features are store-week; product features are SKU-week.
"""

from __future__ import annotations

import pandas as pd
from icdn import ICDNConfig, PanelSchema
from icdn.data.features import FeatureBuilder


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
        self._train_tail = (
            panel.sort_values([schema.store, schema.product, schema.period])
            .groupby([schema.store, schema.product], group_keys=False)
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