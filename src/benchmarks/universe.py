"""Frozen product universe for MLP and ICDN.

SKU identity and slot order are part of the model. They are fixed on the
full panel before any split. Training slices that miss a frozen SKU raise
UniverseError instead of silently dropping a column and changing the Jacobian.

The gate is the *effective* fitting slice (MLP fit_raw, ICDN internal train),
not the outer-train window that still includes early-stopping weeks.
The shared `split_plan.json` starts outer folds late enough that the shortest
nested fit — first inner fold of the first outer fold, after the 80% fit
split — already contains every SKU. OLS and Ridge use those same outer cuts.

Validation and early-stopping slices may omit a frozen SKU. That is mask
zero for those cells, not an invalid fold.
"""

from __future__ import annotations

import pandas as pd

from src.benchmarks.constants import PERIOD_COL


class UniverseError(ValueError):
    """A training split does not contain every frozen product."""


def freeze_products(panel: pd.DataFrame) -> list[str]:
    """Sorted unique SKUs on the full panel. This list is the layout for every split."""
    return sorted(panel["product_code"].astype(str).unique())


def _keep_frozen(df: pd.DataFrame, frozen: list[str]) -> pd.DataFrame:
    return df[df["product_code"].astype(str).isin(frozen)].copy()


def require_training_products(df: pd.DataFrame, frozen: list[str], label: str = "df") -> pd.DataFrame:
    """Keep frozen SKUs; error if any frozen SKU is absent from a parameter-fitting slice."""
    have = set(df["product_code"].astype(str))
    missing = [p for p in frozen if p not in have]
    if missing:
        raise UniverseError(f"{label} missing frozen products: {missing}")
    return _keep_frozen(df, frozen)


def allow_missing_validation_products(df: pd.DataFrame, frozen: list[str]) -> pd.DataFrame:
    """Keep frozen SKUs only. Absence of a SKU is m_ist = 0, not a failed split."""
    return _keep_frozen(df, frozen)


def first_complete_period_count(
    panel: pd.DataFrame,
    frozen: list[str],
    period_col: str = PERIOD_COL,
) -> int:
    """How many leading periods the shortest fit needs so every frozen SKU has appeared."""
    periods = sorted(panel[period_col].unique().tolist())
    first = (
        panel.assign(_sku=panel["product_code"].astype(str))
        .groupby("_sku")[period_col]
        .min()
    )
    missing = [p for p in frozen if p not in set(first.index)]
    if missing:
        raise UniverseError(f"frozen products never appear in the panel: {missing}")
    last_intro = max(int(first[p]) for p in frozen)
    for i, week in enumerate(periods):
        if int(week) >= last_intro:
            return i + 1
    raise UniverseError(f"frozen universe is not covered by {period_col} up to {last_intro}")


def assert_layout(got, frozen: list[str], label: str) -> None:
    """Require exact product order, not just the same set."""
    got = [str(p) for p in got]
    frozen = [str(p) for p in frozen]
    if got != frozen:
        raise UniverseError(f"{label} products {got} != frozen {frozen}")


def patch_icdn_universe(frozen: list[str]) -> None:
    """Freeze ICDN product slots on the full-panel list.

    `_select_products` cannot invent rows. `fit` must not then drop a frozen
    SKU from the layout because `_filter_sparse` removed its cells. Missing
    from the internal train is a UniverseError, not a smaller Jacobian.
    """
    from icdn.data.encoders import LabelEncoder
    from icdn.data.features import LOG_PRICE
    from icdn.data.panel import PanelBuilder, PanelLayout

    frozen = [str(p) for p in frozen]

    def _select_products(self, df):
        col = self.schema.product
        have = set(df[col].astype(str))
        missing = [p for p in frozen if p not in have]
        if missing:
            raise UniverseError(f"ICDN train missing frozen products: {missing}")
        mapping = {str(v): v for v in df[col].unique()}
        return [mapping[p] for p in frozen]

    def fit(self, df, shared_features, product_features):
        products = self._select_products(df)
        filtered = self._filter_sparse(df, products)
        have = set(filtered[self.schema.product].astype(str))
        lost = [str(p) for p in products if str(p) not in have]
        if lost:
            raise UniverseError(
                f"ICDN internal train dropped frozen products from the layout: {lost}"
            )
        if len(products) < 2:
            raise ValueError(
                "fewer than two products survived the density filter. "
                "Lower min_coverage or provide a denser panel."
            )
        store_encoder = LabelEncoder().fit(filtered[self.schema.store])
        layout = PanelLayout(
            products=products,
            shared_features=list(shared_features),
            product_features=list(product_features),
            store_encoder=store_encoder,
        )
        self._attach_metadata(filtered, layout)
        self.layout = layout
        self._price_fallback_mean = self._pivot(filtered, LOG_PRICE, products).mean()
        return self

    PanelBuilder._select_products = _select_products
    PanelBuilder.fit = fit
