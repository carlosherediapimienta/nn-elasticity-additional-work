"""Frozen product universe for MLP and ICDN.

SKU identity and slot order are part of the model. They are fixed on the
full panel before any split. A train/val/bootstrap slice that is missing a
frozen SKU raises UniverseError (skip that replica) instead of silently
dropping a column and changing the Jacobian layout.
"""

from __future__ import annotations

import pandas as pd


class UniverseError(ValueError):
    """A split does not contain every frozen product."""


def freeze_products(panel: pd.DataFrame) -> list[str]:
    """Sorted unique SKUs on the full panel. This list is the layout for every split."""
    return sorted(panel["product_code"].astype(str).unique())


def require_products(df: pd.DataFrame, frozen: list[str], label: str = "df") -> pd.DataFrame:
    """Keep only frozen SKUs; error if any frozen SKU is absent."""
    have = set(df["product_code"].astype(str))
    missing = [p for p in frozen if p not in have]
    if missing:
        raise UniverseError(f"{label} missing frozen products: {missing}")
    return df[df["product_code"].astype(str).isin(frozen)].copy()


def assert_layout(got, frozen: list[str], label: str) -> None:
    """Require exact product order, not just the same set."""
    got = [str(p) for p in got]
    frozen = [str(p) for p in frozen]
    if got != frozen:
        raise UniverseError(f"{label} products {got} != frozen {frozen}")


def patch_icdn_universe(frozen: list[str]) -> None:
    """Force ICDN's PanelBuilder to use the frozen list instead of train uniques."""
    from icdn.data.panel import PanelBuilder

    frozen = [str(p) for p in frozen]

    def _select_products(self, df):
        col = self.schema.product
        have = set(df[col].astype(str))
        missing = [p for p in frozen if p not in have]
        if missing:
            raise UniverseError(f"ICDN train missing frozen products: {missing}")
        mapping = {str(v): v for v in df[col].unique()}
        return [mapping[p] for p in frozen]

    PanelBuilder._select_products = _select_products