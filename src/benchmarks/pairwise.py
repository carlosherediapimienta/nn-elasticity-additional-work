"""Shared diagnostics for pairwise log-log demand equations (OLS and Ridge).

Each equation is one (store, i, j) series. Controls that do not vary in
train are dropped so the design is full rank. VIF is computed on the
continuous block (prices + controls) with an intercept column; the two
price VIFs must be finite and below MAX_VIF before a cross equation is kept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.benchmarks.constants import MAX_VIF, SHORT


class PairwiseLinear:
    """DOF, varying-column filter, and VIF. Subclasses only change the estimator."""

    def __init__(self, control_cols=None):
        self.control_cols = list(control_cols or SHORT)

    def _n_params(self, n_rhs):
        """Intercept plus every right-hand-side column."""
        return 1 + n_rhs

    def _dof(self, n_train, n_rhs):
        return n_train - self._n_params(n_rhs)

    def _varying(self, df, cols):
        return [c for c in cols if df[c].nunique(dropna=True) >= 2]

    def _vif(self, df, price_cols):
        cont = self._varying(df, price_cols + self.control_cols)
        out = {f"vif_{c}": np.nan for c in price_cols + self.control_cols}
        if not cont:
            return out
        X = np.column_stack([np.ones(len(df)), df[cont].to_numpy(float)])
        with np.errstate(divide="ignore", invalid="ignore"):
            for i, c in enumerate(cont):
                v = float(variance_inflation_factor(X, i + 1))
                out[f"vif_{c}"] = v if np.isfinite(v) else np.inf
        return out

    @staticmethod
    def _pred_frame(store, product_i, product_j, week_id, y_true, y_pred) -> pd.DataFrame:
        return pd.DataFrame({
            "store_code": store,
            "product_i": product_i,
            "product_j": product_j,
            "week_id": np.asarray(week_id),
            "y_true": y_true,
            "y_pred": y_pred,
        })

    @staticmethod
    def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _cross_vif_ok(self, vif: dict) -> bool:
        return not (
            not np.isfinite(vif["vif_log_p_i"]) or not np.isfinite(vif["vif_log_p_j"])
            or vif["vif_log_p_i"] >= MAX_VIF or vif["vif_log_p_j"] >= MAX_VIF
        )
