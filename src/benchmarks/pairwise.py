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

from src.benchmarks.constants import MAX_VIF, SHORT, MIN_DOF
from src.benchmarks.store_panel import build_store_panel, extract_pair

def equation_key(store: str, product_i: str, product_j: str) -> tuple[str, str, str]:
    return str(store), str(product_i), str(product_j)

class PairwiseLinear:
    """DOF, varying-column filter, and VIF. Subclasses only change the estimator."""

    def __init__(self, control_cols: list[str] = None):
        self.control_cols = list(control_cols or SHORT)

    def _n_params(self, n_rhs: int) -> int:
        """Intercept plus every right-hand-side column."""
        return 1 + n_rhs

    def _dof(self, n_train: int, n_rhs: int) -> int:
        return n_train - self._n_params(n_rhs)

    def _varying(self, df: pd.DataFrame, cols: list[str]) -> list[str]:
        return [c for c in cols if df[c].nunique(dropna=True) >= 2]

    def _vif(self, df: pd.DataFrame, price_cols: list[str]) -> dict[str, float]:
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

    def _prepare_equation(self, g_tr: pd.DataFrame, g_va: pd.DataFrame) -> tuple[list[str], dict]:
        if g_tr.empty or g_va.empty:
            return None
        ctrl_cols= self._varying(g_tr, self.control_cols)
        if self._dof(len(g_tr), 2 + len(ctrl_cols)) < MIN_DOF:
            return None
        if (
            g_tr["log_p_i"].nunique() < 2
            or g_tr["log_p_j"].nunique() < 2
            or g_tr["log_v_i"].nunique() < 2
        ):
            return None
        vif = self._vif(g_tr, ["log_p_i", "log_p_j"])
        if not self._cross_vif_ok(vif):
            return None
        return ctrl_cols, vif

    def _estimate_pair(
        self,
        g_tr: pd.DataFrame,
        g_va: pd.DataFrame,
        ctrl_cols: list[str],
        alpha=None
    ):
        raise NotImplementedError

    def run_cross(
        self, 
        train: pd.DataFrame,
        val: pd.DataFrame,
        selected_alphas=None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows, preds = [], []
        for store, tr_s in train.groupby("store_code"):
            va_s = val.loc[val["store_code"] == store]
            cache_tr = build_store_panel(tr_s, self.control_cols)
            cache_va = build_store_panel(va_s, self.control_cols)
            for i in cache_tr.products:
                for j in cache_tr.products:
                    if i == j:
                        continue
                    if selected_alphas is not None:
                        alpha = selected_alphas.get(equation_key(store, i, j))
                        if alpha is None:
                            continue
                    else:
                        alpha = None
                    g_tr, g_va = extract_pair(cache_tr, cache_va, i, j)
                    prepared = self._prepare_equation(g_tr, g_va)
                    if prepared is None:
                        continue
                    ctrl_cols, vif = prepared
                    out = self._estimate_pair(g_tr, g_va, ctrl_cols, alpha=alpha)
                    preds.append(self._pred_frame(
                        store, i, j, g_va["week_id"].to_numpy(),
                        out["y_true"], out["y_hat"]
                    ))
                    rows.append({
                        "store_code": store,
                        "product_i": i,
                        "product_j": j,
                        "n_train": len(g_tr),
                        "own_elasticity": out["own_elasticity"],
                        "cross_elasticity": out["cross_elasticity"],
                        **vif,
                        "n_val": len(g_va),
                        "n_params":1 + 2 + len(ctrl_cols),
                        **out["extra"],
                    })
        return pd.DataFrame(rows), self._concat(preds)