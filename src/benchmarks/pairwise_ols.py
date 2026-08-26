"""Pairwise OLS: one HC1 equation per (store, i, j).

Own and cross elasticities both come from `run_cross` (log p_i and log p_j).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.benchmarks.constants import MIN_DOF
from src.benchmarks.pairwise import PairwiseLinear


class PairwiseOLS(PairwiseLinear):
    def run_cross(self, pairs_train, pairs_val):
        """One HC1 equation per (store, i, j). Own = β on log p_i; cross = β on log p_j."""
        rows, preds = [], []
        keys = ["store_code", "product_i", "product_j"]
        for (store, pi, pj), g_tr in pairs_train.groupby(keys):
            rhs_cols = ["log_p_i", "log_p_j"] + self._varying(g_tr, self.control_cols)
            dof = self._dof(len(g_tr), len(rhs_cols))
            g_va = pairs_val[
                (pairs_val.store_code == store)
                & (pairs_val.product_i == pi)
                & (pairs_val.product_j == pj)
            ]
            if (dof < MIN_DOF
                    or g_tr["log_p_i"].nunique() < 2 or g_tr["log_p_j"].nunique() < 2
                    or g_tr["log_v_i"].nunique() < 2 or g_va.empty):
                continue

            vif = self._vif(g_tr, ["log_p_i", "log_p_j"])
            if not self._cross_vif_ok(vif):
                continue

            fit = smf.ols(f"log_v_i ~ {' + '.join(rhs_cols)}", g_tr).fit(cov_type="HC1")
            y_hat = np.asarray(fit.predict(g_va))
            y_true = g_va["log_v_i"].to_numpy(float)
            preds.append(self._pred_frame(
                store, pi, pj, g_va["week_id"].to_numpy(), y_true, y_hat,
            ))
            rows.append({
                "store_code": store, "product_i": pi, "product_j": pj,
                "n_train": len(g_tr),
                "own_elasticity": fit.params["log_p_i"],
                "cross_elasticity": fit.params["log_p_j"],
                **vif,
                "cond": float(np.linalg.cond(fit.model.exog)),
                "n_val": len(g_va),
                "n_params": 1 + len(rhs_cols),
            })
        return pd.DataFrame(rows), self._concat(preds)
