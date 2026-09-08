"""Pairwise OLS: one HC1 equation per (store, i, j).

Own and cross elasticities both come from `run_cross` (log p_i and log p_j).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.benchmarks.pairwise import PairwiseLinear

class PairwiseOLS(PairwiseLinear):
    def _estimate_pair(
        self,
        g_tr: pd.DataFrame,
        g_va: pd.DataFrame,
        ctrl_cols: list[str],
        alpha=None
    ) -> dict:
        rhs_cols = ["log_p_i", "log_p_j"] + ctrl_cols
        fit = smf.ols(f"log_v_i ~ {' + '.join(rhs_cols)}", g_tr).fit(cov_type="HC1")
        y_hat = np.asarray(fit.predict(g_va))
        y_true = g_va["log_v_i"].to_numpy(float)
        return {
            "y_true": y_true,
            "y_hat": y_hat,
            "own_elasticity": fit.params["log_p_i"],
            "cross_elasticity": fit.params["log_p_j"],
            "extra": {"cond": float(np.linalg.cond(fit.model.exog))},
        }