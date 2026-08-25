import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

SHORT = ["promo", "sin_52", "cos_52"]
MIN_DOF = 30
MAX_VIF = 10.0


class PairwiseOLS:
    def __init__(self, control_cols=None):
        self.control_cols = list(control_cols or SHORT)

    def _n_params(self, n_rhs):
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

    def run_own(self, train, val):
        rows = []
        for (store, product), g_tr in train.groupby(["store_code", "product_code"]):
            rhs_cols = ["log_price"] + self._varying(g_tr, self.control_cols)
            dof = self._dof(len(g_tr), len(rhs_cols))
            g_va = val[(val.store_code == store) & (val.product_code == product)]
            if (dof < MIN_DOF or g_tr["log_price"].nunique() < 2
                    or g_tr["log_demand"].nunique() < 2 or g_va.empty):
                continue
            fit = smf.ols(f"log_demand ~ {' + '.join(rhs_cols)}", g_tr).fit(cov_type="HC1")
            resid = g_va["log_demand"] - fit.predict(g_va)
            rows.append({
                "store_code": store, "product_code": product,
                "n_train": len(g_tr),
                "own_elasticity": fit.params["log_price"],
                "own_se": fit.bse["log_price"],
                **self._vif(g_tr, ["log_price"]),
                "cond": float(np.linalg.cond(fit.model.exog)),
                "mae_val": float(resid.abs().mean()),
                "rmse_val": float(np.sqrt((resid ** 2).mean())),
            })
        return pd.DataFrame(rows)

    def run_cross(self, pairs_train, pairs_val):
        rows = []
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
            if (not np.isfinite(vif["vif_log_p_i"]) or not np.isfinite(vif["vif_log_p_j"])
                    or vif["vif_log_p_i"] >= MAX_VIF or vif["vif_log_p_j"] >= MAX_VIF):
                continue

            fit = smf.ols(f"log_v_i ~ {' + '.join(rhs_cols)}", g_tr).fit(cov_type="HC1")
            resid = g_va["log_v_i"] - fit.predict(g_va)
            rows.append({
                "store_code": store, "product_i": pi, "product_j": pj,
                "n_train": len(g_tr),
                "own_elasticity": fit.params["log_p_i"],
                "cross_elasticity": fit.params["log_p_j"],
                **vif,
                "cond": float(np.linalg.cond(fit.model.exog)),
                "mae_val": float(resid.abs().mean()),
                "rmse_val": float(np.sqrt((resid ** 2).mean())),
            })
        return pd.DataFrame(rows)