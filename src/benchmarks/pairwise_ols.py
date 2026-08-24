import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

SHORT = ["promo", "sin_52", "cos_52"]
MIN_DOF = 30
MIN_STORES_CROSS = 3
MAX_VIF = 10.0


class PairwiseOLS:
    def __init__(self, control_cols=None):
        self.control_cols = list(control_cols or SHORT)

    def _n_params(self, n_stores, n_prices):
        return 1 + n_prices + len(self.control_cols) + max(n_stores - 1, 0)

    def _dof(self, n_train, n_stores, n_prices):
        return n_train - self._n_params(n_stores, n_prices)

    def _vif(self, df, price_cols):
        cont = price_cols + self.control_cols
        X = np.column_stack([np.ones(len(df)), df[cont].to_numpy(float)])
        return {f"vif_{c}": float(variance_inflation_factor(X, i + 1))
                for i, c in enumerate(cont)}

    def run_own(self, train, val):
        rows = []
        rhs = "log_price + " + " + ".join(self.control_cols) + " + C(store_code)"
        for product, g_tr in train.groupby("product_code"):
            n_stores = g_tr["store_code"].nunique()
            dof = self._dof(len(g_tr), n_stores, 1)
            g_va = val[(val.product_code == product)
                & val.store_code.isin(g_tr.store_code.unique())]
            if dof < MIN_DOF or g_tr["log_price"].nunique() < 2 or g_va.empty:
                continue
            fit = smf.ols(f"log_demand ~ {rhs}", g_tr).fit(cov_type="HC1")
            resid = g_va["log_demand"] - fit.predict(g_va)
            rows.append({
                "product_code": product, "n_train": len(g_tr), "n_stores": n_stores,
                "own_elasticity": fit.params["log_price"],
                "own_se": fit.bse["log_price"],
                **self._vif(g_tr, ["log_price"]),
                "cond": float(np.linalg.cond(fit.model.exog)),
                "mae_val": float(resid.abs().mean()),
                "rmse_val": float(np.sqrt((resid ** 2).mean())),
            })
        return pd.DataFrame(rows)

    def run_cross(self, pairs_train, pairs_val):
        rhs = "log_p_i + log_p_j + " + " + ".join(self.control_cols) + " + C(store_code)"
        rows = []
        for (pi, pj), g_tr in pairs_train.groupby(["product_i", "product_j"]):
            n_stores = g_tr["store_code"].nunique()
            dof = self._dof(len(g_tr), n_stores, 2)
            g_va = pairs_val[
                (pairs_val.product_i == pi) & (pairs_val.product_j == pj)
                & pairs_val.store_code.isin(g_tr.store_code.unique())
            ]
            if (n_stores < MIN_STORES_CROSS or dof < MIN_DOF
                    or g_tr["log_p_i"].nunique() < 2 or g_tr["log_p_j"].nunique() < 2
                    or g_va.empty):
                continue

            vif = self._vif(g_tr, ["log_p_i", "log_p_j"])
            if vif["vif_log_p_i"] >= MAX_VIF or vif["vif_log_p_j"] >= MAX_VIF:
                continue

            fit = smf.ols(f"log_v_i ~ {rhs}", g_tr).fit(cov_type="HC1")
            resid = g_va["log_v_i"] - fit.predict(g_va)
            rows.append({
                "product_i": pi, "product_j": pj, "n_train": len(g_tr), "n_stores": n_stores,
                "own_elasticity": fit.params["log_p_i"],
                "cross_elasticity": fit.params["log_p_j"],
                **vif,
                "cond": float(np.linalg.cond(fit.model.exog)),
                "mae_val": float(resid.abs().mean()),
                "rmse_val": float(np.sqrt((resid ** 2).mean())),
            })
        return pd.DataFrame(rows)