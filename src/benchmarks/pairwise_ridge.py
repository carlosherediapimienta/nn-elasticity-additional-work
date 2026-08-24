# src/benchmarks/pairwise_ridge.py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

SHORT = ["promo", "sin_52", "cos_52"]
MIN_DOF = 30
MIN_STORES_CROSS = 3
ALPHAS = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0)
MAX_VIF = 10.0


class PairwiseRidge:
    def __init__(self, control_cols=None, alphas=ALPHAS, cv_folds=5):
        self.control_cols = list(control_cols or SHORT)
        self.alphas = np.asarray(alphas)
        self.cv_folds = cv_folds

    def _n_params(self, n_stores, n_prices):
        return 1 + n_prices + len(self.control_cols) + max(n_stores - 1, 0)

    def _dof(self, n_train, n_stores, n_prices):
        return n_train - self._n_params(n_stores, n_prices)

    def _design(self, df, price_cols, store_levels):
        X_price = df[price_cols].to_numpy(float)
        X_ctrl = df[self.control_cols].to_numpy(float)
        dummies = pd.get_dummies(df["store_code"]).reindex(columns=store_levels, fill_value=0)
        return X_price, dummies.to_numpy(float), X_ctrl

    def _vif(self, df, price_cols):
        cont = price_cols + self.control_cols
        X = np.column_stack([np.ones(len(df)), df[cont].to_numpy(float)])
        return {f"vif_{c}": float(variance_inflation_factor(X, i + 1))
                for i, c in enumerate(cont)}

    @staticmethod
    def _fit_penalized(y, X_price, X_fe, X_ctrl, alpha):
        # Penaliza solo el bloque de controles; precios y dummies de tienda, libres.
        X = np.column_stack([np.ones(len(y)), X_price, X_fe, X_ctrl])
        n_free = 1 + X_price.shape[1] + X_fe.shape[1]
        D = np.diag([1e-8] * n_free + [alpha] * X_ctrl.shape[1])
        return np.linalg.solve(X.T @ X + D, X.T @ y)

    def _cv_alpha(self, y, X_price, X_fe, X_ctrl):
        n = len(y)
        folds = min(self.cv_folds, n)
        if folds < 2:
            return float(self.alphas[0])
        kf = KFold(n_splits=folds, shuffle=True, random_state=0)
        errs = []
        for alpha in self.alphas:
            sse = 0.0
            for tr_idx, va_idx in kf.split(X_price):
                beta = self._fit_penalized(
                    y[tr_idx], X_price[tr_idx], X_fe[tr_idx], X_ctrl[tr_idx], alpha
                )
                X_va = np.column_stack(
                    [np.ones(len(va_idx)), X_price[va_idx], X_fe[va_idx], X_ctrl[va_idx]]
                )
                sse += np.sum((y[va_idx] - X_va @ beta) ** 2)
            errs.append(sse)
        return float(self.alphas[int(np.argmin(errs))])

    def _run(self, g_tr, g_va, price_cols, y_col):
        store_levels = sorted(g_tr["store_code"].unique())[1:]  # baseline = intercepto
        X_price, X_fe, X_ctrl = self._design(g_tr, price_cols, store_levels)
        y = g_tr[y_col].to_numpy(float)
        alpha = self._cv_alpha(y, X_price, X_fe, X_ctrl)
        beta = self._fit_penalized(y, X_price, X_fe, X_ctrl, alpha)

        Xp_va, Xfe_va, Xc_va = self._design(g_va, price_cols, store_levels)
        X_va = np.column_stack([np.ones(len(g_va)), Xp_va, Xfe_va, Xc_va])
        resid = g_va[y_col].to_numpy(float) - X_va @ beta
        return beta, alpha, resid

    def run_own(self, train, val):
        rows = []
        for product, g_tr in train.groupby("product_code"):
            n_stores = g_tr["store_code"].nunique()
            dof = self._dof(len(g_tr), n_stores, 1)
            g_va = val[(val.product_code == product)
                       & val.store_code.isin(g_tr.store_code.unique())]
            if dof < MIN_DOF or g_tr["log_price"].nunique() < 2 or g_va.empty:
                continue
            beta, alpha, resid = self._run(g_tr, g_va, ["log_price"], "log_demand")
            rows.append({
                "product_code": product, "n_train": len(g_tr), "n_stores": n_stores,
                "alpha_selected": alpha, "own_elasticity": float(beta[1]),
                "mae_val": float(np.mean(np.abs(resid))),
                "rmse_val": float(np.sqrt(np.mean(resid ** 2))),
            })
        return pd.DataFrame(rows)

    def run_cross(self, pairs_train, pairs_val):
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

            beta, alpha, resid = self._run(g_tr, g_va, ["log_p_i", "log_p_j"], "log_v_i")
            rows.append({
                "product_i": pi, "product_j": pj, "n_train": len(g_tr), "n_stores": n_stores,
                "alpha_selected": alpha,
                "own_elasticity": float(beta[1]), "cross_elasticity": float(beta[2]),
                **vif,
                "mae_val": float(np.mean(np.abs(resid))),
                "rmse_val": float(np.sqrt(np.mean(resid ** 2))),
            })
        return pd.DataFrame(rows)