import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

SHORT = ["promo", "sin_52", "cos_52"]
MIN_DOF = 30
ALPHAS = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0)
MAX_VIF = 10.0


class PairwiseRidge:
    def __init__(self, control_cols=None, alphas=ALPHAS, cv_folds=5):
        self.control_cols = list(control_cols or SHORT)
        self.alphas = np.asarray(alphas)
        self.cv_folds = cv_folds

    def _n_params(self, n_rhs):
        return 1 + n_rhs

    def _dof(self, n_train, n_rhs):
        return n_train - self._n_params(n_rhs)

    def _varying(self, df, cols):
        return [c for c in cols if df[c].nunique(dropna=True) >= 2]

    def _design(self, df, price_cols, ctrl_cols):
        X_price = df[price_cols].to_numpy(float)
        if ctrl_cols:
            X_ctrl = df[ctrl_cols].to_numpy(float)
        else:
            X_ctrl = np.zeros((len(df), 0))
        return X_price, X_ctrl

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
    def _moments(X):
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return mu, sd

    @staticmethod
    def _z(X, mu, sd):
        return (X - mu) / sd

    @staticmethod
    def _unscale(beta_z, mu, sd):
        b = beta_z[1:] / sd
        a = beta_z[0] - np.dot(beta_z[1:], mu / sd)
        return np.concatenate([[a], b])

    @staticmethod
    def _fit_penalized(y, X_price, X_ctrl, alpha):
        X = np.column_stack([np.ones(len(y)), X_price, X_ctrl])
        d = [1e-8] + [alpha] * (X_price.shape[1] + X_ctrl.shape[1])
        D = np.diag(d)
        return np.linalg.solve(X.T @ X + D, X.T @ y)

    def _cv_alpha(self, y, X_price, X_ctrl):
        n = len(y)
        folds = min(self.cv_folds, n)
        if folds < 2:
            return float(self.alphas[0])
        kf = KFold(n_splits=folds, shuffle=True, random_state=0)
        n_p = X_price.shape[1]
        errs = []
        for alpha in self.alphas:
            sse = 0.0
            for tr_idx, va_idx in kf.split(X_price):
                X_tr = np.column_stack([X_price[tr_idx], X_ctrl[tr_idx]])
                X_va = np.column_stack([X_price[va_idx], X_ctrl[va_idx]])
                mu, sd = self._moments(X_tr)
                Z_tr = self._z(X_tr, mu, sd)
                Z_va = self._z(X_va, mu, sd)
                beta_z = self._fit_penalized(
                    y[tr_idx], Z_tr[:, :n_p], Z_tr[:, n_p:], alpha
                )
                pred = np.column_stack([np.ones(len(va_idx)), Z_va]) @ beta_z
                sse += np.sum((y[va_idx] - pred) ** 2)
            errs.append(sse)
        return float(self.alphas[int(np.argmin(errs))])

    def _run(self, g_tr, g_va, price_cols, y_col, ctrl_cols):
        X_price, X_ctrl = self._design(g_tr, price_cols, ctrl_cols)
        y = g_tr[y_col].to_numpy(float)
        n_p = X_price.shape[1]
        X_tr = np.column_stack([X_price, X_ctrl])
        mu, sd = self._moments(X_tr)
        Z_tr = self._z(X_tr, mu, sd)

        alpha = self._cv_alpha(y, X_price, X_ctrl)
        beta_z = self._fit_penalized(y, Z_tr[:, :n_p], Z_tr[:, n_p:], alpha)
        beta = self._unscale(beta_z, mu, sd)

        Xp_va, Xc_va = self._design(g_va, price_cols, ctrl_cols)
        X_va = np.column_stack([np.ones(len(g_va)), Xp_va, Xc_va])
        resid = g_va[y_col].to_numpy(float) - X_va @ beta
        return beta, alpha, resid

    def run_own(self, train, val):
        rows = []
        for (store, product), g_tr in train.groupby(["store_code", "product_code"]):
            ctrl_cols = self._varying(g_tr, self.control_cols)
            dof = self._dof(len(g_tr), 1 + len(ctrl_cols))
            g_va = val[(val.store_code == store) & (val.product_code == product)]
            if (dof < MIN_DOF or g_tr["log_price"].nunique() < 2
                    or g_tr["log_demand"].nunique() < 2 or g_va.empty):
                continue
            beta, alpha, resid = self._run(
                g_tr, g_va, ["log_price"], "log_demand", ctrl_cols
            )
            rows.append({
                "store_code": store, "product_code": product,
                "n_train": len(g_tr), "alpha_selected": alpha,
                "own_elasticity": float(beta[1]),
                "mae_val": float(np.mean(np.abs(resid))),
                "rmse_val": float(np.sqrt(np.mean(resid ** 2))),
            })
        return pd.DataFrame(rows)

    def run_cross(self, pairs_train, pairs_val):
        rows = []
        keys = ["store_code", "product_i", "product_j"]
        for (store, pi, pj), g_tr in pairs_train.groupby(keys):
            ctrl_cols = self._varying(g_tr, self.control_cols)
            dof = self._dof(len(g_tr), 2 + len(ctrl_cols))
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

            beta, alpha, resid = self._run(
                g_tr, g_va, ["log_p_i", "log_p_j"], "log_v_i", ctrl_cols
            )
            rows.append({
                "store_code": store, "product_i": pi, "product_j": pj,
                "n_train": len(g_tr), "alpha_selected": alpha,
                "own_elasticity": float(beta[1]),
                "cross_elasticity": float(beta[2]),
                **vif,
                "mae_val": float(np.mean(np.abs(resid))),
                "rmse_val": float(np.sqrt(np.mean(resid ** 2))),
            })
        return pd.DataFrame(rows)