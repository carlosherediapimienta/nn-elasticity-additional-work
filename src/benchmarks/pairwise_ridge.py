"""Pairwise Ridge: same (store, i, j) system as OLS, L2 on scaled slopes.

Alpha is chosen by chronological KFold (`shuffle=False`) on the training
weeks of that equation, then the selected alpha is refit on all train weeks.
Prices and controls are z-scored using train moments only; coefficients are
unscaled back to log-log units before they are reported as elasticities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.benchmarks.constants import MIN_DOF
from src.benchmarks.pairwise import PairwiseLinear

ALPHAS = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0)


class PairwiseRidge(PairwiseLinear):
    def __init__(self, control_cols=None, alphas=ALPHAS, cv_folds=5):
        super().__init__(control_cols)
        self.alphas = np.asarray(alphas)
        self.cv_folds = cv_folds

    def _design(self, df, price_cols, ctrl_cols):
        X_price = df[price_cols].to_numpy(float)
        if ctrl_cols:
            X_ctrl = df[ctrl_cols].to_numpy(float)
        else:
            X_ctrl = np.zeros((len(df), 0))
        return X_price, X_ctrl

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
        kf = KFold(n_splits=folds, shuffle=False)
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
        # Sort so KFold slices are chronological rather than groupby order.
        g_tr = g_tr.sort_values("week_id")
        g_va = g_va.sort_values("week_id")
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
        y_true = g_va[y_col].to_numpy(float)
        y_hat = X_va @ beta
        return beta, alpha, y_true, y_hat

    def run_cross(self, pairs_train, pairs_val):
        """Same (store, i, j) filter as OLS; α chosen by chronological CV on that equation."""
        rows, preds = [], []
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
            if not self._cross_vif_ok(vif):
                continue

            beta, alpha, y_true, y_hat = self._run(
                g_tr, g_va, ["log_p_i", "log_p_j"], "log_v_i", ctrl_cols
            )
            preds.append(self._pred_frame(
                store, pi, pj, g_va["week_id"].to_numpy(), y_true, y_hat,
            ))
            rows.append({
                "store_code": store, "product_i": pi, "product_j": pj,
                "n_train": len(g_tr), "alpha_selected": alpha,
                "own_elasticity": float(beta[1]),
                "cross_elasticity": float(beta[2]),
                **vif,
                "n_val": len(g_va),
                "n_params": 1 + 2 + len(ctrl_cols),
            })
        return pd.DataFrame(rows), self._concat(preds)
