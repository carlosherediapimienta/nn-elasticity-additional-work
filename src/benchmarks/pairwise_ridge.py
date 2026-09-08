"""Pairwise Ridge: same (store, i, j) system as OLS, L2 on scaled slopes.

Alpha is chosen by expanding inner CV on that equation's training weeks
(TemporalSplitter: past → future), minimizing mean fold MAE of log demand.
The selected alpha is then refit on all train weeks.

Prices and controls are z-scored using train moments only; coefficients are
unscaled back to log-log units before they are reported as elasticities.
Bootstrap reuses holdout alphas (conditional on the selected hyperparameters).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f
from icdn.data.splits import TemporalSplitter

from src.benchmarks.constants import MIN_INNER_FRAC, N_INNER_FOLDS, PERIOD_COL
from src.benchmarks.pairwise import PairwiseLinear, equation_key

ALPHAS = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0)

def freeze_alphas(cross: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    """Map (store, i, j) -> alpha from a holdout `run_cross` table.

    An empty dict means nothing was selected, so bootstrap must not re-search.
    """
    if cross is None or cross.empty or "alpha_selected" not in cross.columns:
        return {}
    return {
        equation_key(r.store_code, r.product_i, r.product_j): float(r.alpha_selected)
        for r in cross.itertuples(index=False)
    }

class PairwiseRidge(PairwiseLinear):
    def __init__(self, control_cols: list[str] = None, alphas=ALPHAS):
        super().__init__(control_cols)
        self.alphas = np.asarray(alphas)
    
    def _design(self, df: pd.DataFrame, price_cols: list[str], ctrl_cols: list[str]) -> np.ndarray:
        X_price = df[price_cols].to_numpy(float)
        if ctrl_cols:
            X_ctrl = df[ctrl_cols].to_numpy(float)
        else:
            X_ctrl = np.zeros((len(df), 0))
        return X_price, X_ctrl
    
    @staticmethod
    def _moments(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = X.mean(axis=0)
        sd = X.std(axis=0, ddof=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return mu, sd
    
    @staticmethod
    def _z(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
        return (X - mu) / sd
    
    @staticmethod
    def _unscale(beta_z: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
        b = beta_z[1:] / sd
        a = beta_z[0] - np.dot(beta_z[1:], mu / sd)
        return np.concatenate([[a], b])
    
    @staticmethod
    def _fit_penalized(
        y: np.ndarray,
        X_price: np.ndarray,
        X_ctrl: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        X = np.column_stack([np.ones(len(y)), X_price, X_ctrl])
        d = [1e-8] + [alpha] * (X_price.shape[1] + X_ctrl.shape[1])
        D = np.diag(d)
        return np.linalg.solve(X.T @ X + D, X.T @ y)

    def _expanding_inner(self, g_tr: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        n_peridos = int(g_tr[PERIOD_COL].nunique())
        min_train = max(1, int(n_peridos * MIN_INNER_FRAC))
        n_folds = min(int(N_INNER_FOLDS), n_peridos - min_train)
        if n_folds < 1:
            return []
        return TemporalSplitter(period_col=PERIOD_COL).expanding_splits(
            g_tr, n_folds=n_folds, min_train_frac=MIN_INNER_FRAC
        )

    def _cv_alpha(
        self,
        g_tr: pd.DataFrame,
        price_cols: list[str],
        y_col: str,
        ctrl_cols: list[str],
    ) -> float:
        """ alpha* = argmin_alpha mean_fold MAE(log q,k) over expanding inner CV. """
        try:
            folds = self._expanding_inner(g_tr)
        except ValueError:
            folds = []
        if not folds:
            return float(self.alphas[0])
        n_p = len(price_cols)
        mean_maes = []
        for alpha in self.alphas:
            maes = []
            for inner_tr, inner_va in folds:
                Xp_tr, Xc_tr = self._design(inner_tr, price_cols, ctrl_cols)
                Xp_va, Xc_va = self._design(inner_va, price_cols, ctrl_cols)
                X_tr = np.column_stack([Xp_tr, Xc_tr])
                X_va = np.column_stack([Xp_va, Xc_va])
                mu, sd = self._moments(X_tr)
                Z_tr = self._z(X_tr, mu, sd)
                Z_va = self._z(X_va, mu, sd)
                y_tr = inner_tr[y_col].to_numpy(float)
                y_val = inner_va[y_col].to_numpy(float)
                beta_z = self._fit_penalized(y_tr, Z_tr[:, :n_p], Z_tr[:, n_p:], alpha)
                pred = np.column_stack([np.ones(len(inner_va)), Z_va]) @ beta_z
                maes.append(float(np.mean(np.abs(y_val - pred))))
            mean_maes.append(np.mean(maes))
        return float(self.alphas[np.argmin(mean_maes)])

    def _run(
        self,
        g_tr: pd.DataFrame,
        g_va: pd.DataFrame,
        price_cols: list[str],
        y_col: str,
        ctrl_cols: list[str],
        alpha=None,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        g_tr = g_tr.sort_values(PERIOD_COL)
        g_va = g_va.sort_values(PERIOD_COL)
        X_price, X_ctrl = self._design(g_tr, price_cols, ctrl_cols)
        y = g_tr[y_col].to_numpy(float)
        n_p = X_price.shape[1]
        X_tr = np.column_stack([X_price, X_ctrl])
        mu, sd = self._moments(X_tr)
        Z_tr = self._z(X_tr, mu, sd)

        if alpha is None:
            alpha = self._cv_alpha(g_tr, price_cols, y_col, ctrl_cols)
        beta_z = self._fit_penalized(y, Z_tr[:, :n_p], Z_tr[:, n_p:], alpha)
        beta = self._unscale(beta_z, mu, sd)

        Xp_va, Xc_va = self._design(g_va, price_cols, ctrl_cols)
        X_va = np.column_stack([np.ones(len(g_va)), Xp_va, Xc_va])
        y_true = g_va[y_col].to_numpy(float)
        y_hat = X_va @ beta
        return beta, float(alpha), y_true, y_hat

    def _estimate_pair(
        self,
        g_tr: pd.DataFrame,
        g_va: pd.DataFrame,
        ctrl_cols: list[str],
        alpha=None,
    ) -> dict:
        beta, alpha, y_true, y_hat = self._run(
            g_tr, g_va, ["log_p_i", "log_p_j"], "log_v_i", ctrl_cols, alpha=alpha
        )
        return {
            "own_elasticity": float(beta[1]),
            "cross_elasticity": float(beta[2]),
            "y_true": y_true,
            "y_hat": y_hat,
            "extra": {"alpha_selected": alpha},
        }