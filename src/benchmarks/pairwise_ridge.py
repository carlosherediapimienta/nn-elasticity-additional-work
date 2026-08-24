# src/benchmarks/pairwise_ridge.py
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


class PairwiseRidge:
    GROUP_KEYS = ["store_code", "pair_id", "product_i", "product_j"]

    def __init__(
        self,
        control_cols: list[str],
        min_obs: int = 15,
        alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0),
        cv_folds: int = 5,
        standardize: bool = True,
    ):
        self.control_cols = control_cols
        self.min_obs = min_obs
        self.alphas = np.asarray(alphas)
        self.cv_folds = cv_folds
        self.standardize = standardize

    def run(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        needed = ["log_v_i", "log_p_i", "log_p_j", "week_id"] + self.control_cols
        train_groups = {k: g[needed].dropna() for k, g in train_df.groupby(self.GROUP_KEYS)}
        val_groups = {k: g[needed].dropna() for k, g in val_df.groupby(self.GROUP_KEYS)}

        rows, pred_frames = [], []
        skipped = 0
        for key in sorted(set(train_groups) & set(val_groups)):
            g_train, g_val = train_groups[key], val_groups[key]
            if len(g_train) < self.min_obs or len(g_val) == 0:
                skipped += 1
                continue
            if g_train["log_p_i"].nunique() < 2 or g_train["log_p_j"].nunique() < 2:
                skipped += 1
                continue
            row, preds = self._fit_one(key, g_train, g_val)
            rows.append(row)
            if preds is not None:
                pred_frames.append(preds)

        summary = pd.DataFrame(rows)
        preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
        return summary, preds, skipped

    def _fit_one(self, key, g_train, g_val):
        store, pair_id, product_i, product_j = key
        feature_cols = ["log_p_i", "log_p_j"] + self.control_cols
        try:
            scaler = StandardScaler().fit(g_train[feature_cols]) if self.standardize else None
            Xtr = scaler.transform(g_train[feature_cols]) if scaler else g_train[feature_cols].to_numpy()
            Xval = scaler.transform(g_val[feature_cols]) if scaler else g_val[feature_cols].to_numpy()
            ytr, yval = g_train["log_v_i"], g_val["log_v_i"]

            n_folds = min(self.cv_folds, len(g_train))
            n_folds = max(n_folds, 2)
            model = RidgeCV(alphas=self.alphas, cv=n_folds).fit(Xtr, ytr)
            y_hat = model.predict(Xval)

            resid = yval.to_numpy() - y_hat
            ss_tot = np.sum((yval.to_numpy() - yval.mean()) ** 2)
            scale = scaler.scale_ if scaler is not None else np.ones(len(feature_cols))
            coefs = dict(zip(feature_cols, model.coef_ / scale))

            summary = {
                "store_code": store, "pair_id": pair_id,
                "product_i": product_i, "product_j": product_j,
                "status": "ok",
                "n_train": len(g_train), "n_val": len(g_val),
                "alpha_selected": float(model.alpha_),
                "own_elasticity": coefs["log_p_i"],
                "cross_elasticity": coefs["log_p_j"],
                "mae_val": float(np.mean(np.abs(resid))),
                "rmse_val": float(np.sqrt(np.mean(resid ** 2))),
                "r2_val": np.nan if ss_tot == 0 else float(1 - np.sum(resid ** 2) / ss_tot),
            }
            preds = pd.DataFrame({
                "store_code": store, "product_i": product_i, "product_j": product_j,
                "week_id": g_val["week_id"].to_numpy(),
                "y_true_i": yval.to_numpy(), "y_hat_i": y_hat,
            })
            return summary, preds
        except Exception as exc:
            return {
                "store_code": store, "pair_id": pair_id,
                "product_i": product_i, "product_j": product_j,
                "status": "error", "error_message": str(exc),
            }, None