# src/benchmarks/pairwise_ols.py
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


class PairwiseOLS:
    GROUP_KEYS = ["store_code", "pair_id", "product_i", "product_j"]

    def __init__(self, control_cols: list[str], min_obs: int = 15, robust_cov_type: str = "HC1"):
        self.control_cols = control_cols
        self.min_obs = min_obs
        rhs = "log_p_i + log_p_j"
        if control_cols:
            rhs += " + " + " + ".join(control_cols)
        self._formula = f"log_v_i ~ {rhs}"

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
        try:
            fit = smf.ols(self._formula, data=g_train).fit()
            y_hat = fit.predict(g_val)
            y = g_val["log_v_i"]
            resid = y - y_hat
            ss_tot = np.sum((y - y.mean()) ** 2)
            summary = {
                "store_code": store, "pair_id": pair_id,
                "product_i": product_i, "product_j": product_j,
                "status": "ok",
                "n_train": len(g_train), "n_val": len(g_val),
                "own_elasticity": fit.params.get("log_p_i", np.nan),
                "cross_elasticity": fit.params.get("log_p_j", np.nan),
                "mae_val": float(np.mean(np.abs(resid))),
                "rmse_val": float(np.sqrt(np.mean(resid ** 2))),
                "r2_val": np.nan if ss_tot == 0 else float(1 - np.sum(resid ** 2) / ss_tot),
            }
            preds = pd.DataFrame({
                "store_code": store, "product_i": product_i, "product_j": product_j,
                "week_id": g_val["week_id"].values,
                "y_true_i": y.values, "y_hat_i": y_hat.values,
            })
            return summary, preds
        except Exception as exc:
            return {
                "store_code": store, "pair_id": pair_id,
                "product_i": product_i, "product_j": product_j,
                "status": "error", "error_message": str(exc),
            }, None