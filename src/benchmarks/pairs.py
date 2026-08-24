# src/benchmarks/pairs.py
import numpy as np
import pandas as pd


class PairDatasetBuilder:
    def __init__(self, control_cols: list[str]) -> None:
        self.control_cols = control_cols

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = ["store_code", "week_id", "product_code"]
        base = df[keys + ["log_demand", "log_price"] + self.control_cols].copy()

        left = base.rename(columns={
            "product_code": "product_i",
            "log_demand": "log_v_i",
            "log_price": "log_p_i",
        })
        right = base[["store_code", "week_id", "product_code", "log_price"]].rename(
            columns={"product_code": "product_j", "log_price": "log_p_j"}
        )

        pair_df = left.merge(right, on=["store_code", "week_id"], how="inner")
        pair_df = pair_df[pair_df["product_i"] != pair_df["product_j"]].copy()

        a = pair_df["product_i"].astype(str)
        b = pair_df["product_j"].astype(str)
        pair_df["pair_id"] = np.where(a < b, a + "__" + b, b + "__" + a)
        return pair_df