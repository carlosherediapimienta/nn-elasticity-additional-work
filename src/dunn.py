"""Build a Dunnhumby Complete Journey weekly panel for ICDN.

Observation unit
----------------
(store, product, week) -> (price, units, promo, metadata)

ICDN default columns
--------------------
store_code, product_code, week_id, price, units, on_promo
Optional: category, brand.

Source files
------------
Used:    transaction_data.csv, product.csv, causal_data.csv
Unused:  coupon.csv, coupon_redempt.csv, campaign_desc.csv,
         campaign_table.csv, hh_demographic.csv

Coupons already leave a footprint in RETAIL_DISC / COUPON_DISC /
COUPON_MATCH_DISC. Campaign and household tables would make the
benchmark less comparable with Walmart and Dominick's.

Unlike M5
---------
Complete Journey has no shelf-price panel. Prices are recovered from
household-panel transactions. Do *not* impute missing store-product-weeks
as units = 0 with a forward-filled price: that invents unobserved prices.

units are purchases by panel households, not store-wide sales. That must
be stated in the paper.

Price
-----
Primary:     P = sum(SALES_VALUE) / sum(QUANTITY)
Robustness:  P_customer, P_regular  (kept on the master only)

WEEK_NO is already a consecutive index 1..102. Do not remap it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DunnConfig:
    """Paths and pre-specified selection rules.

    Store, category and SKU choices use only week_id <= selection_cutoff.
    """

    data_dir: Path = Path("./data/dunnhumby")
    out_dir: Path | None = None

    selection_cutoff: int = 51
    min_store_weeks: int = 45
    n_core_stores: int = 20

    min_stores: int = 5
    min_weeks: int = 35
    min_unique_prices: int = 8
    min_price_cv: float = 0.03
    min_products_in_group: int = 10

    n_candidate_skus: int = 30
    n_skus: int = 10
    min_promo_coverage: float = 0.90

    causal_chunksize: int = 1_000_000

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.out_dir is None:
            self.out_dir = self.data_dir / "processed"
        else:
            self.out_dir = Path(self.out_dir)


@dataclass
class DunnSampleSelection:
    """Frozen universe used by OLS, Ridge, MLP and ICDN."""

    cutoff_week_id: int
    grouping_level: str
    category: str
    sub_commodity: str | None
    core_stores: list[str]
    candidate_product_codes: list[str]
    product_codes: list[str]
    n_eligible_in_group: int
    joint_coverage_ge_8_of_10: float | None = None
    joint_coverage_all_10: float | None = None
    criteria: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cutoff_week_id": int(self.cutoff_week_id),
            "grouping_level": self.grouping_level,
            "category": self.category,
            "sub_commodity": self.sub_commodity,
            "core_stores": list(self.core_stores),
            "candidate_product_codes": list(self.candidate_product_codes),
            "product_codes": list(self.product_codes),
            "n_eligible_in_group": int(self.n_eligible_in_group),
            "joint_coverage_ge_8_of_10": self.joint_coverage_ge_8_of_10,
            "joint_coverage_all_10": self.joint_coverage_all_10,
            "criteria": self.criteria,
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class DunnWeeklyPanelBuilder:
    """Transaction master (observed purchases only) and ICDN subset.

    Outputs
    -------
    dunnhumby_weekly_transaction_master.parquet
        Store-product-week with observed q > 0, p > 0. No causal flags yet.

    dunnhumby_candidate_panel.parquet
        Shortlist inside core stores, with display/mailer merged.

    dunnhumby_icdn_panel.parquet
        Frozen 10 SKUs; rows with observed promo only.
    """

    DISCOUNT_COLS = ["RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"]

    def __init__(self, config: DunnConfig | None = None) -> None:
        self.config = config or DunnConfig()
        self.config.out_dir.mkdir(parents=True, exist_ok=True)

        self.products: pd.DataFrame | None = None
        self.tx_audit: dict = {}
        self.line_qty_stats: pd.DataFrame | None = None

        self.weekly: pd.DataFrame | None = None
        self.product_stats: pd.DataFrame | None = None
        self.eligible: pd.DataFrame | None = None
        self.candidates: pd.DataFrame | None = None
        self.core_stores: list[str] = []
        self.promo_weekly: pd.DataFrame | None = None
        self.candidate_panel: pd.DataFrame | None = None
        self.selection: DunnSampleSelection | None = None
        self.icdn_panel: pd.DataFrame | None = None

    # ======================================================================
    # Orchestration
    # ======================================================================

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, DunnSampleSelection]:
        self.build_transaction_master()
        self.select_without_lookahead()
        self.attach_causal_promo()
        self.finalize_skus_with_promo_support()
        self.build_icdn_panel()
        return self.weekly, self.icdn_panel, self.selection

    def build_transaction_master(self) -> pd.DataFrame:
        """Observed store-product-weeks only. No zero-sales imputation."""
        self.load_products()
        tx = self.load_transactions()
        tx_valid = self.drop_invalid_transactions(tx)
        tx_valid = self.construct_value_variables(tx_valid)
        self.line_qty_stats = self.line_quantity_stats(tx_valid)
        weekly = self.aggregate_to_weekly(tx_valid)
        weekly = self.compute_weekly_prices(weekly)
        weekly = self.attach_product_metadata(weekly)
        weekly = self.to_icdn_identifiers(weekly)
        self.weekly = self.save_transaction_master(weekly)
        return self.weekly

    # ======================================================================
    # Products
    # ======================================================================

    def load_products(self) -> pd.DataFrame:
        path = self.config.data_dir / "product.csv"
        products = pd.read_csv(
            path,
            dtype={
                "PRODUCT_ID": "int64",
                "MANUFACTURER": "string",
                "DEPARTMENT": "string",
                "BRAND": "string",
                "COMMODITY_DESC": "string",
                "SUB_COMMODITY_DESC": "string",
                "CURR_SIZE_OF_PRODUCT": "string",
            },
        )
        products.columns = products.columns.str.strip()

        for col in [
            "DEPARTMENT",
            "BRAND",
            "COMMODITY_DESC",
            "SUB_COMMODITY_DESC",
            "CURR_SIZE_OF_PRODUCT",
        ]:
            products[col] = products[col].fillna("UNKNOWN").str.strip()

        assert not products["PRODUCT_ID"].duplicated().any()
        print(products.shape)
        print(products.head())
        self.products = products
        return products

    # ======================================================================
    # Transactions
    # ======================================================================

    def load_transactions(self) -> pd.DataFrame:
        path = self.config.data_dir / "transaction_data.csv"
        tx = pd.read_csv(
            path,
            dtype={
                "household_key": "int32",
                "BASKET_ID": "int64",
                "DAY": "int16",
                "PRODUCT_ID": "int64",
                "QUANTITY": "float32",
                "SALES_VALUE": "float32",
                "STORE_ID": "int32",
                "RETAIL_DISC": "float32",
                "TRANS_TIME": "int32",
                "WEEK_NO": "int16",
                "COUPON_DISC": "float32",
                "COUPON_MATCH_DISC": "float32",
            },
        )
        tx.columns = tx.columns.str.strip()

        self.tx_audit = {
            "rows": len(tx),
            "households": int(tx["household_key"].nunique()),
            "stores": int(tx["STORE_ID"].nunique()),
            "products": int(tx["PRODUCT_ID"].nunique()),
            "weeks": int(tx["WEEK_NO"].nunique()),
            "week_min": int(tx["WEEK_NO"].min()),
            "week_max": int(tx["WEEK_NO"].max()),
            "quantity_le_0": int((tx["QUANTITY"] <= 0).sum()),
            "sales_value_le_0": int((tx["SALES_VALUE"] <= 0).sum()),
            "missing_product": int(tx["PRODUCT_ID"].isna().sum()),
            "missing_store": int(tx["STORE_ID"].isna().sum()),
            "missing_week": int(tx["WEEK_NO"].isna().sum()),
        }
        print(pd.Series(self.tx_audit))
        audit_path = self.config.out_dir / "dunnhumby_transaction_audit.json"
        audit_path.write_text(json.dumps(self.tx_audit, indent=2))
        return tx

    def drop_invalid_transactions(self, tx: pd.DataFrame) -> pd.DataFrame:
        """Keep q > 0 and retailer value > 0. No winsorizing.

        Extreme QUANTITY is typical of weighted/bulk UPCs; those are
        screened later by choosing a packaged-goods commodity.
        """
        tx[self.DISCOUNT_COLS] = tx[self.DISCOUNT_COLS].fillna(0.0)
        tx_valid = tx[
            (tx["QUANTITY"] > 0)
            & (tx["SALES_VALUE"] > 0)
            & tx["PRODUCT_ID"].notna()
            & tx["STORE_ID"].notna()
            & tx["WEEK_NO"].notna()
        ].copy()
        print(
            f"Kept {len(tx_valid):,} / {len(tx):,} "
            f"rows ({len(tx_valid) / len(tx):.2%})"
        )
        return tx_valid

    def construct_value_variables(self, tx_valid: pd.DataFrame) -> pd.DataFrame:
        """SALES_VALUE is retailer receipts after some discounts, not
        always the household's out-of-pocket spend when a manufacturer
        coupon is present.
        """
        tx_valid["retailer_value"] = tx_valid["SALES_VALUE"]
        tx_valid["coupon_amount"] = tx_valid["COUPON_DISC"].abs()
        tx_valid["retail_discount_amount"] = tx_valid["RETAIL_DISC"].abs()
        tx_valid["coupon_match_amount"] = tx_valid["COUPON_MATCH_DISC"].abs()

        tx_valid["customer_spend"] = (
            tx_valid["SALES_VALUE"] - tx_valid["coupon_amount"]
        )
        tx_valid["regular_value"] = (
            tx_valid["SALES_VALUE"]
            + tx_valid["retail_discount_amount"]
            + tx_valid["coupon_match_amount"]
        )
        tx_valid["has_retail_discount"] = (
            tx_valid["retail_discount_amount"] > 0
        ).astype("int8")
        tx_valid["has_coupon"] = (tx_valid["coupon_amount"] > 0).astype("int8")
        return tx_valid

    @staticmethod
    def line_quantity_stats(tx_valid: pd.DataFrame) -> pd.DataFrame:
        """Flag weighted/bulk UPCs (huge line quantities)."""
        stats = (
            tx_valid.groupby("PRODUCT_ID", observed=True)["QUANTITY"]
            .agg(
                median_line_qty="median",
                p99_line_qty=lambda x: x.quantile(0.99),
                max_line_qty="max",
            )
            .reset_index()
        )
        stats["product_code"] = stats["PRODUCT_ID"].astype("string")
        return stats

    # ======================================================================
    # Weekly aggregation
    # ======================================================================

    def aggregate_to_weekly(self, tx_valid: pd.DataFrame) -> pd.DataFrame:
        """household × basket × product  ->  store × product × week."""
        weekly = (
            tx_valid.groupby(["STORE_ID", "PRODUCT_ID", "WEEK_NO"], observed=True)
            .agg(
                units=("QUANTITY", "sum"),
                sales_value=("retailer_value", "sum"),
                customer_spend=("customer_spend", "sum"),
                regular_value=("regular_value", "sum"),
                retail_discount=("retail_discount_amount", "sum"),
                coupon_discount=("coupon_amount", "sum"),
                coupon_match=("coupon_match_amount", "sum"),
                n_baskets=("BASKET_ID", "nunique"),
                n_households=("household_key", "nunique"),
                retail_discount_events=("has_retail_discount", "sum"),
                coupon_events=("has_coupon", "sum"),
            )
            .reset_index()
        )
        return weekly

    def compute_weekly_prices(self, weekly: pd.DataFrame) -> pd.DataFrame:
        """Quantity-weighted unit values, not the mean of line unit prices.

            P         = sum(SALES_VALUE) / sum(QUANTITY)
            P_customer = sum(SALES_VALUE - |COUPON_DISC|) / sum(QUANTITY)
            P_regular  = sum(SALES_VALUE + |RETAIL_DISC| + |COUPON_MATCH_DISC|)
                         / sum(QUANTITY)
        """
        weekly["price"] = weekly["sales_value"] / weekly["units"]
        weekly["price_customer"] = weekly["customer_spend"] / weekly["units"]
        weekly["price_regular"] = weekly["regular_value"] / weekly["units"]
        weekly["retail_discount_share"] = weekly["retail_discount"] / weekly[
            "regular_value"
        ].replace(0, np.nan)
        weekly["coupon_share"] = weekly["coupon_discount"] / weekly[
            "regular_value"
        ].replace(0, np.nan)

        print(
            weekly[["price", "price_customer", "price_regular", "units"]].describe(
                percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]
            )
        )

        weekly = weekly[
            np.isfinite(weekly["price"])
            & (weekly["price"] > 0)
            & np.isfinite(weekly["units"])
            & (weekly["units"] > 0)
        ].copy()
        return weekly

    def attach_product_metadata(self, weekly: pd.DataFrame) -> pd.DataFrame:
        if self.products is None:
            raise RuntimeError("Call load_products() first.")
        weekly = weekly.merge(
            self.products,
            on="PRODUCT_ID",
            how="left",
            validate="many_to_one",
        )
        missing = float(weekly["COMMODITY_DESC"].isna().mean())
        print("Missing product metadata:", missing)
        return weekly

    def to_icdn_identifiers(self, weekly: pd.DataFrame) -> pd.DataFrame:
        """ICDN category = COMMODITY_DESC; brand = Private/National flag.

        WEEK_NO is already sequential. Do not remap.
        """
        weekly = weekly.rename(
            columns={
                "STORE_ID": "store_code",
                "PRODUCT_ID": "product_code",
                "WEEK_NO": "week_id",
                "COMMODITY_DESC": "category",
                "BRAND": "brand",
                "SUB_COMMODITY_DESC": "sub_commodity",
                "CURR_SIZE_OF_PRODUCT": "size",
                "DEPARTMENT": "department",
                "MANUFACTURER": "manufacturer",
            }
        )
        weekly["store_code"] = weekly["store_code"].astype("string")
        weekly["product_code"] = weekly["product_code"].astype("string")
        weekly["week_id"] = weekly["week_id"].astype("int16")

        weeks = np.sort(weekly["week_id"].unique())
        print("week_id range:", int(weeks.min()), "→", int(weeks.max()))
        return weekly

    def save_transaction_master(self, weekly: pd.DataFrame) -> pd.DataFrame:
        weekly = weekly.sort_values(
            ["week_id", "store_code", "product_code"]
        ).reset_index(drop=True)
        path = self.config.out_dir / "dunnhumby_weekly_transaction_master.parquet"
        weekly.to_parquet(path, index=False)
        print("Wrote", path)
        return weekly

    # ======================================================================
    # Selection without lookahead
    # ======================================================================

    def selection_sample(self) -> pd.DataFrame:
        if self.weekly is None:
            raise RuntimeError("Call build_transaction_master() first.")
        cutoff = self.config.selection_cutoff
        weeks = np.sort(self.weekly["week_id"].unique())
        print("weeks:", int(weeks.min()), int(weeks.max()), "cutoff:", cutoff)
        return self.weekly[self.weekly["week_id"] <= cutoff].copy()

    def select_core_stores(self, selection: pd.DataFrame) -> list[str]:
        """High-activity stores with a long enough history. Not the final set."""
        cfg = self.config
        store_stats = (
            selection.groupby("store_code", observed=True)
            .agg(
                n_obs=("units", "size"),
                n_products=("product_code", "nunique"),
                n_weeks=("week_id", "nunique"),
                total_units=("units", "sum"),
                total_value=("sales_value", "sum"),
            )
            .reset_index()
        )
        store_stats = store_stats[store_stats["n_weeks"] >= cfg.min_store_weeks]
        core_stores = (
            store_stats.sort_values(["n_obs", "total_units"], ascending=False)
            .head(cfg.n_core_stores)["store_code"]
            .astype(str)
            .tolist()
        )
        print("core stores:", core_stores)
        self.core_stores = core_stores
        return core_stores

    def compute_product_stats(self, selection_core: pd.DataFrame) -> pd.DataFrame:
        """coverage_rate = share of core store-weeks with an observed purchase.

        This is *not* M5's positive_rate: we do not observe retailer zeros.
        """
        stats = (
            selection_core.groupby(
                ["product_code", "category", "sub_commodity", "brand", "department"],
                observed=True,
            )
            .agg(
                n_obs=("units", "size"),
                n_stores=("store_code", "nunique"),
                n_weeks=("week_id", "nunique"),
                total_units=("units", "sum"),
                unique_prices=("price", "nunique"),
                mean_price=("price", "mean"),
                std_price=("price", "std"),
                mean_baskets=("n_baskets", "mean"),
                mean_households=("n_households", "mean"),
                discount_rate=("retail_discount_events", lambda x: (x > 0).mean()),
                coupon_rate=("coupon_events", lambda x: (x > 0).mean()),
            )
            .reset_index()
        )
        stats["price_cv"] = stats["std_price"] / stats["mean_price"]

        n_weeks = int(selection_core["week_id"].nunique())
        possible_cells = len(self.core_stores) * n_weeks
        stats["coverage_rate"] = stats["n_obs"] / possible_cells

        if self.line_qty_stats is not None:
            stats = stats.merge(
                self.line_qty_stats[
                    ["product_code", "median_line_qty", "p99_line_qty", "max_line_qty"]
                ],
                on="product_code",
                how="left",
            )

        stats = stats.sort_values(
            ["coverage_rate", "n_stores", "n_weeks", "unique_prices", "total_units"],
            ascending=False,
        )
        path = self.config.out_dir / "dunnhumby_product_diagnostics.csv"
        stats.to_csv(path, index=False)
        print("Wrote", path)
        self.product_stats = stats
        return stats

    def screen_eligible(self, stats: pd.DataFrame) -> pd.DataFrame:
        """Household-panel screen. Do not start at coverage_rate >= 0.90."""
        cfg = self.config
        eligible = stats[
            (stats["n_stores"] >= cfg.min_stores)
            & (stats["n_weeks"] >= cfg.min_weeks)
            & (stats["unique_prices"] >= cfg.min_unique_prices)
            & (stats["price_cv"] >= cfg.min_price_cv)
        ].copy()
        eligible = eligible.sort_values(
            ["coverage_rate", "n_stores", "n_weeks", "unique_prices", "total_units"],
            ascending=False,
        )
        print("eligible SKUs:", len(eligible))
        print(
            eligible[
                [
                    "product_code",
                    "category",
                    "sub_commodity",
                    "coverage_rate",
                    "n_stores",
                    "n_weeks",
                    "unique_prices",
                    "price_cv",
                    "total_units",
                    "p99_line_qty",
                ]
            ].head(50)
        )
        self.eligible = eligible
        return eligible

    def choose_product_group(self, eligible: pd.DataFrame) -> tuple[str, str, str | None]:
        """Prefer SUB_COMMODITY_DESC (closer substitutes), else COMMODITY_DESC.

        Pre-specified: first group with >= min_products_in_group, ranked by
        median coverage then volume. Not a hand-picked yogurt/cereal list.
        """
        cfg = self.config
        sub = (
            eligible.groupby(["category", "sub_commodity"], observed=True)
            .agg(
                n_products=("product_code", "nunique"),
                median_coverage=("coverage_rate", "median"),
                median_price_cv=("price_cv", "median"),
                total_units=("total_units", "sum"),
            )
            .reset_index()
        )
        sub = sub[sub["n_products"] >= cfg.min_products_in_group].sort_values(
            ["median_coverage", "total_units"], ascending=False
        )
        print("sub-commodities with enough SKUs:")
        print(sub.head(30))

        if not sub.empty:
            row = sub.iloc[0]
            print("chosen sub_commodity:", row["category"], "/", row["sub_commodity"])
            return "sub_commodity", str(row["category"]), str(row["sub_commodity"])

        com = (
            eligible.groupby("category", observed=True)
            .agg(
                n_products=("product_code", "nunique"),
                median_coverage=("coverage_rate", "median"),
                total_units=("total_units", "sum"),
            )
            .reset_index()
        )
        com = com[com["n_products"] >= cfg.min_products_in_group].sort_values(
            ["median_coverage", "total_units"], ascending=False
        )
        print("commodities with enough SKUs:")
        print(com.head(30))
        if com.empty:
            raise ValueError("No commodity/sub-commodity has enough eligible SKUs.")
        row = com.iloc[0]
        print("chosen category:", row["category"])
        return "category", str(row["category"]), None

    def shortlist_candidates(
        self,
        eligible: pd.DataFrame,
        grouping_level: str,
        category: str,
        sub_commodity: str | None,
    ) -> pd.DataFrame:
        """Keep ~30 SKUs; causal coverage decides the final 10."""
        if grouping_level == "sub_commodity":
            mask = (eligible["category"] == category) & (
                eligible["sub_commodity"] == sub_commodity
            )
        else:
            mask = eligible["category"] == category
        candidates = eligible.loc[mask].head(self.config.n_candidate_skus).copy()
        print("candidate SKUs:", len(candidates))
        self.candidates = candidates
        return candidates

    def select_without_lookahead(self) -> pd.DataFrame:
        selection = self.selection_sample()
        self.select_core_stores(selection)
        selection_core = selection[selection["store_code"].isin(self.core_stores)]
        stats = self.compute_product_stats(selection_core)
        eligible = self.screen_eligible(stats)
        level, category, sub = self.choose_product_group(eligible)
        candidates = self.shortlist_candidates(eligible, level, category, sub)
        self.selection = DunnSampleSelection(
            cutoff_week_id=self.config.selection_cutoff,
            grouping_level=level,
            category=category,
            sub_commodity=sub,
            core_stores=list(self.core_stores),
            candidate_product_codes=candidates["product_code"].astype(str).tolist(),
            product_codes=[],
            n_eligible_in_group=int(len(candidates)),
            criteria={
                "min_store_weeks": self.config.min_store_weeks,
                "n_core_stores": self.config.n_core_stores,
                "min_stores": self.config.min_stores,
                "min_weeks": self.config.min_weeks,
                "min_unique_prices": self.config.min_unique_prices,
                "min_price_cv": self.config.min_price_cv,
                "min_promo_coverage": self.config.min_promo_coverage,
                "n_skus": self.config.n_skus,
            },
        )
        return candidates

    # ======================================================================
    # Causal promo (do not read the full 36.8M-row file at once)
    # ======================================================================

    def load_causal_for_candidates(self) -> pd.DataFrame:
        if self.candidates is None or not self.core_stores:
            raise RuntimeError("Call select_without_lookahead() first.")

        candidate_products = set(
            self.candidates["product_code"].astype("int64").tolist()
        )
        candidate_stores = {int(x) for x in self.core_stores}

        promo_frames: list[pd.DataFrame] = []
        path = self.config.data_dir / "causal_data.csv"
        for chunk in pd.read_csv(
            path,
            usecols=["PRODUCT_ID", "STORE_ID", "WEEK_NO", "display", "mailer"],
            dtype={
                "PRODUCT_ID": "int64",
                "STORE_ID": "int32",
                "WEEK_NO": "int16",
                "display": "string",
                "mailer": "string",
            },
            chunksize=self.config.causal_chunksize,
        ):
            mask = chunk["PRODUCT_ID"].isin(candidate_products) & chunk[
                "STORE_ID"
            ].isin(candidate_stores)
            subset = chunk.loc[mask]
            if len(subset):
                promo_frames.append(subset.copy())

        if not promo_frames:
            raise ValueError("No causal_data rows matched the candidate universe.")
        causal = pd.concat(promo_frames, ignore_index=True)
        print("causal subset:", causal.shape)
        return causal

    @staticmethod
    def construct_on_promo(causal: pd.DataFrame) -> pd.DataFrame:
        """Observed merchandising, not a markdown proxy.

        display == "0" / mailer == "0" -> no feature.
        Any other code -> some display position / mailer letter.
        Codes are categorical, not ordinal.
        """
        causal = causal.copy()
        causal["display"] = causal["display"].fillna("0").str.strip()
        causal["mailer"] = causal["mailer"].fillna("0").str.strip()
        causal["on_display"] = (causal["display"] != "0").astype("int8")
        causal["in_mailer"] = (causal["mailer"] != "0").astype("int8")
        causal["on_promo"] = (
            (causal["on_display"] == 1) | (causal["in_mailer"] == 1)
        ).astype("int8")

        promo_weekly = (
            causal.groupby(["PRODUCT_ID", "STORE_ID", "WEEK_NO"], observed=True)
            .agg(
                on_promo=("on_promo", "max"),
                on_display=("on_display", "max"),
                in_mailer=("in_mailer", "max"),
            )
            .reset_index()
        )
        promo_weekly["product_code"] = promo_weekly["PRODUCT_ID"].astype("string")
        promo_weekly["store_code"] = promo_weekly["STORE_ID"].astype("string")
        promo_weekly["week_id"] = promo_weekly["WEEK_NO"].astype("int16")
        return promo_weekly

    def attach_causal_promo(self) -> pd.DataFrame:
        """Left-join causal flags. Missing is *not* recoded as on_promo = 0."""
        if self.weekly is None or self.candidates is None:
            raise RuntimeError("Need transaction master and candidate shortlist.")

        causal = self.load_causal_for_candidates()
        self.promo_weekly = self.construct_on_promo(causal)

        panel = self.weekly[
            self.weekly["product_code"].isin(self.candidates["product_code"])
            & self.weekly["store_code"].isin(self.core_stores)
        ].copy()
        panel = panel.merge(
            self.promo_weekly[
                [
                    "product_code",
                    "store_code",
                    "week_id",
                    "on_promo",
                    "on_display",
                    "in_mailer",
                ]
            ],
            on=["product_code", "store_code", "week_id"],
            how="left",
            validate="one_to_one",
        )
        panel["promo_observed"] = panel["on_promo"].notna()
        self.candidate_panel = panel
        return panel

    def measure_promo_coverage(self) -> pd.DataFrame:
        if self.candidate_panel is None:
            raise RuntimeError("Call attach_causal_promo() first.")
        coverage = (
            self.candidate_panel.groupby("product_code", observed=True)
            .agg(n_rows=("units", "size"), promo_rows=("promo_observed", "sum"))
            .reset_index()
        )
        coverage["promo_coverage"] = coverage["promo_rows"] / coverage["n_rows"]
        print(coverage.sort_values("promo_coverage", ascending=False))
        return coverage

    def finalize_skus_with_promo_support(self) -> DunnSampleSelection:
        if self.candidates is None or self.selection is None:
            raise RuntimeError("Call select_without_lookahead() first.")

        coverage = self.measure_promo_coverage()
        candidates = self.candidates.merge(coverage, on="product_code", how="left")
        candidates = candidates[
            candidates["promo_coverage"] >= self.config.min_promo_coverage
        ].copy()
        candidates = candidates.sort_values(
            [
                "coverage_rate",
                "promo_coverage",
                "n_stores",
                "n_weeks",
                "unique_prices",
                "total_units",
            ],
            ascending=False,
        )
        if len(candidates) < self.config.n_skus:
            print(
                f"Warning: only {len(candidates)} SKUs reach "
                f"promo_coverage >= {self.config.min_promo_coverage}."
            )

        selected = candidates.head(self.config.n_skus)["product_code"].astype(str).tolist()
        print("SELECTED_PRODUCTS:", selected)

        self.candidates = candidates
        self.selection.product_codes = selected
        self.selection.n_eligible_in_group = int(len(candidates))
        self._measure_joint_coverage()

        path = self.config.out_dir / "dunnhumby_selected_skus.json"
        path.write_text(json.dumps(self.selection.to_dict(), indent=2))
        print("Wrote", path)
        return self.selection

    def _measure_joint_coverage(self) -> None:
        """Cross-elasticities need P_i and P_j in the same store-week."""
        assert self.candidate_panel is not None and self.selection is not None
        selected = self.selection.product_codes
        window = self.candidate_panel[
            (self.candidate_panel["week_id"] <= self.config.selection_cutoff)
            & self.candidate_panel["product_code"].isin(selected)
        ]
        presence = (
            window.assign(observed=1)
            .pivot_table(
                index=["store_code", "week_id"],
                columns="product_code",
                values="observed",
                aggfunc="max",
                fill_value=0,
            )
        )
        if presence.empty:
            print("Joint coverage: empty presence matrix.")
            return
        joint = presence.sum(axis=1) / len(selected)
        print(joint.describe())
        ge8 = float((joint >= 0.8).mean())
        all10 = float((joint == 1).mean())
        print("store-weeks with >=8/10 products observed:", ge8)
        print("store-weeks with 10/10 observed:", all10)
        self.selection.joint_coverage_ge_8_of_10 = ge8
        self.selection.joint_coverage_all_10 = all10

    # ======================================================================
    # ICDN panel
    # ======================================================================

    def build_icdn_panel(self) -> pd.DataFrame:
        """Main spec: keep rows with q > 0, p > 0 and *observed* promo.

        Missing causal is dropped, not filled with on_promo = 0.
        A robustness copy can recode missing promo as 0 via
        ``fill_missing_promo_zero``.
        """
        if self.candidate_panel is None or self.selection is None:
            raise RuntimeError("Need candidate panel and frozen SKUs.")

        panel = self.candidate_panel[
            self.candidate_panel["product_code"].isin(self.selection.product_codes)
        ].copy()
        n_missing_promo = int((~panel["promo_observed"]).sum())
        print("rows dropped for unobserved promo:", n_missing_promo)
        panel = panel[panel["promo_observed"]].copy()

        icdn = panel[
            [
                "store_code",
                "product_code",
                "week_id",
                "price",
                "units",
                "on_promo",
                "category",
                "brand",
            ]
        ].copy()
        icdn["on_promo"] = icdn["on_promo"].astype("int8")
        icdn = icdn.sort_values(
            ["week_id", "store_code", "product_code"]
        ).reset_index(drop=True)

        self.validate_icdn_panel(icdn)
        self.save_outputs(icdn)
        self.icdn_panel = icdn
        return icdn

    def validate_icdn_panel(self, icdn: pd.DataFrame) -> None:
        required = [
            "store_code",
            "product_code",
            "week_id",
            "price",
            "units",
            "on_promo",
        ]
        assert not icdn[required].isna().any().any()
        assert (icdn["price"] > 0).all()
        assert (icdn["units"] > 0).all()
        assert set(icdn["on_promo"].unique()).issubset({0, 1})
        assert not icdn.duplicated(
            ["store_code", "product_code", "week_id"]
        ).any()
        n_skus = icdn["product_code"].nunique()
        if n_skus != self.config.n_skus:
            print(f"Warning: icdn panel has {n_skus} SKUs, expected {self.config.n_skus}.")

    def save_outputs(self, icdn: pd.DataFrame) -> None:
        out = self.config.out_dir
        if self.weekly is not None:
            self.weekly.to_parquet(out / "dunnhumby_weekly_master.parquet", index=False)
        if self.candidate_panel is not None:
            self.candidate_panel.to_parquet(
                out / "dunnhumby_candidate_panel.parquet", index=False
            )
        icdn.to_parquet(out / "dunnhumby_icdn_panel.parquet", index=False)
        if self.candidates is not None:
            self.candidates.to_csv(
                out / "dunnhumby_selected_product_diagnostics.csv", index=False
            )
        print("Wrote ICDN panel", out / "dunnhumby_icdn_panel.parquet")

    @staticmethod
    def fill_missing_promo_zero(panel: pd.DataFrame) -> pd.DataFrame:
        """Robustness only: recode unobserved causal as on_promo = 0."""
        out = panel.copy()
        out["on_promo"] = out["on_promo"].fillna(0).astype("int8")
        return out