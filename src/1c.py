"""Build a 1C Predict Future Sales weekly panel for ICDN.

Observation unit
----------------
(store, product, week) -> (price, units, promo, category)

ICDN default columns
--------------------
store_code, product_code, week_id, price, units, on_promo
Optional: category. Brand / style / size are not parsed from item_name.

Source files
------------
Used:    sales_train.csv, items.csv, item_categories.csv, shops.csv
Unused:  test.csv (Nov 2015 shop-item pairs, no sales/prices),
         sample_submission.csv

Demand
------
Main analysis uses gross positive units, not net of returns:

    Q^{+}_ist = sum_{d in t} max(q_isd, 0)

Negative item_cnt_day rows are stored as returns for audit only.

Price
-----
Quantity-weighted unit value on positive days only:

    P_ist = sum (p_isd * q_isd) / sum q_isd ,   q_isd > 0

Not the simple mean of item_price.

Zeros
-----
No shelf-price table, so a missing shop-item-week is not a confirmed
zero sale. Do not fill gaps with units = 0 or invented prices.

Promo
-----
There is no observed promotion flag. on_promo is a *backward-looking
markdown proxy* (90th percentile of the previous 13 observed prices).
Never call it "observed promotion". Robustness: on_promo = 0.

week_id
-------
date_block_num is a monthly index 0..33. We build Monday-Sunday weeks
from date and map complete weeks to consecutive integers 1..T.
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
class OneCConfig:
    """Paths and pre-specified selection rules.

    Store / category / SKU choice uses only week_id <= selection cutoff.
    ``target_category_id`` is *not* set by default: inspect category_stats
    first (skip gift cards, services, face-value items, etc.).
    """

    data_dir: Path = Path("./data/predict-future-sales-1c")
    out_dir: Path | None = None

    selection_frac: float = 0.50

    min_store_week_coverage: float = 0.85
    n_core_stores: int = 20

    min_stores: int = 10
    min_week_coverage: float = 0.70
    min_unique_prices: int = 8
    min_price_cv: float = 0.03
    min_promo_ref_coverage: float = 0.80
    max_return_rate: float = 0.05
    min_products_in_category: int = 10

    n_candidate_skus: int = 30
    n_skus: int = 10
    jaccard_weight: float = 0.7
    coverage_weight: float = 0.3

    promo_lookback: int = 13
    promo_min_periods: int = 4
    promo_quantile: float = 0.90
    promo_threshold: float = 0.95

    # Set after inspecting 1c_category_diagnostics.csv. Example 40 is NOT a default.
    target_category_id: int | None = None

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.out_dir is None:
            self.out_dir = self.data_dir / "processed"
        else:
            self.out_dir = Path(self.out_dir)


@dataclass
class OneCSampleSelection:
    """Frozen universe used by OLS, Ridge, MLP and ICDN."""

    cutoff_week_id: int
    category_id: int
    category: str
    core_stores: list[str]
    candidate_product_codes: list[str]
    product_codes: list[str]
    n_eligible_in_category: int
    joint_coverage_ge_8_of_10: float | None = None
    joint_coverage_all_10: float | None = None
    criteria: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cutoff_week_id": int(self.cutoff_week_id),
            "category_id": int(self.category_id),
            "category": self.category,
            "core_stores": list(self.core_stores),
            "candidate_product_codes": list(self.candidate_product_codes),
            "product_codes": list(self.product_codes),
            "n_eligible_in_category": int(self.n_eligible_in_category),
            "joint_coverage_ge_8_of_10": self.joint_coverage_ge_8_of_10,
            "joint_coverage_all_10": self.joint_coverage_all_10,
            "criteria": self.criteria,
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class OneCWeeklyPanelBuilder:
    """Observed-purchase weekly master and ICDN-compatible subset.

    Outputs
    -------
    1c_weekly_master.parquet
        All valid positive shop-item-weeks, including markdown proxy.

    1c_returns_audit.parquet
        Raw negative item_cnt_day rows.

    1c_icdn_panel.parquet
        Frozen stores × 10 SKUs; rows with a promo reference only.
    """

    def __init__(self, config: OneCConfig | None = None) -> None:
        self.config = config or OneCConfig()
        self.config.out_dir.mkdir(parents=True, exist_ok=True)

        self.sales: pd.DataFrame | None = None
        self.items: pd.DataFrame | None = None
        self.categories: pd.DataFrame | None = None
        self.shops: pd.DataFrame | None = None
        self.sales_audit: dict = {}

        self.week_map: pd.DataFrame | None = None
        self.complete_week_starts: list = []
        self.n_selection_weeks: int = 0
        self.selection_cutoff: int | None = None

        self.weekly: pd.DataFrame | None = None
        self.store_stats: pd.DataFrame | None = None
        self.product_stats: pd.DataFrame | None = None
        self.category_stats: pd.DataFrame | None = None
        self.eligible: pd.DataFrame | None = None
        self.candidates: pd.DataFrame | None = None
        self.core_stores: list[str] = []
        self.presence: pd.DataFrame | None = None
        self.selection: OneCSampleSelection | None = None
        self.icdn_panel: pd.DataFrame | None = None

    # ======================================================================
    # Orchestration
    # ======================================================================

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, OneCSampleSelection]:
        """Requires ``config.target_category_id`` after inspecting categories."""
        if self.config.target_category_id is None:
            raise ValueError(
                "Inspect category_stats (see rank_categories), then set "
                "OneCConfig.target_category_id. Do not choose a category from "
                "downstream ICDN vs MLP results."
            )
        self.build_weekly_master()
        self.diagnose_selection_window()
        self.freeze_universe(self.config.target_category_id)
        self.build_icdn_panel()
        return self.weekly, self.icdn_panel, self.selection

    def build_weekly_master(self) -> pd.DataFrame:
        self.load_tables()
        self.parse_dates()
        self.audit_sales()
        positive, returns = self.split_sales_and_returns()
        positive = self.build_week_index(positive)
        weekly = self.aggregate_daily_to_weekly(positive)
        weekly = self.attach_weekly_returns(weekly, returns)
        weekly = self.attach_metadata(weekly)
        weekly = self.construct_markdown_promo(weekly)
        self.weekly = self.save_weekly_master(weekly)
        return self.weekly

    # ======================================================================
    # Load
    # ======================================================================

    def load_tables(self) -> None:
        data = self.config.data_dir
        self.sales = pd.read_csv(
            data / "sales_train.csv",
            dtype={
                "date_block_num": "int16",
                "shop_id": "int16",
                "item_id": "int32",
                "item_price": "float32",
                "item_cnt_day": "float32",
            },
        )
        self.items = pd.read_csv(
            data / "items.csv",
            dtype={
                "item_name": "string",
                "item_id": "int32",
                "item_category_id": "int16",
            },
        )
        self.categories = pd.read_csv(
            data / "item_categories.csv",
            dtype={
                "item_category_name": "string",
                "item_category_id": "int16",
            },
        )
        self.shops = pd.read_csv(
            data / "shops.csv",
            dtype={"shop_name": "string", "shop_id": "int16"},
        )
        print("sales", self.sales.shape, "items", self.items.shape)

    def parse_dates(self) -> None:
        if self.sales is None:
            raise RuntimeError("Call load_tables() first.")
        self.sales["date"] = pd.to_datetime(
            self.sales["date"], format="%d.%m.%Y", errors="raise"
        )
        self.sales = self.sales.sort_values("date").reset_index(drop=True)

    def audit_sales(self) -> dict:
        if self.sales is None:
            raise RuntimeError("Call parse_dates() first.")
        s = self.sales
        self.sales_audit = {
            "rows": int(len(s)),
            "shops": int(s["shop_id"].nunique()),
            "items": int(s["item_id"].nunique()),
            "months": int(s["date_block_num"].nunique()),
            "min_date": str(s["date"].min().date()),
            "max_date": str(s["date"].max().date()),
            "negative_units": int((s["item_cnt_day"] < 0).sum()),
            "zero_units": int((s["item_cnt_day"] == 0).sum()),
            "nonpositive_price": int((s["item_price"] <= 0).sum()),
            "missing_price": int(s["item_price"].isna().sum()),
        }
        print(pd.Series(self.sales_audit))
        path = self.config.out_dir / "1c_sales_audit.json"
        path.write_text(json.dumps(self.sales_audit, indent=2))
        return self.sales_audit

    def split_sales_and_returns(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Gross demand vs returns. No global Kaggle-style winsorizing."""
        if self.sales is None:
            raise RuntimeError("Call audit_sales() first.")
        returns = self.sales[self.sales["item_cnt_day"] < 0].copy()
        positive = self.sales[
            (self.sales["item_cnt_day"] > 0)
            & (self.sales["item_price"] > 0)
            & self.sales["item_price"].notna()
        ].copy()
        positive["revenue"] = positive["item_price"] * positive["item_cnt_day"]

        ret_path = self.config.out_dir / "1c_returns_audit.parquet"
        returns.to_parquet(ret_path, index=False)
        print(
            f"positive rows {len(positive):,} | return rows {len(returns):,} "
            f"(wrote {ret_path})"
        )
        return positive, returns

    # ======================================================================
    # Weeks: not date_block_num, not ISO week numbers
    # ======================================================================

    def build_week_index(self, positive: pd.DataFrame) -> pd.DataFrame:
        """Monday-Sunday weeks; drop truncated first/last weeks; week_id = 1..T."""
        if self.sales is None:
            raise RuntimeError("Need dated sales for the calendar span.")

        positive = positive.copy()
        positive["week_start"] = (
            positive["date"].dt.to_period("W-SUN").dt.start_time
        )

        full_calendar = pd.DataFrame(
            {
                "date": pd.date_range(
                    self.sales["date"].min(),
                    self.sales["date"].max(),
                    freq="D",
                )
            }
        )
        full_calendar["week_start"] = (
            full_calendar["date"].dt.to_period("W-SUN").dt.start_time
        )
        week_lengths = (
            full_calendar.groupby("week_start").size().rename("n_days").reset_index()
        )
        print(week_lengths["n_days"].value_counts().sort_index())

        complete = (
            week_lengths.loc[week_lengths["n_days"] == 7, "week_start"]
            .sort_values()
            .tolist()
        )
        self.complete_week_starts = complete
        positive = positive[positive["week_start"].isin(complete)].copy()

        week_map = pd.DataFrame(
            {"week_start": sorted(positive["week_start"].unique())}
        )
        week_map["week_id"] = np.arange(1, len(week_map) + 1, dtype=np.int32)
        self.week_map = week_map

        positive = positive.merge(
            week_map, on="week_start", how="left", validate="many_to_one"
        )
        print(
            week_map.head(8).to_string(index=False),
            "\n…",
            len(week_map),
            "complete weeks with positive sales",
        )
        return positive

    # ======================================================================
    # Daily → weekly
    # ======================================================================

    def aggregate_daily_to_weekly(self, positive: pd.DataFrame) -> pd.DataFrame:
        weekly = (
            positive.groupby(
                ["shop_id", "item_id", "week_id", "week_start"],
                observed=True,
            )
            .agg(
                units=("item_cnt_day", "sum"),
                revenue=("revenue", "sum"),
                n_sales_days=("date", "nunique"),
                n_price_points=("item_price", "nunique"),
                min_daily_price=("item_price", "min"),
                max_daily_price=("item_price", "max"),
            )
            .reset_index()
        )
        weekly["price"] = weekly["revenue"] / weekly["units"]
        weekly = weekly[
            np.isfinite(weekly["price"])
            & np.isfinite(weekly["units"])
            & (weekly["price"] > 0)
            & (weekly["units"] > 0)
        ].copy()
        print(weekly[["price", "units"]].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
        return weekly

    def attach_weekly_returns(
        self, weekly: pd.DataFrame, returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Audit only. Returns do not enter log-demand."""
        if self.week_map is None:
            raise RuntimeError("Call build_week_index() first.")
        returns = returns.copy()
        returns["week_start"] = returns["date"].dt.to_period("W-SUN").dt.start_time
        returns = returns[returns["week_start"].isin(self.complete_week_starts)]
        returns = returns.merge(
            self.week_map, on="week_start", how="inner", validate="many_to_one"
        )
        weekly_returns = (
            returns.groupby(["shop_id", "item_id", "week_id"], observed=True)
            .agg(
                return_units=("item_cnt_day", lambda x: float(-x.sum())),
                n_return_rows=("item_cnt_day", "size"),
            )
            .reset_index()
        )
        weekly = weekly.merge(
            weekly_returns,
            on=["shop_id", "item_id", "week_id"],
            how="left",
            validate="one_to_one",
        )
        weekly[["return_units", "n_return_rows"]] = weekly[
            ["return_units", "n_return_rows"]
        ].fillna(0)
        return weekly

    def attach_metadata(self, weekly: pd.DataFrame) -> pd.DataFrame:
        if self.items is None or self.categories is None or self.shops is None:
            raise RuntimeError("Call load_tables() first.")
        product_master = self.items.merge(
            self.categories,
            on="item_category_id",
            how="left",
            validate="many_to_one",
        )
        weekly = weekly.merge(
            product_master, on="item_id", how="left", validate="many_to_one"
        )
        weekly = weekly.merge(
            self.shops, on="shop_id", how="left", validate="many_to_one"
        )
        weekly = weekly.rename(
            columns={
                "shop_id": "store_code",
                "item_id": "product_code",
                "item_category_name": "category",
            }
        )
        weekly["store_code"] = weekly["store_code"].astype("string")
        weekly["product_code"] = weekly["product_code"].astype("string")
        weekly["week_id"] = weekly["week_id"].astype("int32")
        return weekly

    def construct_markdown_promo(self, weekly: pd.DataFrame) -> pd.DataFrame:
        """Backward-looking markdown proxy, not observed promotion.

            P_ref(i,s,t) = Q_0.90 { P(i,s,t-13), ..., P(i,s,t-1) }

        Lookback is over *prior observed* store-product weeks (the panel is
        not dense). P90 is used instead of max so one spike does not label
        the next 13 weeks as promotions.

            on_promo = 1[ P_t <= 0.95 * P_ref ]  if P_ref exists, else NA
        """
        cfg = self.config
        weekly = weekly.sort_values(
            ["store_code", "product_code", "week_id"]
        ).reset_index(drop=True)

        lagged = weekly.groupby(
            ["store_code", "product_code"], observed=True
        )["price"].shift(1)
        weekly["regular_price_ref"] = (
            lagged.groupby(
                [weekly["store_code"], weekly["product_code"]],
                observed=True,
            )
            .rolling(window=cfg.promo_lookback, min_periods=cfg.promo_min_periods)
            .quantile(cfg.promo_quantile)
            .reset_index(level=[0, 1], drop=True)
        )
        weekly["promo_ref_available"] = weekly["regular_price_ref"].notna()
        weekly["on_promo"] = np.where(
            weekly["promo_ref_available"],
            (
                weekly["price"]
                <= cfg.promo_threshold * weekly["regular_price_ref"]
            ).astype("int8"),
            np.nan,
        )
        print(
            "promo_ref available:",
            weekly["promo_ref_available"].mean(),
            "| on_promo rate among those:",
            weekly.loc[weekly["promo_ref_available"], "on_promo"].mean(),
        )
        return weekly

    def save_weekly_master(self, weekly: pd.DataFrame) -> pd.DataFrame:
        weekly = weekly.sort_values(
            ["week_id", "store_code", "product_code"]
        ).reset_index(drop=True)
        path = self.config.out_dir / "1c_weekly_master.parquet"
        weekly.to_parquet(path, index=False)
        print("Wrote", path)
        return weekly

    @staticmethod
    def zero_promo_copy(panel: pd.DataFrame) -> pd.DataFrame:
        """Robustness: ICDN vs MLP must not hinge on the markdown proxy."""
        out = panel.copy()
        out["on_promo"] = np.int8(0)
        return out

    # ======================================================================
    # Selection without lookahead
    # ======================================================================

    def selection_sample(self) -> pd.DataFrame:
        if self.weekly is None:
            raise RuntimeError("Call build_weekly_master() first.")
        weeks = np.sort(self.weekly["week_id"].unique())
        cutoff = int(weeks[int(len(weeks) * self.config.selection_frac) - 1])
        self.selection_cutoff = cutoff
        print("Selection cutoff:", cutoff, "of", int(weeks.min()), "→", int(weeks.max()))
        sample = self.weekly[self.weekly["week_id"] <= cutoff].copy()
        self.n_selection_weeks = int(sample["week_id"].nunique())
        return sample

    def select_core_stores(self, selection: pd.DataFrame) -> list[str]:
        cfg = self.config
        store_stats = (
            selection.groupby("store_code", observed=True)
            .agg(
                n_obs=("units", "size"),
                n_items=("product_code", "nunique"),
                n_weeks=("week_id", "nunique"),
                total_units=("units", "sum"),
                total_revenue=("revenue", "sum"),
            )
            .reset_index()
        )
        store_stats["week_coverage"] = store_stats["n_weeks"] / self.n_selection_weeks
        eligible_stores = store_stats[
            store_stats["week_coverage"] >= cfg.min_store_week_coverage
        ].copy()
        n_core = min(cfg.n_core_stores, len(eligible_stores))
        core_stores = (
            eligible_stores.sort_values(["n_obs", "total_units"], ascending=False)
            .head(n_core)["store_code"]
            .astype(str)
            .tolist()
        )
        path = self.config.out_dir / "1c_store_diagnostics.csv"
        store_stats.to_csv(path, index=False)
        print("core stores:", core_stores)
        self.store_stats = store_stats
        self.core_stores = core_stores
        return core_stores

    def compute_product_stats(self, selection_core: pd.DataFrame) -> pd.DataFrame:
        stats = (
            selection_core.groupby(
                ["product_code", "item_category_id", "category"],
                observed=True,
            )
            .agg(
                n_obs=("units", "size"),
                n_stores=("store_code", "nunique"),
                n_weeks=("week_id", "nunique"),
                total_units=("units", "sum"),
                total_revenue=("revenue", "sum"),
                unique_prices=("price", "nunique"),
                mean_price=("price", "mean"),
                std_price=("price", "std"),
                median_units=("units", "median"),
                max_units=("units", "max"),
                return_units=("return_units", "sum"),
                promo_ref_rows=("promo_ref_available", "sum"),
                promo_rate=("on_promo", "mean"),
            )
            .reset_index()
        )
        stats["price_cv"] = stats["std_price"] / stats["mean_price"]
        possible = len(self.core_stores) * self.n_selection_weeks
        stats["coverage_rate"] = stats["n_obs"] / possible
        stats["promo_ref_coverage"] = stats["promo_ref_rows"] / stats["n_obs"]
        stats["return_rate"] = stats["return_units"] / (
            stats["total_units"] + stats["return_units"]
        )
        path = self.config.out_dir / "1c_product_diagnostics.csv"
        stats.to_csv(path, index=False)
        print("Wrote", path)
        self.product_stats = stats
        return stats

    def screen_eligible(self, stats: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        min_weeks = cfg.min_week_coverage * self.n_selection_weeks
        eligible = stats[
            (stats["n_stores"] >= cfg.min_stores)
            & (stats["n_weeks"] >= min_weeks)
            & (stats["unique_prices"] >= cfg.min_unique_prices)
            & (stats["price_cv"] >= cfg.min_price_cv)
            & (stats["promo_ref_coverage"] >= cfg.min_promo_ref_coverage)
            & (stats["return_rate"] <= cfg.max_return_rate)
        ].copy()
        eligible = eligible.sort_values(
            [
                "coverage_rate",
                "n_stores",
                "n_weeks",
                "unique_prices",
                "price_cv",
                "total_units",
            ],
            ascending=False,
        )
        print("eligible SKUs:", len(eligible))
        print(
            eligible[
                [
                    "product_code",
                    "category",
                    "coverage_rate",
                    "n_stores",
                    "n_weeks",
                    "unique_prices",
                    "price_cv",
                    "promo_rate",
                    "return_rate",
                    "total_units",
                ]
            ].head(50)
        )
        self.eligible = eligible
        return eligible

    def rank_categories(self, eligible: pd.DataFrame) -> pd.DataFrame:
        """Stop here and inspect. Do not auto-pick gift cards / services."""
        stats = (
            eligible.groupby(["item_category_id", "category"], observed=True)
            .agg(
                n_products=("product_code", "nunique"),
                median_coverage=("coverage_rate", "median"),
                median_price_cv=("price_cv", "median"),
                median_n_stores=("n_stores", "median"),
                total_units=("total_units", "sum"),
            )
            .reset_index()
        )
        stats = stats[
            stats["n_products"] >= self.config.min_products_in_category
        ].sort_values(
            ["median_coverage", "median_n_stores", "total_units"],
            ascending=False,
        )
        path = self.config.out_dir / "1c_category_diagnostics.csv"
        stats.to_csv(path, index=False)
        print(stats.head(30))
        self.category_stats = stats
        return stats

    def diagnose_selection_window(self) -> pd.DataFrame:
        selection = self.selection_sample()
        self.select_core_stores(selection)
        selection_core = selection[selection["store_code"].isin(self.core_stores)]
        stats = self.compute_product_stats(selection_core)
        eligible = self.screen_eligible(stats)
        return self.rank_categories(eligible)

    # ======================================================================
    # Greedy co-occurrence SKU pick
    # ======================================================================

    @staticmethod
    def pairwise_jaccard(presence: pd.DataFrame) -> pd.DataFrame:
        """J(i,j) = |A_i ∩ A_j| / |A_i ∪ A_j| on store-week presence."""
        cols = presence.columns.tolist()
        x = presence.to_numpy().astype(bool)
        inter = x.T @ x
        size = x.sum(axis=0)
        union = size[:, None] + size[None, :] - inter
        scores = np.divide(
            inter,
            union,
            out=np.zeros(inter.shape, dtype=float),
            where=union > 0,
        )
        return pd.DataFrame(scores, index=cols, columns=cols)

    def greedy_select_products(
        self,
        candidates: pd.DataFrame,
        presence: pd.DataFrame,
    ) -> list[str]:
        cfg = self.config
        rank = candidates.set_index("product_code")
        jaccard = self.pairwise_jaccard(presence)
        first = (
            candidates.sort_values("coverage_rate", ascending=False)
            .iloc[0]["product_code"]
        )
        selected = [str(first)]
        remaining_pool = [str(p) for p in candidates["product_code"].tolist()]

        while len(selected) < cfg.n_skus:
            remaining = [p for p in remaining_pool if p not in selected]
            if not remaining:
                break
            scores = {}
            for p in remaining:
                mean_j = float(jaccard.loc[p, selected].mean())
                cov = float(rank.loc[p, "coverage_rate"])
                scores[p] = cfg.jaccard_weight * mean_j + cfg.coverage_weight * cov
            selected.append(max(scores, key=scores.get))
        return selected

    def freeze_universe(self, category_id: int) -> OneCSampleSelection:
        """Lock stores, category and 10 SKUs. Never revise after seeing models."""
        if self.eligible is None or self.weekly is None:
            raise RuntimeError("Call diagnose_selection_window() first.")
        if self.selection_cutoff is None:
            raise RuntimeError("selection_cutoff is not set.")

        candidates = (
            self.eligible[self.eligible["item_category_id"] == category_id]
            .sort_values(
                [
                    "coverage_rate",
                    "n_stores",
                    "n_weeks",
                    "unique_prices",
                    "total_units",
                ],
                ascending=False,
            )
            .head(self.config.n_candidate_skus)
            .copy()
        )
        if candidates.empty:
            raise ValueError(f"No eligible SKUs in item_category_id={category_id}.")

        selection_core = self.weekly[
            (self.weekly["week_id"] <= self.selection_cutoff)
            & self.weekly["store_code"].isin(self.core_stores)
            & self.weekly["product_code"].isin(candidates["product_code"])
        ]
        presence = (
            selection_core.assign(observed=1)
            .pivot_table(
                index=["store_code", "week_id"],
                columns="product_code",
                values="observed",
                aggfunc="max",
                fill_value=0,
            )
        )
        selected = self.greedy_select_products(candidates, presence)
        print("SELECTED_PRODUCTS:", selected)

        selected_presence = presence[selected]
        joint = selected_presence.sum(axis=1) / len(selected)
        print(joint.describe())
        ge8 = float((joint >= 0.80).mean())
        all10 = float((joint == 1.0).mean())
        print("store-weeks >= 8/10:", ge8)
        print("store-weeks = 10/10:", all10)
        print(selected_presence.mean().sort_values())

        cat_name = str(candidates["category"].iloc[0])
        self.candidates = candidates
        self.presence = presence
        self.selection = OneCSampleSelection(
            cutoff_week_id=self.selection_cutoff,
            category_id=int(category_id),
            category=cat_name,
            core_stores=list(self.core_stores),
            candidate_product_codes=candidates["product_code"].astype(str).tolist(),
            product_codes=selected,
            n_eligible_in_category=int(
                (self.eligible["item_category_id"] == category_id).sum()
            ),
            joint_coverage_ge_8_of_10=ge8,
            joint_coverage_all_10=all10,
            criteria={
                "selection_frac": self.config.selection_frac,
                "min_store_week_coverage": self.config.min_store_week_coverage,
                "n_core_stores": self.config.n_core_stores,
                "min_stores": self.config.min_stores,
                "min_week_coverage": self.config.min_week_coverage,
                "min_unique_prices": self.config.min_unique_prices,
                "min_price_cv": self.config.min_price_cv,
                "min_promo_ref_coverage": self.config.min_promo_ref_coverage,
                "max_return_rate": self.config.max_return_rate,
                "n_skus": self.config.n_skus,
                "jaccard_weight": self.config.jaccard_weight,
            },
        )
        self._save_frozen_lists()
        return self.selection

    def _save_frozen_lists(self) -> None:
        assert self.selection is not None
        out = self.config.out_dir
        pd.Series(self.selection.core_stores, name="store_code").to_csv(
            out / "1c_selected_stores.csv", index=False
        )
        pd.Series(self.selection.product_codes, name="product_code").to_csv(
            out / "1c_selected_products.csv", index=False
        )
        (out / "1c_selected_skus.json").write_text(
            json.dumps(self.selection.to_dict(), indent=2)
        )
        print("Wrote frozen store/SKU lists under", out)

    # ======================================================================
    # ICDN panel
    # ======================================================================

    def build_icdn_panel(self) -> pd.DataFrame:
        """Full horizon, frozen universe, rows with a promo reference only."""
        if self.weekly is None or self.selection is None:
            raise RuntimeError("Need weekly master and freeze_universe().")

        panel = self.weekly[
            self.weekly["store_code"].isin(self.selection.core_stores)
            & self.weekly["product_code"].isin(self.selection.product_codes)
        ].copy()
        n_no_ref = int((~panel["promo_ref_available"]).sum())
        print("rows dropped without promo reference:", n_no_ref)
        panel = panel[panel["promo_ref_available"]].copy()
        panel["on_promo"] = panel["on_promo"].astype("int8")

        icdn = panel[
            [
                "store_code",
                "product_code",
                "week_id",
                "price",
                "units",
                "on_promo",
                "category",
            ]
        ].copy()
        icdn = icdn.sort_values(
            ["week_id", "store_code", "product_code"]
        ).reset_index(drop=True)

        self.validate_icdn_panel(icdn)
        path = self.config.out_dir / "1c_icdn_panel.parquet"
        icdn.to_parquet(path, index=False)
        print("Wrote", path)
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
            print(f"Warning: {n_skus} SKUs in ICDN panel, expected {self.config.n_skus}.")
        week_ids = np.sort(icdn["week_id"].unique())
        assert (np.diff(week_ids) >= 1).all()