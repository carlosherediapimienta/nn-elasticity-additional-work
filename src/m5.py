"""Build a Walmart M5 weekly panel for ICDN.

Observation unit
----------------
(store, product, week) -> (price, units, promo)

ICDN default columns
--------------------
store_code, product_code, week_id, price, units, on_promo
Optional: category, brand, style, size.

week_id must be a consecutive integer time index. Using M5's wm_yr_wk
codes (11152, 11201, ...) would insert false gaps into ICDN lags.

Source files
------------
Use sales_train_evaluation.csv when available: it already contains the
validation history (d_1..d_1913) and extends it to d_1941. Do not
concatenate sales_train_validation.csv. sample_submission.csv is unused.

ICDN 1.0.0 units constraint
---------------------------
Docs say units >= 0, but 1.0.0 validates units > 0 and later takes
log(units). Zero-sales weeks cannot be passed to ICDN, and zeros must
not be recoded as ones. SKU selection is therefore restricted to dense,
high-turnover products.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class M5Config:
    """Paths and sample-selection criteria.

    Selection thresholds are applied only on the initial window
    (see ``selection_frac``), never on the full sample.
    """

    data_dir: Path = Path("./m5")
    out_dir: Path | None = None

    # Markdown-proxy promotion: P_t <= threshold * max(P_{t-L}, ..., P_{t-1}).
    promo_lookback: int = 13
    promo_min_periods: int = 4
    promo_threshold: float = 0.95

    # Selection window: first fraction of complete weeks, no lookahead.
    selection_frac: float = 0.50

    # Pre-specified density / price-variation screen (not manual SKU picking).
    min_positive_rate: float = 0.90
    min_stores: int = 8
    min_unique_prices: int = 8
    min_eligible_in_department: int = 15
    n_skus: int = 10

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.out_dir is None:
            self.out_dir = self.data_dir / "processed"
        else:
            self.out_dir = Path(self.out_dir)


@dataclass
class SampleSelection:
    """Frozen universe used by OLS, Ridge, MLP and ICDN."""

    cutoff_week_id: int
    category: str
    product_codes: list[str]
    n_eligible_in_category: int
    criteria: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cutoff_week_id": int(self.cutoff_week_id),
            "category": self.category,
            "product_codes": list(self.product_codes),
            "n_eligible_in_category": int(self.n_eligible_in_category),
            "criteria": self.criteria,
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class M5WeeklyPanelBuilder:
    """Construct the scientific master panel and the ICDN-compatible subset.

    Two outputs
    -----------
    m5_weekly_master.parquet
        Includes units == 0. Source of diagnostics and sample selection.

    m5_icdn_panel.parquet
        Only ICDN-compatible rows (units > 0) for the frozen SKU list.
    """

    META_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

    def __init__(self, config: M5Config | None = None) -> None:
        self.config = config or M5Config()
        self.config.out_dir.mkdir(parents=True, exist_ok=True)

        self.sales_file: Path | None = None
        self.calendar: pd.DataFrame | None = None
        self.week_map: pd.DataFrame | None = None
        self.sales: pd.DataFrame | None = None
        self.day_cols: list[str] = []
        self.calendar_train: pd.DataFrame | None = None
        self.full_weeks: set[int] = set()
        self.n_imputed_zero_units: int = 0

        self.master: pd.DataFrame | None = None
        self.product_stats: pd.DataFrame | None = None
        self.selection: SampleSelection | None = None
        self.icdn_panel: pd.DataFrame | None = None

    # ======================================================================
    # Public orchestration
    # ======================================================================

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, SampleSelection]:
        """Build master panel, freeze SKUs, then export the ICDN panel."""
        self.build_master()
        self.diagnose_sparsity()
        self.select_products_without_lookahead()
        self.build_icdn_panel()
        return self.master, self.icdn_panel, self.selection

    def build_master(self) -> pd.DataFrame:
        """Full weekly panel including zero-sales weeks."""
        self.resolve_sales_file()
        self.load_calendar()
        self.build_week_index()
        self.load_sales()
        self.restrict_calendar_to_observed_days()
        self.identify_complete_weeks()
        weekly_sales = self.aggregate_daily_to_weekly()
        weekly_sales = self.attach_wm_yr_wk(weekly_sales)
        prices = self.load_prices()
        prices = self.construct_markdown_promo(prices)
        master = self.merge_availability_and_sales(prices, weekly_sales)
        master = self.to_icdn_schema(master)
        self.validate_master(master)
        self.master = self.save_master(master)
        return self.master

    # ======================================================================
    # 1. File resolution
    # ======================================================================

    def resolve_sales_file(self) -> Path:
        """Prefer evaluation: it already contains validation history.

        sales_train_evaluation.csv covers d_1..d_1941.
        Do not concatenate sales_train_validation.csv.
        sample_submission.csv is not used.
        """
        data_dir = self.config.data_dir
        evaluation = data_dir / "sales_train_evaluation.csv"
        validation = data_dir / "sales_train_validation.csv"

        if evaluation.exists():
            self.sales_file = evaluation
        elif validation.exists():
            self.sales_file = validation
        else:
            raise FileNotFoundError(
                f"Neither evaluation nor validation sales file found in {data_dir}"
            )

        print("Using:", self.sales_file.name)
        return self.sales_file

    # ======================================================================
    # 2. Calendar and sequential week index
    # ======================================================================

    def load_calendar(self) -> pd.DataFrame:
        path = self.config.data_dir / "calendar.csv"
        calendar = pd.read_csv(path, parse_dates=["date"])
        calendar = calendar.sort_values("date").reset_index(drop=True)
        self.calendar = calendar
        return calendar

    def build_week_index(self) -> pd.DataFrame:
        """Map wm_yr_wk -> consecutive week_id.

        M5 codes such as 11152, 11201 are *not* a linear time index.
        ICDN builds lags on integer week_id, so those codes would create
        spurious gaps. week_id is 1, 2, ..., T in calendar order.
        """
        if self.calendar is None:
            raise RuntimeError("Call load_calendar() first.")

        week_map = (
            self.calendar[["wm_yr_wk"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        week_map["week_id"] = np.arange(1, len(week_map) + 1, dtype=np.int32)

        self.calendar = self.calendar.merge(
            week_map,
            on="wm_yr_wk",
            how="left",
            validate="many_to_one",
        )
        self.week_map = week_map

        print(
            self.calendar[["date", "d", "wm_yr_wk", "week_id"]].head(10)
        )
        return week_map

    # ======================================================================
    # 3. Sales (wide format; do not melt all days)
    # ======================================================================

    def load_sales(self) -> pd.DataFrame:
        """Load the wide item-store x day matrix.

        A full melt of ~1,941 days would produce ~59 million rows.
        We aggregate days into weeks first, then melt the weekly columns.
        """
        if self.sales_file is None:
            raise RuntimeError("Call resolve_sales_file() first.")

        header = pd.read_csv(self.sales_file, nrows=0).columns
        self.day_cols = [c for c in header if c.startswith("d_")]

        dtype_sales = {
            "id": "string",
            "item_id": "category",
            "dept_id": "category",
            "cat_id": "category",
            "store_id": "category",
            "state_id": "category",
        }
        dtype_sales.update({c: "float32" for c in self.day_cols})

        self.sales = pd.read_csv(self.sales_file, dtype=dtype_sales)
        print(self.sales.shape)
        print(len(self.day_cols))
        return self.sales

    # ======================================================================
    # 4. Restrict calendar to observed sales days
    # ======================================================================

    def restrict_calendar_to_observed_days(self) -> pd.DataFrame:
        """calendar.csv also contains forecast dates with no sales columns."""
        if self.calendar is None:
            raise RuntimeError("Call build_week_index() first.")

        calendar_train = self.calendar[
            self.calendar["d"].isin(self.day_cols)
        ].copy()

        assert calendar_train["d"].nunique() == len(self.day_cols)
        self.calendar_train = calendar_train
        return calendar_train

    def identify_complete_weeks(self) -> set[int]:
        """Keep scientific weeks with seven observed days.

        The first and/or last M5 week can be truncated. Incomplete weeks
        are dropped rather than rescaled.
        """
        if self.calendar_train is None:
            raise RuntimeError("Call restrict_calendar_to_observed_days() first.")

        days_per_week = (
            self.calendar_train.groupby(["week_id", "wm_yr_wk"], observed=True)["d"]
            .nunique()
            .reset_index(name="n_days")
        )
        print(days_per_week["n_days"].value_counts().sort_index())

        self.full_weeks = set(
            days_per_week.loc[days_per_week["n_days"] == 7, "week_id"]
        )
        print(f"Complete weeks retained: {len(self.full_weeks)}")
        return self.full_weeks

    def _compact_week_id(self, df: pd.DataFrame, col: str = "week_id") -> pd.DataFrame:
        """Reindex remaining complete weeks to 1..T.

        After dropping truncated calendar weeks, the raw week_id may
        start at 2. Compacting avoids a hole at the beginning of the
        ICDN integer grid without using wm_yr_wk.
        """
        remaining = np.sort(df[col].unique())
        remap = {int(old): np.int32(new) for new, old in enumerate(remaining, start=1)}
        df[col] = df[col].map(remap).astype("int32")
        return df

    # ======================================================================
    # 5. Daily -> weekly units
    # ======================================================================

    def aggregate_daily_to_weekly(self) -> pd.DataFrame:
        """Q_isw = sum_{d in w} Q_isd, then melt to long format."""
        if self.sales is None or self.calendar_train is None:
            raise RuntimeError("Sales and calendar_train must be loaded.")

        weekly_sales = self.sales[self.META_COLS].copy()
        week_columns: list[str] = []

        for week_id, group in self.calendar_train.groupby("week_id", sort=True):
            if week_id not in self.full_weeks:
                continue

            dcols = group["d"].tolist()
            col = f"week_{int(week_id)}"
            weekly_sales[col] = (
                self.sales[dcols].sum(axis=1).astype("float32")
            )
            week_columns.append(col)

        weekly_sales = weekly_sales.melt(
            id_vars=self.META_COLS,
            value_vars=week_columns,
            var_name="week",
            value_name="units",
        )
        weekly_sales["week_id"] = (
            weekly_sales["week"]
            .str.replace("week_", "", regex=False)
            .astype("int32")
        )
        weekly_sales = weekly_sales.drop(columns="week")
        weekly_sales["units"] = weekly_sales["units"].astype("float32")

        # Free the wide daily matrix; it is no longer needed.
        self.sales = None
        return weekly_sales

    # ======================================================================
    # 6. Recover wm_yr_wk for the price join
    # ======================================================================

    def attach_wm_yr_wk(self, weekly_sales: pd.DataFrame) -> pd.DataFrame:
        week_lookup = (
            self.calendar_train[["week_id", "wm_yr_wk"]].drop_duplicates()
        )
        return weekly_sales.merge(
            week_lookup,
            on="week_id",
            how="left",
            validate="many_to_one",
        )

    # ======================================================================
    # 7–8. Prices and markdown promo proxy
    # ======================================================================

    def load_prices(self) -> pd.DataFrame:
        """sell_prices.csv is defined at store_id x item_id x wm_yr_wk.

        A missing price is the M5 signal that the item was not offered
        in that store-week (pre-launch / not in assortment).
        """
        if self.week_map is None:
            raise RuntimeError("Call build_week_index() first.")

        prices = pd.read_csv(
            self.config.data_dir / "sell_prices.csv",
            dtype={
                "store_id": "category",
                "item_id": "category",
                "wm_yr_wk": "int32",
                "sell_price": "float32",
            },
        )
        prices = prices.merge(
            self.week_map,
            on="wm_yr_wk",
            how="left",
            validate="many_to_one",
        )
        return prices

    def construct_markdown_promo(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Markdown-based promotional proxy (not an observed promotion).

        Reference price (current week excluded via shift(1)):

            P_ref(i,s,t) = max{P(i,s,t-L), ..., P(i,s,t-1)}

        Promo flag:

            on_promo = 1[ P(i,s,t) <= 0.95 * P_ref(i,s,t) ]

        SNAP is not used: it is a calendar/state programme, not a
        SKU-week promotion. In the paper this should be labelled a
        markdown-based promotional proxy. A robustness check with
        on_promo = 0 is provided by ``zero_promo_copy``.
        """
        cfg = self.config
        prices = prices.sort_values(
            ["store_id", "item_id", "week_id"]
        ).reset_index(drop=True)

        grouped = prices.groupby(["store_id", "item_id"], observed=True)[
            "sell_price"
        ]
        # Vectorized equivalent of shift(1).rolling(L).max(); avoids
        # a slow Python lambda inside transform().
        lagged = grouped.shift(1)
        prices["regular_price_ref"] = (
            lagged.groupby(
                [prices["store_id"], prices["item_id"]],
                observed=True,
            )
            .rolling(window=cfg.promo_lookback, min_periods=cfg.promo_min_periods)
            .max()
            .reset_index(level=[0, 1], drop=True)
        )

        prices["on_promo"] = (
            prices["regular_price_ref"].notna()
            & (prices["sell_price"] <= cfg.promo_threshold * prices["regular_price_ref"])
        ).astype("int8")
        return prices

    @staticmethod
    def zero_promo_copy(panel: pd.DataFrame) -> pd.DataFrame:
        """Robustness check: ICDN vs MLP must not hinge on the promo proxy."""
        out = panel.copy()
        out["on_promo"] = np.int8(0)
        return out

    # ======================================================================
    # 9. Merge: price availability is the universe
    # ======================================================================

    def merge_availability_and_sales(
        self,
        prices: pd.DataFrame,
        weekly_sales: pd.DataFrame,
    ) -> pd.DataFrame:
        """Left-join sales onto the price panel.

        Rows in the sales file include structural zeros before an item
        is actually offered. The price file is the assortment universe:
        no price => not offered; price but missing sales => units = 0.
        """
        price_panel = prices[
            [
                "store_id",
                "item_id",
                "wm_yr_wk",
                "week_id",
                "sell_price",
                "on_promo",
            ]
        ].copy()

        observed_weeks = set(weekly_sales["week_id"].unique())
        price_panel = price_panel[
            price_panel["week_id"].isin(observed_weeks)
        ].copy()

        master = price_panel.merge(
            weekly_sales[
                [
                    "store_id",
                    "item_id",
                    "dept_id",
                    "cat_id",
                    "state_id",
                    "wm_yr_wk",
                    "week_id",
                    "units",
                ]
            ],
            on=["store_id", "item_id", "wm_yr_wk", "week_id"],
            how="left",
            validate="one_to_one",
        )

        n_missing = int(master["units"].isna().sum())
        print("Missing units after merge:", n_missing)
        self.n_imputed_zero_units = n_missing
        # Impute only after recording the count for the paper.
        master["units"] = master["units"].fillna(0)

        return master

    # ======================================================================
    # 10. ICDN column names
    # ======================================================================

    def to_icdn_schema(self, master: pd.DataFrame) -> pd.DataFrame:
        """dept_id is the economic category (e.g. FOODS_3), not cat_id.

        cat_id and state_id are kept only for audit.
        """
        master = master.rename(
            columns={
                "store_id": "store_code",
                "item_id": "product_code",
                "sell_price": "price",
                "dept_id": "category",
            }
        )
        master = self._compact_week_id(master)
        master = master[
            [
                "store_code",
                "product_code",
                "week_id",
                "price",
                "units",
                "on_promo",
                "category",
                "cat_id",
                "state_id",
                "wm_yr_wk",
            ]
        ].copy()
        return master

    # ======================================================================
    # 11. Validation and persistence
    # ======================================================================

    @staticmethod
    def validate_master(master: pd.DataFrame) -> None:
        required = [
            "store_code",
            "product_code",
            "week_id",
            "price",
            "units",
            "on_promo",
        ]
        assert not master[required].isna().any().any()
        assert (master["price"] > 0).all()
        assert (master["units"] >= 0).all()
        assert set(master["on_promo"].unique()).issubset({0, 1})
        # ICDN rejects duplicate store-product-period keys.
        assert not master.duplicated(
            ["store_code", "product_code", "week_id"]
        ).any()

    def save_master(self, master: pd.DataFrame) -> pd.DataFrame:
        """Scientific dataset: zeros are retained. Do not delete this file."""
        master = master.sort_values(
            ["week_id", "store_code", "product_code"]
        ).reset_index(drop=True)

        path = self.config.out_dir / "m5_weekly_master.parquet"
        master.to_parquet(path, index=False)
        print("Wrote", path)
        return master

    # ======================================================================
    # 12. Sparsity diagnostics (full sample, for the paper)
    # ======================================================================

    def diagnose_sparsity(self) -> pd.DataFrame:
        """Do *not* drop units == 0 here.

        Filtering zeros before ranking SKUs would hide sparsity and
        bias selection toward products that merely have fewer listed
        weeks rather than genuinely dense trajectories.
        """
        if self.master is None:
            raise RuntimeError("Call build_master() first.")

        master = self.master
        print("Rows:", len(master))
        print("Stores:", master["store_code"].nunique())
        print("Products:", master["product_code"].nunique())
        print("Weeks:", master["week_id"].nunique())
        print("Zero-sales share:", (master["units"] == 0).mean())
        print("Promo share:", master["on_promo"].mean())
        print("Imputed zero units:", self.n_imputed_zero_units)

        stats = self._product_stats(master)
        stats_path = self.config.out_dir / "m5_product_diagnostics.csv"
        stats.to_csv(stats_path, index=False)
        print("Wrote", stats_path)

        self.product_stats = stats
        return stats

    @staticmethod
    def _product_stats(panel: pd.DataFrame) -> pd.DataFrame:
        stats = (
            panel.groupby(["product_code", "category"], observed=True)
            .agg(
                n_obs=("units", "size"),
                positive_obs=("units", lambda x: (x > 0).sum()),
                total_units=("units", "sum"),
                mean_units=("units", "mean"),
                n_stores=("store_code", "nunique"),
                n_weeks=("week_id", "nunique"),
                unique_prices=("price", "nunique"),
                mean_price=("price", "mean"),
                std_price=("price", "std"),
                promo_rate=("on_promo", "mean"),
            )
            .reset_index()
        )
        stats["positive_rate"] = stats["positive_obs"] / stats["n_obs"]
        stats["price_cv"] = stats["std_price"] / stats["mean_price"]
        return stats.sort_values(
            ["positive_rate", "n_stores", "unique_prices", "total_units"],
            ascending=False,
        )

    # ======================================================================
    # 13. Sample selection without lookahead
    # ======================================================================

    def select_products_without_lookahead(self) -> SampleSelection:
        """Freeze department and SKUs using only the initial window.

        The product universe is chosen by a pre-specified density and
        price-variation rule, not by inspecting the full sample or by
        hand-picking SKUs. The same list is then used for OLS, Ridge,
        MLP and ICDN on the full horizon.
        """
        if self.master is None:
            raise RuntimeError("Call build_master() first.")

        cfg = self.config
        weeks = np.sort(self.master["week_id"].unique())
        cutoff = int(weeks[int(len(weeks) * cfg.selection_frac)])
        selection_sample = self.master[
            self.master["week_id"] <= cutoff
        ].copy()

        stats = self._product_stats(selection_sample)
        stats.to_csv(
            self.config.out_dir / "m5_selection_window_diagnostics.csv",
            index=False,
        )

        eligible = stats[
            (stats["positive_rate"] >= cfg.min_positive_rate)
            & (stats["n_stores"] >= cfg.min_stores)
            & (stats["unique_prices"] >= cfg.min_unique_prices)
        ].copy()

        dept_counts = (
            eligible.groupby("category", observed=True)
            .size()
            .sort_values(ascending=False)
        )
        print("Eligible SKUs by department (selection window):")
        print(dept_counts)

        if dept_counts.empty:
            raise ValueError(
                "No SKU passed the density/price screen in the selection window. "
                "Relax min_positive_rate, min_stores or min_unique_prices."
            )

        category = str(dept_counts.index[0])
        n_eligible = int(dept_counts.iloc[0])
        if n_eligible < cfg.min_eligible_in_department:
            print(
                f"Warning: chosen department {category} has only "
                f"{n_eligible} eligible SKUs "
                f"(requested >= {cfg.min_eligible_in_department})."
            )

        chosen = (
            eligible[eligible["category"] == category]
            .sort_values(
                ["positive_rate", "n_stores", "unique_prices", "total_units"],
                ascending=False,
            )
            .head(cfg.n_skus)
        )
        product_codes = chosen["product_code"].astype(str).tolist()

        self.selection = SampleSelection(
            cutoff_week_id=cutoff,
            category=category,
            product_codes=product_codes,
            n_eligible_in_category=n_eligible,
            criteria={
                "selection_frac": cfg.selection_frac,
                "min_positive_rate": cfg.min_positive_rate,
                "min_stores": cfg.min_stores,
                "min_unique_prices": cfg.min_unique_prices,
                "n_skus": cfg.n_skus,
            },
        )

        sel_path = self.config.out_dir / "m5_selected_skus.json"
        sel_path.write_text(json.dumps(self.selection.to_dict(), indent=2))
        print("Selected department:", category)
        print("Selected SKUs:", product_codes)
        print("Wrote", sel_path)
        return self.selection

    # ======================================================================
    # 14. ICDN-compatible panel
    # ======================================================================

    def build_icdn_panel(self) -> pd.DataFrame:
        """Keep frozen SKUs over the full horizon; drop zero-sales weeks.

        Zeros are not recoded as ones. ICDN 1.0.0 requires units > 0
        because it takes log(units).
        """
        if self.master is None or self.selection is None:
            raise RuntimeError("Need build_master() and select_products_without_lookahead().")

        panel = self.master[
            self.master["product_code"].isin(self.selection.product_codes)
        ].copy()

        n_zeros = int((panel["units"] == 0).sum())
        print("Zero-sales rows dropped for ICDN:", n_zeros)

        panel = panel[panel["units"] > 0].copy()
        panel = panel.sort_values(
            ["week_id", "store_code", "product_code"]
        ).reset_index(drop=True)

        icdn_cols = [
            "store_code",
            "product_code",
            "week_id",
            "price",
            "units",
            "on_promo",
            "category",
        ]
        panel = panel[icdn_cols]

        assert (panel["units"] > 0).all()
        assert (panel["price"] > 0).all()
        assert not panel.duplicated(
            ["store_code", "product_code", "week_id"]
        ).any()

        path = self.config.out_dir / "m5_icdn_panel.parquet"
        panel.to_parquet(path, index=False)
        print("Wrote", path)
        print(
            panel.groupby("product_code", observed=True)
            .agg(
                n_obs=("units", "size"),
                n_weeks=("week_id", "nunique"),
                n_stores=("store_code", "nunique"),
                mean_units=("units", "mean"),
            )
        )

        self.icdn_panel = panel
        return panel    