"""Build a dunnhumby "Breakfast at the Frat" weekly panel for ICDN.

Observation unit
-----------------
(store, UPC, week) -> (price, units, observed promo)

Same semantics as Walmart (M5) and Dominick, different raw columns:

    concept      | Dominick        | Walmart (M5)     | dunnhumby
    -------------|-----------------|------------------|-----------------
    store        | store_code      | store_id         | store_num
    product      | UPC             | item_id          | upc
    time         | week            | wm_yr_wk->week_id| week_end_date->week_id
    price        | price/liter     | sell_price       | price
    demand       | liters_sold     | weekly units     | units
    promotion    | observed (BSC)  | markdown proxy   | observed (feature/display/tpr)
    category     | beer/imported/. | dept_id          | category
    product meta | brand/style/size| limited          | manufacturer/sub_category/size
    store meta   | none            | state             | state/segment/size

Source files (raw, semicolon-separated, decimal comma, Spanish month
abbreviations in dates, utf-8-sig BOM):

    dunnhumby-transaction.csv  524,950 real rows + ~157k blank trailing
                                rows that must be dropped, not parsed.
    dunnhumby-products.csv     58 real UPCs + blank trailing rows.
    dunnhumby-store.csv        79 rows; STORE_ID 4503 and 17627 each
                                appear twice, differing only in
                                seg_value_name. Resolved deterministically
                                (first occurrence kept, conflict logged),
                                never silently averaged or dropped.

Already weekly
---------------
dh Transaction Data is one row per (store, UPC, week). This module does
not aggregate across weeks. It only validates uniqueness, then maps
columns 1:1 onto the ICDN schema.

week_id
-------
Built from calendar arithmetic, not from row order and not from ISO week
numbers:

    week_id_t = 1 + (date_t - date_1) / 7 days

If a calendar week is missing, the resulting week_id sequence has a hole.
That hole is *not* compacted: ICDN lags must correspond to real elapsed
weeks. (Empirically, this file has zero calendar holes: 156 dates, all
exactly 7 days apart.)

Price
-----
``price`` (actual shelf price charged) is the model price P_ist, exactly
as defined by the manual. ``price_value = spend/units`` (units > 0 only)
is computed purely as an audit cross-check, not as a substitute price.
``discount_depth = 1 - price/base_price`` and ``markdown_flag =
1[price < base_price]`` are audit-only diagnostics: they are direct
functions of the current price and are not fed into ICDN/MLP as
features.

Promotion
---------
``on_promo = 1[feature=1 or display=1 or tpr_only=1]`` is built from the
three *observed* promotional indicators the manual documents explicitly
(circular feature, in-store display, temporary price reduction). This is
categorically different from the M5 markdown-proxy promo flag: here
promotion is observed, not inferred from a price-drop heuristic. That
makes this dataset a natural robustness check on whether ICDN's
advantage over MLP in Walmart depends on the promo-proxy definition.
``feature``, ``display`` and ``tpr_only`` are kept individually on the
master for later analysis.

Leakage guard
-------------
``spend``, ``hhs`` and ``visits`` are contemporaneous outcomes of the
same demand realization being modeled (spend = price*units by
construction; hhs/visits move with units within the week). They are
kept on the scientific master for audit only and are never written to
the ICDN panel.

Zeros
-----
A missing (store, UPC, week) combination is *not* treated as a zero.
The manual explicitly warns that some low/absent sales reflect
out-of-stock or discontinued items, not confirmed zero demand. Only
rows that are actually present in the source file with units == 0 are
kept as zeros on the scientific master. The ICDN panel then drops
units <= 0 (ICDN needs log(units)); zeros are never recoded as ones.

Outliers
--------
Per the manual, units_per_visit and visits_per_hh are computed as
*diagnostics* to spot the < 0.5% of rows that look extreme (or that
may reflect stockouts). They are not used as filters on the master.
Downstream robustness lives in the Huber loss and in an optional
sensitivity analysis, not in ad hoc row deletion here.

Sample selection (no lookahead)
--------------------------------
Every selection decision (core stores, category, candidate SKUs, frozen
SKUs) is made using only week_id <= cutoff (first half of the 156
weeks, i.e. cutoff = 78), and never revisited after inspecting model
results. Order follows the pipeline below:

    scientific master (observed zeros kept)
      -> weeks 1-78 only
      -> select core stores (overall activity, any category)
      -> select category (density/variation screen, no model output)
      -> price/density screen on candidate SKUs within that category
      -> greedy Jaccard co-occurrence selection
      -> freeze stores + SKUs
      -> units > 0, price > 0
      -> dunnhumby_icdn_panel.parquet

Everything after the freeze is dataset-agnostic: it is the same
preprocessing/splitter/Optuna/OLS/Ridge/MLP/ICDN/bootstrap pipeline used
for Walmart and Dominick.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import re

import numpy as np
import pandas as pd

SPANISH_MONTHS = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}

_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)\s*$")

# Concept mapping shown to the user before any column is touched.
CONCEPT_MAP = {
    "store_num": "store_code",
    "upc": "product_code",
    "week_end_date": "week_id (calendar-derived) + week_end_date (kept for audit)",
    "price": "price",
    "base_price": "base_price (audit: discount_depth, markdown_flag)",
    "units": "units",
    "feature|display|tpr_only": "on_promo = 1[feature or display or tpr_only]; kept individually",
    "spend|hhs|visits": "master only (audit) -- never in the ICDN panel: contemporaneous outcomes",
    "manufacturer": "brand",
    "sub_category": "style",
    "product_size": "product_size_raw (audit) + size (float, only if unit is unique in the frozen category)",
    "state|seg_value_name|sales_area_size_num": "store metadata (state/store_segment/sales_area_size)",
}


@dataclass
class DunnConfig:
    """Paths and pre-specified selection rules.

    All *_frac / min_* thresholds are applied only on week_id <= cutoff
    (see ``selection_frac``); nothing here is tuned by looking at the
    full horizon or at any model output.
    """

    data_dir: Path = Path("./data/Dunnhumby")
    out_dir: Path | None = None
    transaction_file: str = "dunnhumby-transaction.csv"
    products_file: str = "dunnhumby-products.csv"
    stores_file: str = "dunnhumby-store.csv"

    # Selection window: first half of the 156 observed weeks -> cutoff = 78.
    selection_frac: float = 0.50

    # Core-store screen (overall activity, not category-specific).
    min_store_week_coverage: float = 0.85
    max_core_stores: int | None = None  # None = keep every store that passes

    # SKU density / price-variation screen (within core stores, selection window).
    min_observed_coverage: float = 0.80
    min_positive_rate: float = 0.90
    min_unique_prices: int = 8         
    min_price_cv: float = 0.03              
    min_store_variation_share: float = 0.60  
    min_within_store_median_cv: float = 0.02
    min_core_store_presence: float = 0.80
    min_eligible_in_category: int = 10  # warn only, not a hard requirement

    # Jaccard co-occurrence freeze.
    n_candidate_skus: int = 20
    n_skus: int = 10
    jaccard_weight: float = 0.7
    coverage_weight: float = 0.3

    # Set only after inspecting category_stats; never from model output.
    target_category: str | None = None

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.out_dir = Path(self.out_dir) if self.out_dir is not None else self.data_dir / "panel"


@dataclass
class DunnSampleSelection:
    """Frozen universe used by OLS, Ridge, MLP and ICDN."""

    cutoff_week_id: int
    category: str
    core_stores: list[str]
    candidate_product_codes: list[str]
    product_codes: list[str]
    n_eligible_in_category: int
    joint_coverage_ge80: float | None = None
    joint_coverage_all: float | None = None
    criteria: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cutoff_week_id": int(self.cutoff_week_id),
            "category": self.category,
            "stores": list(self.core_stores),
            "candidate_products": list(self.candidate_product_codes),
            "products": list(self.product_codes),
            "n_eligible_in_category": int(self.n_eligible_in_category),
            "joint_coverage_ge80": self.joint_coverage_ge80,
            "joint_coverage_all": self.joint_coverage_all,
            "selection_criteria": self.criteria,
        }


class DunnhumbyPanelBuilder:
    """Map dh Transaction/Product/Store lookups onto the ICDN schema.

    Two outputs
    -----------
    dunnhumby_weekly_master.parquet
        Scientific dataset. Includes observed units == 0. spend, hhs,
        visits, discount_depth, units_per_visit, visits_per_hh,
        markdown_flag and price_value are audit-only columns.

    dunnhumby_icdn_panel.parquet
        Frozen stores x SKUs, units > 0 and price > 0 only. No spend,
        hhs or visits.
    """

    MASTER_COLUMNS = [
        "store_code", "product_code", "week_id", "week_end_date",
        "price", "base_price", "units",
        "feature", "display", "tpr_only", "on_promo",
        "category", "brand", "style",
        "product_size_raw", "size_value", "size_unit",
        "state", "store_segment", "sales_area_size",
        "spend", "hhs", "visits",
        "discount_depth", "units_per_visit", "visits_per_hh",
        "markdown_flag", "price_value",
    ]

    ICDN_META_COLUMNS = ["brand", "style"]

    def __init__(self, config: DunnConfig | None = None) -> None:
        self.config = config or DunnConfig()
        self.config.out_dir.mkdir(parents=True, exist_ok=True)

        self.raw_transactions: pd.DataFrame | None = None
        self.products: pd.DataFrame | None = None
        self.stores: pd.DataFrame | None = None

        self.master: pd.DataFrame | None = None

        self.selection_cutoff: int | None = None
        self.n_selection_weeks: int = 0
        self.store_stats: pd.DataFrame | None = None
        self.core_stores: list[str] = []
        self.product_stats: pd.DataFrame | None = None
        self.eligible: pd.DataFrame | None = None
        self.category_stats: pd.DataFrame | None = None
        self.candidates: pd.DataFrame | None = None
        self.presence: pd.DataFrame | None = None
        self.selection: DunnSampleSelection | None = None
        self.icdn_panel: pd.DataFrame | None = None

    # ======================================================================
    # Public orchestration
    # ======================================================================

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, DunnSampleSelection]:
        """Requires ``config.target_category`` after inspecting category_stats."""
        if self.config.target_category is None:
            raise ValueError(
                "Inspect category_stats (see diagnose_selection_window/"
                "rank_categories), then set DunnConfig.target_category. "
                "Do not choose a category from downstream ICDN vs MLP results."
            )
        self.build_weekly_master()
        self.diagnose_selection_window()
        self.freeze_universe(self.config.target_category)
        self.build_icdn_panel()
        return self.master, self.icdn_panel, self.selection

    def build_weekly_master(self) -> pd.DataFrame:
        txn = self.load_transactions()
        self.validate_uniqueness(txn)
        txn = self.build_week_index(txn)

        products = self.load_products()
        stores = self.load_stores()
        merged = self.merge_metadata(txn, products, stores)
        merged = self.compute_promo_and_audit(merged)

        master = self.assemble_master(merged)
        self.validate_master(master)
        self.master = self.save_master(master)
        return self.master

    # ======================================================================
    # 1. Raw loaders
    # ======================================================================

    @staticmethod
    def _parse_spanish_date(value: str) -> pd.Timestamp:
        day, month_abbr, year = value.split("-")
        year_i = int(year)
        year_i += 2000 if year_i < 100 else 0
        return pd.Timestamp(year=year_i, month=SPANISH_MONTHS[month_abbr], day=int(day))

    @staticmethod
    def _parse_pack_size(raw: str) -> tuple[float | None, str | None]:
        m = _SIZE_RE.match(str(raw))
        if m is None:
            return None, None
        return float(m.group(1)), m.group(2).upper()

    def load_transactions(self) -> pd.DataFrame:
        """dh Transaction Data: 524,950 real rows + blank trailing rows.

        The CSV has ~157k fully-empty rows appended after the real data
        (an Excel export artifact). They must be dropped by content, not
        by a hardcoded row count.
        """
        path = self.config.data_dir / self.config.transaction_file
        raw = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig", low_memory=False)
        n_raw = len(raw)
        raw = raw.dropna(how="all").reset_index(drop=True)
        print(f"transactions: dropped {n_raw - len(raw)} blank trailing rows; {len(raw)} real rows remain")

        raw["store_num"] = raw["STORE_NUM"].astype("int64").astype(str)
        raw["upc"] = raw["UPC"].astype("int64").astype(str)
        raw["week_end_date"] = raw["WEEK_END_DATE"].map(self._parse_spanish_date)

        raw = raw.rename(columns={
            "UNITS": "units", "VISITS": "visits", "HHS": "hhs", "SPEND": "spend",
            "PRICE": "price", "BASE_PRICE": "base_price",
            "FEATURE": "feature", "DISPLAY": "display", "TPR_ONLY": "tpr_only",
        })
        for col in ("feature", "display", "tpr_only"):
            raw[col] = raw[col].astype("int8")

        self.raw_transactions = raw
        return raw

    def load_products(self) -> pd.DataFrame:
        """dh Product Lookup: 58 real UPCs + blank trailing rows."""
        path = self.config.data_dir / self.config.products_file
        raw = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        raw = raw.dropna(how="all").dropna(subset=["UPC"]).copy()

        raw["product_code"] = raw["UPC"].astype("int64").astype(str)
        assert not raw["product_code"].duplicated().any(), "duplicate UPCs in product lookup"

        products = raw.rename(columns={
            "CATEGORY": "category",
            "SUB_CATEGORY": "style",
            "MANUFACTURER": "brand",
        })
        products["product_size_raw"] = products["PRODUCT_SIZE"].astype("string")
        parsed = products["product_size_raw"].map(self._parse_pack_size)
        products["size_value"] = [p[0] for p in parsed]
        products["size_unit"] = [p[1] for p in parsed]
        products = products[[
            "product_code", "category", "brand", "style",
            "product_size_raw", "size_value", "size_unit",
        ]]

        print(f"products: {len(products)} UPCs, categories={sorted(products['category'].unique())}")
        self.products = products
        return products

    def load_stores(self) -> pd.DataFrame:
        """dh Store Lookup: 79 rows, but STORE_ID 4503 and 17627 each
        appear twice, differing only in seg_value_name. Resolved
        deterministically (first row kept) with an explicit warning,
        never silently averaged.
        """
        path = self.config.data_dir / self.config.stores_file
        raw = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        raw.columns = [c.strip() for c in raw.columns]
        raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
        raw = raw.dropna(subset=["STORE_ID"]).copy()
        raw["STORE_ID"] = raw["STORE_ID"].astype("int64")

        dup_ids = raw.loc[raw["STORE_ID"].duplicated(keep=False), "STORE_ID"].unique()
        other_cols = [c for c in raw.columns if c != "STORE_ID"]
        for sid in dup_ids:
            rows = raw.loc[raw["STORE_ID"] == sid, other_cols]
            differing = [c for c in other_cols if rows[c].nunique(dropna=False) > 1]
            print(f"store_id {sid}: {len(rows)} duplicate rows differ only in {differing}; keeping the first.")

        raw = raw.drop_duplicates(subset=["STORE_ID"], keep="first").reset_index(drop=True)

        stores = raw.rename(columns={
            "STORE_ID": "store_code",
            "ADDRESS_STATE_PROV_CODE": "state",
            "SEG_VALUE_NAME": "store_segment",
            "SALES_AREA_SIZE_NUM": "sales_area_size",
        })
        stores["store_code"] = stores["store_code"].astype(str)
        stores = stores[["store_code", "state", "store_segment", "sales_area_size"]]

        print(f"stores: {len(stores)} unique store_code after dedup")
        self.stores = stores
        return stores

    # ======================================================================
    # 2. Uniqueness (validate, never sum silently)
    # ======================================================================

    @staticmethod
    def validate_uniqueness(txn: pd.DataFrame) -> None:
        """The manual defines one observation per (store, UPC, week).

        Duplicates are not summed automatically: if any appear, this
        raises so the cause can be inspected first.
        """
        dup_mask = txn.duplicated(["store_num", "upc", "week_end_date"], keep=False)
        n_dup = int(dup_mask.sum())
        print(f"duplicate (store_num, upc, week_end_date) rows: {n_dup}")
        if n_dup:
            print(txn.loc[dup_mask].sort_values(["store_num", "upc", "week_end_date"]).head(20))
            raise ValueError(
                f"{n_dup} rows violate store x UPC x week uniqueness. "
                "Investigate before aggregating -- do not sum automatically."
            )

    # ======================================================================
    # 3. Calendar week_id (holes are never compacted)
    # ======================================================================

    def build_week_index(self, txn: pd.DataFrame) -> pd.DataFrame:
        """week_id_t = 1 + (date_t - date_1) / 7 days.

        Uses calendar distance, not row order and not ISO week numbers.
        If a calendar week is missing, week_id skips that integer; ICDN
        lags must reflect real elapsed weeks, so the hole is not closed.
        """
        dates = np.sort(txn["week_end_date"].unique())
        d0 = dates[0]
        delta_days = (dates - d0) / np.timedelta64(1, "D")

        if not np.allclose(delta_days % 7, 0):
            bad = dates[~np.isclose(delta_days % 7, 0)]
            raise ValueError(f"week_end_date values not aligned to a 7-day grid: {bad}")

        week_id_vals = (1 + delta_days // 7).astype(np.int32)
        calendar_span = int(week_id_vals.max())
        n_holes = calendar_span - len(dates)
        print(
            f"week_id spans 1..{calendar_span} from {len(dates)} observed weeks "
            f"({n_holes} missing calendar weeks, NOT compacted)"
        )

        week_map = pd.Series(week_id_vals, index=pd.Index(dates, name="week_end_date"))
        txn = txn.copy()
        txn["week_id"] = txn["week_end_date"].map(week_map).astype("int32")
        return txn

    # ======================================================================
    # 4. Metadata merge
    # ======================================================================

    def merge_metadata(
        self, txn: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame
    ) -> pd.DataFrame:
        txn = txn.rename(columns={"store_num": "store_code", "upc": "product_code"})
        merged = txn.merge(
            products, on="product_code", how="left", validate="many_to_one"
        )
        n_missing_product = int(merged["category"].isna().sum())
        if n_missing_product:
            raise ValueError(
                f"{n_missing_product} rows have a UPC absent from the product lookup."
            )
        merged = merged.merge(
            stores, on="store_code", how="left", validate="many_to_one"
        )
        n_missing_store = int(merged["state"].isna().sum())
        if n_missing_store:
            raise ValueError(
                f"{n_missing_store} rows have a store_num absent from the store lookup."
            )
        return merged

    # ======================================================================
    # 5. Observed promo + audit-only diagnostics
    # ======================================================================

    @staticmethod
    def compute_promo_and_audit(df: pd.DataFrame) -> pd.DataFrame:
        """on_promo from the three *observed* flags -- not a price proxy.

        markdown_flag, discount_depth and price_value are audit-only:
        direct functions of price/base_price/spend, never fed to a model.
        """
        df = df.copy()
        df["on_promo"] = (
            (df["feature"] == 1) | (df["display"] == 1) | (df["tpr_only"] == 1)
        ).astype("int8")

        has_prices = df["price"].notna() & df["base_price"].notna()
        df["markdown_flag"] = np.where(has_prices, (df["price"] < df["base_price"]).astype("float32"), np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            df["discount_depth"] = 1.0 - df["price"] / df["base_price"]
            df["price_value"] = np.where(df["units"] > 0, df["spend"] / df["units"], np.nan)
            df["units_per_visit"] = np.where(df["visits"] > 0, df["units"] / df["visits"], np.nan)
            df["visits_per_hh"] = np.where(df["hhs"] > 0, df["visits"] / df["hhs"], np.nan)

        price_gap = (df["price_value"] - df["price"]).abs()
        print(
            "audit: price vs spend/units -> max abs diff =", float(price_gap.max(skipna=True)),
            " mean =", float(price_gap.mean(skipna=True)),
        )
        return df

    # ======================================================================
    # 6. Assemble + validate + save the scientific master
    # ======================================================================

    def assemble_master(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[self.MASTER_COLUMNS].copy()

        out["store_code"] = out["store_code"].astype("string")
        out["product_code"] = out["product_code"].astype("string")
        out["week_id"] = out["week_id"].astype("int32")
        for col in (
            "price", "base_price", "units", "spend", "hhs", "visits", "sales_area_size",
            "discount_depth", "units_per_visit", "visits_per_hh", "markdown_flag", "price_value",
        ):
            out[col] = out[col].astype("float32")
        for col in ("feature", "display", "tpr_only", "on_promo"):
            out[col] = out[col].astype("int8")

        return out.sort_values(["week_id", "store_code", "product_code"]).reset_index(drop=True)

    @staticmethod
    def validate_master(master: pd.DataFrame) -> None:
        assert not master.duplicated(["store_code", "product_code", "week_id"]).any()
        assert (master["units"] >= 0).all()
        assert set(master["on_promo"].unique()).issubset({0, 1})

        n_zero_units = int((master["units"] == 0).sum())
        n_price_missing = int(master["price"].isna().sum())
        n_price_nonpos = int(((master["price"] <= 0) & master["price"].notna()).sum())
        n_anomalous = n_price_missing + n_price_nonpos
        print(f"rows={len(master)}  zero-unit rows={n_zero_units} ({n_zero_units / len(master):.4%})")
        print(
            f"missing price={n_price_missing}  non-positive price={n_price_nonpos}  "
            f"({n_anomalous / len(master):.4%} of rows; manual reports <0.5% outliers)"
        )

    def save_master(self, master: pd.DataFrame) -> pd.DataFrame:
        """Scientific dataset: observed zeros are retained. Do not delete this file."""
        path = self.config.out_dir / "dunnhumby_weekly_master.parquet"
        master.to_parquet(path, index=False)
        print("Wrote", path)
        return master

    # ======================================================================
    # 7. Selection window (weeks 1..78 only, no lookahead)
    # ======================================================================

    def selection_sample(self) -> pd.DataFrame:
        if self.master is None:
            raise RuntimeError("Call build_weekly_master() first.")
        weeks = np.sort(self.master["week_id"].unique())
        cutoff = int(weeks[int(len(weeks) * self.config.selection_frac) - 1])
        self.selection_cutoff = cutoff
        self.n_selection_weeks = int((weeks <= cutoff).sum())
        print(f"selection window: week_id <= {cutoff} ({self.n_selection_weeks} of {len(weeks)} weeks)")
        return self.master[self.master["week_id"] <= cutoff].copy()

    # ======================================================================
    # 8. Core stores (overall activity, not category-specific)
    # ======================================================================

    def select_core_stores(self, selection: pd.DataFrame) -> list[str]:
        cfg = self.config
        stats = (
            selection.groupby("store_code", observed=True)
            .agg(
                n_obs=("units", "size"),
                n_weeks=("week_id", "nunique"),
                n_products=("product_code", "nunique"),
                total_units=("units", "sum"),
            )
            .reset_index()
        )
        stats["week_coverage"] = stats["n_weeks"] / self.n_selection_weeks

        eligible_stores = stats[stats["week_coverage"] >= cfg.min_store_week_coverage].copy()
        eligible_stores = eligible_stores.sort_values(["week_coverage", "n_obs"], ascending=False)
        if cfg.max_core_stores is not None:
            eligible_stores = eligible_stores.head(cfg.max_core_stores)

        core_stores = eligible_stores["store_code"].astype(str).tolist()
        print(
            f"core stores: {len(core_stores)} of {len(stats)} pass "
            f"week_coverage >= {cfg.min_store_week_coverage}"
        )

        path = self.config.out_dir / "dunnhumby_store_diagnostics.csv"
        stats.to_csv(path, index=False)
        print("Wrote", path)

        self.store_stats = stats
        self.core_stores = core_stores
        return core_stores

    # ======================================================================
    # 9. Product diagnostics within core stores + selection window
    # ======================================================================

    def compute_product_stats(self, selection_core: pd.DataFrame) -> pd.DataFrame:
        n_core = len(self.core_stores)
        possible = n_core * self.n_selection_weeks
        # ── within-store price variation (the quantity that matters for ID) ──
        # Restrict to positive-price, positive-unit rows to avoid spurious price=0 artefacts
        obs = selection_core[(selection_core["units"] > 0) & (selection_core["price"] > 0)]
        sp = (
            obs.groupby(["store_code", "product_code"])
            .agg(
                n_price_levels=("price", "nunique"),
                mean_price_s=("price", "mean"),
                sd_price_s=("price", "std"),
            )
            .reset_index()
        )
        sp["cv_s"] = sp["sd_price_s"] / sp["mean_price_s"]
        # per-product summaries of within-store variation
        within = (
            sp.groupby("product_code")
            .agg(
                median_within_store_cv=("cv_s", "median"),
                share_stores_with_variation=(
                    "n_price_levels",
                    lambda x: (x >= 3).mean(),
                ),
            )
            .reset_index()
        )
        # ── global aggregation (density, coverage, audit) ──
        stats = (
            selection_core.groupby(
                ["product_code", "category", "brand", "style",
                "product_size_raw", "size_value", "size_unit"],
                observed=True,
            )
            .agg(
                n_obs=("units", "size"),
                positive_obs=("units", lambda x: (x > 0).sum()),
                n_stores=("store_code", "nunique"),
                global_unique_prices=("price", "nunique"),   # audit only
                mean_price=("price", "mean"),
                std_price=("price", "std"),
                total_units=("units", "sum"),
                promo_rate=("on_promo", "mean"),
            )
            .reset_index()
        )
        stats["coverage"] = stats["n_obs"] / possible
        stats["positive_rate"] = stats["positive_obs"] / stats["n_obs"]
        stats["global_price_cv"] = stats["std_price"] / stats["mean_price"]  # audit only
        stats["store_presence"] = stats["n_stores"] / n_core
        stats = stats.merge(within, on="product_code", how="left")

        path = self.config.out_dir / "dunnhumby_product_diagnostics.csv"
        stats.to_csv(path, index=False)
        print("Wrote", path)

        self.product_stats = stats
        return stats

    # ======================================================================
    # 10. Density / price-variation screen
    # ======================================================================

    def screen_eligible(self, stats: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        eligible = stats[
            (stats["coverage"] >= cfg.min_observed_coverage)
            & (stats["positive_rate"] >= cfg.min_positive_rate)
            & (stats["store_presence"] >= cfg.min_core_store_presence)
            & (stats["share_stores_with_variation"] >= cfg.min_store_variation_share)
            & (stats["median_within_store_cv"] >= cfg.min_within_store_median_cv)
        ].copy()
        eligible = eligible.sort_values(
            ["coverage", "store_presence", "share_stores_with_variation",
            "median_within_store_cv"],
            ascending=False,
        )
        print(f"eligible SKUs: {len(eligible)} of {len(stats)}")
        self.eligible = eligible
        return eligible

    # ======================================================================
    # 11. Category ranking -- STOP and inspect before freezing
    # ======================================================================

    def rank_categories(self, eligible: pd.DataFrame) -> pd.DataFrame:
        """Category = the one with the most eligible SKUs; ties broken by
        joint store-week co-occurrence quality (median coverage here as a
        cheap proxy; recompute exact Jaccard co-occurrence if a real tie
        appears). No model output is used anywhere in this ranking.
        """
        stats = (
            eligible.groupby("category", observed=True)
            .agg(
                n_eligible=("product_code", "nunique"),
                median_coverage=("coverage", "median"),
                median_within_store_cv=("median_within_store_cv", "median"),
                median_store_presence=("store_presence", "median"),
                total_units=("total_units", "sum"),
            )
            .reset_index()
            .sort_values(["n_eligible", "median_coverage"], ascending=False)
        )

        path = self.config.out_dir / "dunnhumby_category_diagnostics.csv"
        stats.to_csv(path, index=False)
        print("Wrote", path)
        print(stats.to_string(index=False))

        for _, row in stats.iterrows():
            if row["n_eligible"] < self.config.min_eligible_in_category:
                print(
                    f"[WARN] category {row['category']!r}: {int(row['n_eligible'])} eligible SKUs "
                    f"< {self.config.min_eligible_in_category} required. "
                    f"freeze_universe will raise if you choose this category."
                )

        self.category_stats = stats
        return stats

    def diagnose_selection_window(self) -> pd.DataFrame:
        """Runs the full pre-registration screen (steps 7-11)."""
        selection = self.selection_sample()
        self.select_core_stores(selection)
        selection_core = selection[selection["store_code"].isin(self.core_stores)]
        stats = self.compute_product_stats(selection_core)
        eligible = self.screen_eligible(stats)
        return self.rank_categories(eligible)

    # ======================================================================
    # 12. Greedy Jaccard co-occurrence freeze
    # ======================================================================

    @staticmethod
    def pairwise_jaccard(presence: pd.DataFrame) -> pd.DataFrame:
        cols = presence.columns.tolist()
        x = presence.to_numpy(dtype=np.int32)
        inter = x.T @ x
        size = x.sum(axis=0)
        union = size[:, None] + size[None, :] - inter
        scores = np.divide(inter, union, out=np.zeros(inter.shape, dtype=float), where=union > 0)
        return pd.DataFrame(scores, index=cols, columns=cols)

    def greedy_select_products(
        self, candidates: pd.DataFrame, presence: pd.DataFrame, n_skus: int
    ) -> list[str]:
        cfg = self.config
        rank = candidates.set_index("product_code")
        jaccard = self.pairwise_jaccard(presence)

        first = candidates.sort_values("coverage", ascending=False).iloc[0]["product_code"]
        selected = [str(first)]
        pool = [str(p) for p in candidates["product_code"].tolist()]

        while len(selected) < n_skus:
            remaining = [p for p in pool if p not in selected]
            if not remaining:
                break
            scores = {
                p: cfg.jaccard_weight * float(jaccard.loc[p, selected].mean())
                + cfg.coverage_weight * float(rank.loc[p, "coverage"])
                for p in remaining
            }
            selected.append(max(scores, key=scores.get))
        return selected

    def freeze_universe(self, category: str) -> DunnSampleSelection:
        """Lock stores, category and SKUs. Never revise after seeing models."""
        if self.eligible is None or self.master is None or self.category_stats is None:
            raise RuntimeError("Call diagnose_selection_window() first.")
        if category not in set(self.category_stats["category"]):
            raise ValueError(f"{category!r} is not in category_stats; inspect the ranking before freezing.")

        cfg = self.config
        self.config.target_category = category

        candidates = (
            self.eligible[self.eligible["category"] == category]
            .sort_values(
                ["coverage", "store_presence", "share_stores_with_variation", "median_within_store_cv"],
                ascending=False,
            )
            .head(cfg.n_candidate_skus)
            .copy()
        )
        if candidates.empty:
            raise ValueError(f"No eligible SKUs in category={category!r}.")
        n_eligible = int((self.eligible["category"] == category).sum())

        selection_core = self.master[
            (self.master["week_id"] <= self.selection_cutoff)
            & self.master["store_code"].isin(self.core_stores)
            & self.master["product_code"].isin(candidates["product_code"])
        ]
        usable = selection_core[
            (selection_core["units"] > 0) & (selection_core["price"] > 0)
        ]
        presence = usable.assign(observed=1).pivot_table(
            index=["store_code", "week_id"],
            columns="product_code",
            values="observed",
            aggfunc="max",
            fill_value=0,
        )

        if len(candidates) < cfg.n_skus:
            raise ValueError(
                f"freeze_universe: need {cfg.n_skus} eligible SKUs in {category!r}, "
                f"only {len(candidates)} pass all screens. "
                f"Lower cfg.n_skus or relax selection criteria, then re-document."
            )
        selected = self.greedy_select_products(candidates, presence, cfg.n_skus)
        print("SELECTED_PRODUCTS:", selected)

        selected_presence = presence[selected]
        joint = selected_presence.sum(axis=1) / len(selected)
        k_ge = int(np.ceil(0.80 * len(selected)))
        joint_ge80 = float((joint >= 0.80).mean())
        joint_all = float((joint == 1.0).mean())
        print(f"store-weeks with >= {k_ge}/{len(selected)} SKUs jointly observed: {joint_ge80:.4f}")
        print(f"store-weeks with all {len(selected)}/{len(selected)} SKUs jointly observed: {joint_all:.4f}")

        self.candidates = candidates
        self.presence = presence
        self.selection = DunnSampleSelection(
            cutoff_week_id=self.selection_cutoff,
            category=category,
            core_stores=list(self.core_stores),
            candidate_product_codes=candidates["product_code"].astype(str).tolist(),
            product_codes=selected,
            n_eligible_in_category=n_eligible,
            joint_coverage_ge80=joint_ge80,
            joint_coverage_all=joint_all,
            criteria={
                "selection_frac": cfg.selection_frac,
                "min_store_week_coverage": cfg.min_store_week_coverage,
                "min_observed_coverage": cfg.min_observed_coverage,
                "min_positive_rate": cfg.min_positive_rate,
                "min_unique_prices": cfg.min_unique_prices,
                "min_price_cv": cfg.min_price_cv,
                "min_core_store_presence": cfg.min_core_store_presence,
                "n_candidate_skus": cfg.n_candidate_skus,
                "n_skus": cfg.n_skus,
                "jaccard_weight": cfg.jaccard_weight,
                "coverage_weight": cfg.coverage_weight,
            },
        )
        self._save_frozen_lists()
        return self.selection

    def _save_frozen_lists(self) -> None:
        assert self.selection is not None
        out = self.config.out_dir
        pd.Series(self.selection.core_stores, name="store_code").to_csv(
            out / "dunnhumby_selected_stores.csv", index=False
        )
        pd.Series(self.selection.product_codes, name="product_code").to_csv(
            out / "dunnhumby_selected_products.csv", index=False
        )
        (out / "dunnhumby_selection.json").write_text(json.dumps(self.selection.to_dict(), indent=2))
        print("Wrote frozen store/SKU manifest under", out)

    # ======================================================================
    # 13. ICDN-compatible panel (units > 0, price > 0, no spend/hhs/visits)
    # ======================================================================

    def build_icdn_panel(self) -> pd.DataFrame:
        if self.master is None or self.selection is None:
            raise RuntimeError("Need build_weekly_master() and freeze_universe().")

        panel = self.master[
            self.master["store_code"].isin(self.selection.core_stores)
            & self.master["product_code"].isin(self.selection.product_codes)
        ].copy()

        n_zero = int((panel["units"] == 0).sum())
        n_bad_price = int((panel["price"].isna() | (panel["price"] <= 0)).sum())
        print(
            f"dropping {n_zero} zero-unit rows and {n_bad_price} missing/non-positive "
            "price rows for ICDN (never recoded as positive)"
        )

        panel = panel[(panel["units"] > 0) & panel["price"].notna() & (panel["price"] > 0)].copy()

        sku_sizes = (
            panel[["product_code", "size_value", "size_unit"]]
            .drop_duplicates("product_code")
        )
        units = sku_sizes["size_unit"].dropna().unique()
        if len(units) == 1 and sku_sizes["size_value"].notna().all():
            panel["size"] = panel["size_value"].astype("float32")
            icdn_meta = self.ICDN_META_COLUMNS + ["size"]
            print(f"size: numeric {units[0]}, values={sorted(sku_sizes['size_value'].unique())}")
        else:
            icdn_meta = self.ICDN_META_COLUMNS
            print(f"size: NOT included (units={units})")

        cols = ["store_code", "product_code", "week_id", "price", "units", "on_promo", "category"]
        cols += icdn_meta
        panel = panel[cols].sort_values(["week_id", "store_code", "product_code"]).reset_index(drop=True)

        assert (panel["units"] > 0).all()
        assert (panel["price"] > 0).all()
        assert not panel.duplicated(["store_code", "product_code", "week_id"]).any()
        for col in ("category", "brand", "style"):
            assert col in panel.columns
            assert panel[col].notna().all()
        if "size" in panel.columns:
            assert pd.api.types.is_numeric_dtype(panel["size"])
            assert panel["size"].gt(0).all()
        assert not {"product_size_raw", "size_value", "size_unit",
                    "state", "store_segment", "sales_area_size"} & set(panel.columns)
        assert "spend" not in panel.columns and "hhs" not in panel.columns and "visits" not in panel.columns

        path = self.config.out_dir / "dunnhumby_icdn_panel.parquet"
        panel.to_parquet(path, index=False)
        print("Wrote", path)
        print(
            panel.groupby("product_code", observed=True).agg(
                n_obs=("units", "size"),
                n_weeks=("week_id", "nunique"),
                n_stores=("store_code", "nunique"),
                mean_units=("units", "mean"),
            )
        )

        self.icdn_panel = panel
        return panel