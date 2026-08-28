"""Build a Dominick beer weekly panel for ICDN.

Observation unit
----------------
(store, UPC, week) -> (price per liter, liters sold, observed promo)

The source file is already weekly. This module does not reconstruct
demand or invent promotions. It maps columns, remaps week_id to a
consecutive integer grid, and freezes stores/SKUs without lookahead.

ICDN columns
------------
store_code, product_code, week_id, price, units, on_promo
Optional: category, brand, style, size (competitor-selection features).

Quantity and price
------------------
``units`` is liters sold and ``price`` is price per liter. Pack sizes
differ (6 / 12 / 24), so unit-count and shelf price are not comparable
across UPCs. ICDN logs both internally; do not pass the precomputed
log columns.

Zeros
-----
Rows with units <= 0 also have price <= 0: they are empty placeholders,
not confirmed zero sales with a shelf price. They are dropped, never
recoded as ones.

week_id
-------
The source week codes have three calendar holes. ICDN lags treat
week_id as consecutive integers, so remaining weeks are remapped to
1..T. The original code is kept as source_week_id on the master.

Promo
-----
``on_promo`` is the observed Dominick promotion flag (B/S/C), not a
markdown proxy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORY_LABELS = {
    26: "imported_beer",
    27: "beer",
    28: "nonalcoholic_beer",
}

SOURCE_COLUMNS = [
    "category_code",
    "upc_code",
    "product_description",
    "pack_size_text",
    "store_code",
    "week_id",
    "units_sold",
    "unit_price",
    "liters_sold",
    "price_per_liter",
    "on_promo",
    "brand_family_norm",
    "style_segment_norm",
]


@dataclass
class DominickConfig:
    """Paths and pre-specified selection rules.

    Store / category / SKU choice uses only week_id <= selection cutoff.
    ``target_category_id`` is not set by default: inspect category_stats
    first (skip non-alcoholic if the research question is beer demand).
    """

    data_dir: Path = Path("./data/Dominick")
    source_file: str = "dominick_features.csv"
    out_dir: Path | None = None

    selection_frac: float = 0.50

    min_store_week_coverage: float = 0.85
    n_core_stores: int = 20

    min_stores: int = 10
    min_week_coverage: float = 0.70
    min_unique_prices: int = 8
    min_price_cv: float = 0.03
    min_products_in_category: int = 10

    n_candidate_skus: int = 40
    n_skus: int = 20
    jaccard_weight: float = 0.7
    coverage_weight: float = 0.3

    target_category_id: int | None = None

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.out_dir is None:
            self.out_dir = self.data_dir / "panel"
        else:
            self.out_dir = Path(self.out_dir)


@dataclass
class DominickSampleSelection:
    """Frozen universe used by OLS, Ridge, MLP and ICDN."""

    cutoff_week_id: int
    category_id: int
    category: str
    core_stores: list[str]
    candidate_product_codes: list[str]
    product_codes: list[str]
    n_eligible_in_category: int
    joint_coverage_ge: float | None = None
    joint_coverage_all: float | None = None
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
            "joint_coverage_ge": self.joint_coverage_ge,
            "joint_coverage_all": self.joint_coverage_all,
            "criteria": self.criteria,
        }


class DominickPanelBuilder:
    """Map Dominick features onto the ICDN schema and freeze a SKU universe.

    Outputs
    -------
    dominick_weekly_master.parquet
        Observed positive store-UPC-weeks, with source_week_id.

    dominick_icdn_panel.parquet
        Frozen stores × SKUs; ICDN columns only.
    """

    def __init__(self, config: DominickConfig | None = None) -> None:
        self.config = config or DominickConfig()
        self.config.out_dir.mkdir(parents=True, exist_ok=True)

        self.raw: pd.DataFrame | None = None
        self.week_map: dict[int, int] | None = None
        self.n_placeholder_rows: int = 0

        self.weekly: pd.DataFrame | None = None
        self.n_selection_weeks: int = 0
        self.selection_cutoff: int | None = None

        self.store_stats: pd.DataFrame | None = None
        self.product_stats: pd.DataFrame | None = None
        self.category_stats: pd.DataFrame | None = None
        self.eligible: pd.DataFrame | None = None
        self.candidates: pd.DataFrame | None = None
        self.core_stores: list[str] = []
        self.presence: pd.DataFrame | None = None
        self.selection: DominickSampleSelection | None = None
        self.icdn_panel: pd.DataFrame | None = None

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, DominickSampleSelection]:
        """Requires ``config.target_category_id`` after inspecting categories."""
        if self.config.target_category_id is None:
            raise ValueError(
                "Inspect category_stats (see rank_categories), then set "
                "DominickConfig.target_category_id. Do not choose a category "
                "from downstream ICDN vs MLP results."
            )
        self.build_weekly_master()
        self.diagnose_selection_window()
        self.freeze_universe(self.config.target_category_id)
        self.build_icdn_panel()
        return self.weekly, self.icdn_panel, self.selection

    def load_features(self) -> pd.DataFrame:
        path = self.config.data_dir / self.config.source_file
        if not path.exists():
            raise FileNotFoundError(path)
        raw = pd.read_csv(path, usecols=SOURCE_COLUMNS)
        print("source", path.name, raw.shape)
        print("columns used:", SOURCE_COLUMNS)
        self.raw = raw
        return raw

    def column_inventory(self, raw: pd.DataFrame | None = None) -> pd.DataFrame:
        """Which source columns become ICDN fields, and which are unused."""
        raw = self.raw if raw is None else raw
        if raw is None:
            raise RuntimeError("Call load_features() first.")
        mapped = {
            "upc_code": "product_code",
            "store_code": "store_code",
            "week_id": "week_id (compacted to 1..T; original kept as source_week_id)",
            "liters_sold": "units",
            "price_per_liter": "price",
            "on_promo": "on_promo",
            "category_code": "category (via CATEGORY_LABELS)",
            "brand_family_norm": "brand",
            "style_segment_norm": "style",
            "pack_size_text": "size",
        }
        unused = {
            "units_sold": "pack-count demand; not comparable across pack sizes",
            "unit_price": "shelf price per pack; ICDN uses price per liter",
            "product_description": "audit / diagnostics only",
        }
        rows = []
        for col in raw.columns:
            if col in mapped:
                rows.append({"column": col, "role": "mapped", "icdn": mapped[col]})
            elif col in unused:
                rows.append({"column": col, "role": "unused", "icdn": unused[col]})
            else:
                rows.append({"column": col, "role": "unused", "icdn": ""})
        out = pd.DataFrame(rows)
        print(out.to_string(index=False))
        return out

    def to_icdn_schema(self, raw: pd.DataFrame | None = None) -> pd.DataFrame:
        """Map columns, compact week_id, drop placeholder non-positive rows."""
        if raw is None:
            raw = self.raw
        if raw is None:
            raise RuntimeError("Call load_features() first.")

        n_placeholder = int(
            (
                (raw["units_sold"] <= 0)
                | (raw["liters_sold"] <= 0)
                | (raw["price_per_liter"] <= 0)
            ).sum()
        )
        self.n_placeholder_rows = n_placeholder
        print("placeholder rows (units<=0 or liters<=0 or price<=0):", n_placeholder)

        weekly = raw[
            (raw["units_sold"] > 0)
            & (raw["liters_sold"] > 0)
            & (raw["price_per_liter"] > 0)
        ].copy()
        weekly["source_week_id"] = pd.to_numeric(weekly["week_id"], errors="coerce").astype("int32")
        weekly = self._compact_week_id(weekly)

        weekly["store_code"] = weekly["store_code"].astype("string")
        weekly["product_code"] = weekly["upc_code"].astype("string")
        weekly["price"] = weekly["price_per_liter"].astype("float32")
        weekly["units"] = weekly["liters_sold"].astype("float32")
        weekly["on_promo"] = weekly["on_promo"].astype(bool).astype("int8")
        weekly["category_id"] = pd.to_numeric(weekly["category_code"], errors="coerce").astype("int32")
        weekly["category"] = weekly["category_id"].map(CATEGORY_LABELS).astype("string")
        weekly["brand"] = weekly["brand_family_norm"].astype("string").fillna("unknown")
        weekly["style"] = weekly["style_segment_norm"].astype("string").fillna("unknown")
        weekly["size"] = weekly["pack_size_text"].astype("string").fillna("unknown")
        weekly["product_description"] = weekly["product_description"].astype("string")

        keep = [
            "store_code",
            "product_code",
            "week_id",
            "source_week_id",
            "price",
            "units",
            "on_promo",
            "category_id",
            "category",
            "brand",
            "style",
            "size",
            "product_description",
        ]
        weekly = weekly[keep].sort_values(
            ["week_id", "store_code", "product_code"]
        ).reset_index(drop=True)

        assert not weekly.duplicated(["store_code", "product_code", "week_id"]).any()
        assert (weekly["units"] > 0).all()
        assert (weekly["price"] > 0).all()
        print(
            "observed rows", len(weekly),
            "stores", weekly["store_code"].nunique(),
            "UPCs", weekly["product_code"].nunique(),
            "weeks", weekly["week_id"].nunique(),
        )
        self.weekly = weekly
        return weekly

    def _compact_week_id(self, df: pd.DataFrame) -> pd.DataFrame:
        remaining = np.sort(df["source_week_id"].dropna().unique().astype(int))
        remap = {int(old): np.int32(new) for new, old in enumerate(remaining, start=1)}
        gaps = np.diff(remaining)
        print(
            "source weeks", int(remaining.min()), "→", int(remaining.max()),
            "n=", len(remaining),
            "n_gaps", int((gaps > 1).sum()),
            "max_gap", int(gaps.max()) if len(gaps) else 0,
        )
        df = df.copy()
        df["week_id"] = df["source_week_id"].map(remap).astype("int32")
        self.week_map = remap
        return df

    def save_weekly_master(self, weekly: pd.DataFrame | None = None) -> pd.DataFrame:
        if weekly is None:
            weekly = self.weekly
        if weekly is None:
            raise RuntimeError("Call to_icdn_schema() first.")
        path = self.config.out_dir / "dominick_weekly_master.parquet"
        weekly.to_parquet(path, index=False)
        print("Wrote", path)
        self.weekly = weekly
        return weekly

    def build_weekly_master(self) -> pd.DataFrame:
        self.load_features()
        self.to_icdn_schema()
        return self.save_weekly_master()

    def selection_sample(self) -> pd.DataFrame:
        if self.weekly is None:
            raise RuntimeError("Call to_icdn_schema() first.")
        weeks = np.sort(self.weekly["week_id"].unique())
        cutoff = int(weeks[int(len(weeks) * self.config.selection_frac) - 1])
        self.selection_cutoff = cutoff
        print("Selection cutoff:", cutoff, "of", int(weeks.min()), "→", int(weeks.max()))
        sample = self.weekly[self.weekly["week_id"] <= cutoff].copy()
        self.n_selection_weeks = int(sample["week_id"].nunique())
        print("selection weeks:", self.n_selection_weeks)
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
        path = self.config.out_dir / "dominick_store_diagnostics.csv"
        store_stats.to_csv(path, index=False)
        print("core stores:", core_stores)
        self.store_stats = store_stats
        self.core_stores = core_stores
        return core_stores

    def compute_product_stats(self, selection_core: pd.DataFrame) -> pd.DataFrame:
        stats = (
            selection_core.groupby(
                ["product_code", "category_id", "category", "brand", "style", "size", "product_description"],
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
                promo_rate=("on_promo", "mean"),
            )
            .reset_index()
        )
        stats["price_cv"] = stats["std_price"] / stats["mean_price"]
        possible = len(self.core_stores) * self.n_selection_weeks
        stats["coverage_rate"] = stats["n_obs"] / possible
        path = self.config.out_dir / "dominick_product_diagnostics.csv"
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
        self.eligible = eligible
        return eligible

    def rank_categories(self, eligible: pd.DataFrame) -> pd.DataFrame:
        """Stop here and inspect. Category 28 is non-alcoholic beer."""
        stats = (
            eligible.groupby(["category_id", "category"], observed=True)
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
        path = self.config.out_dir / "dominick_category_diagnostics.csv"
        stats.to_csv(path, index=False)
        print(stats.to_string(index=False))
        self.category_stats = stats
        return stats

    def diagnose_selection_window(self) -> pd.DataFrame:
        selection = self.selection_sample()
        self.select_core_stores(selection)
        selection_core = selection[selection["store_code"].isin(self.core_stores)]
        stats = self.compute_product_stats(selection_core)
        eligible = self.screen_eligible(stats)
        return self.rank_categories(eligible)

    @staticmethod
    def pairwise_jaccard(presence: pd.DataFrame) -> pd.DataFrame:
        cols = presence.columns.tolist()
        x = presence.to_numpy(dtype=np.int32)
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

    def freeze_universe(self, category_id: int) -> DominickSampleSelection:
        """Lock stores, category and SKUs. Never revise after seeing models."""
        if self.eligible is None or self.weekly is None:
            raise RuntimeError("Call diagnose_selection_window() first.")
        if self.selection_cutoff is None:
            raise RuntimeError("selection_cutoff is not set.")

        candidates = (
            self.eligible[self.eligible["category_id"] == category_id]
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
            raise ValueError(f"No eligible SKUs in category_id={category_id}.")
        self.config.target_category_id = int(category_id)

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
        n = len(selected)
        k80 = int(np.ceil(0.80 * n))
        ge80 = float((joint >= 0.80).mean())
        all_n = float((joint == 1.0).mean())
        print(f"store-weeks >= {k80}/{n}:", ge80)
        print(f"store-weeks = {n}/{n}:", all_n)

        cat_name = str(candidates["category"].iloc[0])
        self.candidates = candidates
        self.presence = presence
        self.selection = DominickSampleSelection(
            cutoff_week_id=self.selection_cutoff,
            category_id=int(category_id),
            category=cat_name,
            core_stores=list(self.core_stores),
            candidate_product_codes=candidates["product_code"].astype(str).tolist(),
            product_codes=selected,
            n_eligible_in_category=int(
                (self.eligible["category_id"] == category_id).sum()
            ),
            joint_coverage_ge=ge80,
            joint_coverage_all=all_n,
            criteria={
                "selection_frac": self.config.selection_frac,
                "min_store_week_coverage": self.config.min_store_week_coverage,
                "n_core_stores": self.config.n_core_stores,
                "min_stores": self.config.min_stores,
                "min_week_coverage": self.config.min_week_coverage,
                "min_unique_prices": self.config.min_unique_prices,
                "min_price_cv": self.config.min_price_cv,
                "n_skus": self.config.n_skus,
                "jaccard_weight": self.config.jaccard_weight,
                "price": "price_per_liter",
                "units": "liters_sold",
            },
        )
        self._save_frozen_lists()
        return self.selection

    def _save_frozen_lists(self) -> None:
        assert self.selection is not None
        out = self.config.out_dir
        pd.Series(self.selection.core_stores, name="store_code").to_csv(
            out / "dominick_selected_stores.csv", index=False
        )
        pd.Series(self.selection.product_codes, name="product_code").to_csv(
            out / "dominick_selected_products.csv", index=False
        )
        (out / "dominick_selected_skus.json").write_text(
            json.dumps(self.selection.to_dict(), indent=2)
        )
        print("Wrote frozen store/SKU lists under", out)

    def build_icdn_panel(self) -> pd.DataFrame:
        if self.weekly is None or self.selection is None:
            raise RuntimeError("Need weekly master and freeze_universe().")

        panel = self.weekly[
            self.weekly["store_code"].isin(self.selection.core_stores)
            & self.weekly["product_code"].isin(self.selection.product_codes)
        ].copy()

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
                "style",
                "size",
            ]
        ].copy()
        icdn = icdn.sort_values(
            ["week_id", "store_code", "product_code"]
        ).reset_index(drop=True)

        self.validate_icdn_panel(icdn)
        path = self.config.out_dir / "dominick_icdn_panel.parquet"
        icdn.to_parquet(path, index=False)
        print("Wrote", path)
        print(icdn.groupby("product_code", observed=True).agg(
            n_obs=("units", "size"),
            n_weeks=("week_id", "nunique"),
            n_stores=("store_code", "nunique"),
            mean_units=("units", "mean"),
        ))
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
        for col in ("category", "brand", "style", "size"):
            assert col in icdn.columns
            assert icdn[col].notna().all()
        n_skus = icdn["product_code"].nunique()
        if n_skus != self.config.n_skus:
            print(f"Warning: {n_skus} SKUs in ICDN panel, expected {self.config.n_skus}.")
        week_ids = np.sort(icdn["week_id"].unique())
        assert (np.diff(week_ids) >= 1).all()
