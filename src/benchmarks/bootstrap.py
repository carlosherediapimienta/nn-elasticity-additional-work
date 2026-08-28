"""Non-overlapping block bootstrap with two clocks.

`source_week_id` is the real calendar week (Fourier, lifecycle).
`bootstrap_order` is the synthetic sequence (lags, rollings, causal prices).
`week_id` is set to `bootstrap_order` so ICDN's schema.period stays valid.
`bootstrap_block_id` isolates stateful operators across concatenated blocks:
lags, rollings, price ffill, and ICDN warmup smoothing do not see another
block. Validation keeps the last train block's id so val lags see the end
of the bootstrap train (no barrier before val).

The length of each replicate is exactly the original train length: blocks
are drawn with replacement until that many weeks are filled. A shared
`bootstrap_plan.json` stores the draws and the frozen calendar from the
original holdout train. All four models read that plan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmarks.constants import (
    BLOCK_SIZE,
    N_BOOT_ICDN,
    N_BOOT_LINEAR,
    N_BOOT_MLP,
    PERIOD_COL,
    SEED,
)

SOURCE_WEEK_COL = "source_week_id"
BOOTSTRAP_ORDER_COL = "bootstrap_order"
BLOCK_ID_COL = "bootstrap_block_id"
PANEL_KEYS = ["store_code", "product_code", PERIOD_COL]
BOOTSTRAP_PLAN_JSON = "bootstrap_plan.json"


class BootstrapError(ValueError):
    """A replicate does not have a valid two-clock chronology."""


@dataclass(frozen=True)
class FrozenCalendar:
    """Holdout-train calendar. Replicates must not re-estimate these."""

    origin: int
    max_train_rank: int
    product_first_rank: dict[str, int]
    store_product_first_rank: tuple[tuple[str, str, int], ...]

    def to_dict(self) -> dict:
        return {
            "origin": int(self.origin),
            "max_train_rank": int(self.max_train_rank),
            "product_first_rank": {str(k): int(v) for k, v in self.product_first_rank.items()},
            "store_product_first_rank": [
                {"store": s, "product": p, "rank": int(r)}
                for s, p, r in self.store_product_first_rank
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FrozenCalendar":
        sp = tuple(
            (str(row["store"]), str(row["product"]), int(row["rank"]))
            for row in payload.get("store_product_first_rank") or []
        )
        return cls(
            origin=int(payload["origin"]),
            max_train_rank=int(payload["max_train_rank"]),
            product_first_rank={str(k): int(v) for k, v in (payload.get("product_first_rank") or {}).items()},
            store_product_first_rank=sp,
        )

    @classmethod
    def from_panel(cls, panel: pd.DataFrame, period_col: str = PERIOD_COL) -> "FrozenCalendar":
        weeks = pd.to_numeric(panel[period_col], errors="coerce").dropna().astype(int)
        origin = int(weeks.min())
        rank = weeks - origin + 1
        tmp = panel.assign(_rank=rank)
        product_first = (
            tmp.groupby(tmp["product_code"].astype(str))["_rank"].min().astype(int).to_dict()
        )
        sp = (
            tmp.assign(_store=tmp["store_code"].astype(str), _product=tmp["product_code"].astype(str))
            .groupby(["_store", "_product"])["_rank"]
            .min()
            .astype(int)
        )
        return cls(
            origin=origin,
            max_train_rank=int(weeks.max() - origin + 1),
            product_first_rank={str(k): int(v) for k, v in product_first.items()},
            store_product_first_rank=tuple((s, p, int(r)) for (s, p), r in sp.items()),
        )

    def store_product_series(self) -> pd.Series:
        if not self.store_product_first_rank:
            return pd.Series(dtype=float)
        idx = pd.MultiIndex.from_tuples(
            [(s, p) for s, p, _ in self.store_product_first_rank],
            names=["store_code", "product_code"],
        )
        return pd.Series(
            [r for _, _, r in self.store_product_first_rank],
            index=idx,
            dtype=float,
        )


@dataclass
class BootstrapDraw:
    train: pd.DataFrame
    val: pd.DataFrame
    manifest: dict


def restore_source_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Write real calendar weeks back to `week_id` for saved tables."""
    if SOURCE_WEEK_COL not in df.columns:
        return df
    out = df.copy()
    out[PERIOD_COL] = out[SOURCE_WEEK_COL]
    return out


def validate_draw(
    train: pd.DataFrame,
    val: pd.DataFrame,
    n_original_train_periods: int,
) -> None:
    train_max = int(train[PERIOD_COL].max())
    val_min = int(val[PERIOD_COL].min())
    if not train_max < val_min:
        raise BootstrapError(
            f"bootstrap_train_max ({train_max}) is not < bootstrap_validation_min ({val_min})"
        )
    n_sampled = int(train[PERIOD_COL].nunique())
    if n_sampled != n_original_train_periods:
        raise BootstrapError(
            f"n_sampled_periods ({n_sampled}) != n_original_train_periods ({n_original_train_periods})"
        )
    for label, df in ("train", train), ("val", val):
        if df.duplicated(PANEL_KEYS).any():
            raise BootstrapError(f"duplicate store-product-period in bootstrap {label}")


def save_bootstrap_manifest(out_dir: Path, records: list[dict], *, block_size: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "non-overlapping-block-bootstrap",
        "block_size": int(block_size),
        "seed": int(seed),
        "replicates": records,
    }
    path = out_dir / "bootstrap_manifest.json"
    path.write_text(json.dumps(payload, indent=2))
    #print("  wrote", path)


def protocol_n_boot() -> int:
    return max(int(N_BOOT_LINEAR), int(N_BOOT_MLP), int(N_BOOT_ICDN))


class NonOverlappingBlockBootstrap:
    """Resample the non-overlapping partition of train weeks, with replacement.

    Draws whole blocks until the replicate has exactly `n_periods` weeks. The
    last take may be a prefix of a sampled block. No gap is inserted between
    train blocks; isolation is `bootstrap_block_id`, not a missing week.
    """

    def __init__(
        self,
        period_col: str = PERIOD_COL,
        block_size: int = BLOCK_SIZE,
        rng: np.random.Generator | None = None,
        seed: int = SEED,
    ):
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.period_col = period_col
        self.block_size = block_size
        self.seed = seed
        self.rng = rng or np.random.default_rng(seed)

    def sample_manifest(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        periods = [int(w) for w in sorted(train[self.period_col].unique().tolist())]
        n_periods = len(periods)
        mapping, block_records, next_id = self._sample_train_order(periods)
        val_weeks = [int(w) for w in sorted(val[self.period_col].unique().tolist())]
        last_block = int(block_records[-1]["block_id"]) if block_records else 0
        val_order = [int(next_id + i) for i in range(len(val_weeks))]
        return {
            "n_original_train_periods": n_periods,
            "n_sampled_periods": len(mapping),
            "n_blocks": len(block_records),
            "train_order_max": int(next_id - 1) if next_id else 0,
            "val_order_min": int(val_order[0]) if val_order else None,
            "blocks": block_records,
            "val": {
                "source_weeks": val_weeks,
                "bootstrap_order": val_order,
                "bootstrap_block_id": last_block,
            },
        }

    def draw(self, train: pd.DataFrame, val: pd.DataFrame) -> BootstrapDraw:
        rec = self.sample_manifest(train, val)
        return materialize_draw(train, val, rec, period_col=self.period_col)

    def _sample_train_order(self, periods: list[int]) -> tuple[list[tuple[int, int, int]], list[dict], int]:
        n_periods = len(periods)
        if n_periods < 1:
            raise BootstrapError("train has no periods to resample")
        if n_periods < self.block_size:
            mapping = [(int(w), i, 0) for i, w in enumerate(periods)]
            return mapping, [{
                "block_id": 0,
                "start_index": 0,
                "source_weeks": [int(w) for w in periods],
                "bootstrap_order": list(range(n_periods)),
            }], n_periods

        length = self.block_size
        starts = list(range(0, n_periods, length))
        mapping: list[tuple[int, int, int]] = []
        block_records: list[dict] = []
        next_id = 0
        kept = 0
        block_id = 0
        while kept < n_periods:
            idx = int(self.rng.integers(0, len(starts)))
            start = int(starts[idx])
            block_periods = periods[start : start + length]
            take = min(len(block_periods), n_periods - kept)
            block_periods = block_periods[:take]
            orders = []
            for src in block_periods:
                mapping.append((int(src), int(next_id), int(block_id)))
                orders.append(int(next_id))
                next_id += 1
                kept += 1
            block_records.append({
                "block_id": int(block_id),
                "start_index": start,
                "source_weeks": [int(s) for s in block_periods],
                "bootstrap_order": orders,
            })
            block_id += 1
        if kept != n_periods:
            raise BootstrapError(f"sampled {kept} periods, need {n_periods}")
        return mapping, block_records, next_id


def _slots_frame(pairs: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=[SOURCE_WEEK_COL, BOOTSTRAP_ORDER_COL, BLOCK_ID_COL])


def apply_slots(df: pd.DataFrame, slots: pd.DataFrame, period_col: str = PERIOD_COL) -> pd.DataFrame:
    """Attach one synthetic slot per (source week, order, block). Repeats a week if sampled twice."""
    base = df.copy()
    base[SOURCE_WEEK_COL] = base[period_col].astype(int)
    out = base.merge(slots, on=SOURCE_WEEK_COL, how="inner")
    out[period_col] = out[BOOTSTRAP_ORDER_COL].astype(int)
    return out.reset_index(drop=True)


def materialize_draw(
    train: pd.DataFrame,
    val: pd.DataFrame,
    rec: dict,
    period_col: str = PERIOD_COL,
) -> BootstrapDraw:
    train_pairs = []
    for block in rec.get("blocks") or []:
        bid = int(block["block_id"])
        for src, order in zip(block["source_weeks"], block["bootstrap_order"]):
            train_pairs.append((int(src), int(order), bid))
    val_rec = rec.get("val") or {}
    val_bid = int(val_rec.get("bootstrap_block_id", (rec.get("blocks") or [{}])[-1].get("block_id", 0)))
    val_pairs = [
        (int(src), int(order), val_bid)
        for src, order in zip(val_rec.get("source_weeks") or [], val_rec.get("bootstrap_order") or [])
    ]
    train_b = apply_slots(train, _slots_frame(train_pairs), period_col=period_col)
    val_b = apply_slots(val, _slots_frame(val_pairs), period_col=period_col)
    n_orig = int(rec.get("n_original_train_periods") or train[period_col].nunique())
    validate_draw(train_b, val_b, n_orig)
    return BootstrapDraw(train=train_b, val=val_b, manifest=dict(rec))


@dataclass
class BootstrapPlan:
    dataset: str
    n_train_periods: int
    block_size: int
    seed: int
    n_boot: int
    calendar: FrozenCalendar
    train_weeks: tuple[int, ...]
    val_weeks: tuple[int, ...]
    replicates: tuple[dict, ...]

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "n_train_periods": self.n_train_periods,
            "block_size": self.block_size,
            "seed": self.seed,
            "n_boot": self.n_boot,
            "calendar": self.calendar.to_dict(),
            "train_weeks": list(self.train_weeks),
            "val_weeks": list(self.val_weeks),
            "replicates": [dict(r) for r in self.replicates],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BootstrapPlan":
        return cls(
            dataset=str(payload["dataset"]),
            n_train_periods=int(payload["n_train_periods"]),
            block_size=int(payload["block_size"]),
            seed=int(payload["seed"]),
            n_boot=int(payload["n_boot"]),
            calendar=FrozenCalendar.from_dict(payload["calendar"]),
            train_weeks=tuple(int(w) for w in payload.get("train_weeks") or []),
            val_weeks=tuple(int(w) for w in payload.get("val_weeks") or []),
            replicates=tuple(dict(r) for r in payload.get("replicates") or []),
        )

    def draw(self, train: pd.DataFrame, val: pd.DataFrame, bootstrap_id: int) -> BootstrapDraw:
        rec = dict(self.replicates[int(bootstrap_id)])
        rec["bootstrap_id"] = int(bootstrap_id)
        return materialize_draw(train, val, rec)


def build_bootstrap_plan(
    train: pd.DataFrame,
    val: pd.DataFrame,
    dataset: str,
    *,
    n_boot: int | None = None,
    block_size: int = BLOCK_SIZE,
    seed: int = SEED,
) -> BootstrapPlan:
    n_boot = protocol_n_boot() if n_boot is None else int(n_boot)
    sampler = NonOverlappingBlockBootstrap(
        period_col=PERIOD_COL,
        block_size=block_size,
        rng=np.random.default_rng(seed),
        seed=seed,
    )
    replicates = []
    for b in range(n_boot):
        rec = sampler.sample_manifest(train, val)
        rec["bootstrap_id"] = b
        replicates.append(rec)
    return BootstrapPlan(
        dataset=dataset,
        n_train_periods=int(train[PERIOD_COL].nunique()),
        block_size=int(block_size),
        seed=int(seed),
        n_boot=n_boot,
        calendar=FrozenCalendar.from_panel(train),
        train_weeks=tuple(int(w) for w in sorted(train[PERIOD_COL].unique().tolist())),
        val_weeks=tuple(int(w) for w in sorted(val[PERIOD_COL].unique().tolist())),
        replicates=tuple(replicates),
    )


def bootstrap_plan_path(panel_dir: Path) -> Path:
    return Path(panel_dir) / BOOTSTRAP_PLAN_JSON


def save_bootstrap_plan(plan: BootstrapPlan, panel_dir: Path) -> None:
    panel_dir = Path(panel_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)
    path = bootstrap_plan_path(panel_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plan.to_dict(), indent=2))
    tmp.replace(path)
    #print("  wrote", path)


def load_bootstrap_plan(path: Path) -> BootstrapPlan:
    return BootstrapPlan.from_dict(json.loads(Path(path).read_text()))
