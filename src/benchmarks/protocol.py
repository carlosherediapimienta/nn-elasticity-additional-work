"""Shared evaluation protocol: panels, splits, tables, and summaries.

Native cell metrics (MAE/RMSE/R² plus prediction_coverage) are each model on
the cells it can predict. Direct comparison uses the intersection of those
cells (`pred_matched_kfold.csv` / `pred_matched_holdout.csv`). Cross-elasticity
comparisons use matched directed edges (`edge_matched_*.csv`): ICDN∩MLP, then
ICDN∩Ridge and ICDN∩OLS. Pairwise own-price is reported as a product mean
(`kind=own`) and as equation-level β_ij^own (`kind=own_eq`); OLS vs Ridge own
stability uses common partners (`own_matched_*.csv`). Native full-set
`cross_mean` in kfold.csv is not comparable across models. Elasticity
uncertainty stays series-level.

Outer folds and the holdout are a single per-dataset plan (`split_plan.json`).
The cut is the most restrictive of the default floor, MLP nested fit, and
ICDN nested fit. OLS, Ridge, MLP, and ICDN consume that plan; they do not
call the splitter independently. Bootstrap draws live in `bootstrap_plan.json`
(exact train length, `bootstrap_block_id`, frozen calendar). Models read that
file; they do not each call the sampler.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from icdn import ICDNConfig, PanelSchema
from icdn.data.splits import TemporalSplitter

from src.benchmarks.bootstrap import BootstrapPlan, NonOverlappingBlockBootstrap, build_bootstrap_plan, bootstrap_plan_path, load_bootstrap_plan, protocol_n_boot, save_bootstrap_plan

from src.benchmarks.constants import (
    BLOCK_SIZE,
    HOLDOUT_TRAIN_FRAC,
    INNER_TRAIN_FRAC,
    MIN_INNER_FRAC,
    MIN_TRAIN_FRAC,
    N_FOLDS,
    N_INNER_FOLDS,
    PERIOD_COL,
    SEED,
)
from src.benchmarks.features import ICDNFeaturePipeline
from src.benchmarks.predict import (
    COMPARE_MODELS,
    EDGE_COMPARISONS,
    EDGE_KEYS,
    OWN_EQ_KIND,
    attach_pred,
    boot_fold_ratio,
    bootstrap_series_report,
    cross_only,
    edge_fold_sd_summary,
    edge_presence_coverage,
    fixed_partner_product_own,
    fold_series_stats,
    intersect_cross_keys,
    intersect_kind_keys,
    kind_only,
    matched_edge_rows,
    matched_eval_rows,
    matched_global,
    matched_own_rows,
    native_metrics,
    normalize_series_keys,
    own_eq_only,
    own_product_diagnostics,
    point_in_boot_ci,
    product_own_on_keys,
    series_fold_sd_summary,
    share_abs_le_1,
    val_cells,
)


def project_root(cwd: Path | None = None) -> Path:
    """Repo root whether the kernel was started in the repo or in notebooks/."""
    cwd = Path.cwd() if cwd is None else cwd
    return cwd if cwd.name != "notebooks" else cwd.parent


def model_datasets(root: Path, model: str, extras: dict | None = None) -> dict:
    """Walmart M5, 1C, and Dominick panels, with per-model output directory `panel/<model>/`.

    `extras` is merged per dataset (ICDN bounds, category rule).
    """
    extras = extras or {}
    specs = {
        "walmart": {
            "path": root / "data" / "M5-walmart" / "panel" / "m5_icdn_panel.parquet",
            "out": root / "data" / "M5-walmart" / "panel" / model,
            "schema": PanelSchema(category="category"),
        },
        "dunnhumby": {
            "path": root / "data" / "Dunnhumby" / "panel" / "dunnhumby_icdn_panel.parquet",
            "out": root / "data" / "Dunnhumby" / "panel" / model,
            "schema": PanelSchema(
                category="category",
                brand="brand",
                style="style",
                size="size",
            ),
        },
        "dominick": {
            "path": root / "data" / "Dominick" / "panel" / "dominick_icdn_panel.parquet",
            "out": root / "data" / "Dominick" / "panel" / model,
            "schema": PanelSchema(
                category="category",
                brand="brand",
                style="style",
                size="size",
            ),
        },
    }
    for name, extra in extras.items():
        if name not in specs:
            continue
        specs[name] = {**specs[name], **extra}
    return {name: spec for name, spec in specs.items() if Path(spec["path"]).exists()}


def load_panel(spec: dict) -> pd.DataFrame:
    """Drop non-positive prices or units. ICDN and the linear models both need logs."""
    panel = pd.read_parquet(spec["path"])
    return panel[(panel["price"] > 0) & (panel["units"] > 0)].copy()


def save_table(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    """Write a CSV and echo the path so notebook logs stay searchable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df.to_csv(path, index=False)
    #print("  wrote", path)


def temporal_splitter() -> TemporalSplitter:
    return TemporalSplitter(period_col=PERIOD_COL)


def _frac_for_count(n_periods: int, n_train: int) -> float:
    """train_frac so TemporalSplitter.int(n * frac) equals n_train."""
    return min((n_train + 0.5) / n_periods, 1.0 - 1e-9)


def n_fit_periods(n_periods: int, train_frac: float) -> int:
    """Length of TemporalSplitter.single_split(..., train_frac) training side."""
    if n_periods < 2:
        return 0
    return min(max(int(n_periods * train_frac), 1), n_periods - 1)


def shortest_nested_fit_periods(
    n_outer: int,
    *,
    n_inner_folds: int = N_INNER_FOLDS,
    min_inner_frac: float = MIN_INNER_FRAC,
    fit_frac: float = INNER_TRAIN_FRAC,
) -> int:
    """Periods in MLP fit_raw / ICDN internal train of inner fold 0 of this outer train."""
    min_inner = max(1, int(n_outer * min_inner_frac))
    if n_outer - min_inner < n_inner_folds:
        return 0
    return n_fit_periods(min_inner, fit_frac)


def delayed_min_train(
    panel: pd.DataFrame,
    products: list[str],
    *,
    fit_frac: float,
    n_folds: int = N_FOLDS,
    min_train_frac: float = MIN_TRAIN_FRAC,
    n_inner_folds: int = N_INNER_FOLDS,
    min_inner_frac: float = MIN_INNER_FRAC,
) -> int:
    """Smallest outer-train length whose shortest nested fit contains every frozen SKU.

    Inner fold 0 of outer fold 0, then the 80% parameter-fitting split, is the
    shortest train in the protocol. Outer folds start no earlier than that.
    """
    from src.benchmarks.universe import UniverseError, first_complete_period_count

    periods = sorted(panel[PERIOD_COL].unique().tolist())
    n_periods = len(periods)
    need = first_complete_period_count(panel, products)
    n0 = max(1, int(n_periods * min_train_frac))
    n_max = n_periods - n_folds
    if n_max < n0:
        raise UniverseError(
            f"not enough periods ({n_periods}) for {n_folds} folds "
            f"with min_train_frac={min_train_frac}"
        )
    for n_outer in range(n0, n_max + 1):
        if shortest_nested_fit_periods(
            n_outer,
            n_inner_folds=n_inner_folds,
            min_inner_frac=min_inner_frac,
            fit_frac=fit_frac,
        ) >= need:
            return n_outer
    raise UniverseError(
        f"need {need} periods in the shortest nested fit for the frozen universe, "
        f"but outer train cannot exceed {n_max} periods and still leave {n_folds} val folds"
    )


def delayed_holdout_n_train(
    panel: pd.DataFrame,
    products: list[str],
    *,
    fit_frac: float,
    train_frac: float = HOLDOUT_TRAIN_FRAC,
    n_inner_folds: int = N_INNER_FOLDS,
    min_inner_frac: float = MIN_INNER_FRAC,
) -> int:
    """Holdout train length whose first inner fit contains every frozen SKU."""
    from src.benchmarks.universe import UniverseError, first_complete_period_count

    n_periods = panel[PERIOD_COL].nunique()
    need = first_complete_period_count(panel, products)
    n0 = n_fit_periods(n_periods, train_frac)
    for n_train in range(n0, n_periods):
        if shortest_nested_fit_periods(
            n_train,
            n_inner_folds=n_inner_folds,
            min_inner_frac=min_inner_frac,
            fit_frac=fit_frac,
        ) >= need:
            return n_train
    raise UniverseError(
        f"need {need} periods in the shortest nested holdout fit for the frozen universe"
    )


def expanding_folds(
    panel: pd.DataFrame,
    n_folds: int = N_FOLDS,
    min_train_frac: float = MIN_TRAIN_FRAC,
    min_train: int | None = None,
):
    if min_train is not None:
        min_train_frac = _frac_for_count(panel[PERIOD_COL].nunique(), min_train)
    folds = temporal_splitter().expanding_splits(
        panel, n_folds=n_folds, min_train_frac=min_train_frac
    )
    if min_train is not None and folds[0][0][PERIOD_COL].nunique() < min_train:
        raise ValueError(
            f"first outer train has {folds[0][0][PERIOD_COL].nunique()} periods, need {min_train}"
        )
    return folds


def holdout_split(
    panel: pd.DataFrame,
    train_frac: float = HOLDOUT_TRAIN_FRAC,
    n_train: int | None = None,
):
    if n_train is not None:
        train_frac = _frac_for_count(panel[PERIOD_COL].nunique(), n_train)
    return temporal_splitter().single_split(panel, train_frac=train_frac)


def icdn_fit_frac() -> float:
    """Parameter-fitting share of an ICDN train window (1 - validation_fraction)."""
    return 1.0 - float(ICDNConfig().validation_fraction)


def _protocol_fingerprint() -> dict:
    return {
        "N_FOLDS": N_FOLDS,
        "MIN_TRAIN_FRAC": MIN_TRAIN_FRAC,
        "HOLDOUT_TRAIN_FRAC": HOLDOUT_TRAIN_FRAC,
        "N_INNER_FOLDS": N_INNER_FOLDS,
        "MIN_INNER_FRAC": MIN_INNER_FRAC,
        "INNER_TRAIN_FRAC": INNER_TRAIN_FRAC,
        "icdn_fit_frac": icdn_fit_frac(),
        "PERIOD_COL": PERIOD_COL,
    }


def _period_tuple(df: pd.DataFrame) -> tuple[int, ...]:
    return tuple(int(x) for x in sorted(df[PERIOD_COL].unique().tolist()))


def _products_hash(products) -> str:
    toks = sorted({str(p) for p in products})
    return hashlib.sha1("|".join(toks).encode()).hexdigest()[:12]


SPLIT_PLAN_JSON = "split_plan.json"
SPLIT_PLAN_CSV = "split_plan.csv"


@dataclass(frozen=True)
class SplitWindow:
    """One outer train/validation cut, stored as the exact period lists."""

    name: str
    train_periods: tuple[int, ...]
    val_periods: tuple[int, ...]

    def slice(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        col = panel[PERIOD_COL]
        train = panel[col.isin(self.train_periods)].copy()
        val = panel[col.isin(self.val_periods)].copy()
        if train.empty or val.empty:
            raise ValueError(
                f"split window {self.name} produced empty train or val "
                f"(n_train={len(train)}, n_val={len(val)})"
            )
        return train, val

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "train_periods": list(self.train_periods),
            "val_periods": list(self.val_periods),
            "train_week_min": min(self.train_periods) if self.train_periods else None,
            "train_week_max": max(self.train_periods) if self.train_periods else None,
            "val_week_min": min(self.val_periods) if self.val_periods else None,
            "val_week_max": max(self.val_periods) if self.val_periods else None,
            "n_train_periods": len(self.train_periods),
            "n_val_periods": len(self.val_periods),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SplitWindow":
        return cls(
            name=str(payload["name"]),
            train_periods=tuple(int(x) for x in payload["train_periods"]),
            val_periods=tuple(int(x) for x in payload["val_periods"]),
        )


@dataclass(frozen=True)
class SplitPlan:
    """Dataset-level outer folds and holdout shared by OLS, Ridge, MLP, and ICDN."""

    dataset: str
    n_periods: int
    min_train: int
    n_holdout_train: int
    default_min_train: int
    mlp_min_train: int
    icdn_min_train: int
    default_holdout_n_train: int
    mlp_holdout_n_train: int
    icdn_holdout_n_train: int
    frozen_products: tuple[str, ...]
    frozen_products_hash: str
    protocol: dict
    folds: tuple[SplitWindow, ...]
    holdout: SplitWindow

    def materialize_folds(self, panel: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        return [window.slice(panel) for window in self.folds]

    def materialize_holdout(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.holdout.slice(panel)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "n_periods": self.n_periods,
            "min_train": self.min_train,
            "n_holdout_train": self.n_holdout_train,
            "requirements": {
                "default_min_train": self.default_min_train,
                "mlp_min_train": self.mlp_min_train,
                "icdn_min_train": self.icdn_min_train,
                "default_holdout_n_train": self.default_holdout_n_train,
                "mlp_holdout_n_train": self.mlp_holdout_n_train,
                "icdn_holdout_n_train": self.icdn_holdout_n_train,
            },
            "frozen_products": list(self.frozen_products),
            "frozen_products_hash": self.frozen_products_hash,
            "protocol": self.protocol,
            "folds": [window.to_dict() for window in self.folds],
            "holdout": self.holdout.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SplitPlan":
        req = payload["requirements"]
        return cls(
            dataset=str(payload["dataset"]),
            n_periods=int(payload["n_periods"]),
            min_train=int(payload["min_train"]),
            n_holdout_train=int(payload["n_holdout_train"]),
            default_min_train=int(req["default_min_train"]),
            mlp_min_train=int(req["mlp_min_train"]),
            icdn_min_train=int(req["icdn_min_train"]),
            default_holdout_n_train=int(req["default_holdout_n_train"]),
            mlp_holdout_n_train=int(req["mlp_holdout_n_train"]),
            icdn_holdout_n_train=int(req["icdn_holdout_n_train"]),
            frozen_products=tuple(str(p) for p in payload["frozen_products"]),
            frozen_products_hash=str(payload["frozen_products_hash"]),
            protocol=dict(payload["protocol"]),
            folds=tuple(SplitWindow.from_dict(w) for w in payload["folds"]),
            holdout=SplitWindow.from_dict(payload["holdout"]),
        )


def common_min_train(panel: pd.DataFrame, products: list[str]) -> tuple[int, dict[str, int]]:
    """Outer-train length required by the most restrictive model."""
    n_periods = panel[PERIOD_COL].nunique()
    default_min_train = max(1, int(n_periods * MIN_TRAIN_FRAC))
    mlp_min_train = delayed_min_train(panel, products, fit_frac=INNER_TRAIN_FRAC)
    icdn_min_train = delayed_min_train(panel, products, fit_frac=icdn_fit_frac())
    req = {
        "default_min_train": int(default_min_train),
        "mlp_min_train": int(mlp_min_train),
        "icdn_min_train": int(icdn_min_train),
    }
    return max(req.values()), req


def common_holdout_n_train(panel: pd.DataFrame, products: list[str]) -> tuple[int, dict[str, int]]:
    """Holdout-train length required by the most restrictive model."""
    n_periods = panel[PERIOD_COL].nunique()
    default_holdout = n_fit_periods(n_periods, HOLDOUT_TRAIN_FRAC)
    mlp_holdout = delayed_holdout_n_train(panel, products, fit_frac=INNER_TRAIN_FRAC)
    icdn_holdout = delayed_holdout_n_train(panel, products, fit_frac=icdn_fit_frac())
    req = {
        "default_holdout_n_train": int(default_holdout),
        "mlp_holdout_n_train": int(mlp_holdout),
        "icdn_holdout_n_train": int(icdn_holdout),
    }
    return max(req.values()), req


def _window_from_split(name: str, train: pd.DataFrame, val: pd.DataFrame) -> SplitWindow:
    return SplitWindow(name=str(name), train_periods=_period_tuple(train), val_periods=_period_tuple(val))


def build_split_plan(panel: pd.DataFrame, dataset: str, products: list[str] | None = None) -> SplitPlan:
    """Build the common outer folds and holdout. Call this once per dataset."""
    from src.benchmarks.universe import freeze_products

    products = list(products) if products is not None else freeze_products(panel)
    n_periods = panel[PERIOD_COL].nunique()
    min_train, min_req = common_min_train(panel, products)
    n_holdout, hold_req = common_holdout_n_train(panel, products)
    folds = expanding_folds(panel, n_folds=N_FOLDS, min_train=min_train)
    holdout_train, holdout_val = holdout_split(panel, n_train=n_holdout)
    return SplitPlan(
        dataset=dataset,
        n_periods=int(n_periods),
        min_train=int(min_train),
        n_holdout_train=int(n_holdout),
        frozen_products=tuple(str(p) for p in products),
        frozen_products_hash=_products_hash(products),
        protocol=_protocol_fingerprint(),
        folds=tuple(_window_from_split(i, tr, va) for i, (tr, va) in enumerate(folds, 1)),
        holdout=_window_from_split("holdout", holdout_train, holdout_val),
        **min_req,
        **hold_req,
    )


def split_plan_path(panel_dir: Path) -> Path:
    return Path(panel_dir) / SPLIT_PLAN_JSON


def _save_split_plan(plan: SplitPlan, panel_dir: Path) -> None:
    panel_dir = Path(panel_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)
    path = split_plan_path(panel_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plan.to_dict(), indent=2))
    tmp.replace(path)
    #print("  wrote", path)
    rows = []
    lead = [
        "dataset", "fold",
        "train_week_min", "train_week_max",
        "val_week_min", "val_week_max",
        "n_train_periods", "n_val_periods",
        "min_train", "n_holdout_train",
    ]
    for window in (*plan.folds, plan.holdout):
        rec = window.to_dict()
        rec["dataset"] = plan.dataset
        rec["fold"] = rec.pop("name")
        rec["min_train"] = plan.min_train
        rec["n_holdout_train"] = plan.n_holdout_train
        rec.pop("train_periods", None)
        rec.pop("val_periods", None)
        rows.append(rec)
    save_table(pd.DataFrame(rows)[lead], panel_dir, SPLIT_PLAN_CSV)


def _load_split_plan(path: Path) -> SplitPlan:
    return SplitPlan.from_dict(json.loads(path.read_text()))


def _print_split_plan(plan: SplitPlan, *, source: str) -> None:
    print(
        f"  common split plan ({source})  "
        f"min_train={plan.min_train}/{plan.n_periods}  "
        f"holdout_train={plan.n_holdout_train}/{plan.n_periods}"
    )
    print(
        f"  requirements  default_min={plan.default_min_train}  "
        f"mlp={plan.mlp_min_train}  icdn={plan.icdn_min_train}  "
        f"default_holdout={plan.default_holdout_n_train}  "
        f"mlp_holdout={plan.mlp_holdout_n_train}  icdn_holdout={plan.icdn_holdout_n_train}"
    )
    if plan.min_train > plan.default_min_train:
        print(
            f"  delayed outer train to {plan.min_train}/{plan.n_periods} periods "
            "so the shortest nested fit contains every frozen SKU"
        )


def _canon_protocol(payload: dict) -> dict:
    return json.loads(json.dumps(payload))


def _plan_is_current(plan: SplitPlan, panel: pd.DataFrame, products: list[str]) -> bool:
    if _canon_protocol(plan.protocol) != _canon_protocol(_protocol_fingerprint()):
        return False
    if tuple(plan.frozen_products) != tuple(str(p) for p in products):
        return False
    have = {int(x) for x in panel[PERIOD_COL].unique()}
    for window in (*plan.folds, plan.holdout):
        if any(p not in have for p in (*window.train_periods, *window.val_periods)):
            return False
    return True


def dataset_split_plan(
    panel: pd.DataFrame,
    spec: dict,
    dataset: str,
    products: list[str] | None = None,
) -> SplitPlan:
    """Load the dataset split plan, or build and persist it on first use.

    Later models read `split_plan.json`. They do not re-run expanding_folds
    or holdout_split to choose their own outer cuts.
    """
    from src.benchmarks.universe import freeze_products

    products = list(products) if products is not None else freeze_products(panel)
    panel_dir = Path(spec["out"]).parent
    path = split_plan_path(panel_dir)
    if path.exists():
        try:
            plan = _load_split_plan(path)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"  could not load {path.name} ({e}); rebuilding")
        else:
            if _plan_is_current(plan, panel, products):
                _print_split_plan(plan, source=f"read {path.name}")
                return plan
            print(f"  {path.name} is stale; rebuilding")
    plan = build_split_plan(panel, dataset, products=products)
    _save_split_plan(plan, panel_dir)
    _print_split_plan(plan, source=f"wrote {path.name}")
    return plan


def first_inner_train(train_raw: pd.DataFrame) -> pd.DataFrame:
    """Train slice of inner fold 0. This is the shortest nested training window."""
    return temporal_splitter().expanding_splits(
        train_raw, n_folds=N_INNER_FOLDS, min_train_frac=MIN_INNER_FRAC
    )[0][0]


def block_sampler(seed: int = SEED, block_size: int = BLOCK_SIZE) -> NonOverlappingBlockBootstrap:
    """Non-overlapping block bootstrap (starts 0, 4, 8, ...), two clocks."""
    return NonOverlappingBlockBootstrap(
        period_col=PERIOD_COL,
        block_size=block_size,
        rng=np.random.default_rng(seed),
        seed=seed,
    )


def _bootstrap_plan_is_current(
    plan: BootstrapPlan,
    train: pd.DataFrame,
    val: pd.DataFrame,
    n_boot: int,
) -> bool:
    if int(plan.n_boot) < int(n_boot):
        return False
    if int(plan.block_size) != int(BLOCK_SIZE) or int(plan.seed) != int(SEED):
        return False
    if int(plan.n_train_periods) != int(train[PERIOD_COL].nunique()):
        return False
    train_weeks = tuple(int(w) for w in sorted(train[PERIOD_COL].unique().tolist()))
    val_weeks = tuple(int(w) for w in sorted(val[PERIOD_COL].unique().tolist()))
    if tuple(plan.train_weeks) and tuple(plan.train_weeks) != train_weeks:
        return False
    return tuple(plan.val_weeks) == val_weeks


def dataset_bootstrap_plan(
    train: pd.DataFrame,
    val: pd.DataFrame,
    spec: dict,
    dataset: str,
    n_boot: int | None = None,
) -> BootstrapPlan:
    """Load the shared bootstrap draws, or build and persist them on first use."""
    n_boot = protocol_n_boot() if n_boot is None else int(n_boot)
    panel_dir = Path(spec["out"]).parent
    path = bootstrap_plan_path(panel_dir)
    if path.exists():
        try:
            plan = load_bootstrap_plan(path)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"  could not load {path.name} ({e}); rebuilding")
        else:
            if _bootstrap_plan_is_current(plan, train, val, n_boot):
                print(
                    f"  common bootstrap plan (read {path.name})  "
                    f"n_boot={plan.n_boot}  n_train={plan.n_train_periods}"
                )
                return plan
            print(f"  {path.name} is stale; rebuilding")
    plan = build_bootstrap_plan(train, val, dataset, n_boot=n_boot)
    save_bootstrap_plan(plan, panel_dir)
    print(
        f"  common bootstrap plan (wrote {path.name})  "
        f"n_boot={plan.n_boot}  n_train={plan.n_train_periods}"
    )
    return plan


def featurize(spec: dict, train_raw: pd.DataFrame, val_raw: pd.DataFrame):
    """Fit ICDN features on train only; transform val with the train tail (no leakage)."""
    feats = ICDNFeaturePipeline(schema=spec["schema"])
    train = feats.fit(train_raw).transform(train_raw)
    val = feats.transform_val(val_raw)
    return train, val


def print_dataset_banner(name: str, panel: pd.DataFrame) -> None:
    print(
        f"\n=== {name} === {panel.shape}  "
        f"products={panel['product_code'].nunique()}  "
        f"stores={panel['store_code'].nunique()}"
    )


def print_fold_banner(name: str, k: int, n_folds: int, train_raw: pd.DataFrame, val_raw: pd.DataFrame) -> None:
    print(
        f"--- {name} fold {k}/{n_folds}  "
        f"weeks {train_raw.week_id.min()}–{train_raw.week_id.max()} | "
        f"{val_raw.week_id.min()}–{val_raw.week_id.max()}"
    )


def print_holdout_banner(name: str, train_raw: pd.DataFrame, val_raw: pd.DataFrame) -> None:
    print(
        f"--- {name} holdout  train weeks {train_raw.week_id.min()}–{train_raw.week_id.max()} | "
        f"val {val_raw.week_id.min()}–{val_raw.week_id.max()}"
    )


def save_pred_grid(val, cells, dataset, fold, out_dir: Path) -> dict:
    """Join ŷ onto the val grid, write pred_cells, refresh matched eval, return native metrics."""
    filename = "holdout_pred_cells.csv" if fold == "holdout" else f"fold{fold}_pred_cells.csv"
    grid = attach_pred(val_cells(val, dataset, fold), cells)
    save_table(grid, out_dir, filename)
    refresh_matched_eval(out_dir.parent)
    return native_metrics(grid)


def save_pred_ij(pred_ij: pd.DataFrame, dataset, fold, out_dir: Path) -> None:
    """Pairwise ŷ_ijst with dataset / outer_fold so OLS/Ridge equations can be audited."""
    out = pred_ij.copy()
    if "dataset" not in out.columns:
        out.insert(0, "dataset", dataset)
    if "outer_fold" not in out.columns:
        out.insert(1, "outer_fold", fold)
    name = "holdout_pred_ij.csv" if fold == "holdout" else f"fold{fold}_pred_ij.csv"
    save_table(out, out_dir, name)


def save_elasticities_long(df: pd.DataFrame, dataset, model, fold, out_dir: Path) -> None:
    """Week-level MLP/ICDN elasticities (own/cross) with observed_i / observed_j."""
    from src.benchmarks.predict import elasticity_long

    if df is None or len(df) == 0:
        return
    out = elasticity_long(df)
    out.insert(0, "dataset", dataset)
    out.insert(1, "model", model)
    out.insert(2, "outer_fold", fold)
    save_table(out, out_dir, f"fold{fold}_elasticities_long.csv")


def _rebuild_derived(panel_dir: Path) -> None:
    from src.benchmarks.reporting import rebuild_panel_tables

    rebuild_panel_tables(panel_dir)


def refresh_matched_eval(panel_dir: Path) -> None:
    """Rewrite panel-level matched-cell tables from whatever model pred_cells exist."""
    panel_dir = Path(panel_dir)
    folds: set[str] = set()
    for model in COMPARE_MODELS:
        for path in (panel_dir / model).glob("fold*_pred_cells.csv"):
            tag = path.name.removeprefix("fold").removesuffix("_pred_cells.csv")
            folds.add(tag)
    kfold_rows = []
    for tag in sorted(folds, key=lambda x: int(x) if str(x).isdigit() else x):
        grids = {}
        for model in COMPARE_MODELS:
            path = panel_dir / model / f"fold{tag}_pred_cells.csv"
            if path.exists():
                grids[model] = pd.read_csv(path)
        kfold_rows.extend(matched_eval_rows(grids, outer_fold=tag))
    if kfold_rows:
        save_table(pd.DataFrame(kfold_rows), panel_dir, "pred_matched_kfold.csv")

    holdout_grids = {}
    for model in COMPARE_MODELS:
        path = panel_dir / model / "holdout_pred_cells.csv"
        if path.exists():
            holdout_grids[model] = pd.read_csv(path)
    holdout_rows = matched_eval_rows(holdout_grids, outer_fold="holdout")
    if holdout_rows:
        save_table(pd.DataFrame(holdout_rows), panel_dir, "pred_matched_holdout.csv")
    refresh_matched_edges(panel_dir)


def _load_model_csv(panel_dir: Path, model: str, name: str) -> pd.DataFrame | None:
    path = panel_dir / model / name
    if not path.exists():
        return None
    return normalize_series_keys(pd.read_csv(path))


def refresh_matched_edges(panel_dir: Path) -> None:
    """Matched-edge cross-elasticity tables: native vs ICDN∩MLP / ICDN∩Ridge / ICDN∩OLS."""
    panel_dir = Path(panel_dir)
    series_by_model = {}
    folds: set[str] = set()
    for model in COMPARE_MODELS:
        df = _load_model_csv(panel_dir, model, "kfold_series.csv")
        if df is None or df.empty:
            continue
        series_by_model[model] = df
        if "outer_fold" in df.columns:
            folds.update(df["outer_fold"].astype(str).unique())
    kfold_rows, long_parts = [], []
    for tag in sorted(folds, key=lambda x: int(x) if str(x).isdigit() else x):
        by_fold = {}
        for model, df in series_by_model.items():
            part = df[df["outer_fold"].astype(str) == str(tag)]
            if len(part):
                by_fold[model] = part
        rows, parts = matched_edge_rows(by_fold, outer_fold=tag)
        kfold_rows.extend(rows)
        long_parts.extend(parts)
    if kfold_rows:
        save_table(pd.DataFrame(kfold_rows), panel_dir, "edge_matched_kfold.csv")
    sd_parts = []
    for model, df in series_by_model.items():
        native_stats = fold_series_stats(df)
        if native_stats.empty:
            continue
        native_stats = native_stats.copy()
        native_stats["eval"] = "native"
        sd_parts.append(native_stats)
    if long_parts:
        long_df = pd.concat(long_parts, ignore_index=True)
        save_table(long_df, panel_dir, "edge_matched_kfold_series.csv")
        stats = fold_series_stats(long_df, extra_keys=("eval",))
        save_table(stats, panel_dir, "edge_matched_kfold_stats.csv")
        sd_parts.append(stats)
    if sd_parts:
        all_stats = pd.concat(sd_parts, ignore_index=True)
        coverage = edge_presence_coverage(all_stats)
        if len(coverage):
            save_table(coverage, panel_dir, "edge_matched_kfold_coverage.csv")
        sd = edge_fold_sd_summary(all_stats)
        if len(sd):
            save_table(sd, panel_dir, "edge_matched_kfold_sd.csv")

    holdouts = {}
    for model in COMPARE_MODELS:
        df = _load_model_csv(panel_dir, model, "holdout_elasticities.csv")
        if df is not None and len(df):
            holdouts[model] = df
    holdout_rows, _ = matched_edge_rows(holdouts, outer_fold="holdout")
    if holdout_rows:
        save_table(pd.DataFrame(holdout_rows), panel_dir, "edge_matched_holdout.csv")

    boot_rows = _matched_edge_bootstrap_rows(panel_dir, holdouts)
    if boot_rows:
        save_table(pd.DataFrame(boot_rows), panel_dir, "edge_matched_bootstrap.csv")
    refresh_matched_own(panel_dir)


def refresh_matched_own(panel_dir: Path) -> None:
    """Own-price tables: product mean vs equation-level, OLS∩Ridge on common partners."""
    panel_dir = Path(panel_dir)
    series_by_model = {}
    folds: set[str] = set()
    for model in COMPARE_MODELS:
        df = _load_model_csv(panel_dir, model, "kfold_series.csv")
        if df is None or df.empty:
            continue
        series_by_model[model] = df
        if "outer_fold" in df.columns:
            folds.update(df["outer_fold"].astype(str).unique())
    kfold_rows, long_parts = [], []
    sd_parts = []
    for tag in sorted(folds, key=lambda x: int(x) if str(x).isdigit() else x):
        by_fold = {}
        for model, df in series_by_model.items():
            part = df[df["outer_fold"].astype(str) == str(tag)]
            if len(part):
                by_fold[model] = part
        rows, parts = matched_own_rows(by_fold, outer_fold=tag)
        kfold_rows.extend(rows)
        long_parts.extend(parts)
    if kfold_rows:
        save_table(pd.DataFrame(kfold_rows), panel_dir, "own_matched_kfold.csv")
    for model, df in series_by_model.items():
        native_stats = fold_series_stats(df)
        if native_stats.empty:
            continue
        native_stats = native_stats.copy()
        native_stats["eval"] = "native"
        sd_parts.append(native_stats)
        fixed = fixed_partner_product_own(df)
        if len(fixed):
            fixed_stats = fold_series_stats(fixed)
            if len(fixed_stats):
                fixed_stats = fixed_stats.copy()
                fixed_stats["eval"] = "own_fixed"
                sd_parts.append(fixed_stats)
    if long_parts:
        long_df = pd.concat(long_parts, ignore_index=True)
        save_table(long_df, panel_dir, "own_matched_kfold_series.csv")
        stats = fold_series_stats(long_df, extra_keys=("eval",))
        save_table(stats, panel_dir, "own_matched_kfold_stats.csv")
        sd_parts.append(stats)
    if sd_parts:
        all_stats = pd.concat(sd_parts, ignore_index=True)
        sd_frames = [
            series_fold_sd_summary(all_stats, "own", mean_name="own_mean", sd_name="own_fold_sd"),
            series_fold_sd_summary(all_stats, OWN_EQ_KIND, mean_name="own_mean", sd_name="own_fold_sd"),
            series_fold_sd_summary(all_stats, "own_fixed", mean_name="own_mean", sd_name="own_fold_sd"),
        ]
        sd = pd.concat([f for f in sd_frames if len(f)], ignore_index=True)
        if len(sd):
            save_table(sd, panel_dir, "own_matched_kfold_sd.csv")

    holdouts = {}
    for model in COMPARE_MODELS:
        df = _load_model_csv(panel_dir, model, "holdout_elasticities.csv")
        if df is not None and len(df):
            holdouts[model] = df
    holdout_rows, _ = matched_own_rows(holdouts, outer_fold="holdout")
    if holdout_rows:
        save_table(pd.DataFrame(holdout_rows), panel_dir, "own_matched_holdout.csv")

    boot_rows = _matched_own_bootstrap_rows(panel_dir, holdouts)
    if boot_rows:
        save_table(pd.DataFrame(boot_rows), panel_dir, "own_matched_bootstrap.csv")


def _product_own_boot_ci(boot_eq: pd.DataFrame, keys: pd.DataFrame, universe_eq: pd.DataFrame) -> pd.DataFrame:
    if boot_eq.empty or keys is None or keys.empty:
        return pd.DataFrame()
    parts = []
    for boot_id, g in boot_eq.groupby("bootstrap_id"):
        prod = product_own_on_keys(g, keys)
        if prod.empty:
            continue
        prod["bootstrap_id"] = boot_id
        parts.append(prod)
    if not parts:
        return pd.DataFrame()
    uni = product_own_on_keys(universe_eq, keys)
    return bootstrap_series_report(pd.concat(parts, ignore_index=True), universe=uni if len(uni) else None)


def _matched_own_bootstrap_rows(panel_dir: Path, holdouts: dict) -> list[dict]:
    boots = {}
    for model in COMPARE_MODELS:
        df = _load_model_csv(panel_dir, model, "bootstrap_replicates.csv")
        if df is not None and len(df):
            boots[model] = df
    if not boots:
        return []
    rows = []
    dataset = None
    for df in holdouts.values():
        if df is not None and len(df) and "dataset" in df.columns:
            dataset = df["dataset"].iloc[0]
            break

    def summarize(eval_name, model, models, ci, *, n_native) -> dict:
        part = ci if len(ci) else ci
        return {
            "dataset": dataset,
            "outer_fold": "holdout",
            "eval": eval_name,
            "model": model,
            "models": ",".join(sorted(models)),
            "n_models": len(models),
            "n_series": int(len(part)),
            "n_own": int(len(part)),
            "n_own_native": int(n_native),
            "own_mean": float(part["mean"].mean()) if len(part) and "mean" in part.columns else np.nan,
            "own_bootstrap_sd": float(pd.to_numeric(part["sd"], errors="coerce").mean()) if len(part) and "sd" in part.columns else np.nan,
        }

    for model, boot in boots.items():
        uni = holdouts.get(model)
        if uni is None:
            continue
        ci = bootstrap_series_report(boot, universe=uni)
        own_n = int(len(kind_only(uni, "own").drop_duplicates(["store_code", "product_i"]))) if len(uni) else 0
        eq_n = int(len(own_eq_only(uni)[EDGE_KEYS].drop_duplicates())) if len(own_eq_only(uni)) else 0
        if len(ci) and "kind" in ci.columns:
            rows.append(summarize("native_own", model, [model], ci[ci["kind"].astype(str) == "own"], n_native=own_n))
            eq_ci = ci[ci["kind"].astype(str) == OWN_EQ_KIND]
            if eq_n and len(eq_ci):
                rows.append(summarize("native_own_eq", model, [model], eq_ci, n_native=eq_n))
    if "ols" in holdouts and "ridge" in holdouts and "ols" in boots and "ridge" in boots:
        keys = intersect_kind_keys(holdouts["ols"], holdouts["ridge"], OWN_EQ_KIND)
        pair = ["ols", "ridge"]
        if len(keys):
            for model in pair:
                uni_eq = own_eq_only(holdouts[model])
                uni_eq = keys.merge(uni_eq, on=EDGE_KEYS, how="inner")
                boot_eq = own_eq_only(boots[model])
                boot_eq = keys.merge(boot_eq, on=EDGE_KEYS, how="inner")
                eq_n = int(len(own_eq_only(holdouts[model])[EDGE_KEYS].drop_duplicates())) if len(own_eq_only(holdouts[model])) else 0
                own_n = int(len(kind_only(holdouts[model], "own").drop_duplicates(["store_code", "product_i"]))) if len(holdouts[model]) else 0
                if uni_eq.empty or boot_eq.empty:
                    continue
                ci_eq = bootstrap_series_report(boot_eq, universe=uni_eq)
                rows.append(summarize("matched_own_eq_ols_ridge", model, pair, ci_eq, n_native=eq_n))
                ci_prod = _product_own_boot_ci(boot_eq, keys, uni_eq)
                rows.append(summarize("matched_own_product_ols_ridge", model, pair, ci_prod, n_native=own_n))
    return rows


def _matched_edge_bootstrap_rows(panel_dir: Path, holdouts: dict) -> list[dict]:
    boots = {}
    for model in COMPARE_MODELS:
        df = _load_model_csv(panel_dir, model, "bootstrap_replicates.csv")
        if df is not None and len(df):
            boots[model] = df
    if not boots:
        return []
    rows = []
    dataset = None
    for df in holdouts.values():
        if df is not None and len(df) and "dataset" in df.columns:
            dataset = df["dataset"].iloc[0]
            break

    def summarize_ci(eval_name, model, models, ci, holdout_cross) -> dict:
        cross = ci[ci["kind"].astype(str) == "cross"] if len(ci) and "kind" in ci.columns else ci
        return {
            "dataset": dataset,
            "outer_fold": "holdout",
            "eval": eval_name,
            "model": model,
            "models": ",".join(sorted(models)),
            "n_models": len(models),
            "n_series": int(len(cross)),
            "n_cross": int(len(holdout_cross[EDGE_KEYS].drop_duplicates())) if len(holdout_cross) and all(c in holdout_cross.columns for c in EDGE_KEYS) else int(len(holdout_cross)),
            "cross_mean": float(cross["mean"].mean()) if len(cross) and "mean" in cross.columns else np.nan,
            "cross_bootstrap_sd": float(pd.to_numeric(cross["sd"], errors="coerce").mean()) if len(cross) and "sd" in cross.columns else np.nan,
            "share_abs_le_1": share_abs_le_1(holdout_cross["elasticity"] if "elasticity" in holdout_cross.columns else []),
        }

    for model, boot in boots.items():
        uni = holdouts.get(model)
        if uni is None:
            continue
        ci = bootstrap_series_report(boot, universe=uni)
        rows.append(summarize_ci("native", model, [model], ci, cross_only(uni)))
    for eval_name, left, right in EDGE_COMPARISONS:
        if left not in holdouts or right not in holdouts or left not in boots or right not in boots:
            continue
        keys = intersect_cross_keys(holdouts[left], holdouts[right])
        pair = [left, right]
        for model in pair:
            uni = cross_only(holdouts[model])
            uni = keys.merge(uni, on=EDGE_KEYS, how="inner") if len(keys) else uni.iloc[0:0]
            if len(uni):
                uni = uni.drop_duplicates(EDGE_KEYS)
            boot = cross_only(boots[model])
            boot = keys.merge(boot, on=EDGE_KEYS, how="inner") if len(keys) else boot.iloc[0:0]
            if uni.empty or boot.empty:
                rows.append(summarize_ci(eval_name, model, pair, pd.DataFrame(), uni))
                continue
            ci = bootstrap_series_report(boot, universe=uni)
            rows.append(summarize_ci(eval_name, model, pair, ci, uni))
    return rows


def _pred_metric_fields(metrics: dict) -> dict:
    mae = float(metrics.get("mae_native", metrics["mae_val"]))
    rmse = float(metrics.get("rmse_native", metrics["rmse_val"]))
    r2 = float(metrics.get("r2_native", metrics.get("r2_val", np.nan)))
    n_cells = int(metrics.get("n_cells", 0))
    n_val = int(metrics.get("n_val_cells", n_cells))
    cov = metrics.get("prediction_coverage")
    return {
        "mae_native": mae,
        "rmse_native": rmse,
        "r2_native": r2,
        "mae_val": mae,
        "rmse_val": rmse,
        "r2_val": r2,
        "n_cells": n_cells,
        "n_val_cells": n_val,
        "n_validation_cells": n_val,
        "prediction_coverage": float(cov) if cov is not None else np.nan,
    }


def summarize_pairwise(own: pd.DataFrame, cross: pd.DataFrame, metrics: dict, model: str, **extra) -> dict:
    """One row of fold/bootstrap metrics from pairwise own/cross tables."""
    row = dict(extra)
    row["model"] = model
    row["n_own"] = len(own)
    row["n_cross"] = len(cross)
    row["n_own_eq"] = int(len(cross))
    row["own_mean"] = float(own.own_elasticity.mean()) if len(own) else np.nan
    row["own_eq_mean"] = float(cross.own_elasticity.mean()) if len(cross) and "own_elasticity" in cross.columns else np.nan
    row["n_partners_mean"] = float(own.n_partners.mean()) if len(own) and "n_partners" in own.columns else np.nan
    row["n_partner_sets"] = int(own.partner_set_id.nunique()) if len(own) and "partner_set_id" in own.columns else np.nan
    row["cross_mean"] = float(cross.cross_elasticity.mean()) if len(cross) else np.nan
    row["share_abs_le_1"] = share_abs_le_1(cross.cross_elasticity if len(cross) else [])
    row.update(_pred_metric_fields(metrics))
    return row


def summarize_kind(metrics: dict, table: pd.DataFrame, model: str, *, n_cells_required: bool = True, **extra) -> dict:
    """One row of fold/bootstrap metrics from a long table with a `kind` column.

    MLP uses `metrics.get("n_cells", 0)`; ICDN indexes `metrics["n_cells"]`.
    `n_cells_required` preserves that difference.
    """
    own = table[table.kind == "own"]
    cross = table[table.kind == "cross"]
    row = dict(extra)
    row["model"] = model
    row["n_own"] = len(own)
    row["n_cross"] = len(cross)
    row["own_mean"] = float(own.elasticity.mean()) if len(own) else np.nan
    row["cross_mean"] = float(cross.elasticity.mean()) if len(cross) else np.nan
    row["share_abs_le_1"] = share_abs_le_1(cross.elasticity if len(cross) else [])
    fields = _pred_metric_fields(metrics)
    if not n_cells_required:
        fields["n_cells"] = int(metrics.get("n_cells", 0))
    row.update(fields)
    return row


def save_kfold_tables(out_dir: Path, rows: list, series: list, *, allow_empty: bool = False) -> pd.DataFrame:
    """Write kfold.csv, kfold_series.csv, and (when appropriate) kfold_series_stats.csv.

    Linear models always concatenate `series` (empty list raises). Neural models
    allow an empty skip-all-folds run and only write stats if any series exist.
    """
    out = pd.DataFrame(rows)
    if allow_empty:
        fold_long = pd.concat(series, ignore_index=True) if series else pd.DataFrame()
    else:
        fold_long = pd.concat(series, ignore_index=True)
    metric_cols = [
        "own_mean", "own_eq_mean", "n_partners_mean", "cross_mean", "share_abs_le_1",
        "mae_native", "rmse_native", "r2_native", "prediction_coverage",
    ]
    present = [c for c in metric_cols if c in out.columns]
    if len(out) and present:
        print(out[present].agg(["mean", "std"]))
    else:
        print("  no completed folds")
    save_table(out, out_dir, "kfold.csv")
    save_table(fold_long, out_dir, "kfold_series.csv")
    if (not allow_empty) or len(fold_long):
        save_table(fold_series_stats(fold_long), out_dir, "kfold_series_stats.csv")
        diag = own_product_diagnostics(fold_long)
        if len(diag):
            save_table(diag, out_dir, "kfold_own_diagnostics.csv")
            if "partner_set_stable" in diag.columns:
                stable = diag["partner_set_stable"].fillna(False).astype(bool)
                sd = pd.to_numeric(diag["fold_sd"], errors="coerce")
                print(
                    "  own partner_set_stable", float(stable.mean()),
                    "n_shift", int((~stable).sum()),
                    "fold_sd stable", float(sd[stable].mean()) if stable.any() else np.nan,
                    "fold_sd shift", float(sd[~stable].mean()) if (~stable).any() else np.nan,
                )
        fixed = fixed_partner_product_own(fold_long)
        if len(fixed):
            save_table(fixed, out_dir, "kfold_own_fixed_series.csv")
            save_table(fold_series_stats(fixed), out_dir, "kfold_own_fixed_stats.csv")
    refresh_matched_edges(out_dir.parent)
    _rebuild_derived(out_dir.parent)
    return out


def save_bootstrap_report(out_dir: Path, rows: list, replicates: list, holdout: pd.DataFrame) -> pd.DataFrame:
    """Matched series CIs, holdout-in-CI, and boot/fold SD ratio if kfold stats exist."""
    import json

    boots = pd.DataFrame(rows)
    n_ok = int(len(boots))
    n_att = n_ok
    manifest_path = out_dir / "bootstrap_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        n_att = max(n_ok, int(len(payload.get("replicates") or [])))
    if len(boots):
        if "n_attempted" not in boots.columns:
            boots["n_attempted"] = n_att
        if "n_successful" not in boots.columns:
            boots["n_successful"] = n_ok
        if "r2_val" not in boots.columns and "r2_native" in boots.columns:
            boots["r2_val"] = boots["r2_native"]
    boot_long = pd.concat(replicates, ignore_index=True)
    boot_ci = bootstrap_series_report(boot_long, universe=holdout)
    matched = boot_ci[boot_ci["matched"]]
    save_table(boots, out_dir, "bootstrap.csv")
    save_table(boot_long, out_dir, "bootstrap_replicates.csv")
    save_table(boot_ci, out_dir, "bootstrap_series_ci.csv")
    save_table(matched, out_dir, "bootstrap_matched.csv")
    save_table(matched_global(boot_ci), out_dir, "bootstrap_matched_global.csv")
    save_table(point_in_boot_ci(holdout, matched), out_dir, "holdout_in_boot_ci.csv")
    fold_stats_path = out_dir / "kfold_series_stats.csv"
    if fold_stats_path.exists():
        fold_stats = pd.read_csv(fold_stats_path)
        save_table(boot_fold_ratio(matched, fold_stats), out_dir, "boot_fold_ratio.csv")
    print(
        "  series", len(boot_ci),
        "matched", int(boot_ci["matched"].sum()),
        "mean freq", float(boot_ci["freq"].mean()),
    )
    print(matched_global(boot_ci))
    refresh_matched_edges(out_dir.parent)
    _rebuild_derived(out_dir.parent)
    return boot_long


def run_all_datasets(datasets: dict, run_kfold, run_bootstrap, extra_print=None):
    """Cell-2 loop: banner, common split plan, common bootstrap plan, kfold, then bootstrap."""
    kfold_tables, boot_tables = {}, {}
    for name, spec in datasets.items():
        panel = load_panel(spec)
        print_dataset_banner(name, panel)
        if extra_print is not None:
            extra_print()
        split_plan = dataset_split_plan(panel, spec, name)
        train_raw, val_raw = split_plan.materialize_holdout(panel)
        boot_plan = dataset_bootstrap_plan(train_raw, val_raw, spec, name)
        print("\n Start kfold")
        kfold_tables[name] = run_kfold(name, spec, split_plan)
        print("\n Start bootstrap")
        boot_tables[name] = run_bootstrap(name, spec, split_plan, boot_plan)
    return kfold_tables, boot_tables
