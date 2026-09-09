# Nested-protocol elasticity benchmarks

Extension of [nn-elasticity](https://github.com/carlosherediapimienta/nn-elasticity) (*Integrable Elasticity via Neural Demand Surfaces*). That repository introduces **ICDN** (Integrable Context-Dependent Demand Network): a demand-first neural surface whose own- and cross-price elasticities are exact derivatives of a single log-demand map.

This repository does **not** re-implement ICDN. It reuses the published `icdn` package and asks a different question: **does the integrable demand surface still look better than a comparable demand-first MLP, and than pairwise log-log OLS/Ridge, when the comparison protocol is nested, leakage-controlled, and shared across models and datasets?**

The original project evaluates ICDN on Dominick's beer against directed pairwise OLS (plus Ridge and MLP ablations in the same codebase). Here the same four estimators run on three scanner panels under one frozen SKU universe, one temporal split plan, and one block-bootstrap plan per dataset.

---

## What this extension adds

1. **Three panels, one schema.** Walmart M5 (FOODS_3), dunnhumby *Breakfast at the Frat* (frozen category after a no-lookahead screen), and Dominick beer. Each builder maps `(store, product, week) → (price, units, promo)` onto ICDN columns and writes a scientific master (zeros kept where they are real) plus an ICDN subset (`units > 0`, `price > 0`).
2. **Shared nested protocol.** Outer expanding-window CV and holdout cuts live in `split_plan.json`. The start week is the most restrictive of the default floor and the MLP/ICDN nested-fit requirement (every frozen SKU must appear in the shortest inner train). OLS and Ridge consume that plan; they do not pick their own folds.
3. **Shared block bootstrap.** `bootstrap_plan.json` stores non-overlapping week-block draws of exact train length, `bootstrap_block_id` isolation, and a calendar frozen on the original holdout train. All four models read the same draws.
4. **Causal prices.** ICDN's default panel builder can backfill missing prices. This repo patches that path: observed price, else last *past* store–product price, else the train SKU mean. Never a future price.
5. **Matched comparisons.** Predictive MAE is compared on the intersection of cells each model can predict. Cross-price elasticities are compared on matched directed edges. Series-level bootstrap CIs are not treated as a CI for a global parameter.
6. **Paired ICDN vs MLP.** Week-block bootstrap of already-fitted outer-validation residuals (no retraining) with a pre-specified non-inferiority margin (`PRED_NI_MARGIN`).

ICDN's architecture, spline bases, sparse neighbor graph, and elasticity penalties are documented in the [parent README](https://github.com/carlosherediapimienta/nn-elasticity). This repo treats ICDN as a fitted object and focuses on *how* it is trained and compared.

---

## Datasets

| Panel | Source | Observation | Promotion | Frozen screen (selection window only) |
| ----- | ------ | ----------- | --------- | ------------------------------------- |
| **Walmart M5** | M5 competition (`sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`) | Weekly units and `sell_price` | Markdown *proxy*: price vs lookback max | Density / unique prices in FOODS_3; `week_id` compacted over complete 7-day weeks |
| **dunnhumby** | *Breakfast at the Frat* (semicolon CSVs, decimal comma, Spanish month abbreviations in dates) | Already weekly `(store, UPC, week)` | Observed `feature` / `display` / `tpr_only` | Core stores, then category + Jaccard co-occurrence SKUs. `spend` / `hhs` / `visits` stay on the master only |
| **Dominick** | `dominick_features.csv` from the parent Dominick pipeline | Liters sold and price per liter | Observed B/S/C flag | Core stores, then beer category + Jaccard SKUs. Placeholder rows with zero price/units are dropped, never recoded as ones |

SKU and store identity is frozen using only the first half of observed weeks (`selection_frac = 0.5`). Nothing in that screen is tuned on later weeks or on model output.

Place raw files under:

```
data/M5-walmart/          # M5 CSVs (see notebooks/walmart-data.ipynb)
data/Dunnhumby/           # dunnhumby-transaction.csv, dunnhumby-products.csv, dunnhumby-store.csv
data/Dominick/            # dominick_features.csv
```

Derived panels (`*.parquet`, Optuna DBs) are gitignored except frozen-selection diagnostics.

---

## Evaluation protocol

| Piece | What is shared | Where it lives |
| ----- | -------------- | -------------- |
| Outer expanding CV (`N_FOLDS=3`) + holdout | All four models | `data/<dataset>/panel/split_plan.json` |
| Inner expanding CV for Optuna / Ridge α | MLP, ICDN, Ridge | never sees outer validation |
| Early stopping | last 20% of the *outer train* | MLP and ICDN; matches `ICDNConfig.validation_fraction` |
| Frozen product order | MLP and ICDN Jacobians | full-panel SKU list; a missing SKU in a **fit** slice is an error, in val it is a zero mask |
| Block bootstrap (`BLOCK_SIZE=20`) | all four models | `data/<dataset>/panel/bootstrap_plan.json` |
| Two clocks | bootstrap replicates | `source_week_id` = calendar; `week_id` = bootstrap order; `bootstrap_block_id` isolates lags / ffill / smoothing |
| Design knobs | trial counts, seeds, NI margin | `src/benchmarks/constants.py` |

Protocol constants are part of the research design. Changing them changes the experiment.

---

## Project structure

```
nn-elasticity-additional-work/
├── data/
│   ├── M5-walmart/
│   ├── Dunnhumby/
│   └── Dominick/
├── notebooks/
│   ├── walmart-data.ipynb      # M5 → ICDN panel (step-by-step, no builder.run())
│   ├── dunn-data.ipynb         # dunnhumby → ICDN panel
│   ├── dominick-data.ipynb     # Dominick features → ICDN panel
│   ├── ols.ipynb               # Pairwise OLS
│   ├── ridge.ipynb             # Pairwise Ridge
│   ├── mlp.ipynb               # Demand-first MLP
│   ├── icdn.ipynb              # ICDN nested eval (patches ICDN on import)
│   ├── analysis.ipynb          # Matched tables, paired ICDN–MLP, figures
│   └── figures/
├── src/
│   ├── m5.py                   # Walmart weekly panel builder
│   ├── dunn.py                 # dunnhumby weekly panel builder
│   ├── dominick.py             # Dominick weekly panel builder
│   └── benchmarks/             # Shared protocol + four experiments
│       ├── constants.py
│       ├── protocol.py         # splits, tables, run_all_datasets
│       ├── bootstrap.py        # two-clock block bootstrap
│       ├── features.py         # ICDN FeatureBuilder reused by OLS/Ridge/MLP
│       ├── prices.py           # CausalPriceFill (no bfill)
│       ├── universe.py         # Frozen SKU layout
│       ├── linear.py           # OLS / Ridge experiment
│       ├── mlp.py
│       ├── icdn.py             # Do not import ICDNExperiment from package __init__
│       ├── reporting.py        # Derived comparison CSVs (rebuild_all)
│       └── paired.py           # ICDN vs MLP inference
└── requirements.txt
```

Do **not** `from src.benchmarks import ICDNExperiment`. Loading ICDN monkey-patches ICDN's `PanelBuilder`. Import it from `src.benchmarks.icdn` (as `icdn.ipynb` does).

---

## Pipeline

```
Raw scanner files
        │
        ▼
walmart-data / dunn-data / dominick-data
        │
        ▼
*_icdn_panel.parquet  +  selected stores/SKUs  +  diagnostics
        │
        ├──────────────┬──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
     ols.ipynb     ridge.ipynb     mlp.ipynb     icdn.ipynb
        │              │              │              │
        └──────────────┴──────────────┴──────────────┘
                               │
                               ▼
                    analysis.ipynb  (rebuild_all → figures)
```

Each experiment notebook is a thin wrapper: construct the experiment class and call `run_all()`. Datasets whose parquet is missing are skipped. The first fitted model on a panel writes `split_plan.json` and `bootstrap_plan.json`; later models **read** them.

`analysis.ipynb` starts with `rebuild_all()`, which rebuilds matched-cell / matched-edge CSVs from files already on disk. It does not retrain.

---

## Estimators

The four estimators share features and chronology where that is scientifically required. They are **not** forced to be symmetric in compute budget (see `N_TRIALS_*` and `N_BOOT_*` in `constants.py`).

1. **Pairwise OLS** (`PairwiseExperiment.ols()`) — directed log-log equation per product pair within store, promo + annual Fourier controls, HC1-style pairwise machinery from the parent design.
2. **Pairwise Ridge** — same grouping and formula; RidgeCV α is selected on inner expanding folds of the holdout train and reused.
3. **Demand MLP** — one global network on the multiproduct panel; elasticities by autodiff. No ICDN splines, sparse attention, bilinear cross potential, or elasticity penalties.
4. **ICDN** — published integrable head. Dataset extras (own/cross bounds, `beta_prior`, `same_category_first`) are in `ICDN_EXTRAS` and are treated as design constants, not holdout-tuned.

---

## Setup

```bash
git clone https://github.com/carlosherediapimienta/nn-elasticity-additional-work.git
cd nn-elasticity-additional-work
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`icdn==1.1.0` is the model package from the parent project. Jupyter should be started with the repo root (or `notebooks/`) on `sys.path`; the notebooks insert the parent of `notebooks/` automatically.

M5, dunnhumby, and Dominick files are third-party and are not redistributed here. Dominick's Finer Foods data is from the Kilts Center at Chicago Booth, as in the parent repo.

---

## Relation to nn-elasticity

| | [nn-elasticity](https://github.com/carlosherediapimienta/nn-elasticity) | This repo |
| --- | --- | --- |
| Core contribution | ICDN architecture and Dominick beer study | Nested, multi-dataset protocol around the same ICDN |
| Data builders | Dominick raw → features → elasticity dataset | M5, dunnhumby, Dominick → ICDN parquet with a frozen universe |
| Baselines | Pairwise OLS, Ridge, demand MLP | Same families, shared splits/bootstrap/features/causal prices |
| Hyperparameters | Optuna in `hparam-search.ipynb` | Nested Optuna *inside* each outer fold (`optuna.db` per fold) |
| Analysis | `analysis-results.ipynb` | `analysis.ipynb` (matched cells/edges + paired ICDN–MLP) |

If you need to change ICDN layers, losses, or spline code, do that in the parent package. Change this repo when the **comparison** changes (panels, leakage guards, matched tables, inference).

---

## Authors

**Researchers:** Carlos Heredia, PhD & Daniel Roncel

**Affiliation:** IAMMResearch
