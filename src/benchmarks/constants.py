"""Protocol knobs shared by OLS, Ridge, MLP, and ICDN.

These numbers are part of the research design, not implementation details.
Changing them changes the experiment. Model-specific overrides (trial
budget, bootstrap replicates, Optuna pruner warmup) stay next to that
model so a reader can see the comparison is not fully symmetric.
"""

from __future__ import annotations

# Calendar column used by TemporalSplitter and the block bootstrap.
PERIOD_COL = "week_id"

# Outer expanding window. First train slice is the later of MIN_TRAIN_FRAC of
# the panel and the MLP/ICDN nested-fit requirement (every frozen SKU in the
# shortest inner train). All four models share that common start via
# split_plan.json; they do not choose their own outer cuts.
N_FOLDS = 3
MIN_TRAIN_FRAC = 0.5

# Inner expanding window for MLP/ICDN Optuna and Ridge α. Outer val is never
# seen during search.
N_INNER_FOLDS = 3
MIN_INNER_FRAC = 0.5

# Last 20% of an outer (or inner) train slice is held out for MLP early stopping.
# Must match ICDNConfig.validation_fraction: ICDN uses the same 80/20 split internally.
INNER_TRAIN_FRAC = 0.8

# Holdout used for the point estimate and as the bootstrap validation window.
HOLDOUT_TRAIN_FRAC = 0.8

# Non-overlapping block bootstrap on raw train weeks (starts 0, 4, 8, ...).
# Draws blocks until n_sampled equals the original train length. Isolation is
# bootstrap_block_id (lags, rolling, price ffill, smoothing), not a 1-week gap.
# Calendar ranks are frozen from the original holdout train. All models read
# bootstrap_plan.json. Validation weeks are never resampled.
BLOCK_SIZE = 20
SEED = 42

# Pairwise controls: promo plus annual Fourier terms. Same list for OLS and Ridge.
SHORT = ["promo", "sin_52", "cos_52"]

# Degrees of freedom and collinearity gates for pairwise linear equations.
MIN_DOF = 30
MAX_VIF = 10.0

# Hidden-layer menus shared by MLP and ICDN Optuna searches.
HIDDEN_CHOICES = {
    "256_128_64": (256, 128, 64),
    "128_64_32": (128, 64, 32),
    "128_64": (128, 64),
}

# Replicate counts. Linear models are cheap; ICDN is not, so it uses fewer boots.
N_BOOT_LINEAR = 100
N_BOOT_MLP = 100
N_BOOT_ICDN = 100

# MLP train budget. Optuna trials, outer folds, holdout, and bootstrap share these.
MLP_MAX_EPOCHS = 250
MLP_PATIENCE = 15

# Optuna budgets. MLP is currently a smoke search (1 trial); ICDN is 15.
N_TRIALS_MLP = 100  
N_TRIALS_ICDN = 100

# MedianPruner startup trials: MLP waits longer before pruning than ICDN.
PRUNER_STARTUP_MLP = 5
PRUNER_STARTUP_ICDN = 5

# Linear bootstrap checkpoints every 10 replicates (and always the first).
LINEAR_BOOT_CHECKPOINT = 10

# Week-block bootstrap for ICDN–MLP predictive inference (no retraining).
# Applied to concatenated outer-validation cells; blocks follow BLOCK_SIZE.
N_BOOT_PRED_INFER = 999

# Pre-specified non-inferiority margin on log-demand MAE (ICDN − MLP).
# Fixed as a design constant before inspecting the paired CI. ICDN is
# non-inferior to MLP if the 95% CI upper bound is strictly below this value.
PRED_NI_MARGIN = 0.05
