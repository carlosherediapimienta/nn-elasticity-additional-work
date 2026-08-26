"""Protocol knobs shared by OLS, Ridge, MLP, and ICDN.

These numbers are part of the research design, not implementation details.
Changing them changes the experiment. Model-specific overrides (trial
budget, bootstrap replicates, Optuna pruner warmup) stay next to that
model so a reader can see the comparison is not fully symmetric.
"""

from __future__ import annotations

# Calendar column used by TemporalSplitter and BlockBootstrapSampler.
PERIOD_COL = "week_id"

# Outer expanding window: five folds, first train slice at least half the panel.
N_FOLDS = 2
MIN_TRAIN_FRAC = 0.5

# Inner expanding window used only for Optuna (MLP / ICDN). Outer val is never
# seen during search.
N_INNER_FOLDS = 3
MIN_INNER_FRAC = 0.5

# Last 20% of an outer (or inner) train slice is held out for MLP early stopping.
INNER_TRAIN_FRAC = 0.8

# Holdout used for the point estimate and as the bootstrap validation window.
HOLDOUT_TRAIN_FRAC = 0.8

# Moving-block bootstrap on raw weeks. Validation weeks are never resampled.
BLOCK_SIZE = 4
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
N_BOOT_LINEAR = 2
N_BOOT_MLP = 2
N_BOOT_ICDN = 2

# Optuna budgets. MLP is currently a smoke search (1 trial); ICDN is 15.
N_TRIALS_MLP = 5
N_TRIALS_ICDN = 5

# MedianPruner startup trials: MLP waits longer before pruning than ICDN.
PRUNER_STARTUP_MLP = 5
PRUNER_STARTUP_ICDN = 4

# Linear bootstrap checkpoints every 10 replicates (and always the first).
LINEAR_BOOT_CHECKPOINT = 10
