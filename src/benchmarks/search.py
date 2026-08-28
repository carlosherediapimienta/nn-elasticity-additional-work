"""Optuna helpers shared by MLP and ICDN nested searches.

Search never sees the outer validation window. Inner expanding folds of the
outer train are the only MAE signal. Best params are written next to the
SQLite study so a fold can be resumed with `load_if_exists=True`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna

from src.benchmarks.constants import HIDDEN_CHOICES, SEED


def dump_best(study, out_dir: Path, hidden_choices: dict = HIDDEN_CHOICES):
    """Persist best params (hidden as a list in JSON, tuple in Python) and the trial table."""
    best = dict(study.best_params)
    best["hidden"] = list(hidden_choices[best["hidden"]])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best_params.json").write_text(json.dumps(best, indent=2))
    study.trials_dataframe().to_csv(out_dir / "optuna_trials.csv", index=False)
    print("  best MAE", study.best_value)
    print("  best params", best)
    #print("  wrote", out_dir / "best_params.json")
    best["hidden"] = tuple(best["hidden"])
    return best, study


def make_study(study_name: str, out_dir: Path, seed: int = SEED, n_startup_trials: int = 5):
    """TPE + median pruner. Storage is `optuna.db` under `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{out_dir / 'optuna.db'}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=0),
    )


def report_and_maybe_prune(trial, maes: list[float], step: int) -> None:
    """Mean MAE so far is the prune statistic; per-fold MAE is stored as a user attr."""
    trial.set_user_attr(f"fold{step}_mae", maes[-1])
    trial.report(float(np.mean(maes)), step)
    if trial.should_prune():
        raise optuna.TrialPruned()
