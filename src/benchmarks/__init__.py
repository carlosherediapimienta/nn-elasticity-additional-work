"""Demand-elasticity benchmarks: OLS, Ridge, MLP, ICDN.

Notebooks import an experiment class and call `run_all()`. Do not import
`ICDNExperiment` from this package init: loading it monkey-patches ICDN's
PanelBuilder. Import it from `src.benchmarks.icdn` instead.
"""

from src.benchmarks.linear import PairwiseExperiment
from src.benchmarks.mlp import MLPExperiment

__all__ = ["PairwiseExperiment", "MLPExperiment"]
