"""
netgames -- evolutionary game dynamics on networks.

Simulation framework for the evolutionary Prisoner's Dilemma on structured
populations, supporting a factorial sweep over network topology, strategy
update rule, and payoff structure.

Quick start
-----------
>>> from netgames import build, Config, Payoff, run_once
>>> net = build("lattice", L=100, neighbourhood="moore")
>>> cfg = Config(payoff=Payoff.weak_pd(1.4), rule="imitate_best", generations=500)
>>> run_once(net, cfg).equilibrium          # doctest: +SKIP
0.31...
"""

from .analysis import (cluster_statistics, cooperation_by_degree,
                       critical_threshold, full_diagnostics, hub_cooperation,
                       structural_regression)
from .game import (Config, EvolutionaryGame, Payoff, Result, RULES,
                   compute_payoffs, run_once)
from .networks import GENERATORS, Network, build

__version__ = "0.1.0"

__all__ = [
    "Network", "build", "GENERATORS",
    "Payoff", "Config", "Result", "EvolutionaryGame", "run_once",
    "RULES", "compute_payoffs",
    "cooperation_by_degree", "hub_cooperation", "cluster_statistics",
    "full_diagnostics", "critical_threshold", "structural_regression",
]
