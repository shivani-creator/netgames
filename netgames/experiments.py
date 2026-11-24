"""
Experiment drivers.

Every function here returns a tidy :class:`pandas.DataFrame` -- one row per
completed run, or per averaged cell -- so results can be concatenated, cached to
CSV, and re-plotted without re-simulating.  Long sweeps are worth caching: the
full phase diagram in ``run_demo.py`` is several thousand independent runs.

Replicates matter.  A single run on a finite graph can absorb into all-C or
all-D by chance, so every reported number is a mean over ``replicates`` runs
with different seeds for both the graph and the initial condition.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import networks as nw
from .analysis import full_diagnostics
from .game import Config, EvolutionaryGame, Payoff


def _run_task(task: dict) -> dict:
    """One replicate.  Top-level so it survives pickling to a worker process."""
    net_kind = task["net_kind"]
    net_kwargs = dict(task["net_kwargs"])
    if "seed" in net_kwargs or net_kind != "lattice":
        net_kwargs.setdefault("seed", task["seed"])
    network = nw.build(net_kind, **net_kwargs)

    config = replace(task["config"], seed=task["seed"])
    result = EvolutionaryGame(network, config).run()

    row = result.as_row()
    row.update(full_diagnostics(network, result.final_state))
    row.update(task.get("extra", {}))
    row["net_kind"] = net_kind
    return row


def _execute(tasks: Sequence[dict], workers: int | None = None,
             progress: bool = True) -> pd.DataFrame:
    """Run tasks, in parallel when it is worth the process-spawn overhead."""
    if workers is None:
        workers = 1 if len(tasks) < 8 else min(8, len(tasks))

    rows = []
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            rows.append(_run_task(task))
            if progress and i % max(1, len(tasks) // 10) == 0:
                print(f"  {i}/{len(tasks)} runs", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, row in enumerate(pool.map(_run_task, tasks, chunksize=4), 1):
                rows.append(row)
                if progress and i % max(1, len(tasks) // 10) == 0:
                    print(f"  {i}/{len(tasks)} runs", flush=True)
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Average over replicates, keeping the spread."""
    numeric = df.select_dtypes(include=[np.number]).columns
    value_cols = [c for c in numeric if c not in group_cols and c != "seed"]
    agg = df.groupby(list(group_cols), as_index=False)[value_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c
                   for c in agg.columns]
    agg = agg.reset_index() if agg.index.name else agg
    agg = agg.rename(columns={"equilibrium_mean": "equilibrium",
                              "equilibrium_std_mean": "equilibrium_within_std"})
    counts = df.groupby(list(group_cols), as_index=False).size()
    return agg.merge(counts, on=list(group_cols), how="left")


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------

def sweep_temptation(net_kind: str = "lattice",
                     net_kwargs: dict | None = None,
                     b_values: Iterable[float] = np.linspace(1.0, 2.0, 21),
                     rule: str = "imitate_best",
                     rule_params: dict | None = None,
                     normalise_payoff: bool = False,
                     replicates: int = 5,
                     base_config: Config | None = None,
                     workers: int | None = None) -> pd.DataFrame:
    """Cooperation versus the temptation ``b`` in the weak Prisoner's Dilemma.

    This is the replication baseline: on a Moore lattice under best-takes-over,
    cooperation should survive up to roughly ``b = 1.8``, and collapse to zero
    on a well-mixed population at any ``b > 1``.
    """
    net_kwargs = net_kwargs or {}
    base = base_config or Config(payoff=Payoff.weak_pd(1.0))

    tasks = []
    for b, rep in itertools.product(b_values, range(replicates)):
        cfg = replace(base, payoff=Payoff.weak_pd(float(b)), rule=rule,
                      rule_params=rule_params or {},
                      normalise_payoff=normalise_payoff)
        tasks.append({"net_kind": net_kind, "net_kwargs": net_kwargs,
                      "config": cfg, "seed": 1000 * rep + int(b * 997),
                      "extra": {"b": float(b), "replicate": rep}})

    df = _execute(tasks, workers=workers)
    return _aggregate(df, ["b", "net_kind", "rule", "normalise_payoff"])


def phase_diagram(net_kind: str = "lattice",
                  net_kwargs: dict | None = None,
                  T_values: Iterable[float] = np.linspace(1.0, 2.0, 21),
                  S_values: Iterable[float] = np.linspace(-1.0, 1.0, 21),
                  rule: str = "fermi",
                  rule_params: dict | None = None,
                  normalise_payoff: bool = False,
                  replicates: int = 3,
                  base_config: Config | None = None,
                  workers: int | None = None) -> pd.DataFrame:
    """Cooperation over the T-S plane.

    With R = 1 and P = 0 the point (T, S) fixes the game, and the plane splits
    into four quadrants: harmony (T<1, S>0), snowdrift (T>1, S>0), stag hunt
    (T<1, S<0) and the Prisoner's Dilemma (T>1, S<0).  Sweeping the whole plane
    rather than only the PD quadrant shows how far the effect of topology
    extends into neighbouring social dilemmas.
    """
    net_kwargs = net_kwargs or {}
    base = base_config or Config(payoff=Payoff.ts_plane(1.0, 0.0))

    tasks = []
    for T, S, rep in itertools.product(T_values, S_values, range(replicates)):
        cfg = replace(base, payoff=Payoff.ts_plane(float(T), float(S)),
                      rule=rule, rule_params=rule_params or {},
                      normalise_payoff=normalise_payoff)
        tasks.append({"net_kind": net_kind, "net_kwargs": net_kwargs,
                      "config": cfg,
                      "seed": 7919 * rep + int(T * 131) + int((S + 2) * 617),
                      "extra": {"T_val": float(T), "S_val": float(S),
                                "replicate": rep}})

    df = _execute(tasks, workers=workers)
    return _aggregate(df, ["T_val", "S_val", "net_kind", "rule",
                           "normalise_payoff"])


def topology_comparison(net_specs: Sequence[tuple[str, dict]],
                        b_values: Iterable[float] = np.linspace(1.0, 2.0, 11),
                        rules: Sequence[str] = ("imitate_best", "fermi",
                                                "proportional_imitation"),
                        normalisations: Sequence[bool] = (False, True),
                        replicates: int = 5,
                        base_config: Config | None = None,
                        workers: int | None = None) -> pd.DataFrame:
    """The main factorial sweep: topology x update rule x payoff x normalisation.

    This is the experiment the proposal is built around.  Crossing all four
    factors is what distinguishes the study from the many papers that vary
    topology alone while holding the update rule fixed -- a design that cannot
    separate an effect of structure from an interaction with the dynamics.
    """
    base = base_config or Config(payoff=Payoff.weak_pd(1.0))

    tasks = []
    for (kind, kwargs), b, rule, norm, rep in itertools.product(
            net_specs, b_values, rules, normalisations, range(replicates)):
        cfg = replace(base, payoff=Payoff.weak_pd(float(b)), rule=rule,
                      normalise_payoff=norm)
        tasks.append({"net_kind": kind, "net_kwargs": kwargs, "config": cfg,
                      "seed": 104729 * rep + int(b * 733),
                      "extra": {"b": float(b), "replicate": rep,
                                "net_label": kwargs.get("label", kind)}})

    df = _execute(tasks, workers=workers)
    return _aggregate(df, ["net_kind", "b", "rule", "normalise_payoff"])


def robustness_sweep(net_kind: str = "barabasi_albert",
                     net_kwargs: dict | None = None,
                     b: float = 1.5,
                     mutation_values: Iterable[float] = (0.0, 1e-4, 1e-3, 1e-2),
                     temperature_values: Iterable[float] = (0.01, 0.1, 0.5, 1.0),
                     replicates: int = 5,
                     base_config: Config | None = None,
                     workers: int | None = None) -> pd.DataFrame:
    """Objective 3's robustness check: does the result survive noise?

    Mutation injects strategies at random; the Fermi temperature controls how
    often a node copies a *worse* neighbour.  A structural effect that
    evaporates at any non-zero noise is an artefact of deterministic dynamics
    rather than a property of the network.
    """
    net_kwargs = net_kwargs or {}
    base = base_config or Config(payoff=Payoff.weak_pd(b))

    tasks = []
    for mu, K, rep in itertools.product(mutation_values, temperature_values,
                                        range(replicates)):
        cfg = replace(base, payoff=Payoff.weak_pd(float(b)), rule="fermi",
                      rule_params={"K": float(K)}, mutation=float(mu),
                      absorbing_stop=False)
        tasks.append({"net_kind": net_kind, "net_kwargs": net_kwargs,
                      "config": cfg, "seed": 31337 * rep + int(K * 1000) + int(mu * 1e6),
                      "extra": {"mutation_rate": float(mu), "temperature": float(K),
                                "replicate": rep}})

    df = _execute(tasks, workers=workers)
    return _aggregate(df, ["mutation_rate", "temperature"])


def structural_survey(n: int = 1500, b: float = 1.4, rule: str = "fermi",
                      replicates: int = 3, normalise_payoff: bool = False,
                      workers: int | None = None) -> pd.DataFrame:
    """Many networks spanning the structural space, each with its descriptors.

    Feeds :func:`analysis.structural_regression`: to ask which property of a
    graph predicts cooperation, the design needs graphs whose properties vary
    somewhat independently, not four canonical families.  Sweeping the
    Watts-Strogatz rewiring probability and the Barabasi-Albert attachment
    parameter traces out a continuum between the extremes.
    """
    specs: list[tuple[str, dict]] = []
    for p in (0.0, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0):
        specs.append(("watts_strogatz", {"n": n, "k": 8, "p": p}))
    for m in (2, 3, 4, 6, 8):
        specs.append(("barabasi_albert", {"n": n, "m": m}))
    for k in (4, 6, 8, 12):
        specs.append(("regular_random", {"n": n, "k": k}))
    for kbar in (4.0, 8.0, 12.0):
        specs.append(("erdos_renyi", {"n": n, "mean_degree": kbar}))
    specs.append(("lattice", {"L": int(np.sqrt(n)), "neighbourhood": "moore"}))
    specs.append(("lattice", {"L": int(np.sqrt(n)), "neighbourhood": "von_neumann"}))

    base = Config(payoff=Payoff.weak_pd(b), rule=rule, rule_params={"K": 0.1},
                  normalise_payoff=normalise_payoff,
                  generations=1500, burn_in=1000, absorbing_stop=False)

    tasks = [{"net_kind": kind, "net_kwargs": kwargs,
              "config": replace(base, seed=None), "seed": 555 * rep + i,
              "extra": {"spec_id": i, "replicate": rep}}
             for i, (kind, kwargs) in enumerate(specs)
             for rep in range(replicates)]

    df = _execute(tasks, workers=workers)

    # Attach structural descriptors, computed once per network specification.
    desc = []
    for i, (kind, kwargs) in enumerate(specs):
        net = nw.build(kind, **{**kwargs, "seed": 555})
        row = net.summary()
        row["spec_id"] = i
        desc.append(row)
    desc_df = pd.DataFrame(desc).rename(columns={"name": "network_label"})

    agg = _aggregate(df, ["spec_id", "net_kind", "normalise_payoff"])
    return agg.merge(desc_df, on="spec_id", how="left")
