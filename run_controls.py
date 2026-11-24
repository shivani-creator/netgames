#!/usr/bin/env python3
"""
Control experiments added for the paper (Studies 8-16).

Each study writes raw, replicate-level rows to ``results/controls/`` so that
confidence intervals can be bootstrapped and every number in the manuscript can
be traced to a CSV.  Seeds follow the conventions of ``experiments.py`` so the
replicate-level re-run of Study 3 (C7) and the original-specification part of
the extended survey (C3) must reproduce the stored aggregate tables exactly.

    python run_controls.py              # everything (~1 h on 2 cores)
    python run_controls.py --only C1 C3 # selected studies
"""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from netgames import Config, EvolutionaryGame, Payoff, build
from netgames import experiments as ex
from netgames.analysis import cluster_statistics

OUT = Path("results/controls")
WORKERS = 2

N, L = 2500, 50
GENS, BURN = 3000, 2000
B_GRID = np.round(np.linspace(1.0, 2.0, 21), 4)
REPS = 8

TOPOLOGIES = [
    ("lattice", {"L": L, "neighbourhood": "moore"}),
    ("watts_strogatz", {"n": N, "k": 8, "p": 0.1}),
    ("barabasi_albert", {"n": N, "m": 4}),
    ("erdos_renyi", {"n": N, "mean_degree": 8.0}),
]


def base(**kw) -> Config:
    return Config(**{**dict(payoff=Payoff.weak_pd(1.0), generations=GENS,
                            burn_in=BURN), **kw})


def sweep_tasks(specs, cfg: Config, b_values=B_GRID, reps=REPS, extra=None):
    """Tasks seeded exactly as ``experiments.topology_comparison``."""
    tasks = []
    for (kind, kw), b, rep in itertools.product(specs, b_values, range(reps)):
        c = replace(cfg, payoff=Payoff.weak_pd(float(b)))
        tasks.append({"net_kind": kind, "net_kwargs": kw, "config": c,
                      "seed": 104729 * rep + int(b * 733),
                      "extra": {"b": float(b), "replicate": rep, **(extra or {}),
                                **{f"net_{k}": v for k, v in kw.items()}}})
    return tasks


def save(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"    -> {OUT / (name + '.csv')}  ({len(df)} rows)", flush=True)


# --------------------------------------------------------------------------

def c1_matched_temperature():
    """Averaged payoff with K/<k>: identical to accumulated on regular graphs."""
    cfg = base(rule="fermi", rule_params={"K": 0.1}, normalise_payoff=True,
               match_temperature=True)
    save(ex._execute(sweep_tasks(TOPOLOGIES, cfg), workers=WORKERS),
         "C1_matched_temperature_raw")


def c2_alpha_sweep():
    """Payoff / k**alpha at matched temperature, alpha strictly between 0 and 1."""
    rows = []
    for alpha in (0.25, 0.5, 0.75):
        cfg = base(rule="fermi", rule_params={"K": 0.1}, payoff_exponent=alpha,
                   match_temperature=True)
        specs = [t for t in TOPOLOGIES if t[0] in ("barabasi_albert", "erdos_renyi")]
        rows.append(ex._execute(sweep_tasks(specs, cfg, extra={"alpha": alpha}),
                                workers=WORKERS))
    save(pd.concat(rows, ignore_index=True), "C2_alpha_sweep_raw")


def c3_extended_survey():
    """Structural survey with families that decorrelate the descriptors."""
    specs: list[tuple[str, dict]] = []
    for p in (0.0, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0):
        specs.append(("watts_strogatz", {"n": N, "k": 8, "p": p}))
    for m in (2, 3, 4, 6, 8):
        specs.append(("barabasi_albert", {"n": N, "m": m}))
    for k in (4, 6, 8, 12):
        specs.append(("regular_random", {"n": N, "k": k}))
    for kbar in (4.0, 8.0, 12.0):
        specs.append(("erdos_renyi", {"n": N, "mean_degree": kbar}))
    specs.append(("lattice", {"L": L, "neighbourhood": "moore"}))
    specs.append(("lattice", {"L": L, "neighbourhood": "von_neumann"}))
    # --- new: heterogeneity continuum at <k> ~ 8, and clustered scale-free ---
    for gamma in (2.25, 2.5, 3.0, 3.5, 4.5):
        specs.append(("chung_lu", {"n": N, "gamma": gamma, "mean_degree": 8.0}))
    for p in (0.3, 0.6, 0.9):
        specs.append(("holme_kim", {"n": N, "m": 4, "p": p}))

    conditions = {
        "accumulated": dict(normalise_payoff=False, match_temperature=False),
        "averaged": dict(normalise_payoff=True, match_temperature=False),
        "averaged_matched": dict(normalise_payoff=True, match_temperature=True),
    }
    b_values = (1.05, 1.1, 1.2, 1.3, 1.4, 1.6)
    reps = 5

    tasks = []
    for cond, kw in conditions.items():
        for b in b_values:
            cfg = Config(payoff=Payoff.weak_pd(b), rule="fermi",
                         rule_params={"K": 0.1}, generations=1500, burn_in=1000,
                         absorbing_stop=False, **kw)
            for i, (kind, nkw) in enumerate(specs):
                for rep in range(reps):
                    tasks.append({"net_kind": kind, "net_kwargs": nkw,
                                  "config": cfg, "seed": 555 * rep + i,
                                  "extra": {"spec_id": i, "replicate": rep,
                                            "b": b, "condition": cond}})
    raw = ex._execute(tasks, workers=WORKERS)
    save(raw, "C3_extended_survey_raw")

    desc = []
    for i, (kind, nkw) in enumerate(specs):
        net = build(kind, **{**nkw, "seed": 555})
        row = net.summary()
        row["spec_id"] = i
        row["family"] = kind
        desc.append(row)
    save(pd.DataFrame(desc).rename(columns={"name": "network_label"}),
         "C3_extended_survey_descriptors")


def c4_assortment():
    """Excess assortment against a permutation null on the same graph."""
    net = build("lattice", L=L, neighbourhood="moore")
    rng = np.random.default_rng(2026)
    rows = []
    for b in (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7):
        for rep in range(REPS):
            cfg = base(payoff=Payoff.weak_pd(b), rule="imitate_best",
                       seed=104729 * rep + int(b * 733))
            res = EvolutionaryGame(net, cfg).run()
            st = cluster_statistics(net, res.final_state)
            null = [cluster_statistics(net, rng.permutation(res.final_state))
                    for _ in range(50)]
            null_a = np.array([d["assortment"] for d in null])
            null_c = np.array([d["n_clusters"] for d in null])
            rows.append({"b": b, "replicate": rep,
                         "cooperation": float(res.final_state.mean()),
                         "equilibrium": res.equilibrium,
                         "assortment": st["assortment"],
                         "n_clusters": st["n_clusters"],
                         "largest_cluster_frac": st["largest_cluster_frac"],
                         "null_assortment_mean": float(null_a.mean()),
                         "null_assortment_sd": float(null_a.std()),
                         "null_clusters_mean": float(null_c.mean())})
    save(pd.DataFrame(rows), "C4_assortment")


def c5_hubs():
    """Cooperation by degree (replicate-averaged) and hub-vs-leaf dynamics."""
    rows, traj = [], []
    snaps = tuple(range(0, 201, 2)) + (300, 500, 1000, 2000, 3000)
    for b in (1.5, 1.7, 1.9):
        for rep in range(REPS):
            seed = 104729 * rep + int(b * 733)
            net = build("barabasi_albert", n=N, m=4, seed=seed)
            deg = net.degrees
            cfg = base(payoff=Payoff.weak_pd(b), rule="fermi",
                       rule_params={"K": 0.1}, seed=seed, absorbing_stop=False)
            game = EvolutionaryGame(net, cfg)
            res = game.run(record_snapshots=snaps)
            for k, c in zip(deg, res.final_state):
                rows.append({"b": b, "replicate": rep, "degree": k,
                             "cooperate": int(c)})
            hub = deg >= np.quantile(deg, 0.95)
            leaf = deg <= 4
            for g, s in game.snapshots.items():
                traj.append({"b": b, "replicate": rep, "generation": g,
                             "coop_all": float(s.mean()),
                             "coop_hubs": float(s[hub].mean()),
                             "coop_leaves": float(s[leaf].mean())})
    save(pd.DataFrame(rows), "C5_degree_states")
    save(pd.DataFrame(traj), "C5_hub_trajectories")


def c6_robustness_er():
    """Noise grid of Study 6 on the Erdos-Renyi reference graph."""
    df = ex.robustness_sweep(
        net_kind="erdos_renyi", net_kwargs={"n": N, "mean_degree": 8.0}, b=1.4,
        mutation_values=(0.0, 1e-4, 1e-3, 1e-2, 5e-2),
        temperature_values=(0.01, 0.1, 0.3, 1.0), replicates=REPS,
        base_config=base(absorbing_stop=False), workers=WORKERS)
    save(df, "C6_robustness_erdos_renyi")


def c7_study3_replicates():
    """Study 3 re-run keeping every replicate (bootstrap CIs, reproducibility)."""
    rows = []
    for rule, norm in itertools.product(
            ("imitate_best", "fermi", "proportional_imitation"), (False, True)):
        cfg = base(rule=rule, normalise_payoff=norm)
        rows.append(ex._execute(sweep_tasks(TOPOLOGIES, cfg), workers=WORKERS))
        print(f"    {rule}, norm={norm} done", flush=True)
    save(pd.concat(rows, ignore_index=True), "C7_study3_raw")


def c8_asynchronous():
    """Random-sequential-like updating: 10% of nodes revise per generation."""
    b_values = np.round(np.linspace(1.0, 2.0, 11), 4)
    rows = []
    for rule, norm in (("imitate_best", False), ("imitate_best", True),
                       ("fermi", False)):
        cfg = base(rule=rule, normalise_payoff=norm, update_fraction=0.1,
                   generations=15000, burn_in=10000)
        rows.append(ex._execute(sweep_tasks(TOPOLOGIES, cfg, b_values=b_values,
                                            reps=4), workers=WORKERS))
        print(f"    {rule}, norm={norm} done", flush=True)
    save(pd.concat(rows, ignore_index=True), "C8_asynchronous_raw")


def c9_finite_size():
    """Thresholds versus system size."""
    b_values = np.round(np.linspace(1.0, 2.0, 21), 4)
    rows = []
    for size_L in (20, 30, 50, 80):
        n = size_L * size_L
        specs = [("lattice", {"L": size_L, "neighbourhood": "moore"}),
                 ("barabasi_albert", {"n": n, "m": 4}),
                 ("erdos_renyi", {"n": n, "mean_degree": 8.0})]
        for rule in ("fermi", "imitate_best"):
            cfg = base(rule=rule)
            rows.append(ex._execute(
                sweep_tasks(specs, cfg, b_values=b_values, reps=4,
                            extra={"N": n}), workers=WORKERS))
        print(f"    N={n} done", flush=True)
    save(pd.concat(rows, ignore_index=True), "C9_finite_size_raw")


STUDIES = {"C1": c1_matched_temperature, "C2": c2_alpha_sweep,
           "C4": c4_assortment, "C5": c5_hubs, "C6": c6_robustness_er,
           "C3": c3_extended_survey, "C7": c7_study3_replicates,
           "C9": c9_finite_size, "C8": c8_asynchronous}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=sorted(STUDIES))
    args = ap.parse_args()
    t_all = time.time()
    for key in (args.only or list(STUDIES)):
        t0 = time.time()
        print(f"[{key}] {STUDIES[key].__doc__.strip().splitlines()[0]}", flush=True)
        STUDIES[key]()
        print(f"[{key}] {time.time() - t0:.0f}s\n", flush=True)
    print(f"all done in {time.time() - t_all:.0f}s", flush=True)
