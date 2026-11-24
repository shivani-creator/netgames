#!/usr/bin/env python3
"""
End-to-end experiment pipeline.

Runs the seven studies the proposal promises and writes every figure and every
results table to ``results/``.  Each study is a separate function so individual
pieces can be re-run without repeating the whole sweep.

    python run_demo.py                # ~3-5 minutes, thesis-draft resolution
    python run_demo.py --full         # ~30-60 minutes, publication resolution
    python run_demo.py --only 1 3 5   # selected studies only

Study 1 is the sanity check: if the lattice snapshots do not show compact
cooperator clusters, nothing downstream is worth reading.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from netgames import (Config, EvolutionaryGame, Payoff, build,
                      cooperation_by_degree, critical_threshold, run_once,
                      structural_regression)
from netgames import experiments as ex
from netgames import viz

OUT = Path("results")
FIG = OUT / "figures"
TAB = OUT / "tables"


class Settings:
    """Resolution knobs.  ``--full`` scales every sweep up for final runs."""

    def __init__(self, full: bool):
        self.full = full
        self.n = 2500 if full else 900
        self.L = 50 if full else 30
        self.generations = 3000 if full else 800
        self.burn_in = 2000 if full else 500
        self.b_points = 21 if full else 11
        self.grid = 25 if full else 13
        self.replicates = 8 if full else 3
        self.survey_reps = 5 if full else 2

    def base(self, **kw) -> Config:
        defaults = dict(payoff=Payoff.weak_pd(1.0),
                        generations=self.generations, burn_in=self.burn_in)
        return Config(**{**defaults, **kw})

    def topologies(self) -> list[tuple[str, dict]]:
        return [
            ("lattice", {"L": self.L, "neighbourhood": "moore"}),
            ("watts_strogatz", {"n": self.n, "k": 8, "p": 0.1}),
            ("barabasi_albert", {"n": self.n, "m": 4}),
            ("erdos_renyi", {"n": self.n, "mean_degree": 8.0}),
        ]


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TAB / f"{name}.csv", index=False)
    print(f"    table -> {TAB / (name + '.csv')}")


# --------------------------------------------------------------------------

def study_1_snapshots(st: Settings) -> None:
    """Spatial reciprocity, made visible."""
    print("[1] lattice snapshots", flush=True)
    net = build("lattice", L=st.L * 2, neighbourhood="moore")
    gens = (0, 5, 25, 100, 400)

    for b, tag in [(1.4, "b1.4_cooperation_survives"),
                   (1.9, "b1.9_cooperation_collapses")]:
        cfg = st.base(payoff=Payoff.weak_pd(b), rule="imitate_best",
                      generations=max(gens), burn_in=max(gens) // 2, seed=1,
                      absorbing_stop=False)
        game = EvolutionaryGame(net, cfg)
        res = game.run(record_snapshots=gens)
        viz.plot_lattice_snapshots(
            net, game.snapshots, str(FIG / f"01_snapshots_{tag}.png"),
            title=f"Weak PD on a Moore lattice, best-takes-over, b = {b}")
        print(f"    b={b}: equilibrium cooperation = {res.equilibrium:.3f}")


def study_2_time_series(st: Settings) -> None:
    """Trajectories: how fast, and to what level, each topology settles."""
    print("[2] time series by topology", flush=True)
    series = {}
    for kind, kwargs in st.topologies():
        net = build(kind, **{**kwargs, "seed": 1})
        cfg = st.base(payoff=Payoff.weak_pd(1.4), rule="fermi",
                      rule_params={"K": 0.1}, seed=1, absorbing_stop=False)
        res = run_once(net, cfg)
        series[f"{kind} (<k>={net.mean_degree:.1f})"] = res.cooperation
    viz.plot_time_series(series, str(FIG / "02_time_series.png"),
                         title="Cooperation over time, weak PD at b = 1.4 (Fermi, K = 0.1)",
                         burn_in=st.burn_in)


def study_3_temptation(st: Settings) -> pd.DataFrame:
    """Cooperation versus b, crossed with topology and update rule."""
    print("[3] temptation sweep (topology x rule x normalisation)", flush=True)
    df = ex.topology_comparison(
        net_specs=st.topologies(),
        b_values=np.linspace(1.0, 2.0, st.b_points),
        rules=("imitate_best", "fermi", "proportional_imitation"),
        normalisations=(False, True),
        replicates=st.replicates,
        base_config=st.base())
    save_table(df, "03_temptation_sweep")

    for norm in (False, True):
        sub = df[df["normalise_payoff"] == norm]
        for rule in sub["rule"].unique():
            d = sub[sub["rule"] == rule]
            tag = "averaged" if norm else "accumulated"
            viz.plot_cooperation_curve(
                d, str(FIG / f"03_coop_vs_b_{rule}_{tag}.png"),
                x="b", hue="net_kind", xlabel="temptation  b",
                title=f"{rule.replace('_', ' ')}, {tag} payoff")

    rows = []
    for (kind, rule, norm), d in df.groupby(["net_kind", "rule",
                                             "normalise_payoff"]):
        rows.append({"net_kind": kind, "rule": rule, "normalise_payoff": norm,
                     "b_critical": critical_threshold(d, "b"),
                     "coop_at_b_1.2": float(
                         d.loc[(d["b"] - 1.2).abs().idxmin(), "equilibrium"])})
    thresholds = pd.DataFrame(rows).sort_values(["rule", "normalise_payoff",
                                                 "net_kind"])
    save_table(thresholds, "03_critical_thresholds")
    return df


def study_4_phase_diagrams(st: Settings) -> None:
    """The T-S plane for each topology."""
    print("[4] phase diagrams over the T-S plane", flush=True)
    frames = {}
    for kind, kwargs in st.topologies():
        t0 = time.time()
        df = ex.phase_diagram(
            net_kind=kind, net_kwargs=kwargs,
            T_values=np.linspace(0.0, 2.0, st.grid),
            S_values=np.linspace(-1.0, 1.0, st.grid),
            rule="fermi", rule_params={"K": 0.1},
            replicates=max(2, st.replicates // 2),
            base_config=st.base(absorbing_stop=False))
        frames[kind] = df
        save_table(df, f"04_phase_{kind}")
        viz.plot_phase_diagram(df, str(FIG / f"04_phase_{kind}.png"),
                               title=f"T-S plane: {kind}")
        print(f"    {kind}: {time.time() - t0:.0f}s", flush=True)

    viz.plot_phase_grid(frames, str(FIG / "04_phase_grid.png"),
                        title="Cooperation over the T-S plane by topology "
                              "(Fermi, K = 0.1)")


def study_5_hubs(st: Settings) -> None:
    """Do hubs carry cooperation, and does that survive payoff normalisation?"""
    print("[5] cooperation by degree", flush=True)
    frames = {}
    for kind, kwargs in [("barabasi_albert", {"n": st.n, "m": 4}),
                         ("erdos_renyi", {"n": st.n, "mean_degree": 8.0})]:
        for norm in (False, True):
            net = build(kind, **{**kwargs, "seed": 1})
            cfg = st.base(payoff=Payoff.weak_pd(1.4), rule="fermi",
                          rule_params={"K": 0.1}, normalise_payoff=norm,
                          seed=1, absorbing_stop=False)
            res = run_once(net, cfg)
            tag = "averaged" if norm else "accumulated"
            frames[f"{kind}, {tag}"] = cooperation_by_degree(net, res.final_state)
    viz.plot_cooperation_by_degree(
        frames, str(FIG / "05_cooperation_by_degree.png"),
        title="Cooperation by degree: the hub effect depends on payoff normalisation")

    combined = pd.concat([d.assign(condition=k) for k, d in frames.items()])
    save_table(combined, "05_cooperation_by_degree")


def study_6_robustness(st: Settings) -> None:
    """Mutation and selection temperature: does the effect survive noise?"""
    print("[6] robustness to noise", flush=True)
    df = ex.robustness_sweep(
        net_kind="barabasi_albert", net_kwargs={"n": st.n, "m": 4}, b=1.4,
        mutation_values=(0.0, 1e-4, 1e-3, 1e-2, 5e-2),
        temperature_values=(0.01, 0.1, 0.3, 1.0),
        replicates=st.replicates,
        base_config=st.base(absorbing_stop=False))
    save_table(df, "06_robustness")
    viz.plot_heatmap(df, str(FIG / "06_robustness.png"),
                     x="temperature", y="mutation_rate",
                     title="Cooperation under noise (scale-free, b = 1.4)",
                     xlabel="selection temperature  K",
                     ylabel="mutation rate  mu")


def study_7_structure(st: Settings) -> None:
    """Which structural property actually predicts cooperation?"""
    print("[7] structural survey and regression", flush=True)

    # Fitted separately under each payoff convention.  If degree heterogeneity
    # is the dominant predictor only when payoffs accumulate, then the
    # "scale-free networks promote cooperation" claim is really a statement
    # about the accounting rule, not about topology -- which is precisely the
    # question the thesis is positioned to settle.
    # The two conventions sit at very different points on the b axis: averaged
    # payoff kills cooperation everywhere by b = 1.4, so regressing there would
    # be fitting structure to a column of zeros.  Scan a few temptation levels
    # and fit at the one with the most outcome variance, reporting which.
    b_candidates = (1.05, 1.1, 1.2, 1.4)

    surveys, regressions = {}, {}
    for norm in (False, True):
        tag = "averaged" if norm else "accumulated"

        scanned = []
        for b in b_candidates:
            df = ex.structural_survey(n=st.n, b=b, rule="fermi",
                                      replicates=st.survey_reps,
                                      normalise_payoff=norm)
            scanned.append(df.assign(b=b, payoff_convention=tag))
            print(f"    {tag}, b={b}: mean coop = {df['equilibrium'].mean():.3f}, "
                  f"spread = {df['equilibrium'].std():.3f}", flush=True)

        all_b = pd.concat(scanned, ignore_index=True)
        surveys[tag] = all_b
        save_table(all_b, f"07_structural_survey_{tag}")

        best = max(scanned, key=lambda d: d["equilibrium"].std())
        b_used = float(best["b"].iloc[0])

        try:
            coefs = structural_regression(best)
        except ValueError as err:
            print(f"    {tag}: regression skipped ({err})", flush=True)
            continue
        coefs = coefs.assign(b_used=b_used)
        regressions[tag] = coefs.assign(payoff_convention=tag)
        save_table(coefs, f"07_structural_regression_{tag}")
        viz.plot_regression(
            coefs, str(FIG / f"07_structural_regression_{tag}.png"),
            title=f"Structural predictors of cooperation\n({tag} payoff, b = {b_used:g})")

        print(f"\n    --- {tag} payoff, fitted at b = {b_used:g} ---", flush=True)
        print(coefs.to_string(index=False,
                              float_format=lambda v: f"{v: .3f}"), flush=True)

    if surveys:
        save_table(pd.concat(surveys.values(), ignore_index=True),
                   "07_structural_survey_combined")
    if regressions:
        save_table(pd.concat(regressions.values(), ignore_index=True),
                   "07_structural_regression_combined")


STUDIES = {
    1: study_1_snapshots,
    2: study_2_time_series,
    3: study_3_temptation,
    4: study_4_phase_diagrams,
    5: study_5_hubs,
    6: study_6_robustness,
    7: study_7_structure,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                    help="publication resolution (much slower)")
    ap.add_argument("--only", type=int, nargs="+", choices=sorted(STUDIES),
                    help="run only the listed studies")
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    st = Settings(full=args.full)
    chosen = args.only or sorted(STUDIES)
    print(f"resolution: {'full' if args.full else 'quick'}  "
          f"(n={st.n}, generations={st.generations}, replicates={st.replicates})\n")

    t0 = time.time()
    for k in chosen:
        STUDIES[k](st)
        print()
    print(f"done in {time.time() - t0:.0f}s -> {OUT.resolve()}")


if __name__ == "__main__":
    main()
