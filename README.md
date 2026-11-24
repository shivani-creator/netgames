# netgames

Evolutionary game dynamics on networks: a simulation framework for the
evolutionary Prisoner's Dilemma on structured populations, supporting a
factorial sweep over **network topology × strategy update rule × payoff
structure × payoff normalisation**.

Accompanying paper: Shivani and Dipty Tripathi, *"A Factorial Simulation
Study of Network Topology, Update Rules and Payoff Schemes in the Prisoner's
Dilemma"*, ComSIA 2027 (Springer LNNS). Every number and table in it is
generated from the CSVs in `results/` by `paper_analysis.py`, which writes
`paper/numbers.tex`, `paper/tab_*.tex` and `paper/figs/`. The figures as
printed in the paper are redrawn at the journal's text width by
`paper/figure_source/redraw.py`.

**Random seeds.** Every run is seeded: the seed is a fixed function of the
replicate index and the parameters of the run (see
`netgames/experiments.py` and `run_controls.py`), so all results can be
regenerated exactly. Network-generation parameters are set in the same files.

---

## Install and run

```bash
pip install -r requirements.txt

python -m pytest tests/ -v       # 34 tests, ~2 s
python run_demo.py --full        # Studies 1-7 (publication resolution)
python run_controls.py           # control experiments C1-C9 (~1 h on 2 cores)
python run_lottery.py            # C10: hub broadcasting under best-takes-over
python paper_analysis.py         # figures, tables and numbers.tex for the paper
python paper/figure_source/redraw.py   # figures as printed in the paper
```

Only NumPy, SciPy, NetworkX, pandas and Matplotlib are required. No GPU, no
cluster.

---

## Minimal example

```python
from netgames import build, Config, Payoff, run_once

net = build("lattice", L=100, neighbourhood="moore")
cfg = Config(payoff=Payoff.weak_pd(1.4), rule="imitate_best", generations=500)
print(run_once(net, cfg).equilibrium)

# Averaged payoffs at a temperature matched to <k>: identical to accumulated
# payoffs on any regular graph, so remaining differences are due to degree
# heterogeneity alone.
cfg = Config(payoff=Payoff.weak_pd(1.3), rule="fermi", rule_params={"K": 0.1},
             normalise_payoff=True, match_temperature=True)

# Interpolate between the conventions: payoff / k**alpha.
cfg = Config(payoff=Payoff.weak_pd(1.3), rule="fermi", payoff_exponent=0.5,
             match_temperature=True)
```

---

## What is implemented

**Topologies** (`netgames.networks`): square lattice (Moore / von Neumann,
periodic or open), Watts–Strogatz, Barabási–Albert, Erdős–Rényi,
random-regular, complete graph, **Chung–Lu power-law graphs** (tunable
heterogeneity at fixed mean degree), **Holme–Kim** clustered scale-free
graphs, and an edge-list loader for empirical networks.

**Payoff structures** (`netgames.game`): weak PD `Payoff.weak_pd(b)`, donation
game `Payoff.donation(b, c)`, the general T–S plane `Payoff.ts_plane(T, S)`.

**Payoff accounting**: accumulated, averaged (`normalise_payoff=True`), or the
general `payoff_exponent=alpha` (payoff / k^alpha); `match_temperature=True`
divides the Fermi temperature by <k>^alpha.

**Update rules**: `imitate_best`, `fermi`, `proportional_imitation`,
`death_birth`; synchronous or partial (`update_fraction < 1`).

**Diagnostics** (`netgames.analysis`): cooperation by degree, hub advantage,
cooperator clusters and assortment, critical thresholds, standardised OLS.

A generation is one sparse mat-vec plus CSR segment reductions, so it costs
O(edges) whatever the rule (~10^4 Fermi generations per second at N = 2500 on
one core).

---

## Main results (full resolution, N = 2500, 8 replicates)

| finding | number |
|---|---|
| Fermi, accumulated: critical temptation b* | BA **1.77** · ER 1.15 · WS 1.10 · lattice 1.07 |
| Fermi, averaged | BA and ER never reach 50 % cooperation |
| Fermi, averaged at matched temperature K/<k> | BA **1.03** · ER 1.04 · lattice 1.07 (identical to accumulated) |
| BA b* vs normalisation exponent alpha (matched K) | 1.77 → 1.55 → 1.35 → 1.16 → 1.03 |
| Best-takes-over, accumulated, synchronous | lattice 1.57 · ER 1.47 · WS 1.43 · BA **1.22** (BA last) |
| Best-takes-over, accumulated, 10 % of nodes update per generation | BA **first**: cooperation ≥ 0.92 for every b ≤ 2 |
| Best-takes-over, averaged, asynchronous | BA advantage gone (AUC 0.36 vs ER 0.43, lattice 0.39) |
| Structural regression, accumulated (29 specs) | heterogeneity β = +0.92 (t = 10.1), R² = 0.88 |
| Structural regression, averaged, matched K | heterogeneity β = −0.58 (t = −3.9); clustering β = +0.85 (t = 5.3) |
| Lattice clustering vs permutation null (b = 1.4) | assortment 0.185 vs −0.0003 |

The dependence of the scale-free advantage on accumulated payoffs holds under
every rule and update scheme tested; the "BA last under best-takes-over"
reversal is specific to synchronous updating. Re-running the full-resolution
sweep and survey from scratch reproduces the stored tables to within 2.2e-16.
See the paper for confidence intervals, the remaining robustness checks (noise,
system size) and the T–S plane.

---

## Layout

```
netgames/            package: networks, game, analysis, experiments, viz
tests/               34 tests
run_demo.py          Studies 1-7
run_controls.py      control experiments C1-C9
run_lottery.py       C10
paper_analysis.py    analysis + figures + numbers.tex for the paper
results/tables/      Studies 1-7 (aggregated CSVs, full resolution)
results/figures/     original Study 1-7 figures
results/controls/    C1-C10 raw, replicate-level CSVs
paper/               figs/, generated tables and numbers.tex; figure_source/ (figures as printed)
```

## Limitations and next steps

- All networks are generated; run the survey on empirical networks (the loader
  exists and is tested).
- The factorial sweep uses the weak PD (P = S); repeat it in the strict PD
  quadrant.
- Add memory-one strategies and co-evolving networks.
