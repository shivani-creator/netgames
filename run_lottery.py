#!/usr/bin/env python3
"""C10: does the initial strategy of the hubs decide the outcome on BA graphs?

For each replicate we record the initial strategies of the top 1% of nodes by
degree, the fraction of the largest hub's neighbours that share its initial
strategy after one generation, and the equilibrium.  Run for best-takes-over
and, as a contrast, Fermi dynamics, both with accumulated payoffs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from netgames import Config, EvolutionaryGame, Payoff, build

rows = []
for rule in ("imitate_best", "fermi"):
    for b in (1.1, 1.2, 1.3):
        for rep in range(40):
            seed = 7000 + 100 * rep + int(b * 10)
            net = build("barabasi_albert", n=2500, m=4, seed=seed)
            deg = net.degrees
            top = np.argsort(deg)[::-1][: max(1, int(0.01 * net.n))]
            h0 = top[0]
            cfg = Config(payoff=Payoff.weak_pd(b), rule=rule, rule_params={},
                         generations=3000, burn_in=2000, seed=seed)
            game = EvolutionaryGame(net, cfg)
            res = game.run(record_snapshots=(0, 1))
            s0, s1 = game.snapshots[0], game.snapshots[1]
            nb = net.adj.indices[net.adj.indptr[h0]:net.adj.indptr[h0 + 1]]
            w = deg[top]
            rows.append({
                "rule": rule, "b": b, "replicate": rep,
                "hub_init_coop_wtd": float((s0[top] * w).sum() / w.sum()),
                "top_hub_init": int(s0[h0]),
                "top_hub_nb_follow_t1": float((s1[nb] == s0[h0]).mean()),
                "equilibrium": res.equilibrium,
                "absorbed": res.absorbed or "none",
            })
df = pd.DataFrame(rows)
df.to_csv("results/controls/C10_hub_lottery.csv", index=False)
for (rule, b), d in df.groupby(["rule", "b"]):
    r = np.corrcoef(d["hub_init_coop_wtd"], d["equilibrium"])[0, 1]
    bimodal = ((d["equilibrium"] < 0.05) | (d["equilibrium"] > 0.95)).mean()
    print(f"{rule:13s} b={b}: corr(hub init C, rho)={r:+.2f}  "
          f"follow_t1={d['top_hub_nb_follow_t1'].mean():.2f}  "
          f"extreme outcomes={bimodal:.2f}  mean rho={d['equilibrium'].mean():.2f}")
