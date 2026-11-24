"""
The evolutionary game engine.

Strategies are encoded as a uint8 vector ``s`` with 1 = cooperate, 0 = defect.

Each generation has two phases:

1. **Interaction.**  Every node plays the 2x2 game against all of its
   neighbours and accumulates payoff.  Because payoff depends only on how many
   neighbours cooperate, the whole phase is one sparse mat-vec:

       n_C = A @ s,   n_D = k - n_C
       pi_i = s_i (R n_C + S n_D) + (1 - s_i) (T n_C + P n_D)

2. **Update.**  Every node revises its strategy by imitating a neighbour under
   one of the rules below.  All rules are vectorised over nodes using CSR
   segment reductions, so a generation costs O(edges) regardless of rule.

The payoff normalisation flag matters more than it looks.  With *accumulated*
payoff a hub with degree 100 collects up to 100x the payoff of a leaf, so hubs
dominate imitation purely by virtue of connectivity.  With *averaged* payoff
(divide by degree) that advantage disappears.  Santos & Pacheco's result that
scale-free topology promotes cooperation is largely an accumulated-payoff
effect, so this flag is the single most important control in the sweep and is
exposed as a first-class experimental factor rather than buried as a default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix

from .networks import Network

COOPERATE = np.uint8(1)
DEFECT = np.uint8(0)


# --------------------------------------------------------------------------
# Payoff structure
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Payoff:
    """A symmetric 2x2 game.

    ``T`` temptation (D vs C), ``R`` reward (C vs C),
    ``P`` punishment (D vs D), ``S`` sucker (C vs D).

    The Prisoner's Dilemma is ``T > R > P > S``; relaxing that ordering gives
    the neighbouring games (snowdrift, stag hunt, harmony) that make up the
    rest of the T-S plane.
    """

    T: float
    R: float
    P: float
    S: float

    @classmethod
    def weak_pd(cls, b: float) -> "Payoff":
        """Nowak-May parameterisation: the cost of cooperating is set to zero.

        The single parameter ``b > 1`` is the advantage a defector gains from a
        cooperating neighbour.  Degenerate (P = S) but it is *the* benchmark for
        spatial reciprocity, so replication requires it.
        """
        return cls(T=float(b), R=1.0, P=0.0, S=0.0)

    @classmethod
    def donation(cls, b: float, c: float) -> "Payoff":
        """Donation game: pay ``c`` to give a neighbour ``b``.  A strict PD when b > c."""
        return cls(T=b, R=b - c, P=0.0, S=-c)

    @classmethod
    def ts_plane(cls, T: float, S: float) -> "Payoff":
        """R and P pinned to 1 and 0 so that (T, S) alone indexes the game."""
        return cls(T=float(T), R=1.0, P=0.0, S=float(S))

    @property
    def span(self) -> float:
        """Payoff range; normalises the proportional-imitation acceptance rate."""
        return max(self.T, self.R, self.P, self.S) - min(self.T, self.R, self.P, self.S)

    def as_matrix(self) -> np.ndarray:
        """Row = focal strategy, column = opponent strategy, index 1 = cooperate."""
        return np.array([[self.P, self.T], [self.S, self.R]], dtype=np.float64)

    def classify(self, tol: float = 1e-12) -> str:
        """Which quadrant of the T-S plane this game sits in.

        The boundaries are reported explicitly rather than folded into an
        adjacent quadrant.  The weak PD (``S = P``) is exactly such a boundary
        case: it is the standard benchmark, but it is not a strict Prisoner's
        Dilemma, and silently labelling it as one would misreport every
        replication run in the results table.
        """
        t_hi = self.T > self.R + tol
        t_eq = abs(self.T - self.R) <= tol
        s_lo = self.S < self.P - tol
        s_eq = abs(self.S - self.P) <= tol

        if t_hi and s_eq:
            return "weak_pd"           # boundary: PD / snowdrift
        if t_eq or (s_eq and not t_hi):
            return "boundary"
        if t_hi and s_lo:
            return "prisoners_dilemma"
        if t_hi:
            return "snowdrift"
        if s_lo:
            return "stag_hunt"
        return "harmony"


def compute_payoffs(adj: csr_matrix, s: np.ndarray, payoff: Payoff,
                    degrees: np.ndarray, normalise: bool = False,
                    exponent: float | None = None) -> np.ndarray:
    """Accumulated (or degree-averaged) payoff of every node in one mat-vec.

    ``exponent`` generalises the two conventions: payoff is divided by
    ``k ** exponent``, so 0 is accumulated and 1 is averaged.  When given it
    overrides ``normalise``.
    """
    sf = s.astype(np.float64)
    n_coop = adj @ sf
    n_def = degrees - n_coop
    pi = np.where(
        s == COOPERATE,
        payoff.R * n_coop + payoff.S * n_def,
        payoff.T * n_coop + payoff.P * n_def,
    )
    if exponent is not None:
        if exponent != 0.0:
            pi = pi / degrees ** exponent
    elif normalise:
        pi = pi / degrees
    return pi


# --------------------------------------------------------------------------
# CSR segment helpers
# --------------------------------------------------------------------------

def _segment_argmax(values: np.ndarray, indptr: np.ndarray, counts: np.ndarray,
                    seg_of: np.ndarray, rng: np.random.Generator | None
                    ) -> np.ndarray:
    """Index (into ``values``) of the maximum within each CSR row.

    Ties are broken uniformly at random when ``rng`` is supplied.  This is not
    cosmetic: in a population of all defectors every payoff is identical, and
    always taking the lowest-indexed maximum would impose a systematic spatial
    drift on the dynamics.

    All segments must be non-empty, which :func:`networks._finalise` guarantees
    by dropping isolated nodes.
    """
    starts = indptr[:-1]
    seg_max = np.maximum.reduceat(values, starts)
    is_max = values >= np.repeat(seg_max, counts)

    if rng is not None:
        key = rng.random(values.shape[0])
        key[~is_max] = -1.0
        key_max = np.maximum.reduceat(key, starts)
        is_max = key >= np.repeat(key_max, counts)

    hit = np.flatnonzero(is_max)
    _, first = np.unique(seg_of[hit], return_index=True)
    return hit[first]


def _random_neighbour(indptr: np.ndarray, indices: np.ndarray,
                      degrees: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One uniformly random neighbour per node."""
    offset = (rng.random(degrees.size) * degrees).astype(np.int64)
    return indices[indptr[:-1] + offset]


# --------------------------------------------------------------------------
# Update rules
# --------------------------------------------------------------------------
# Signature: (s, pi, ctx, active, rng) -> new strategies for the active nodes.
# ``ctx`` carries the cached CSR arrays; ``active`` indexes the nodes revising
# their strategy this generation.

def imitate_best(s, pi, ctx, active, rng):
    """Deterministic best-takes-over (Nowak & May 1992).

    Adopt the strategy of the highest-payoff individual in the closed
    neighbourhood; keep the current strategy on a tie with self.
    """
    best_pos = ctx["best_pos"]
    if best_pos is None or ctx["best_stale"]:
        best_pos = _segment_argmax(pi[ctx["indices"]], ctx["indptr"],
                                   ctx["counts"], ctx["seg_of"], rng)
        ctx["best_pos"] = best_pos
    best_nb = ctx["indices"][best_pos]
    take = pi[best_nb][active] > pi[active]
    return np.where(take, s[best_nb][active], s[active])


def fermi(s, pi, ctx, active, rng, K: float = 0.1):
    """Pairwise comparison with a Fermi acceptance function (Szabo & Toke 1998).

    Copy a random neighbour ``j`` with probability ``1 / (1 + exp((pi_i - pi_j) / K))``.
    ``K`` is the selection temperature: ``K -> 0`` recovers deterministic
    imitation of the better player, ``K -> inf`` gives a coin flip.  Non-zero
    ``K`` lets the population escape frozen configurations, which is why this
    rule -- not best-takes-over -- gives smooth phase transitions.
    """
    j = ctx["rand_nb"][active]
    diff = (pi[j] - pi[active]) / K
    prob = 1.0 / (1.0 + np.exp(-np.clip(diff, -60.0, 60.0)))
    switch = rng.random(active.size) < prob
    return np.where(switch, s[j], s[active])


def proportional_imitation(s, pi, ctx, active, rng):
    """Replicator-like rule (Santos & Pacheco 2005).

    Copy a strictly better random neighbour with probability proportional to the
    payoff gap, normalised by the larger of the two degrees so the probability
    stays in [0, 1] on heterogeneous graphs.
    """
    j = ctx["rand_nb"][active]
    k_max = np.maximum(ctx["degrees"][active], ctx["degrees"][j])
    denom = k_max * ctx["span"]
    gap = pi[j] - pi[active]
    prob = np.where(denom > 0, np.clip(gap / np.maximum(denom, 1e-12), 0.0, 1.0), 0.0)
    switch = rng.random(active.size) < prob
    return np.where(switch, s[j], s[active])


def death_birth(s, pi, ctx, active, rng, w: float = 0.1):
    """Moran death-birth updating.

    The focal node is replaced by a neighbour drawn with probability
    proportional to fitness ``f = 1 - w + w * pi`` (``w`` = selection strength).
    Unlike the imitation rules this one is a genuine birth-death process, so it
    is the right setting for the fixation-probability results in the literature.
    """
    indptr, indices = ctx["indptr"], ctx["indices"]
    fitness = np.maximum(1.0 - w + w * pi, 1e-12)

    # Roulette-wheel selection inside each neighbour list, done with a
    # per-row cumulative sum so the whole draw stays vectorised.
    vals = fitness[indices]
    cumulative = np.cumsum(vals)
    row_offset = np.concatenate([[0.0], cumulative[indptr[1:] - 1]])[:-1]
    local_cum = cumulative - np.repeat(row_offset, ctx["counts"])
    totals = local_cum[indptr[1:] - 1]

    targets = rng.random(active.size) * totals[active]
    picked = np.empty(active.size, dtype=np.int64)
    for out, (i, t) in enumerate(zip(active, targets)):
        lo, hi = indptr[i], indptr[i + 1]
        picked[out] = indices[lo + np.searchsorted(local_cum[lo:hi], t)]
    return s[picked]


RULES: dict[str, Callable] = {
    "imitate_best": imitate_best,
    "fermi": fermi,
    "proportional_imitation": proportional_imitation,
    "death_birth": death_birth,
}


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

@dataclass
class Config:
    """Everything that defines one run.  Kept flat so it can be logged as a row."""

    payoff: Payoff
    rule: str = "fermi"
    rule_params: dict = field(default_factory=dict)
    normalise_payoff: bool = False
    init_cooperators: float = 0.5
    mutation: float = 0.0
    update_fraction: float = 1.0   # 1.0 = synchronous; small = quasi-asynchronous
    generations: int = 2000
    burn_in: int = 1000            # generations discarded before measurement
    absorbing_stop: bool = True    # halt once cooperation hits 0 or 1
    seed: int | None = None
    # Generalised normalisation: payoff / k**payoff_exponent (overrides
    # normalise_payoff when not None).
    payoff_exponent: float | None = None
    # Divide the Fermi temperature by <k>**alpha, alpha being the effective
    # normalisation exponent.  On a regular graph this makes every convention
    # dynamically identical to accumulated payoff at the nominal K, so any
    # remaining difference is due to degree heterogeneity alone.
    match_temperature: bool = False

    @property
    def alpha(self) -> float:
        """Effective normalisation exponent (0 accumulated, 1 averaged)."""
        if self.payoff_exponent is not None:
            return float(self.payoff_exponent)
        return 1.0 if self.normalise_payoff else 0.0

    def label(self) -> str:
        return (f"{self.rule}|norm={self.normalise_payoff}|"
                f"T={self.payoff.T:g},S={self.payoff.S:g}")


@dataclass
class Result:
    """Outcome of one run."""

    cooperation: np.ndarray        # fraction of cooperators, per generation
    final_state: np.ndarray        # strategy vector at the end
    equilibrium: float             # mean cooperation after burn-in
    equilibrium_std: float
    absorbed: str | None           # 'all_C', 'all_D', or None
    generations_run: int
    config: Config
    network_name: str

    def as_row(self) -> dict:
        """Flatten to a dict for a results DataFrame."""
        c = self.config
        return {
            "network": self.network_name,
            "rule": c.rule,
            "normalise_payoff": c.normalise_payoff,
            "payoff_exponent": c.alpha,
            "match_temperature": c.match_temperature,
            "T": c.payoff.T, "R": c.payoff.R, "P": c.payoff.P, "S": c.payoff.S,
            "game": c.payoff.classify(),
            "mutation": c.mutation,
            "update_fraction": c.update_fraction,
            "init_cooperators": c.init_cooperators,
            "seed": c.seed,
            "equilibrium": self.equilibrium,
            "equilibrium_std": self.equilibrium_std,
            "absorbed": self.absorbed or "none",
            "generations_run": self.generations_run,
        }


class EvolutionaryGame:
    """Runs one configuration on one network."""

    def __init__(self, network: Network, config: Config):
        self.network = network
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        adj = network.adj
        degrees = network.degrees
        if np.any(degrees == 0):
            raise ValueError("isolated nodes present; rebuild the network without them")

        counts = np.diff(adj.indptr)
        self.ctx = {
            "adj": adj,
            "indptr": adj.indptr,
            "indices": adj.indices,
            "counts": counts,
            "seg_of": np.repeat(np.arange(network.n), counts),
            "degrees": degrees,
            "span": max(config.payoff.span, 1e-12),
            "best_pos": None,
            "best_stale": True,
            "rand_nb": None,
        }

        if config.rule not in RULES:
            raise KeyError(f"unknown rule {config.rule!r}; choose from {sorted(RULES)}")
        self._rule = RULES[config.rule]

        self.rule_params = dict(config.rule_params)
        if config.match_temperature and config.rule == "fermi":
            K = float(self.rule_params.get("K", 0.1))
            self.rule_params["K"] = K / float(degrees.mean()) ** config.alpha

    def _initial_state(self) -> np.ndarray:
        n = self.network.n
        s = np.zeros(n, dtype=np.uint8)
        n_coop = int(round(self.config.init_cooperators * n))
        if n_coop:
            s[self.rng.choice(n, size=n_coop, replace=False)] = COOPERATE
        return s

    def run(self, record_snapshots: tuple[int, ...] = ()) -> Result:
        cfg = self.config
        s = self._initial_state()
        n = self.network.n
        n_active = max(1, int(round(cfg.update_fraction * n)))
        synchronous = n_active == n

        history = np.empty(cfg.generations + 1, dtype=np.float64)
        history[0] = s.mean()
        self.snapshots: dict[int, np.ndarray] = {}
        if 0 in record_snapshots:
            self.snapshots[0] = s.copy()

        absorbed = None
        gen = 0
        for gen in range(1, cfg.generations + 1):
            pi = compute_payoffs(self.ctx["adj"], s, cfg.payoff,
                                 self.ctx["degrees"], cfg.normalise_payoff,
                                 cfg.payoff_exponent)

            # Per-generation caches consumed by the rules.
            self.ctx["best_stale"] = True
            self.ctx["rand_nb"] = _random_neighbour(
                self.ctx["indptr"], self.ctx["indices"],
                self.ctx["degrees"], self.rng)

            active = (np.arange(n) if synchronous
                      else self.rng.choice(n, size=n_active, replace=False))

            new = self._rule(s, pi, self.ctx, active, self.rng, **self.rule_params)

            if cfg.mutation > 0:
                flip = self.rng.random(active.size) < cfg.mutation
                if flip.any():
                    new = new.copy()
                    new[flip] = (self.rng.random(int(flip.sum())) < 0.5).astype(np.uint8)

            s = s.copy()
            s[active] = new
            history[gen] = s.mean()

            if gen in record_snapshots:
                self.snapshots[gen] = s.copy()

            if cfg.absorbing_stop and cfg.mutation == 0:
                if history[gen] == 0.0:
                    absorbed = "all_D"
                elif history[gen] == 1.0:
                    absorbed = "all_C"
                if absorbed:
                    history[gen:] = history[gen]
                    break

        measured = history[cfg.burn_in:]
        return Result(
            cooperation=history,
            final_state=s,
            equilibrium=float(measured.mean()),
            equilibrium_std=float(measured.std()),
            absorbed=absorbed,
            generations_run=gen,
            config=cfg,
            network_name=self.network.name,
        )


def run_once(network: Network, config: Config, **kwargs) -> Result:
    """Convenience wrapper."""
    return EvolutionaryGame(network, config).run(**kwargs)
