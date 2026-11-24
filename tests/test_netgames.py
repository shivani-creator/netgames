"""
Correctness tests.

Two kinds of check here.  The first group verifies mechanics against brute-force
reimplementations -- payoff arithmetic, tie-breaking, neighbour sampling.  The
second group verifies that the simulator reproduces results the literature
already establishes, which is what makes any *new* result from it credible:

  * well-mixed population -> cooperation goes extinct for any T > R;
  * Moore lattice, best-takes-over -> cooperation survives near b = 1.4
    but not near b = 2.0 (Nowak & May 1992);
  * harmony game -> cooperation fixates.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from netgames import (Config, EvolutionaryGame, Payoff, build, compute_payoffs,
                      run_once)
from netgames.analysis import (cluster_statistics, cooperation_by_degree,
                               critical_threshold, hub_cooperation)
from netgames.game import COOPERATE, _random_neighbour, _segment_argmax


# --------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------

def test_lattice_degrees():
    net = build("lattice", L=10, neighbourhood="moore", periodic=True)
    assert net.n == 100
    assert np.all(net.degrees == 8)

    net4 = build("lattice", L=10, neighbourhood="von_neumann", periodic=True)
    assert np.all(net4.degrees == 4)


def test_lattice_open_boundary_has_corners():
    net = build("lattice", L=6, neighbourhood="moore", periodic=False)
    assert net.degrees.min() == 3      # corner
    assert net.degrees.max() == 8      # interior


def test_adjacency_is_symmetric_and_loopless():
    for kind, kwargs in [("lattice", {"L": 8}),
                         ("watts_strogatz", {"n": 200, "k": 6, "p": 0.2, "seed": 1}),
                         ("barabasi_albert", {"n": 200, "m": 3, "seed": 1}),
                         ("erdos_renyi", {"n": 200, "mean_degree": 6, "seed": 1})]:
        adj = build(kind, **kwargs).adj
        assert (adj != adj.T).nnz == 0, f"{kind} not symmetric"
        assert adj.diagonal().sum() == 0, f"{kind} has self-loops"
        assert set(np.unique(adj.data)) <= {1.0}, f"{kind} not binary"


def test_no_isolated_nodes():
    net = build("erdos_renyi", n=300, mean_degree=1.5, seed=3)
    assert net.degrees.min() > 0


# --------------------------------------------------------------------------
# Payoffs
# --------------------------------------------------------------------------

def test_payoff_matches_brute_force():
    net = build("watts_strogatz", n=120, k=6, p=0.3, seed=7)
    rng = np.random.default_rng(0)
    s = (rng.random(net.n) < 0.5).astype(np.uint8)
    pay = Payoff.donation(b=3.0, c=1.0)

    fast = compute_payoffs(net.adj, s, pay, net.degrees, normalise=False)

    matrix = pay.as_matrix()
    dense = net.adj.toarray()
    slow = np.array([
        sum(matrix[s[i], s[j]] for j in np.flatnonzero(dense[i]))
        for i in range(net.n)
    ])
    np.testing.assert_allclose(fast, slow, rtol=1e-10)


def test_normalised_payoff_is_per_neighbour_average():
    net = build("barabasi_albert", n=150, m=3, seed=2)
    s = np.ones(net.n, dtype=np.uint8)
    pay = Payoff.weak_pd(1.5)

    raw = compute_payoffs(net.adj, s, pay, net.degrees, normalise=False)
    avg = compute_payoffs(net.adj, s, pay, net.degrees, normalise=True)

    np.testing.assert_allclose(raw, net.degrees * pay.R)   # all-C: R per link
    np.testing.assert_allclose(avg, pay.R)


def test_game_classification():
    assert Payoff.ts_plane(T=1.5, S=-0.5).classify() == "prisoners_dilemma"
    assert Payoff.ts_plane(T=1.5, S=0.5).classify() == "snowdrift"
    assert Payoff.ts_plane(T=0.5, S=-0.5).classify() == "stag_hunt"
    assert Payoff.ts_plane(T=0.5, S=0.5).classify() == "harmony"
    assert Payoff.donation(b=3.0, c=1.0).classify() == "prisoners_dilemma"


def test_weak_pd_is_reported_as_a_boundary_case():
    """S = P, so the Nowak-May game is not a strict PD and must not claim to be."""
    assert Payoff.weak_pd(1.5).classify() == "weak_pd"


# --------------------------------------------------------------------------
# CSR helpers
# --------------------------------------------------------------------------

def test_segment_argmax_finds_the_maximum():
    net = build("watts_strogatz", n=200, k=4, p=0.2, seed=11)
    adj = net.adj
    counts = np.diff(adj.indptr)
    seg_of = np.repeat(np.arange(net.n), counts)
    values = np.random.default_rng(5).random(adj.nnz)

    pos = _segment_argmax(values, adj.indptr, counts, seg_of, rng=None)
    for i in range(net.n):
        lo, hi = adj.indptr[i], adj.indptr[i + 1]
        assert values[pos[i]] == pytest.approx(values[lo:hi].max())


def test_segment_argmax_breaks_ties_randomly():
    """All-equal payoffs must not systematically select the same neighbour."""
    net = build("lattice", L=12, neighbourhood="moore")
    adj = net.adj
    counts = np.diff(adj.indptr)
    seg_of = np.repeat(np.arange(net.n), counts)
    values = np.ones(adj.nnz)

    rng = np.random.default_rng(0)
    picks = [_segment_argmax(values, adj.indptr, counts, seg_of, rng)
             for _ in range(6)]
    assert not all(np.array_equal(picks[0], p) for p in picks[1:])


def test_random_neighbour_is_a_real_neighbour():
    net = build("barabasi_albert", n=200, m=2, seed=4)
    adj = net.adj
    rng = np.random.default_rng(1)
    j = _random_neighbour(adj.indptr, adj.indices, net.degrees, rng)
    dense = adj.toarray()
    assert all(dense[i, j[i]] == 1 for i in range(net.n))


# --------------------------------------------------------------------------
# Dynamics: reproduce known results
# --------------------------------------------------------------------------

def test_well_mixed_population_kills_cooperation():
    """The textbook result: no structure, no cooperation."""
    net = build("complete", n=300)
    cfg = Config(payoff=Payoff.weak_pd(1.4), rule="imitate_best",
                 generations=200, burn_in=150, seed=0)
    assert run_once(net, cfg).equilibrium < 0.02


def test_harmony_game_fixates_on_cooperation():
    """When cooperating dominates, cooperation must take over."""
    net = build("lattice", L=25, neighbourhood="moore")
    cfg = Config(payoff=Payoff.ts_plane(T=0.5, S=0.5), rule="imitate_best",
                 generations=300, burn_in=200, seed=0)
    assert run_once(net, cfg).equilibrium > 0.98


def test_spatial_reciprocity_sustains_cooperation():
    """Nowak-May: a lattice supports cooperation where a well-mixed graph cannot."""
    lattice = build("lattice", L=40, neighbourhood="moore")
    cfg = Config(payoff=Payoff.weak_pd(1.4), rule="imitate_best",
                 generations=400, burn_in=300, seed=1)
    spatial = run_once(lattice, cfg).equilibrium

    mixed = run_once(build("complete", n=400), cfg).equilibrium
    assert spatial > 0.1
    assert spatial > mixed + 0.1


def test_high_temptation_destroys_cooperation_even_on_lattice():
    """Spatial reciprocity has a limit; b = 2.2 is past it."""
    net = build("lattice", L=40, neighbourhood="moore")
    cfg = Config(payoff=Payoff.weak_pd(2.2), rule="imitate_best",
                 generations=400, burn_in=300, seed=2)
    assert run_once(net, cfg).equilibrium < 0.05


def test_absorbing_state_halts_the_run():
    net = build("lattice", L=20)
    cfg = Config(payoff=Payoff.weak_pd(3.0), rule="imitate_best",
                 generations=5000, burn_in=100, seed=0, absorbing_stop=True)
    res = run_once(net, cfg)
    assert res.absorbed == "all_D"
    assert res.generations_run < 200      # should die out fast


def test_mutation_prevents_absorption():
    net = build("lattice", L=25)
    cfg = Config(payoff=Payoff.weak_pd(3.0), rule="fermi", rule_params={"K": 0.1},
                 mutation=0.02, generations=300, burn_in=200, seed=0,
                 absorbing_stop=False)
    res = run_once(net, cfg)
    assert res.absorbed is None
    assert 0.0 < res.equilibrium < 1.0


@pytest.mark.parametrize("rule", ["imitate_best", "fermi",
                                  "proportional_imitation", "death_birth"])
def test_all_rules_run_and_conserve_state(rule):
    net = build("watts_strogatz", n=200, k=6, p=0.1, seed=3)
    cfg = Config(payoff=Payoff.weak_pd(1.3), rule=rule, generations=60,
                 burn_in=30, seed=0)
    res = run_once(net, cfg)
    assert res.final_state.shape == (net.n,)
    assert set(np.unique(res.final_state)) <= {0, 1}
    assert 0.0 <= res.equilibrium <= 1.0
    assert len(res.cooperation) == cfg.generations + 1


def test_initial_cooperator_fraction_is_respected():
    net = build("lattice", L=20)
    cfg = Config(payoff=Payoff.weak_pd(1.2), init_cooperators=0.3, generations=1,
                 burn_in=0, seed=0)
    game = EvolutionaryGame(net, cfg)
    assert game.run().cooperation[0] == pytest.approx(0.3, abs=1e-9)


def test_seeding_is_reproducible():
    net = build("barabasi_albert", n=200, m=3, seed=1)
    cfg = Config(payoff=Payoff.weak_pd(1.5), rule="fermi", generations=100,
                 burn_in=50, seed=42)
    a, b = run_once(net, cfg), run_once(net, cfg)
    np.testing.assert_array_equal(a.final_state, b.final_state)
    assert a.equilibrium == b.equilibrium


def test_update_fraction_limits_changes_per_generation():
    net = build("lattice", L=30)
    cfg = Config(payoff=Payoff.weak_pd(1.5), rule="fermi", update_fraction=0.05,
                 generations=1, burn_in=0, seed=0)
    res = run_once(net, cfg)
    changed = abs(res.cooperation[1] - res.cooperation[0]) * net.n
    assert changed <= 0.05 * net.n + 1


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def test_cluster_statistics_on_a_known_configuration():
    """One solid 5x5 block of cooperators on a 20x20 lattice = exactly one cluster."""
    L = 20
    net = build("lattice", L=L, neighbourhood="von_neumann", periodic=True)
    state = np.zeros(net.n, dtype=np.uint8)
    grid = state.reshape(L, L)
    grid[5:10, 5:10] = COOPERATE

    stats = cluster_statistics(net, state)
    assert stats["n_clusters"] == 1
    assert stats["largest_cluster"] == 25
    assert stats["assortment"] > 0      # clustered, so more C-C links than chance


def test_cluster_statistics_all_defect():
    net = build("lattice", L=10)
    stats = cluster_statistics(net, np.zeros(net.n, dtype=np.uint8))
    assert stats["n_clusters"] == 0
    assert stats["largest_cluster_frac"] == 0.0


def test_hub_cooperation_detects_hub_bias():
    net = build("barabasi_albert", n=400, m=3, seed=1)
    order = np.argsort(net.degrees)[::-1]
    state = np.zeros(net.n, dtype=np.uint8)
    state[order[:20]] = COOPERATE            # only the hubs cooperate

    out = hub_cooperation(net, state, top_fraction=0.05)
    assert out["hub_cooperation"] == pytest.approx(1.0)
    assert out["hub_advantage"] > 0.9


def test_cooperation_by_degree_covers_every_node():
    net = build("barabasi_albert", n=400, m=2, seed=1)
    state = (np.random.default_rng(0).random(net.n) < 0.4).astype(np.uint8)
    df = cooperation_by_degree(net, state, n_bins=6)
    assert df["n_nodes"].sum() == net.n
    assert df["cooperation"].between(0, 1).all()


def test_critical_threshold_interpolates():
    import pandas as pd
    df = pd.DataFrame({"b": [1.0, 1.5, 2.0], "equilibrium": [1.0, 0.5, 0.0]})
    assert critical_threshold(df, "b") == pytest.approx(1.5)

    never = pd.DataFrame({"b": [1.0, 2.0], "equilibrium": [1.0, 0.9]})
    assert np.isnan(critical_threshold(never, "b"))


# --------------------------------------------------------------------------
# Guard rails
# --------------------------------------------------------------------------

def test_unknown_rule_and_network_raise():
    with pytest.raises(KeyError):
        build("not_a_network")
    with pytest.raises(KeyError):
        EvolutionaryGame(build("lattice", L=10),
                         Config(payoff=Payoff.weak_pd(1.2), rule="nope"))


# --------------------------------------------------------------------------
# Generalised normalisation and temperature matching
# --------------------------------------------------------------------------

def test_payoff_exponent_interpolates_conventions():
    from netgames.game import compute_payoffs
    net = build("barabasi_albert", n=300, m=3, seed=4)
    rng = np.random.default_rng(0)
    s = (rng.random(net.n) < 0.5).astype(np.uint8)
    pay = Payoff.weak_pd(1.3)
    acc = compute_payoffs(net.adj, s, pay, net.degrees, normalise=False)
    avg = compute_payoffs(net.adj, s, pay, net.degrees, normalise=True)
    assert np.allclose(compute_payoffs(net.adj, s, pay, net.degrees, exponent=0.0), acc)
    assert np.allclose(compute_payoffs(net.adj, s, pay, net.degrees, exponent=1.0), avg)
    half = compute_payoffs(net.adj, s, pay, net.degrees, exponent=0.5)
    assert np.allclose(half, acc / np.sqrt(net.degrees))


def test_matched_temperature_is_exact_on_regular_graphs():
    """Averaged payoff at K/k must reproduce accumulated payoff at K exactly."""
    net = build("regular_random", n=400, k=6, seed=2)
    base = Config(payoff=Payoff.weak_pd(1.15), rule="fermi", rule_params={"K": 0.1},
                  generations=300, burn_in=100, absorbing_stop=False, seed=11)
    from dataclasses import replace
    acc = run_once(net, base)
    avg = run_once(net, replace(base, normalise_payoff=True, match_temperature=True))
    assert np.array_equal(acc.cooperation, avg.cooperation)


@pytest.mark.parametrize("kind,kw", [
    ("chung_lu", {"n": 800, "gamma": 2.5, "mean_degree": 8.0}),
    ("holme_kim", {"n": 800, "m": 4, "p": 0.6}),
])
def test_new_generators_are_connected_and_heterogeneous(kind, kw):
    net = build(kind, seed=3, **kw)
    assert np.all(net.degrees > 0)
    s = net.summary()
    assert 6.0 < s["mean_degree"] < 10.0
    assert s["degree_heterogeneity"] > 0.5
