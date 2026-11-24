"""
Diagnostics that turn a raw strategy vector into the quantities the thesis
actually argues about: are cooperators clustered, do hubs cooperate, and which
structural properties predict the outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from .game import COOPERATE
from .networks import Network


def cooperation_by_degree(network: Network, state: np.ndarray,
                          n_bins: int = 8) -> pd.DataFrame:
    """Cooperation frequency as a function of node degree.

    Logarithmic bins, because on a scale-free graph a linear binning puts 99%
    of nodes in the first bin and leaves the hubs -- the whole point of the
    exercise -- in bins of size one.
    """
    deg = network.degrees
    if deg.max() == deg.min():
        return pd.DataFrame([{
            "degree_lo": deg.min(), "degree_hi": deg.max(),
            "degree_mid": deg.min(), "n_nodes": deg.size,
            "cooperation": float(state.mean()),
        }])

    edges = np.unique(np.geomspace(deg.min(), deg.max() + 1, n_bins + 1))
    which = np.clip(np.digitize(deg, edges) - 1, 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        mask = which == b
        if not mask.any():
            continue
        rows.append({
            "degree_lo": float(edges[b]),
            "degree_hi": float(edges[b + 1]),
            "degree_mid": float(np.sqrt(edges[b] * edges[b + 1])),
            "n_nodes": int(mask.sum()),
            "cooperation": float(state[mask].mean()),
        })
    return pd.DataFrame(rows)


def hub_cooperation(network: Network, state: np.ndarray,
                    top_fraction: float = 0.05) -> dict:
    """Compare the best-connected nodes against the rest.

    ``hub_advantage > 0`` means hubs cooperate more than average, which is the
    signature of the mechanism claimed for scale-free networks: a cooperating
    hub feeds a large neighbourhood, accumulates high payoff, and is therefore
    imitated -- so cooperation radiates outward from hubs rather than from the
    compact clusters seen on a lattice.
    """
    deg = network.degrees
    n_top = max(1, int(round(top_fraction * network.n)))
    order = np.argsort(deg)[::-1]
    hubs, rest = order[:n_top], order[n_top:]
    hub_c = float(state[hubs].mean())
    rest_c = float(state[rest].mean()) if rest.size else float("nan")
    return {
        "hub_cooperation": hub_c,
        "non_hub_cooperation": rest_c,
        "hub_advantage": hub_c - rest_c,
        "hub_min_degree": float(deg[hubs].min()),
        "n_hubs": n_top,
    }


def cluster_statistics(network: Network, state: np.ndarray) -> dict:
    """Connected components of the subgraph induced on cooperators.

    Network reciprocity is a claim about *geometry*: cooperators survive by
    forming compact clusters whose interior members are shielded from
    defectors.  A population that is 30% cooperators in one giant cluster is a
    completely different regime from one that is 30% cooperators scattered as
    isolated individuals, and the global cooperation level cannot tell them
    apart -- hence these statistics.
    """
    coop = np.flatnonzero(state == COOPERATE)
    if coop.size == 0:
        return {"n_clusters": 0, "largest_cluster": 0, "largest_cluster_frac": 0.0,
                "mean_cluster_size": 0.0, "coop_coop_edge_frac": 0.0,
                "assortment": 0.0}

    sub = network.adj[coop][:, coop]
    n_comp, labels = connected_components(sub, directed=False)
    sizes = np.bincount(labels)

    # Fraction of a cooperator's links that go to other cooperators, compared
    # against the fraction expected if strategies were placed at random.
    sf = state.astype(np.float64)
    coop_nb = network.adj @ sf
    deg = network.degrees
    observed = float(coop_nb[coop].sum() / deg[coop].sum())
    expected = float(sf.mean())

    return {
        "n_clusters": int(n_comp),
        "largest_cluster": int(sizes.max()),
        "largest_cluster_frac": float(sizes.max() / network.n),
        "mean_cluster_size": float(sizes.mean()),
        "coop_coop_edge_frac": observed,
        "assortment": observed - expected,
    }


def full_diagnostics(network: Network, state: np.ndarray) -> dict:
    """All scalar diagnostics for one final configuration."""
    out = {"cooperation": float(state.mean())}
    out.update(cluster_statistics(network, state))
    out.update(hub_cooperation(network, state))
    return out


def critical_threshold(df: pd.DataFrame, param: str,
                       value_col: str = "equilibrium",
                       level: float = 0.5) -> float:
    """Parameter value at which cooperation falls through ``level``.

    Linear interpolation between the bracketing samples.  Returns NaN if the
    curve never crosses, which is itself informative (cooperation either always
    survives or never does over the sampled range).
    """
    d = df.sort_values(param)
    x = d[param].to_numpy(dtype=float)
    y = d[value_col].to_numpy(dtype=float)
    above = y >= level
    if above.all() or (~above).all():
        return float("nan")
    idx = np.flatnonzero(above[:-1] & ~above[1:])
    if idx.size == 0:
        return float("nan")
    i = int(idx[0])
    if y[i] == y[i + 1]:
        return float(x[i])
    t = (y[i] - level) / (y[i] - y[i + 1])
    return float(x[i] + t * (x[i + 1] - x[i]))


def structural_regression(df: pd.DataFrame,
                          predictors: tuple[str, ...] = (
                              "mean_degree", "degree_heterogeneity",
                              "clustering", "mean_path_length", "assortativity"),
                          target: str = "equilibrium") -> pd.DataFrame:
    """Standardised OLS coefficients of cooperation on structural descriptors.

    This is the quantitative answer to objective 3 -- which property of a graph
    predicts cooperation once the others are held fixed.  Standardising the
    predictors makes the coefficients directly comparable, since raw degree and
    raw clustering live on incomparable scales.

    The usual caveat applies: the topology families are generated, not sampled,
    so the predictors are correlated by construction and these coefficients
    describe association within this design rather than a causal effect.
    """
    cols = [c for c in predictors if c in df.columns]
    d = df.dropna(subset=cols + [target])
    if len(d) <= len(cols):
        raise ValueError("not enough distinct networks to fit the regression")

    X = d[cols].to_numpy(dtype=float)
    y = d[target].to_numpy(dtype=float)

    if y.std() < 1e-12:
        # Every network gave the same outcome, so there is nothing for structure
        # to explain.  That is a real result (the parameter point is past the
        # threshold for all topologies), not a numerical failure, so it gets an
        # explicit error rather than a silent NaN.
        raise ValueError(
            f"target {target!r} is constant at {y.mean():.3f} across all "
            f"{len(y)} networks; no variance to regress on")

    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    y_std = (y - y.mean()) / y.std()

    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y_std, rcond=None)
    resid = y_std - X1 @ beta
    dof = max(len(X) - X1.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))

    r2 = 1.0 - float(resid @ resid) / float(y_std @ y_std)
    return pd.DataFrame({
        "predictor": ["intercept"] + cols,
        "coefficient": beta,
        "std_error": se,
        "t_stat": beta / np.where(se > 0, se, np.nan),
        "r_squared": r2,
    })
