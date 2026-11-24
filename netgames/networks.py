"""
Network generators for evolutionary games on graphs.

Every generator returns a :class:`Network` holding an unweighted, undirected
adjacency matrix in CSR form.  CSR is the right representation here because the
whole simulation reduces to two primitives: a sparse mat-vec (to count
cooperating neighbours) and segment-wise reductions over each node's neighbour
list (to implement imitation rules).  Both are O(edges) and fully vectorised.

Isolated nodes are removed by default: a node with no neighbours has no
opponents, its payoff is undefined, and it breaks the segment reductions in
``game.py``.  Restricting to the largest connected component is the usual
convention in this literature anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


@dataclass
class Network:
    """An undirected, unweighted graph plus bookkeeping metadata."""

    adj: csr_matrix
    name: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.adj.shape[0]

    @property
    def degrees(self) -> np.ndarray:
        """Degree of every node, as float (used as a divisor)."""
        return np.asarray(self.adj.sum(axis=1)).ravel().astype(np.float64)

    @property
    def mean_degree(self) -> float:
        return float(self.degrees.mean())

    def to_networkx(self) -> nx.Graph:
        return nx.from_scipy_sparse_array(self.adj)

    def summary(self) -> dict[str, Any]:
        """Structural descriptors used as predictors in the regression study."""
        g = self.to_networkx()
        deg = self.degrees
        return {
            "name": self.name,
            "n": self.n,
            "edges": int(self.adj.nnz // 2),
            "mean_degree": float(deg.mean()),
            "degree_std": float(deg.std()),
            "degree_heterogeneity": float(deg.var() / deg.mean() ** 2),
            "clustering": float(nx.average_clustering(g)),
            # Undefined when every node has the same degree (regular graphs and
            # periodic lattices): the correlation has zero variance in both
            # margins.  Report 0.0 -- no correlation is the honest reading --
            # rather than propagating a NaN into the regression design matrix.
            "assortativity": (0.0 if deg.std() == 0
                              else float(nx.degree_assortativity_coefficient(g))),
            "mean_path_length": _mean_path_length(g),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Network {self.name}: n={self.n}, <k>={self.mean_degree:.2f}>"


def _mean_path_length(g: nx.Graph, sample: int = 500) -> float:
    """Average shortest path length, estimated by sampling sources on big graphs."""
    if g.number_of_nodes() <= sample:
        return float(nx.average_shortest_path_length(g))
    rng = np.random.default_rng(0)
    sources = rng.choice(g.number_of_nodes(), size=sample, replace=False)
    total, count = 0, 0
    for s in sources:
        lengths = nx.single_source_shortest_path_length(g, int(s))
        total += sum(lengths.values())
        count += len(lengths) - 1
    return total / max(count, 1)


def _finalise(adj: csr_matrix, name: str, meta: dict | None = None,
              keep_isolated: bool = False) -> Network:
    """Symmetrise, binarise, drop self-loops, and (optionally) drop isolates."""
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1.0

    meta = dict(meta or {})
    if not keep_isolated:
        deg = np.asarray(adj.sum(axis=1)).ravel()
        keep = np.flatnonzero(deg > 0)
        if keep.size < adj.shape[0]:
            adj = adj[keep][:, keep].tocsr()
            if "coords" in meta:
                meta["coords"] = meta["coords"][keep]
            meta["dropped_isolated"] = int(deg.size - keep.size)
    return Network(adj=adj.astype(np.float64), name=name, meta=meta)


# --------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------

MOORE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
VON_NEUMANN = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def lattice_2d(L: int = 50, neighbourhood: str = "moore",
               periodic: bool = True, **_) -> Network:
    """Square lattice on an ``L x L`` torus (or open square if not periodic).

    This is the Nowak-May setting.  Node ``i = r * L + c``, and grid coordinates
    are stored in ``meta['coords']`` so snapshots can be reshaped for plotting.
    """
    offsets = {"moore": MOORE, "von_neumann": VON_NEUMANN}[neighbourhood]
    n = L * L
    idx = np.arange(n).reshape(L, L)
    rows, cols = [], []

    for dr, dc in offsets:
        if periodic:
            nb = np.roll(np.roll(idx, -dr, axis=0), -dc, axis=1)
            rows.append(idx.ravel())
            cols.append(nb.ravel())
        else:
            r0, r1 = max(0, -dr), min(L, L - dr)
            c0, c1 = max(0, -dc), min(L, L - dc)
            src = idx[r0:r1, c0:c1]
            dst = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            rows.append(src.ravel())
            cols.append(dst.ravel())

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    adj = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()

    rr, cc = np.divmod(np.arange(n), L)
    meta = {"L": L, "coords": np.column_stack([rr, cc]),
            "neighbourhood": neighbourhood, "periodic": periodic}
    return _finalise(adj, f"lattice_{neighbourhood}_L{L}", meta)


def watts_strogatz(n: int = 2500, k: int = 8, p: float = 0.1,
                   seed: int | None = None, **_) -> Network:
    """Small-world ring: ``k`` nearest neighbours, each edge rewired w.p. ``p``.

    ``p`` interpolates between a ring lattice (0) and a random graph (1), which
    makes it the cleanest single knob for separating the effect of clustering
    from the effect of path length.
    """
    g = nx.connected_watts_strogatz_graph(n, k, p, tries=200, seed=seed)
    adj = nx.to_scipy_sparse_array(g, format="csr", dtype=np.float64)
    return _finalise(csr_matrix(adj), f"ws_n{n}_k{k}_p{p:g}",
                     {"k": k, "p": p, "seed": seed})


def barabasi_albert(n: int = 2500, m: int = 4, seed: int | None = None,
                    **_) -> Network:
    """Scale-free graph by preferential attachment; mean degree approaches ``2m``."""
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    adj = nx.to_scipy_sparse_array(g, format="csr", dtype=np.float64)
    return _finalise(csr_matrix(adj), f"ba_n{n}_m{m}", {"m": m, "seed": seed})


def erdos_renyi(n: int = 2500, mean_degree: float = 8.0,
                seed: int | None = None, **_) -> Network:
    """Random graph with the given expected degree; the homogeneous control."""
    p = mean_degree / (n - 1)
    g = nx.fast_gnp_random_graph(n, p, seed=seed)
    adj = nx.to_scipy_sparse_array(g, format="csr", dtype=np.float64)
    return _finalise(csr_matrix(adj), f"er_n{n}_k{mean_degree:g}",
                     {"mean_degree": mean_degree, "seed": seed})


def complete_graph(n: int = 500, **_) -> Network:
    """Well-mixed population.  The analytic control: cooperation must die out."""
    adj = csr_matrix(np.ones((n, n)) - np.eye(n))
    return _finalise(adj, f"complete_n{n}")


def regular_random(n: int = 2500, k: int = 8, seed: int | None = None,
                   **_) -> Network:
    """Random graph with *every* node of degree ``k``.

    Useful as a control against Barabasi-Albert: same mean degree, zero degree
    heterogeneity, so any difference in cooperation is attributable to the
    degree distribution rather than to mean connectivity.
    """
    g = nx.random_regular_graph(k, n, seed=seed)
    adj = nx.to_scipy_sparse_array(g, format="csr", dtype=np.float64)
    return _finalise(csr_matrix(adj), f"rr_n{n}_k{k}", {"k": k, "seed": seed})


def chung_lu_powerlaw(n: int = 2500, gamma: float = 3.0, mean_degree: float = 8.0,
                      seed: int | None = None, **_) -> Network:
    """Chung-Lu graph with power-law expected degrees, restricted to its giant component.

    Sweeping ``gamma`` fills the gap in degree heterogeneity between the
    homogeneous families (Erdos-Renyi, regular) and Barabasi-Albert, at roughly
    fixed mean degree and with negligible clustering.  Expected weights follow
    ``w_i ~ (i + i0) ** (-1 / (gamma - 1))`` with the offset ``i0`` chosen so the
    largest weight stays below ``sqrt(n * <k>)`` (no self-loop-dominated hubs).
    """
    rng = np.random.default_rng(seed)
    i = np.arange(1, n + 1, dtype=np.float64)
    expo = 1.0 / (gamma - 1.0)
    w = i ** (-expo)
    w *= mean_degree * n / w.sum()
    cap = np.sqrt(mean_degree * n)
    if w.max() > cap:                      # soft structural cutoff
        i0 = 1.0
        for _ in range(60):
            w = (i + i0) ** (-expo)
            w *= mean_degree * n / w.sum()
            if w.max() <= cap:
                break
            i0 *= 1.5
    g = nx.expected_degree_graph(w.tolist(), seed=int(rng.integers(2**31)),
                                 selfloops=False)
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    g = nx.convert_node_labels_to_integers(g)
    adj = nx.to_scipy_sparse_array(g, format="csr", dtype=np.float64)
    return _finalise(csr_matrix(adj), f"cl_n{n}_g{gamma:g}",
                     {"gamma": gamma, "mean_degree": mean_degree, "seed": seed})


def holme_kim(n: int = 2500, m: int = 4, p: float = 0.5,
              seed: int | None = None, **_) -> Network:
    """Preferential attachment with triad formation (Holme & Kim 2002).

    Same scale-free degree distribution as Barabasi-Albert, but each attachment
    step closes a triangle with probability ``p``, so clustering can be raised
    while heterogeneity is held fixed -- which breaks the collinearity between
    the two descriptors in the generated families.
    """
    g = nx.powerlaw_cluster_graph(n, m, p, seed=seed)
    adj = nx.to_scipy_sparse_array(g, format="csr", dtype=np.float64)
    return _finalise(csr_matrix(adj), f"hk_n{n}_m{m}_p{p:g}",
                     {"m": m, "p": p, "seed": seed})


def from_edgelist(path: str, name: str | None = None, **kwargs) -> Network:
    """Load an empirical network (SNAP / Network Repository edge-list format)."""
    g = nx.read_edgelist(path, nodetype=int, **kwargs)
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    g = nx.convert_node_labels_to_integers(g)
    adj = nx.to_scipy_sparse_array(g, format="csr", dtype=np.float64)
    return _finalise(csr_matrix(adj), name or f"empirical_{path.split('/')[-1]}")


GENERATORS = {
    "lattice": lattice_2d,
    "watts_strogatz": watts_strogatz,
    "barabasi_albert": barabasi_albert,
    "erdos_renyi": erdos_renyi,
    "regular_random": regular_random,
    "complete": complete_graph,
    "chung_lu": chung_lu_powerlaw,
    "holme_kim": holme_kim,
}


def build(kind: str, **kwargs) -> Network:
    """Look up a generator by name.  Keeps experiment configs plain dictionaries."""
    if kind not in GENERATORS:
        raise KeyError(f"unknown network {kind!r}; choose from {sorted(GENERATORS)}")
    return GENERATORS[kind](**kwargs)
