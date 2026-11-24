"""
Figures.

Colour convention is fixed across every plot: blue = cooperator, red = defector.
Consistency matters more than palette choice when a reader is comparing a
lattice snapshot against a phase diagram three pages later.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless-safe; must precede the pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from .networks import Network

COOP_COLOUR = "#2166ac"
DEFECT_COLOUR = "#b2182b"
BINARY_CMAP = ListedColormap([DEFECT_COLOUR, COOP_COLOUR])
COOP_CMAP = LinearSegmentedColormap.from_list(
    "coop", [DEFECT_COLOUR, "#f7f7f7", COOP_COLOUR])

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_lattice_snapshots(network: Network, snapshots: dict[int, np.ndarray],
                           path: str, title: str = "") -> str:
    """Strategy configuration on the lattice at selected generations.

    The point of the figure is the *texture*: cooperators surviving as compact
    blobs with smooth boundaries is spatial reciprocity made visible, and it is
    the clearest single image for a defence.
    """
    L = network.meta.get("L")
    if L is None:
        raise ValueError("snapshots require a lattice network")

    gens = sorted(snapshots)
    fig, axes = plt.subplots(1, len(gens), figsize=(3.0 * len(gens), 3.3))
    axes = np.atleast_1d(axes)

    for ax, g in zip(axes, gens):
        grid = snapshots[g].reshape(L, L)
        ax.imshow(grid, cmap=BINARY_CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"generation {g}\ncooperation = {grid.mean():.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    if title:
        fig.suptitle(title, y=1.02)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_time_series(series: dict[str, np.ndarray], path: str,
                     title: str = "Cooperation over time",
                     burn_in: int | None = None) -> str:
    """Cooperation trajectories, one line per condition."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for label, y in series.items():
        ax.plot(y, lw=1.4, label=label)
    if burn_in is not None:
        ax.axvline(burn_in, color="0.4", ls="--", lw=1,
                   label=f"burn-in ends ({burn_in})")
    ax.set_xlabel("generation")
    ax.set_ylabel("fraction of cooperators")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_cooperation_curve(df: pd.DataFrame, path: str, x: str = "b",
                           hue: str | None = "rule",
                           title: str = "Cooperation versus temptation",
                           xlabel: str | None = None) -> str:
    """Equilibrium cooperation against a swept parameter, with replicate spread."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    groups = [(None, df)] if hue is None else list(df.groupby(hue))

    for label, d in groups:
        d = d.sort_values(x)
        y = d["equilibrium"].to_numpy()
        ax.plot(d[x], y, marker="o", ms=3.5, lw=1.5, label=str(label))
        err_col = next((c for c in ("equilibrium_std", "equilibrium_mean_std")
                        if c in d.columns), None)
        if err_col:
            e = d[err_col].fillna(0).to_numpy()
            ax.fill_between(d[x], np.clip(y - e, 0, 1), np.clip(y + e, 0, 1),
                            alpha=0.15, lw=0)

    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel("equilibrium cooperation")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    if hue is not None:
        ax.legend(frameon=False, fontsize=8, title=hue)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_phase_diagram(df: pd.DataFrame, path: str, x: str = "T_val",
                       y: str = "S_val", value: str = "equilibrium",
                       title: str = "Cooperation on the T-S plane") -> str:
    """Heatmap over the T-S plane with the four game quadrants marked."""
    grid = df.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    xs, ys = grid.columns.to_numpy(), grid.index.to_numpy()

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    mesh = ax.pcolormesh(xs, ys, grid.to_numpy(), cmap=COOP_CMAP,
                         vmin=0, vmax=1, shading="nearest")
    fig.colorbar(mesh, ax=ax, label="equilibrium cooperation")

    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(1.0, color="k", lw=0.8)
    for tx, ty, name in [(0.98, 0.95, "stag hunt"), (0.98, -0.95, "PD"),
                         (1.02, 0.95, "harmony"), (1.02, -0.95, "snowdrift")]:
        ha = "right" if tx < 1 else "left"
        va = "top" if ty > 0 else "bottom"
        ax.text(xs.min() if ha == "right" else xs.max(),
                ys.max() if va == "top" else ys.min(),
                name, fontsize=7.5, ha=ha, va=va, color="0.25",
                bbox=dict(fc="white", ec="none", alpha=0.65, pad=1.5))

    ax.set_xlabel("temptation  T")
    ax.set_ylabel("sucker payoff  S")
    ax.set_title(title)
    ax.grid(False)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_phase_grid(frames: dict[str, pd.DataFrame], path: str,
                    value: str = "equilibrium", ncols: int = 2,
                    title: str = "") -> str:
    """Several phase diagrams side by side on a shared colour scale."""
    items = list(frames.items())
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows),
                             squeeze=False)

    mesh = None
    for ax, (label, df) in zip(axes.ravel(), items):
        grid = df.pivot_table(index="S_val", columns="T_val", values=value,
                              aggfunc="mean")
        mesh = ax.pcolormesh(grid.columns.to_numpy(), grid.index.to_numpy(),
                             grid.to_numpy(), cmap=COOP_CMAP, vmin=0, vmax=1,
                             shading="nearest")
        ax.axhline(0.0, color="k", lw=0.7)
        ax.axvline(1.0, color="k", lw=0.7)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("T")
        ax.set_ylabel("S")
        ax.grid(False)

    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.85,
                     label="equilibrium cooperation")
    if title:
        fig.suptitle(title, y=1.0)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_cooperation_by_degree(frames: dict[str, pd.DataFrame], path: str,
                               title: str = "Cooperation by node degree") -> str:
    """Does cooperation concentrate on hubs?  Objective 3, in one figure."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for label, d in frames.items():
        ax.plot(d["degree_mid"], d["cooperation"], marker="o", ms=4, lw=1.5,
                label=label)
    ax.set_xscale("log")
    ax.set_xlabel("node degree (log scale)")
    ax.set_ylabel("fraction cooperating")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_heatmap(df: pd.DataFrame, path: str, x: str, y: str,
                 value: str = "equilibrium", title: str = "",
                 xlabel: str | None = None, ylabel: str | None = None) -> str:
    """Generic two-factor heatmap with the cell values annotated."""
    grid = df.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.1 * len(grid.columns) + 2.2,
                                    0.85 * len(grid.index) + 2.0))
    mesh = ax.pcolormesh(np.arange(len(grid.columns) + 1),
                         np.arange(len(grid.index) + 1),
                         grid.to_numpy(), cmap=COOP_CMAP, vmin=0, vmax=1)
    fig.colorbar(mesh, ax=ax, label=value.replace("_", " "))

    ax.set_xticks(np.arange(len(grid.columns)) + 0.5)
    ax.set_xticklabels([f"{c:g}" if isinstance(c, float) else c
                        for c in grid.columns])
    ax.set_yticks(np.arange(len(grid.index)) + 0.5)
    ax.set_yticklabels([f"{i:g}" if isinstance(i, float) else i
                        for i in grid.index])

    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            v = grid.to_numpy()[i, j]
            if not np.isnan(v):
                ax.text(j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="k" if 0.25 < v < 0.75 else "w")

    ax.set_xlabel(xlabel or x.replace("_", " "))
    ax.set_ylabel(ylabel or y.replace("_", " "))
    ax.set_title(title)
    ax.grid(False)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_regression(coefs: pd.DataFrame, path: str,
                    title: str = "Structural predictors of cooperation") -> str:
    """Standardised regression coefficients with standard-error bars."""
    d = coefs[coefs["predictor"] != "intercept"].copy()
    d = d.sort_values("coefficient")

    fig, ax = plt.subplots(figsize=(6.0, 0.55 * len(d) + 1.8))
    colours = [COOP_COLOUR if c > 0 else DEFECT_COLOUR for c in d["coefficient"]]
    ax.barh(d["predictor"].str.replace("_", " "), d["coefficient"],
            xerr=d["std_error"], color=colours, alpha=0.85,
            error_kw={"lw": 1, "ecolor": "0.3"})
    ax.axvline(0, color="k", lw=0.9)
    ax.set_xlabel("standardised coefficient")
    r2 = float(coefs["r_squared"].iloc[0])
    ax.set_title(f"{title}  (R² = {r2:.2f})")
    fig.savefig(path)
    plt.close(fig)
    return path
