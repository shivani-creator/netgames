#!/usr/bin/env python3
"""
Analysis and figures for the manuscript.

Reads the stored study tables (results/tables, Studies 1-7) and the control
experiments (results/controls, C1-C9), and writes

  paper/figs/*.pdf        every figure in the manuscript
  paper/numbers.tex       every number quoted in the text, as LaTeX macros
  paper/tab_*.tex         table bodies

so that the manuscript is generated from the CSVs without manual transcription.

    python paper_analysis.py            # all parts whose inputs exist
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D

from netgames import Config, EvolutionaryGame, Payoff, build
from netgames.analysis import critical_threshold

TAB = Path("results/tables")
CTL = Path("results/controls")
FIG = Path("paper/figs")
OUTDIR = Path("paper")
FIG.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20261111)
NBOOT = 2000

# ---- style ------------------------------------------------------------------
# Categorical slots validated (adjacent CVD dE >= 9.2) in this order; aqua is
# below 3:1 contrast on white, so every series also carries a marker shape.
TOPO = ["lattice", "watts_strogatz", "barabasi_albert", "erdos_renyi"]
TOPO_LABEL = {"lattice": "Lattice", "watts_strogatz": "Watts–Strogatz",
              "barabasi_albert": "Barabási–Albert", "erdos_renyi": "Erdős–Rényi"}
TOPO_SHORT = {"lattice": "LAT", "watts_strogatz": "WS",
              "barabasi_albert": "BA", "erdos_renyi": "ER"}
TOPO_COLOR = {"lattice": "#2a78d6", "watts_strogatz": "#1baf7a",
              "barabasi_albert": "#eb6834", "erdos_renyi": "#6250d6"}
TOPO_MARKER = {"lattice": "s", "watts_strogatz": "^",
               "barabasi_albert": "o", "erdos_renyi": "D"}
RULES = ["fermi", "imitate_best", "proportional_imitation"]
RULE_LABEL = {"fermi": "Fermi ($K=0.1$)", "imitate_best": "Best-takes-over",
              "proportional_imitation": "Proportional imitation"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_COLOR, D_COLOR = "#2a78d6", "#eb6834"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
       "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)

COL_W, TXT_W = 3.35, 6.5   # inches, single-column layout

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

NUM: dict[str, str] = {}


def num(key: str, value, fmt: str = "{:.2f}") -> None:
    """Register a number for the manuscript as a LaTeX macro."""
    NUM[key] = fmt.format(value) if not isinstance(value, str) else value


def fmt_b(x: float, nd: int = 2) -> str:
    return "---" if not np.isfinite(x) else f"{x:.{nd}f}"


# ---- bootstrap helpers ------------------------------------------------------

def curve_matrix(raw: pd.DataFrame, col: str = "equilibrium"):
    """b values and a list of per-b replicate arrays."""
    bs = np.sort(raw["b"].unique())
    reps = [raw.loc[np.isclose(raw["b"], b), col].to_numpy() for b in bs]
    return bs, reps


def bstar(bs: np.ndarray, y: np.ndarray) -> float:
    return critical_threshold(pd.DataFrame({"b": bs, "equilibrium": y}), "b")


def auc(bs: np.ndarray, y: np.ndarray) -> float:
    """Mean cooperation over the swept range (normalised area under the curve)."""
    return float(np.trapezoid(y, bs) / (bs[-1] - bs[0]))


def bootstrap_curve(raw: pd.DataFrame, nboot: int = NBOOT) -> dict:
    bs, reps = curve_matrix(raw)
    mean = np.array([r.mean() for r in reps])
    boots = np.empty((nboot, len(bs)))
    for j, r in enumerate(reps):
        idx = RNG.integers(0, len(r), size=(nboot, len(r)))
        boots[:, j] = r[idx].mean(axis=1)
    bst = np.array([bstar(bs, row) for row in boots])
    aucs = np.array([auc(bs, row) for row in boots])
    finite = bst[np.isfinite(bst)]
    return {
        "b": bs, "mean": mean,
        "lo": np.percentile(boots, 2.5, axis=0),
        "hi": np.percentile(boots, 97.5, axis=0),
        "bstar": bstar(bs, mean),
        "bstar_lo": np.percentile(finite, 2.5) if finite.size > 0.5 * nboot else np.nan,
        "bstar_hi": np.percentile(finite, 97.5) if finite.size > 0.5 * nboot else np.nan,
        "bstar_defined": finite.size / nboot,
        "auc": auc(bs, mean),
        "auc_lo": np.percentile(aucs, 2.5), "auc_hi": np.percentile(aucs, 97.5),
    }


# =============================================================================
# Figure 1: lattice snapshots
# =============================================================================

def fig_snapshots():
    net = build("lattice", L=100, neighbourhood="moore")
    gens = (0, 5, 25, 100, 400)
    cmap = ListedColormap([D_COLOR, C_COLOR])
    fig, axes = plt.subplots(2, 5, figsize=(TXT_W, 3.05))
    finals = {}
    for row, b in enumerate((1.4, 1.9)):
        cfg = Config(payoff=Payoff.weak_pd(b), rule="imitate_best",
                     generations=max(gens), burn_in=200, seed=1,
                     absorbing_stop=False)
        game = EvolutionaryGame(net, cfg)
        res = game.run(record_snapshots=gens)
        finals[b] = res
        for col, g in enumerate(gens):
            ax = axes[row, col]
            grid = game.snapshots[g].reshape(100, 100)
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(INK2); s.set_linewidth(0.4)
            ax.set_title(f"$t={g}$,  $\\rho_C={grid.mean():.2f}$", fontsize=7.5)
        axes[row, 0].set_ylabel(f"$b={b}$", fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_COLOR, label="cooperator"),
               plt.Rectangle((0, 0), 1, 1, color=D_COLOR, label="defector")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(wspace=0.06, hspace=0.28)
    fig.savefig(FIG / "fig1_snapshots.pdf")
    plt.close(fig)
    num("snapRhoFiveFourteen", float(finals[1.4].cooperation[5]))
    num("snapEqFourteen", finals[1.4].equilibrium)
    num("snapEqNineteen", finals[1.9].equilibrium)


# =============================================================================
# Table 1: network descriptors (N = 2500)
# =============================================================================

def tab_networks():
    s = pd.read_csv(TAB / "07_structural_survey_accumulated.csv")
    s = s[np.isclose(s["b"], 1.2)].set_index("network_label")
    pick = [("lattice_moore_L50", "Lattice (Moore, $50\\times50$)"),
            ("ws_n2500_k8_p0.1", "Watts--Strogatz ($p=0.1$)"),
            ("ba_n2500_m4", "Barab\\'asi--Albert ($m=4$)"),
            ("er_n2500_k8", "Erd\\H{o}s--R\\'enyi")]
    lines = []
    for key, label in pick:
        r = s.loc[key]
        lines.append(f"{label} & {r['n']:.0f} & {r['mean_degree']:.2f} & "
                     f"{r['degree_heterogeneity']:.3f} & {r['clustering']:.3f} & "
                     f"{r['mean_path_length']:.2f} \\\\")
    (OUTDIR / "tab_networks.tex").write_text("\n".join(lines) + "\n")


# =============================================================================
# Figure 2 + Table 2: the factorial sweep (C7 replicate-level re-run)
# =============================================================================

def factorial():
    raw = pd.read_csv(CTL / "C7_study3_raw.csv")
    stored = pd.read_csv(TAB / "03_temptation_sweep.csv")

    # Reproducibility check against the stored aggregate table.
    raw = raw.assign(b=raw["b"].round(4))
    stored = stored.assign(b=stored["b"].round(4))
    agg = (raw.groupby(["net_kind", "b", "rule", "normalise_payoff"])
           ["equilibrium"].mean().reset_index())
    m = agg.merge(stored[["net_kind", "b", "rule", "normalise_payoff",
                          "equilibrium"]],
                  on=["net_kind", "b", "rule", "normalise_payoff"],
                  suffixes=("_rerun", "_stored"))
    diff = float((m["equilibrium_rerun"] - m["equilibrium_stored"]).abs().max())
    num("reproMaxDiff", diff, "{:.1e}")
    num("reproCells", len(m), "{:d}")
    print(f"  reproducibility: max |diff| over {len(m)} cells = {diff:.2e}")

    res = {}
    for (kind, rule, norm), d in raw.groupby(["net_kind", "rule",
                                              "normalise_payoff"]):
        res[(kind, rule, bool(norm))] = bootstrap_curve(d)

    # ---- figure: 2 rows (convention) x 3 cols (rule)
    fig, axes = plt.subplots(2, 3, figsize=(TXT_W, 3.9), sharex=True, sharey=True)
    for i, norm in enumerate((False, True)):
        for j, rule in enumerate(RULES):
            ax = axes[i, j]
            for kind in TOPO:
                r = res[(kind, rule, norm)]
                c = TOPO_COLOR[kind]
                ax.fill_between(r["b"], r["lo"], r["hi"], color=c, alpha=0.16,
                                lw=0)
                ax.plot(r["b"], r["mean"], color=c, lw=1.4,
                        marker=TOPO_MARKER[kind], ms=2.6, mew=0,
                        label=TOPO_LABEL[kind])
            ax.axhline(0.5, color=INK2, lw=0.5, ls=(0, (2, 2)))
            ax.set_ylim(-0.03, 1.03); ax.set_xlim(0.97, 2.03)
            ax.set_xticks([1.0, 1.25, 1.5, 1.75, 2.0])
            ax.set_xticklabels(["1", "1.25", "1.5", "1.75", "2"])
            if i == 0:
                ax.set_title(RULE_LABEL[rule])
            if j == 0:
                ax.set_ylabel(("accumulated" if not norm else "averaged")
                              + "\ncooperation $\\rho_C$")
            if i == 1:
                ax.set_xlabel("temptation $b$")
    h = [Line2D([], [], color=TOPO_COLOR[k], marker=TOPO_MARKER[k], ms=4,
                lw=1.4, label=TOPO_LABEL[k]) for k in TOPO]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(wspace=0.1, hspace=0.12)
    fig.savefig(FIG / "fig2_factorial.pdf")
    plt.close(fig)

    # ---- threshold table: b* [95% CI] and AUC
    lines = []
    for kind in ["barabasi_albert", "erdos_renyi", "lattice", "watts_strogatz"]:
        cells = []
        for rule in RULES:
            for norm in (False, True):
                r = res[(kind, rule, norm)]
                # best within column
                col_vals = [res[(k, rule, norm)]["bstar"] for k in TOPO]
                col_vals = [v for v in col_vals if np.isfinite(v)]
                best = np.isfinite(r["bstar"]) and col_vals and \
                    r["bstar"] == max(col_vals)
                txt = fmt_b(r["bstar"])
                if best:
                    txt = f"\\textbf{{{txt}}}"
                cells.append(txt)
        lines.append(f"{TOPO_LABEL[kind].replace('–', '--').replace('á', chr(92)+chr(39)+'a').replace('ő', chr(92)+'H{o}').replace('é', chr(92)+chr(39)+'e')} & " + " & ".join(cells) + " \\\\")
    (OUTDIR / "tab_thresholds.tex").write_text("\n".join(lines) + "\n")

    lines = []
    for kind in ["barabasi_albert", "erdos_renyi", "lattice", "watts_strogatz"]:
        cells = []
        for rule in RULES:
            for norm in (False, True):
                r = res[(kind, rule, norm)]
                cells.append(f"{r['auc']:.2f}")
        lines.append(f"{TOPO_SHORT[kind]} & " + " & ".join(cells) + " \\\\")
    (OUTDIR / "tab_auc.tex").write_text("\n".join(lines) + "\n")

    # macros
    for kind in TOPO:
        for rule in RULES:
            for norm in (False, True):
                r = res[(kind, rule, norm)]
                key = (f"bs{TOPO_SHORT[kind]}"
                       f"{ {'fermi':'Fe','imitate_best':'Bt','proportional_imitation':'Pi'}[rule]}"
                       f"{'Av' if norm else 'Ac'}")
                num(key, fmt_b(r["bstar"]))
                num(key + "Lo", fmt_b(r["bstar_lo"]))
                num(key + "Hi", fmt_b(r["bstar_hi"]))
                num(key.replace("bs", "auc"), r["auc"])
                num(key.replace("bs", "auc") + "Lo", r["auc_lo"])
                num(key.replace("bs", "auc") + "Hi", r["auc_hi"])
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                                          for kk, vv in v.items()}
               for k, v in res.items()},
              open(CTL / "factorial_bootstrap.json", "w"), indent=1, default=float)
    return res


# =============================================================================
# Figure 3: temperature-matched control and the alpha interpolation
# =============================================================================

def fig_temperature(res_fact):
    c1 = pd.read_csv(CTL / "C1_matched_temperature_raw.csv")
    c2 = pd.read_csv(CTL / "C2_alpha_sweep_raw.csv")
    matched = {k: bootstrap_curve(d) for k, d in c1.groupby("net_kind")}

    # exact equivalence on the lattice
    c7 = pd.read_csv(CTL / "C7_study3_raw.csv")
    a_ = c1[c1["net_kind"] == "lattice"].assign(b=lambda d: d["b"].round(4))
    b_ = c7[(c7["net_kind"] == "lattice") & (c7["rule"] == "fermi")
            & (~c7["normalise_payoff"])].assign(b=lambda d: d["b"].round(4))
    m_ = a_.merge(b_, on=["b", "replicate"])
    num("latMatchedRuns", len(m_), "{:d}")
    num("latMatchedMaxDiff", float((m_["equilibrium_x"] - m_["equilibrium_y"]).abs().max()),
        "{:.1g}")

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    curves = {}
    for kind in ("barabasi_albert", "erdos_renyi"):
        for a in alphas:
            if a == 0.0:
                curves[(kind, a)] = res_fact[(kind, "fermi", False)]
            elif a == 1.0:
                curves[(kind, a)] = matched[kind]
            else:
                d = c2[(c2["net_kind"] == kind) & np.isclose(c2["alpha"], a)]
                curves[(kind, a)] = bootstrap_curve(d)

    fig, axes = plt.subplots(1, 3, figsize=(TXT_W, 2.15),
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})
    # (a) accumulated vs averaged-matched, Fermi
    for ax, (title, src) in zip(axes[:2], [
            ("(a) accumulated, $K=0.1$", lambda k: res_fact[(k, "fermi", False)]),
            ("(b) averaged, $K=0.1/\\langle k\\rangle$", lambda k: matched[k])]):
        for kind in TOPO:
            r = src(kind)
            c = TOPO_COLOR[kind]
            ax.fill_between(r["b"], r["lo"], r["hi"], color=c, alpha=0.16, lw=0)
            ax.plot(r["b"], r["mean"], color=c, lw=1.4, marker=TOPO_MARKER[kind],
                    ms=2.6, mew=0, label=TOPO_LABEL[kind])
        ax.axhline(0.5, color=INK2, lw=0.5, ls=(0, (2, 2)))
        ax.set_xlim(1, 2); ax.set_ylim(-0.03, 1.03)
        ax.set_title(title); ax.set_xlabel("temptation $b$")
    axes[0].set_ylabel("cooperation $\\rho_C$")
    axes[1].set_yticklabels([])
    axes[1].legend(loc="upper right", handlelength=1.6, borderaxespad=0.2)
    # (c) b* vs alpha
    ax = axes[2]
    for kind in ("barabasi_albert", "erdos_renyi"):
        y = np.array([curves[(kind, a)]["bstar"] for a in alphas])
        lo = np.array([curves[(kind, a)]["bstar_lo"] for a in alphas])
        hi = np.array([curves[(kind, a)]["bstar_hi"] for a in alphas])
        c = TOPO_COLOR[kind]
        ax.fill_between(alphas, lo, hi, color=c, alpha=0.16, lw=0)
        ax.plot(alphas, y, color=c, lw=1.4, marker=TOPO_MARKER[kind], ms=3.5,
                mew=0, label=TOPO_LABEL[kind])
        for a, v in zip(alphas, y):
            word = {0.0: "Zero", 0.25: "Quarter", 0.5: "Half", 0.75: "ThreeQ",
                    1.0: "One"}[a]
            num(f"alpha{TOPO_SHORT[kind]}{word}", fmt_b(v))
    lat = res_fact[("lattice", "fermi", False)]["bstar"]
    ax.axhline(lat, color=TOPO_COLOR["lattice"], lw=0.9, ls=(0, (3, 2)))
    ax.text(0.02, lat - 0.012, "lattice (all $\\alpha$)", fontsize=6.5,
            color=INK2, va="top")
    ax.set_xlabel("normalisation exponent $\\alpha$")
    ax.set_ylabel("critical temptation $b^{\\ast}$")
    ax.set_title("(c) $\\pi_i/k_i^{\\alpha}$, matched $K$")
    ax.set_xticks(alphas)
    ax.legend(loc="upper right", handlelength=1.6)
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(FIG / "fig3_temperature.pdf")
    plt.close(fig)

    for kind in TOPO:
        r = matched[kind]
        num(f"bs{TOPO_SHORT[kind]}FeMt", fmt_b(r["bstar"]))
        num(f"bs{TOPO_SHORT[kind]}FeMtLo", fmt_b(r["bstar_lo"]))
        num(f"bs{TOPO_SHORT[kind]}FeMtHi", fmt_b(r["bstar_hi"]))
        num(f"auc{TOPO_SHORT[kind]}FeMt", r["auc"])
    return matched


# =============================================================================
# Table 3: assortment against the permutation null
# =============================================================================

def fmt_null(x: float) -> str:
    txt = f"{x:+.4f}"
    return "0.0000" if txt in ("+0.0000", "-0.0000") else f"${txt.replace('-', '-')}$"


def tab_assortment():
    a = pd.read_csv(CTL / "C4_assortment.csv")
    g = a.groupby("b")
    lines = []
    for b, d in g:
        coop = d["cooperation"]
        alive = d[d["cooperation"] > 0]
        if len(alive) == 0:
            lines.append(f"{b:.1f} & 0.000 & --- & --- & --- \\\\")
            continue
        z = ((alive["assortment"] - alive["null_assortment_mean"])
             / alive["null_assortment_sd"].clip(lower=1e-9))
        lines.append(
            f"{b:.1f} & {coop.mean():.3f} & {d['n_clusters'].mean():.1f} "
            f"& {alive['assortment'].mean():.3f} $\\pm$ {alive['assortment'].std():.3f} "
            f"& {fmt_null(alive['null_assortment_mean'].mean())} \\\\")
    (OUTDIR / "tab_assortment.tex").write_text("\n".join(lines) + "\n")
    d14 = a[np.isclose(a["b"], 1.4)]
    num("assortFourteen", d14["assortment"].mean(), "{:.3f}")
    num("assortFourteenSd", d14["assortment"].std(), "{:.3f}")
    num("nullFourteen", d14["null_assortment_mean"].mean(), "{:+.4f}")
    num("nullSdFourteen", d14["null_assortment_sd"].mean(), "{:.4f}")
    z = ((d14["assortment"] - d14["null_assortment_mean"])
         / d14["null_assortment_sd"]).min()
    num("assortMinZ", z, "{:.0f}")
    num("clustersFourteen", d14["n_clusters"].mean(), "{:.1f}")
    num("nullClustersFourteen", d14["null_clusters_mean"].mean(), "{:.1f}")
    num("latBtCoopFourteen", d14["cooperation"].mean(), "{:.2f}")


# =============================================================================
# Figure 4: hubs
# =============================================================================

def fig_hubs():
    t = pd.read_csv(CTL / "C5_hub_trajectories.csv")
    s = pd.read_csv(CTL / "C5_degree_states.csv")
    fig, axes = plt.subplots(1, 2, figsize=(TXT_W * 0.72, 2.1))
    ax = axes[0]
    d = t[np.isclose(t["b"], 1.7)].groupby("generation")[
        ["coop_all", "coop_hubs", "coop_leaves"]].mean()
    d = d[d.index > 0]
    ax.plot(d.index, d["coop_hubs"], color=TOPO_COLOR["barabasi_albert"], lw=1.4,
            label="hubs (top 5% by degree)")
    ax.plot(d.index, d["coop_leaves"], color=INK2, lw=1.4, ls=(0, (3, 1.5)),
            label="leaves ($k=4$)")
    ax.set_xscale("log"); ax.set_xlim(2, 3000); ax.set_ylim(0, 1)
    ax.set_xlabel("generation $t$"); ax.set_ylabel("cooperation $\\rho_C$")
    ax.set_title("(a) BA, accumulated, $b=1.7$")
    ax.legend(loc="upper left", handlelength=1.8)
    ax = axes[1]
    edges = np.unique(np.round(np.geomspace(4, s["degree"].max() + 1, 9)))
    shades = {1.5: "#86b6ef", 1.7: "#2a78d6", 1.9: "#104281"}
    marks = {1.5: "o", 1.7: "s", 1.9: "D"}
    for b in (1.5, 1.7, 1.9):
        x = s[np.isclose(s["b"], b)]
        which = np.clip(np.digitize(x["degree"], edges) - 1, 0, len(edges) - 2)
        g = x.groupby(which)["cooperate"].mean()
        mids = np.sqrt(edges[:-1] * edges[1:])[g.index]
        ax.plot(mids, g.values, color=shades[b], marker=marks[b], ms=3.2, lw=1.3,
                mew=0, label=f"$b={b}$")
    ax.set_xscale("log"); ax.set_ylim(0, 1.03)
    ax.set_xlabel("degree $k$"); ax.set_title("(b) cooperation by degree")
    ax.legend(loc="lower right", handlelength=1.6)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(FIG / "fig4_hubs.pdf")
    plt.close(fig)
    end = t[t["generation"] == 3000].groupby("b")[["coop_hubs", "coop_leaves"]].mean()
    num("hubEndSeventeen", end.loc[1.7, "coop_hubs"])
    num("leafEndSeventeen", end.loc[1.7, "coop_leaves"])
    early = t[(t["generation"] == 10) & np.isclose(t["b"], 1.7)][
        ["coop_hubs", "coop_leaves"]].mean()
    num("hubTenSeventeen", early["coop_hubs"])
    num("leafTenSeventeen", early["coop_leaves"])


# =============================================================================
# Structural survey (C3)
# =============================================================================

PRED = ["mean_degree", "degree_heterogeneity", "clustering",
        "log_path_length", "assortativity"]
PRED_LABEL = {"mean_degree": "Mean degree $\\langle k\\rangle$",
              "degree_heterogeneity": "Degree heterogeneity $H$",
              "clustering": "Clustering $C$",
              "log_path_length": "$\\log$ mean path length",
              "assortativity": "Degree assortativity $r$"}
COND_LABEL = {"accumulated": "Accumulated",
              "averaged": "Averaged",
              "averaged_matched": "Averaged, matched $K$"}


def ols(X: np.ndarray, y: np.ndarray):
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    ys = (y - y.mean()) / y.std()
    X1 = np.column_stack([np.ones(len(Xs)), Xs])
    beta, *_ = np.linalg.lstsq(X1, ys, rcond=None)
    resid = ys - X1 @ beta
    dof = len(ys) - X1.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X1.T @ X1)))
    r2 = 1 - resid @ resid / (ys @ ys)
    return beta[1:], se[1:], r2


def survey():
    raw = pd.read_csv(CTL / "C3_extended_survey_raw.csv")
    desc = pd.read_csv(CTL / "C3_extended_survey_descriptors.csv")
    desc["log_path_length"] = np.log(desc["mean_path_length"])
    raw = raw.assign(b=raw["b"].round(4))
    agg = (raw.groupby(["condition", "b", "spec_id"])["equilibrium"]
           .agg(["mean", "std"]).reset_index()
           .merge(desc, on="spec_id"))

    # reproducibility on the original 21 specifications
    stored = pd.concat([
        pd.read_csv(TAB / "07_structural_survey_accumulated.csv").assign(condition="accumulated"),
        pd.read_csv(TAB / "07_structural_survey_averaged.csv").assign(condition="averaged")])
    m = agg.merge(stored[["condition", "b", "spec_id", "equilibrium"]],
                  on=["condition", "b", "spec_id"])
    num("surveyReproMaxDiff", float((m["mean"] - m["equilibrium"]).abs().max()), "{:.1e}")
    num("surveyReproCells", len(m), "{:d}")
    print(f"  survey reproducibility: {len(m)} cells, max diff "
          f"{(m['mean'] - m['equilibrium']).abs().max():.2e}")

    # VIF of the design
    X = desc[PRED].to_numpy(float)
    Xs = (X - X.mean(0)) / X.std(0)
    vif = np.diag(np.linalg.inv(np.corrcoef(Xs, rowvar=False)))
    num("vifMax", vif.max(), "{:.1f}")
    num("corrHetClust", np.corrcoef(desc["degree_heterogeneity"],
                                    desc["clustering"])[0, 1], "{:+.2f}")
    num("nSpecs", len(desc), "{:d}")

    rows = []
    for (cond, b), d in agg.groupby(["condition", "b"]):
        y = d["mean"].to_numpy()
        if y.std() < 1e-9:
            continue
        beta, se, r2 = ols(d[PRED].to_numpy(float), y)
        for p, bb, ss in zip(PRED, beta, se):
            rows.append({"condition": cond, "b": b, "predictor": p,
                         "beta": bb, "se": ss, "t": bb / ss, "r2": r2,
                         "y_sd": y.std(), "y_mean": y.mean()})
    reg = pd.DataFrame(rows)
    reg.to_csv(CTL / "C3_regressions.csv", index=False)

    # pick b per condition with maximal response variance (pre-registered rule
    # of Study 7), and also report the fixed b = 1.2 fit.
    picks = {}
    for cond in COND_LABEL:
        r = reg[reg["condition"] == cond]
        if r.empty:
            continue
        picks[cond] = float(r.loc[r["y_sd"].idxmax(), "b"])
        num(f"bPick{cond.replace('_', '').title()}", picks[cond], "{:g}")

    lines = []
    for p in PRED:
        cells = []
        for cond in COND_LABEL:
            r = reg[(reg["condition"] == cond) & np.isclose(reg["b"], picks[cond])
                    & (reg["predictor"] == p)]
            if r.empty:
                cells.append("---"); continue
            r = r.iloc[0]
            star = "^{\\ast}" if abs(r["t"]) > 2.069 else ""   # t_{0.975, 23}
            val = f"{r['beta']:+.2f}{star}"
            cells.append(f"${val}$ & $({r['se']:.2f})$")
        lines.append(PRED_LABEL[p] + " & " + " & ".join(cells) + " \\\\")
    r2s = []
    for cond in COND_LABEL:
        r = reg[(reg["condition"] == cond) & np.isclose(reg["b"], picks[cond])]
        r2s.append(f"\\multicolumn{{2}}{{c}}{{{r['r2'].iloc[0]:.2f}}}")
    lines.append("\\midrule")
    lines.append("$R^2$ & " + " & ".join(r2s) + " \\\\")
    sds = []
    for cond in COND_LABEL:
        r = reg[(reg["condition"] == cond) & np.isclose(reg["b"], picks[cond])]
        sds.append(f"\\multicolumn{{2}}{{c}}{{{r['y_mean'].iloc[0]:.2f} / {r['y_sd'].iloc[0]:.2f}}}")
    lines.append("Response mean / s.d. & " + " & ".join(sds) + " \\\\")
    (OUTDIR / "tab_regression.tex").write_text("\n".join(lines) + "\n")

    for cond in COND_LABEL:
        r = reg[(reg["condition"] == cond) & np.isclose(reg["b"], picks[cond])
                & (reg["predictor"] == "degree_heterogeneity")].iloc[0]
        key = cond.replace("_", "").title()
        num(f"het{key}", r["beta"], "{:+.2f}")
        num(f"hetT{key}", r["t"], "{:+.1f}")
        num(f"rsq{key}", r["r2"])
        rc = reg[(reg["condition"] == cond) & np.isclose(reg["b"], picks[cond])
                 & (reg["predictor"] == "clustering")].iloc[0]
        num(f"clu{key}", rc["beta"], "{:+.2f}")
        num(f"cluT{key}", rc["t"], "{:+.1f}")
        rp = reg[(reg["condition"] == cond) & np.isclose(reg["b"], picks[cond])
                 & (reg["predictor"] == "log_path_length")].iloc[0]
        num(f"pl{key}", rp["beta"], "{:+.2f}")
        num(f"plT{key}", rp["t"], "{:+.1f}")
        rm = reg[(reg["condition"] == cond) & np.isclose(reg["b"], picks[cond])
                 & (reg["predictor"] == "mean_degree")].iloc[0]
        num(f"md{key}", rm["beta"], "{:+.2f}")
        num(f"mdT{key}", rm["t"], "{:+.1f}")
    dm = agg[(agg["condition"] == "averaged_matched")
             & np.isclose(agg["b"], picks["averaged_matched"])]
    ws = dm[dm["family"] == "watts_strogatz"].sort_values("mean_path_length")
    num("wsBestMatched", ws["mean"].max())
    num("wsRingMatched", ws["mean"].iloc[-1])

    for cond in COND_LABEL:
        r = reg[(reg["condition"] == cond) & np.isclose(reg["b"], picks[cond])].iloc[0]
        key = cond.replace("_", "").title()
        num(f"yMean{key}", r["y_mean"], "{:.3f}")
        num(f"ySd{key}", r["y_sd"], "{:.3f}")

    # leave-one-family-out stability of the heterogeneity/clustering effects
    ih, ic = PRED.index("degree_heterogeneity"), PRED.index("clustering")
    for cond in ("accumulated", "averaged_matched"):
        d = agg[(agg["condition"] == cond) & np.isclose(agg["b"], picks[cond])]
        bh, th, bc, tc = [], [], [], []
        for fam in d["family"].unique():
            x = d[d["family"] != fam]
            beta, se, _ = ols(x[PRED].to_numpy(float), x["mean"].to_numpy())
            bh.append(beta[ih]); th.append(beta[ih] / se[ih])
            bc.append(beta[ic]); tc.append(beta[ic] / se[ic])
        key = cond.replace("_", "").title()
        num(f"lofoHetMin{key}", min(bh), "{:+.2f}")
        num(f"lofoHetMax{key}", max(bh), "{:+.2f}")
        num(f"lofoHetAbsTMin{key}", min(abs(t) for t in th), "{:.1f}")
        num(f"lofoCluMin{key}", min(bc), "{:+.2f}")
        num(f"lofoCluMax{key}", max(bc), "{:+.2f}")
        num(f"lofoCluAbsTMin{key}", min(abs(t) for t in tc), "{:.1f}")
        cl = d[d["family"] == "chung_lu"].sort_values("degree_heterogeneity")
        num(f"clCoopLowH{key}", cl["mean"].iloc[0])
        num(f"clCoopHighH{key}", cl["mean"].iloc[-1])
    cl = desc[desc["family"] == "chung_lu"].sort_values("degree_heterogeneity")
    num("clHLow", cl["degree_heterogeneity"].iloc[0])
    num("clHHigh", cl["degree_heterogeneity"].iloc[-1])

    # ---- figure: cooperation vs heterogeneity, by family, two conditions
    fam_style = {"watts_strogatz": ("Watts–Strogatz", TOPO_COLOR["watts_strogatz"], "^"),
                 "barabasi_albert": ("Barabási–Albert", TOPO_COLOR["barabasi_albert"], "o"),
                 "erdos_renyi": ("Erdős–Rényi", TOPO_COLOR["erdos_renyi"], "D"),
                 "lattice": ("Lattice", TOPO_COLOR["lattice"], "s"),
                 "regular_random": ("Random regular", "#52514e", "v"),
                 "chung_lu": ("Chung–Lu (new)", "#eda100", "P"),
                 "holme_kim": ("Holme–Kim (new)", "#e34948", "X")}
    fig, axes = plt.subplots(1, 3, figsize=(TXT_W, 2.2), sharey=False,
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, cond in zip(axes[:2], ["accumulated", "averaged_matched"]):
        d = agg[(agg["condition"] == cond) & np.isclose(agg["b"], picks[cond])]
        for fam, (lab, col, mk) in fam_style.items():
            x = d[d["family"] == fam]
            ax.scatter(x["degree_heterogeneity"], x["mean"], s=16, color=col,
                       marker=mk, linewidths=0.4, edgecolors="white", label=lab,
                       zorder=3)
        ax.set_ylim(-0.04, 1.04); ax.set_xlim(-0.1, 2.75)
        ax.set_xlabel("degree heterogeneity $H$")
        ttl = "accumulated" if cond == "accumulated" else "averaged, matched $K$"
        ax.set_title(f"({'a' if cond == 'accumulated' else 'b'}) {ttl}, "
                     f"$b={picks[cond]:g}$")
    axes[0].set_ylabel("cooperation $\\rho_C$")
    axes[1].set_yticklabels([])
    # (c) coefficient on H across b, three conditions
    ax = axes[2]
    cstyle = {"accumulated": (TOPO_COLOR["lattice"], "o", "-"),
              "averaged": (INK2, "s", (0, (1.5, 1.2))),
              "averaged_matched": (TOPO_COLOR["barabasi_albert"], "D", (0, (4, 1.5)))}
    for cond, (col, mk, ls) in cstyle.items():
        r = reg[(reg["condition"] == cond) & (reg["predictor"] == "degree_heterogeneity")
                & (reg["y_sd"] >= 0.02)]   # drop b where cooperation is extinct everywhere
        ax.errorbar(r["b"], r["beta"], yerr=1.96 * r["se"], color=col, marker=mk,
                    ms=3.2, lw=1.2, ls=ls, capsize=1.5, elinewidth=0.7,
                    label=COND_LABEL[cond])
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("temptation $b$")
    ax.set_ylabel("std. coefficient on $H$")
    ax.set_title("(c) heterogeneity effect vs $b$")
    ax.legend(loc="lower right", fontsize=6.3, handlelength=2.2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, bbox_to_anchor=(0.45, -0.04),
               fontsize=6.5, handletextpad=0.2, columnspacing=0.9)
    fig.subplots_adjust(wspace=0.34)
    fig.savefig(FIG / "fig5_survey.pdf")
    plt.close(fig)

    # descriptor table for the appendix
    lines = []
    for _, r in desc.iterrows():
        lab = r["network_label"].replace("_", "\\_")
        lines.append(f"\\texttt{{{lab}}} & {r['n']:.0f} & {r['mean_degree']:.2f} & "
                     f"{r['degree_heterogeneity']:.3f} & {r['clustering']:.3f} & "
                     f"{r['mean_path_length']:.2f} & {r['assortativity']:+.3f} \\\\")
    (OUTDIR / "tab_survey_specs.tex").write_text("\n".join(lines) + "\n")
    return reg, picks


# =============================================================================
# Robustness: noise (Study 6 + C6), asynchronous (C8), finite size (C9)
# =============================================================================

def tab_noise():
    ba = pd.read_csv(TAB / "06_robustness.csv")
    er = pd.read_csv(CTL / "C6_robustness_erdos_renyi.csv")
    pb = ba.pivot(index="mutation_rate", columns="temperature", values="equilibrium")
    pe = er.pivot(index="mutation_rate", columns="temperature", values="equilibrium")
    lab = {0.0: "$0$", 1e-4: "$10^{-4}$", 1e-3: "$10^{-3}$", 1e-2: "$10^{-2}$",
           5e-2: "$5\\times10^{-2}$"}
    lines = []
    for mu in pb.index:
        cells = [f"{pb.loc[mu, K]:.3f}" for K in pb.columns]
        cells += [f"{pe.loc[mu, 0.1]:.3f}"]
        lines.append(f"{lab[mu]} & " + " & ".join(cells) + " \\\\")
    (OUTDIR / "tab_noise.tex").write_text("\n".join(lines) + "\n")
    num("noiseBAzero", pb.loc[0.0].mean())
    num("noiseBAzeroMin", np.floor(pb.loc[0.0].min() * 100) / 100)
    num("noiseBAzeroMax", pb.loc[0.0].max())
    num("noiseBAmuTwo", pb.loc[1e-2].mean())
    num("noiseBAmuFive", pb.loc[5e-2].mean())
    num("noiseERmuTwo", pe.loc[1e-2, 0.1], "{:.3f}")
    num("noiseERmuFive", pe.loc[5e-2, 0.1], "{:.3f}")
    num("noiseERmuFiveMax", pe.loc[5e-2].max(), "{:.3f}")
    num("noiseERzero", pe.loc[0.0, 0.1], "{:.3f}")


def robustness_fig():
    have_async = (CTL / "C8_asynchronous_raw.csv").exists()
    have_fs = (CTL / "C9_finite_size_raw.csv").exists()
    if not (have_async and have_fs):
        print("  robustness figure skipped (C8/C9 pending)")
        return
    asy = pd.read_csv(CTL / "C8_asynchronous_raw.csv")
    fs = pd.read_csv(CTL / "C9_finite_size_raw.csv")
    syn = pd.read_csv(CTL / "C7_study3_raw.csv")

    fig, axes = plt.subplots(1, 3, figsize=(TXT_W, 2.15))
    # (a)(b): sync vs async curves for best-takes-over and Fermi, accumulated
    asy_res = {}
    for ax, rule, title in [(axes[0], "imitate_best", "(a) best-takes-over, accum."),
                            (axes[1], "fermi", "(b) Fermi, accumulated")]:
        for kind in TOPO:
            c = TOPO_COLOR[kind]
            s = syn[(syn["rule"] == rule) & (~syn["normalise_payoff"]) &
                    (syn["net_kind"] == kind)]
            ms = s.groupby("b")["equilibrium"].mean()
            ax.plot(ms.index, ms.values, color=c, lw=0.9, ls=(0, (2, 1.5)), alpha=0.8)
            a = asy[(asy["rule"] == rule) & (~asy["normalise_payoff"]) &
                    (asy["net_kind"] == kind)]
            r = bootstrap_curve(a)
            asy_res[(kind, rule, False)] = r
            ax.plot(r["b"], r["mean"], color=c, lw=1.4, marker=TOPO_MARKER[kind],
                    ms=2.8, mew=0, label=TOPO_LABEL[kind])
        ax.axhline(0.5, color=INK2, lw=0.5, ls=(0, (2, 2)))
        ax.set_xlim(1, 2); ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("temptation $b$"); ax.set_title(title)
    axes[0].set_ylabel("cooperation $\\rho_C$")
    axes[1].set_yticklabels([])
    h_ab = [Line2D([], [], color=TOPO_COLOR[k], marker=TOPO_MARKER[k], ms=3.5,
                   lw=1.4, label=TOPO_LABEL[k]) for k in TOPO]
    h_ab += [Line2D([], [], color=INK2, lw=1.4, label="asynchronous (10%)"),
             Line2D([], [], color=INK2, lw=0.9, ls=(0, (2, 1.5)), label="synchronous")]
    fig.legend(handles=h_ab, loc="upper center", ncol=3, fontsize=6.4,
               bbox_to_anchor=(0.36, -0.03), handlelength=2.0, columnspacing=1.0)
    # (c) finite size: b* vs N
    ax = axes[2]
    fs_res = {}
    for kind in ("lattice", "barabasi_albert", "erdos_renyi"):
        for rule, ls in (("fermi", "-"), ("imitate_best", (0, (3, 1.5)))):
            xs, ys, los, his = [], [], [], []
            for n, d in fs[(fs["net_kind"] == kind) & (fs["rule"] == rule)].groupby("N"):
                r = bootstrap_curve(d)
                fs_res[(kind, rule, n)] = r
                xs.append(n); ys.append(r["bstar"])
                los.append(r["bstar_lo"]); his.append(r["bstar_hi"])
            ax.plot(xs, ys, color=TOPO_COLOR[kind], ls=ls, lw=1.3,
                    marker=TOPO_MARKER[kind], ms=3, mew=0)
    ax.set_xscale("log")
    ax.set_xticks([400, 900, 2500, 6400]); ax.set_xticklabels(["400", "900", "2500", "6400"])
    ax.minorticks_off()
    ax.set_xlabel("population size $N$"); ax.set_ylabel("$b^{\\ast}$")
    ax.set_title("(c) finite-size dependence")
    h = [Line2D([], [], color=INK2, lw=1.3, label="Fermi"),
         Line2D([], [], color=INK2, lw=1.3, ls=(0, (3, 1.5)), label="best-takes-over")]
    ax.legend(handles=h, loc="upper center", fontsize=6.4, handlelength=2.2,
              bbox_to_anchor=(0.5, -0.27), ncol=1)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(FIG / "fig6_robustness.pdf")
    plt.close(fig)

    # async threshold table
    for rule, norm in (("imitate_best", False), ("imitate_best", True), ("fermi", False)):
        for kind in TOPO:
            a = asy[(asy["rule"] == rule) & (asy["normalise_payoff"] == norm) &
                    (asy["net_kind"] == kind)]
            r = bootstrap_curve(a)
            key = (f"async{TOPO_SHORT[kind]}"
                   f"{'Bt' if rule == 'imitate_best' else 'Fe'}{'Av' if norm else 'Ac'}")
            num(key, fmt_b(r["bstar"]))
            num(key.replace("async", "asyncAuc"), r["auc"])
            num(key.replace("async", "asyncMin"), np.floor(float(r["mean"].min()) * 100) / 100)
    sizes = sorted({k[2] for k in fs_res})
    for kind in ("lattice", "barabasi_albert", "erdos_renyi"):
        for rule, rk in (("fermi", "Fe"), ("imitate_best", "Bt")):
            v = [fs_res[(kind, rule, n)]["bstar"] for n in sizes]
            num(f"fsMin{TOPO_SHORT[kind]}{rk}", fmt_b(min(v)))
            num(f"fsMax{TOPO_SHORT[kind]}{rk}", fmt_b(max(v)))
            a_ = [fs_res[(kind, rule, n)]["auc"] for n in sizes]
            num(f"fsAucMin{TOPO_SHORT[kind]}{rk}", min(a_))
            num(f"fsAucMax{TOPO_SHORT[kind]}{rk}", max(a_))
    v = [fs_res[("barabasi_albert", "fermi", n)]["bstar"] for n in sizes if n >= 900]
    num("fsMinBAFeLarge", fmt_b(min(v)))
    num("fsMaxBAFeLarge", fmt_b(max(v)))
    others = [fs_res[(k, "imitate_best", n)]["auc"] for k in ("lattice", "erdos_renyi")
              for n in sizes]
    num("fsAucMinOthersBt", min(others))
    nword = {400: "Four", 900: "Nine", 2500: "TwentyFive", 6400: "SixtyFour"}
    for (kind, rule, n), r in fs_res.items():
        num(f"fs{TOPO_SHORT[kind]}{'Fe' if rule == 'fermi' else 'Bt'}{nword[int(n)]}",
            fmt_b(r["bstar"]))


# =============================================================================
# Figure 7: T-S phase diagrams
# =============================================================================

def fig_phase():
    fig, axes = plt.subplots(1, 4, figsize=(TXT_W, 1.95), sharey=True)
    pd_means = {}
    for ax, kind in zip(axes, TOPO):
        d = pd.read_csv(TAB / f"04_phase_{kind}.csv")
        p = d.pivot(index="S_val", columns="T_val", values="equilibrium")
        T, S = p.columns.to_numpy(), p.index.to_numpy()
        dT, dS = T[1] - T[0], S[1] - S[0]
        im = ax.imshow(p.to_numpy(), origin="lower", cmap=SEQ_CMAP, vmin=0, vmax=1,
                       extent=[T[0] - dT / 2, T[-1] + dT / 2, S[0] - dS / 2, S[-1] + dS / 2],
                       aspect="auto", interpolation="nearest")
        ax.axvline(1, color="white", lw=0.8); ax.axhline(0, color="white", lw=0.8)
        ax.axvline(1, color=INK, lw=0.4, ls=(0, (2, 2)))
        ax.axhline(0, color=INK, lw=0.4, ls=(0, (2, 2)))
        ax.grid(False)
        ax.set_title(TOPO_LABEL[kind]); ax.set_xlabel("$T$")
        q = d[(d["T_val"] > 1) & (d["S_val"] < 0)]
        pd_means[kind] = q["equilibrium"].mean()
        sd = d[(d["T_val"] > 1) & (d["S_val"] > 0)]
        num(f"sd{TOPO_SHORT[kind]}", sd["equilibrium"].mean())
        sh = d[(d["T_val"] < 1) & (d["S_val"] < 0)]
        num(f"sh{TOPO_SHORT[kind]}", sh["equilibrium"].mean())
        for (tx, sy, lab) in [(0.5, 0.5, "H"), (1.5, 0.5, "SD"), (0.5, -0.5, "SH"),
                              (1.5, -0.5, "PD")]:
            ax.text(tx, sy, lab, ha="center", va="center", fontsize=6.3,
                    color=INK, bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                         ec="none", alpha=0.75))
    axes[0].set_ylabel("$S$")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.015)
    cb.set_label("cooperation $\\rho_C$"); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.5)
    fig.savefig(FIG / "fig7_phase.pdf")
    plt.close(fig)
    for k, v in pd_means.items():
        num(f"pd{TOPO_SHORT[k]}", v)


# =============================================================================

def lottery():
    d = pd.read_csv(CTL / "C10_hub_lottery.csv")
    bt, fe = d[d["rule"] == "imitate_best"], d[d["rule"] == "fermi"]
    num("followBt", 100 * bt["top_hub_nb_follow_t1"].mean(), "{:.0f}")
    num("followFe", 100 * fe["top_hub_nb_follow_t1"].mean(), "{:.0f}")
    ext = ((bt["equilibrium"] < 0.05) | (bt["equilibrium"] > 0.95)).mean()
    num("extremeBt", 100 * ext, "{:.0f}")
    corr = [np.corrcoef(g["hub_init_coop_wtd"], g["equilibrium"])[0, 1]
            for _, g in bt.groupby("b")]
    num("lotteryCorrMax", max(corr), "{:.2f}")


def write_numbers():
    lines = ["% Auto-generated by paper_analysis.py -- do not edit by hand."]
    for k, v in sorted(NUM.items()):
        assert k.isalpha(), k
        lines.append(f"\\newcommand{{\\N{k}}}{{{v}}}")
    (OUTDIR / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"  wrote {len(NUM)} macros")


def main():
    print("[fig1]"); fig_snapshots()
    print("[tab1]"); tab_networks()
    print("[phase]"); fig_phase()
    print("[assort]"); tab_assortment()
    print("[hubs]"); fig_hubs()
    if (CTL / "C10_hub_lottery.csv").exists():
        print("[lottery]"); lottery()
    res = None
    if (CTL / "C7_study3_raw.csv").exists():
        print("[factorial]"); res = factorial()
        print("[temperature]"); fig_temperature(res)
    if (CTL / "C6_robustness_erdos_renyi.csv").exists():
        print("[noise]"); tab_noise()
    if (CTL / "C3_extended_survey_raw.csv").exists():
        print("[survey]"); survey()
    if res is not None:
        print("[robustness]"); robustness_fig()
    write_numbers()


if __name__ == "__main__":
    main()
