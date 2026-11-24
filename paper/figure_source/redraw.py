"""Redraw the four paper figures (run: python figure_source/redraw.py)

 at the Springer svmult text width (117 mm)
with readable type.  Data come from the original vector figures (extracted by
ex_fig*.py and checked against numbers.tex)."""
import json
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

W = 117 / 25.4                      # text width in inches
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": "#525250", "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.grid": True, "grid.color": "#e3e3de", "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "savefig.pad_inches": 0.02,
})
C = {"LAT": "#2878d6", "WS": "#1cb07a", "BA": "#eb6933", "ER": "#614fd6"}
NAME = {"LAT": "Lattice", "WS": "Watts–Strogatz", "BA": "Barabási–Albert", "ER": "Erdős–Rényi"}
MK = {"LAT": "s", "WS": "^", "BA": "o", "ER": "D"}
ORDER = ["LAT", "WS", "BA", "ER"]
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figs") + os.sep


def topo_legend(fig, y, ncol=4, extra=()):
    h = [Line2D([], [], color=C[t], marker=MK[t], ms=3.5, lw=1.4, label=NAME[t]) for t in ORDER]
    fig.legend(handles=h + list(extra), loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, y), handlelength=1.8, columnspacing=1.2)


# ---------------------------------------------------------------- Fig. factorial
d = json.load(open(os.path.join(HERE, "factorial_curves.json")))
fig, ax = plt.subplots(2, 3, figsize=(W, 2.42), sharex=True, sharey=True)
rules = [("fermi", "Fermi ($K=0.1$)"), ("bto", "Best-takes-over"), ("pi", "Proportional imitation")]
for i, (conv, cname) in enumerate([("acc", "accumulated"), ("avg", "averaged")]):
    for j, (rule, title) in enumerate(rules):
        a = ax[i, j]
        for t in ORDER:
            s = d[f"{rule}_{conv}"][t]
            b = np.array(s["b"])
            a.fill_between(b, s["lo"], s["hi"], color=C[t], alpha=0.18, lw=0)
            a.plot(b, s["mean"], color=C[t], lw=1.3, marker=MK[t], ms=2.3, markevery=2)
        a.axhline(0.5, color="#525250", lw=0.7, ls=(0, (2, 2)))
        a.set_xlim(0.98, 2.02); a.set_ylim(-0.03, 1.03)
        a.set_xticks([1, 1.25, 1.5, 1.75, 2]); a.set_xticklabels(["1", "1.25", "1.5", "1.75", "2"])
        a.set_yticks([0, 0.25, 0.5, 0.75, 1])
        if i == 0:
            a.set_title(f"({'abc'[j]}) {title}", pad=3)
        else:
            a.set_title(f"({'def'[j]})", pad=3)
        if j == 0:
            a.set_ylabel(f"{cname}\ncooperation $\\rho_C$")
        if i == 1:
            a.set_xlabel("temptation $b$")
fig.subplots_adjust(left=0.155, right=0.99, top=0.920, bottom=0.270, wspace=0.08, hspace=0.32)
topo_legend(fig, 0.0)
fig.savefig(OUT + "fig_factorial.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. temperature
d = json.load(open(os.path.join(HERE, "temperature_curves.json")))
fig, ax = plt.subplots(1, 3, figsize=(W, 1.68), gridspec_kw={"width_ratios": [1, 1, 1.05]})
for k, (key, title) in enumerate([("a", "(a) accum., $K=0.1$"),
                                  ("b", r"(b) avg., $K=0.1/\langle k\rangle$")]):
    a = ax[k]
    for t in ORDER:
        s = d[key][t]
        x = np.array(s["x"])
        a.fill_between(x, s["lo"], s["hi"], color=C[t], alpha=0.18, lw=0)
        a.plot(x, s["y"], color=C[t], lw=1.3, marker=MK[t], ms=2.3, markevery=2)
    a.axhline(0.5, color="#525250", lw=0.7, ls=(0, (2, 2)))
    a.set_xlim(0.98, 2.02); a.set_ylim(-0.03, 1.03)
    a.set_xticks([1, 1.5, 2]); a.set_xticklabels(["1", "1.5", "2"])
    a.set_yticks([0, 0.25, 0.5, 0.75, 1])
    a.set_title(title, pad=3); a.set_xlabel("temptation $b$")
ax[0].set_ylabel(r"cooperation $\rho_C$")
ax[1].set_yticklabels([])
a = ax[2]
for t in ["BA", "ER"]:
    s = d["c"][t]
    a.fill_between(s["x"], s["lo"], s["hi"], color=C[t], alpha=0.18, lw=0)
    a.plot(s["x"], s["y"], color=C[t], lw=1.3, marker=MK[t], ms=3)
lat = d["c"]["LAT"]["dashed"][0][1]
a.axhline(lat, color=C["LAT"], lw=1.1, ls=(0, (3, 2)))
a.set_xlim(-0.04, 1.04); a.set_ylim(0.95, 1.85)
a.set_xticks([0, 0.5, 1]); a.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8])
a.set_title(r"(c) $\pi_i/k_i^{\alpha}$, matched $K$", pad=3)
a.set_xlabel(r"exponent $\alpha$"); a.set_ylabel(r"critical temptation $b^{*}$", labelpad=2)
fig.subplots_adjust(left=0.1, right=0.995, top=0.884, bottom=0.429, wspace=0.42)
pos0, pos1 = ax[0].get_position(), ax[1].get_position()
ax[1].set_position([pos0.x1 + 0.025, pos1.y0, pos1.width, pos1.height])
topo_legend(fig, 0.0)
fig.savefig(OUT + "fig_temperature.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. survey
d = json.load(open(os.path.join(HERE, "survey_points.json")))
FAM = [("Watts–Strogatz", "#1cb07a", "^"), ("Barabási–Albert", "#eb6933", "h"),
       ("Erdős–Rényi", "#614fd6", "D"), ("Lattice", "#2878d6", "s"),
       ("Random regular", "#525250", "v"), ("Chung–Lu (new)", "#eda100", "p"),
       ("Holme–Kim (new)", "#e34a47", "*")]
fig, ax = plt.subplots(1, 3, figsize=(W, 1.74), gridspec_kw={"width_ratios": [1, 1, 1.1]})
for k, (key, title) in enumerate([("a", "(a) accum., $b=1.3$"),
                                  ("b", "(b) matched $K$, $b=1.05$")]):
    a = ax[k]
    for name, col, mk in FAM:
        pts = [(x, y) for f, x, y in d[key] if f == name]
        a.scatter([p[0] for p in pts], [p[1] for p in pts], s=22 if mk == "*" else 15, marker=mk,
                  color=col, edgecolor="white", linewidth=0.4, zorder=3)
    a.set_xlim(-0.12, 2.75); a.set_ylim(-0.03, 1.05)
    a.set_xticks([0, 1, 2]); a.set_yticks([0, 0.25, 0.5, 0.75, 1])
    a.set_title(title, pad=3); a.set_xlabel("heterogeneity $H$")
ax[0].set_ylabel(r"cooperation $\rho_C$")
ax[1].set_yticklabels([])
# panel c (values extracted from the original figure)
acc_b = [1.05, 1.1, 1.2, 1.3, 1.4, 1.6]
acc = [0.427, 0.6285, 0.8276, 0.9215, 0.9779, 0.7666]
acc_lo = [0.0885, 0.3079, 0.599, 0.7432, 0.8324, 0.5843]
acc_hi = [0.7656, 0.9492, 1.0562, 1.0998, 1.1234, 0.9489]
mt_b = [1.05, 1.1, 1.2]
mt = [-0.585, -0.4768, -0.3317]
mt_lo = [-0.8786, -0.8983, -0.6233]
mt_hi = [-0.2913, -0.0554, -0.0401]
a = ax[2]
a.axhline(0, color="#525250", lw=0.7)
a.errorbar(acc_b, acc, yerr=[np.subtract(acc, acc_lo), np.subtract(acc_hi, acc)], color="#2878d6",
           marker="h", ms=3.5, lw=1.3, capsize=2, elinewidth=0.8, label="accumulated")
a.errorbar(mt_b, mt, yerr=[np.subtract(mt, mt_lo), np.subtract(mt_hi, mt)], color="#eb6933",
           marker="D", ms=3, lw=1.3, ls=(0, (4, 1.5)), capsize=2, elinewidth=0.8,
           label="averaged, matched $K$")
a.set_xlim(1.0, 1.65); a.set_ylim(-1.0, 1.2)
a.set_xticks([1.1, 1.3, 1.5]); a.set_yticks([-1, -0.5, 0, 0.5, 1])
a.set_title(r"(c) coefficient on $H$ vs $b$", pad=3)
a.set_xlabel("temptation $b$"); a.set_ylabel("standardised coefficient", labelpad=1)
a.text(1.36, 0.42, "accumulated", color="#2878d6", fontsize=7.5)
a.text(1.25, -0.8, "avg., matched $K$", color="#eb6933", fontsize=7.5)
fig.subplots_adjust(left=0.1, right=0.995, top=0.884, bottom=0.453, wspace=0.42)
pos0, pos1 = ax[0].get_position(), ax[1].get_position()
ax[1].set_position([pos0.x1 + 0.025, pos1.y0, pos1.width, pos1.height])
h = [Line2D([], [], ls="", marker=mk, color=col, mec="white", mew=0.4, ms=5 if mk == "*" else 4.2, label=name)
     for name, col, mk in FAM]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0),
           handletextpad=0.2, columnspacing=1.0)
fig.savefig(OUT + "fig_survey.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Fig. robustness
d = json.load(open(os.path.join(HERE, "robustness_curves.json")))
fig, ax = plt.subplots(1, 3, figsize=(W, 1.76), gridspec_kw={"width_ratios": [1, 1, 1.05]})
for k, (key, title) in enumerate([("a", "(a) best-takes-over"), ("b", "(b) Fermi")]):
    a = ax[k]
    for t in ORDER:
        s = d[key][t]
        sx, sy = zip(*s["solid"]); dx, dy = zip(*s["dashed"])
        a.plot(dx, dy, color=C[t], lw=1.0, ls=(0, (2, 1.5)))
        a.plot(sx, sy, color=C[t], lw=1.4, marker=MK[t], ms=2.6)
    a.set_xlim(0.98, 2.02); a.set_ylim(-0.03, 1.03)
    a.set_xticks([1, 1.5, 2]); a.set_xticklabels(["1", "1.5", "2"]); a.set_yticks([0, 0.25, 0.5, 0.75, 1])
    a.set_title(title, pad=3); a.set_xlabel("temptation $b$")
ax[0].set_ylabel(r"cooperation $\rho_C$")
ax[1].set_yticklabels([])
a = ax[2]
for t in ["LAT", "BA", "ER"]:
    s = d["c"][t]
    sx, sy = zip(*s["solid"]); dx, dy = zip(*s["dashed"])
    a.plot(sx, sy, color=C[t], lw=1.3, marker=MK[t], ms=3)
    a.plot(dx, dy, color=C[t], lw=1.1, ls=(0, (3, 1.8)), marker=MK[t], ms=3, mfc="white")
a.set_xscale("log"); a.set_xlim(330, 7800)
a.set_xticks([400, 900, 2500, 6400]); a.set_xticklabels(["400", "900", "2500", "6400"])
a.minorticks_off()
a.set_ylim(0.98, 1.82); a.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8])
a.set_title("(c) system size", pad=3); a.set_xlabel("population size $N$")
a.set_ylabel(r"critical temptation $b^{*}$", labelpad=2)
fig.subplots_adjust(left=0.1, right=0.995, top=0.883, bottom=0.492, wspace=0.42)
pos0, pos1 = ax[0].get_position(), ax[1].get_position()
ax[1].set_position([pos0.x1 + 0.025, pos1.y0, pos1.width, pos1.height])
topo = [Line2D([], [], color=C[t], marker=MK[t], ms=3, lw=1.3, label=NAME[t]) for t in ORDER]
fig.legend(handles=topo, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.075),
           handlelength=1.8, columnspacing=1.2)
st = [Line2D([], [], color="#525250", lw=1.4, label="(a, b) asynchronous 10%  /  (c) Fermi"),
      Line2D([], [], color="#525250", lw=1.0, ls=(0, (2, 1.5)), label="(a, b) synchronous  /  (c) best-takes-over")]
fig.legend(handles=st, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01),
           handlelength=2.2, columnspacing=1.2, fontsize=7.5)
fig.savefig(OUT + "fig_robustness.pdf")
plt.close(fig)
print("ok")
