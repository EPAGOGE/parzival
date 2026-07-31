#!/usr/bin/env python
"""Render the gCLM full-simulation phase portrait from runs/full_map.json."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
BLUE, ORANGE, INK, MUT = "#2a78d6", "#c25e00", "#1a1a19", "#757570"

m = json.load(open(HERE / "runs" / "full_map.json"))
rows = [r for r in m["rows"] if r["Astar_est"]]
a = np.array([r["a"] for r in rows])
astar = np.array([r["Astar_est"] for r in rows])
omega = np.array([r["omega_edge"] or np.nan for r in rows])

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(9, 11.5), height_ratios=[3.0, 2.4, 1.2],
    constrained_layout=True)
fig.suptitle("gCLM phase portrait at $\\nu=1$ ($\\nu$ is exact gauge)",
             fontsize=14, x=0.5)

# ---- Panel A: fate boundary + edge branch + pole law
for r in rows:
    br = r["Astar_fp64_bracket"]
    if br:
        ax1.plot([r["a"], r["a"]], br, color=BLUE, lw=3, alpha=0.45,
                 solid_capstyle="butt", zorder=2)
ax1.plot(a, astar, "-o", color=BLUE, lw=2, ms=5, zorder=3)
sel = a >= 0.32
ax1.plot(a[sel], omega[sel], "-o", color=ORANGE, lw=2, ms=5, zorder=3)
ax1.plot(a[~sel], omega[~sel], "o", color=ORANGE, ms=5, alpha=0.35, zorder=2)
up = a >= 0.9
Cfit = float(np.exp(np.mean(np.log(omega[up]) + np.log(1 - a[up]))))
aa = np.linspace(0.35, 0.985, 300)
ax1.plot(aa, Cfit / (1 - aa), "--", color=MUT, lw=1.4, zorder=1)
ax1.axvspan(-0.02, 0.32, color=MUT, alpha=0.08, zorder=0)
ax1.axvline(1.0, color=MUT, lw=1, ls=":", zorder=1)
ax1.set_yscale("log")
ax1.set_xlim(-0.02, 1.03)
ax1.set_ylim(3, 130)
ax1.set_ylabel("amplitude  (sup norm)")
ax1.text(0.755, 41, "fate boundary $A^*(a)$\nfp32 swarm + fp64 brackets",
         color=BLUE, fontsize=9.5, ha="right")
ax1.text(0.93, 12.5, "edge state $\\Omega(a)$\n(unstable rel. equilibrium)",
         color=ORANGE, fontsize=9.5, ha="right")
ax1.text(0.80, 85, f"$\\Omega \\sim {Cfit:.2f}\\,\\nu/(1{{-}}a)$",
         color=INK, fontsize=10, ha="center")
ax1.text(0.15, 60, "branch exchange:\ntracked branch $\\neq$ edge",
         color=MUT, fontsize=9, ha="center")
ax1.text(1.0, 3.6, " DG: total depletion", color=MUT, fontsize=9, rotation=90,
         va="bottom")
ax1.set_title("fate boundary and the organizing edge state", fontsize=11,
              loc="left")

# ---- Panel B: hover-rate surface in (a, A/A*)
ybins = np.linspace(0.5, 1.9, 57)
yc = 0.5 * (ybins[:-1] + ybins[1:])
surf = np.full((len(yc), len(rows)), np.nan)
for j, r in enumerate(rows):
    c = r["cells"]
    cent = np.array(c["centers"]) / r["Astar_est"]
    n = np.array(c["n"], float)
    h = np.array(c["hover"], float)
    tot, hov = np.zeros(len(yc)), np.zeros(len(yc))
    idx = np.clip(np.searchsorted(ybins, cent) - 1, 0, len(yc) - 1)
    ok = (cent >= ybins[0]) & (cent <= ybins[-1])
    np.add.at(tot, idx[ok], (n + h)[ok])
    np.add.at(hov, idx[ok], h[ok])
    with np.errstate(invalid="ignore", divide="ignore"):
        col = np.where(tot >= 20, hov / np.maximum(tot, 1), np.nan)
    surf[:, j] = col
ae = np.concatenate([[a[0] - 0.02], 0.5 * (a[1:] + a[:-1]), [a[-1] + 0.005]])
pc = ax2.pcolormesh(ae, ybins, surf, cmap="Blues", vmin=0,
                    vmax=np.nanquantile(surf, 0.995), shading="flat")
ax2.axhline(1.0, color=MUT, lw=1.2, ls="--")
ax2.text(0.02, 1.03, "fate boundary", color=MUT, fontsize=9)
ax2.set_xlim(-0.02, 1.03)
ax2.set_ylabel("$A\\,/\\,A^*(a)$")
ax2.set_title("hover rate per resolved lane  (lingering near the edge state)",
              fontsize=11, loc="left")
cb = fig.colorbar(pc, ax=ax2, pad=0.01)
cb.set_label("hover rate", fontsize=9)

# ---- Panel C: spectral character of blowup (trust-wire flag fraction)
frac = np.array([r["lowtrust"] / max(r["blowups"], 1) for r in rows]).clip(0, 1)
ax3.plot(a, frac, "-o", color=BLUE, lw=2, ms=4)
ax3.set_xlim(-0.02, 1.03)
ax3.set_ylim(-0.05, 1.1)
ax3.set_ylabel("flagged fraction")
ax3.set_xlabel("advection dial  $a$   (CLM $=0$, De Gregorio $=1$)")
ax3.set_title("spectral character: fraction of blowups tripping the tail "
              "trust wire", fontsize=11, loc="left")
ax3.text(0.65, 0.12, "depletion keeps mid-dial\nblowups spectrally clean",
         color=INK, fontsize=9, ha="center")

for ax in (ax1, ax2, ax3):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=MUT, alpha=0.18, lw=0.6)
fig.text(0.995, 0.002,
         "era: fp32 swarm N=128 cos2 ic, fp64 anchors + branch | single seed | "
         "6.66M fates, 24 points, 85.7 min on M1 Pro | vault 0194058a",
         fontsize=7, color=MUT, ha="right")
out = HERE / "runs" / "phase_portrait.png"
fig.savefig(out, dpi=170)
print(out)
