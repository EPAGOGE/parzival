"""Figures 1-3 for the note. Data-only: every point is a logged measurement, no
smoothing, no invented values. Written to ~/parzival/note/fig/."""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

OUT = pathlib.Path("/Users/epagogellc/parzival/note/fig"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25,
                     "figure.dpi": 160, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
REF = -0.34240009

# ---- Fig 1: the eps_b ladder and the three extrapolations -------------------
e = np.array([1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6])
a = np.array([-0.34541032, -0.34386167, -0.34312079, -0.34268591, -0.34254247])
lin, quad = np.polyfit(e, a, 1)[-1], np.polyfit(e, a, 2)[-1]
f = lambda x, p0, p1, p2: p0 + p1*x + p2*x*np.log(x)
p, _ = curve_fit(f, e, a, p0=[-0.3424, -30, -3], maxfev=40000)
xs = np.linspace(0, 1.05e-4, 400); xs[0] = 1e-12
fig, (ax, axz) = plt.subplots(1, 2, figsize=(7.0, 3.1),
                              gridspec_kw=dict(width_ratios=[1.35, 1], wspace=.28))
def draw(A):
    A.plot(xs, np.polyval(np.polyfit(e, a, 1), xs), lw=1.0, ls="--", color="#888")
    A.plot(xs, np.polyval(np.polyfit(e, a, 2), xs), lw=1.2, color="#1f6fb4")
    A.plot(xs, f(xs, *p), lw=1.0, ls=":", color="#b4501f")
    A.axhline(REF, color="#2a9d4a", lw=.9)
    A.plot(e, a, "o", ms=4.5, color="k", zorder=5)
draw(ax); draw(axz)
for h, lab, st in ((None, "linear", dict(ls="--", lw=1.0, color="#888")),
                   (None, "quadratic", dict(lw=1.2, color="#1f6fb4")),
                   (None, r"$\varepsilon\ln\varepsilon$", dict(ls=":", lw=1.0, color="#b4501f")),
                   (None, "reference", dict(lw=.9, color="#2a9d4a"))):
    ax.plot([], [], label=lab, **st)
ax.plot([], [], "o", ms=4.5, color="k", label="computed rungs")
ax.set_xlabel(r"wedge truncation $\varepsilon_b$"); ax.set_ylabel(r"$\alpha$")
ax.set_title("(a)  the five-rung ladder", fontsize=9, loc="left")
ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.set_xlim(-4e-6, 1.06e-4)
ax.set_xticks([0, 2.5e-5, 5e-5, 7.5e-5, 1e-4])
ax.set_xticklabels(["0", r"$2.5{\times}10^{-5}$", r"$5{\times}10^{-5}$",
                    r"$7.5{\times}10^{-5}$", r"$10^{-4}$"], fontsize=7)

for v, c in ((lin, "#888"), (quad, "#1f6fb4"), (p[0], "#b4501f")):
    axz.plot(0, v, "s", ms=5, color=c, zorder=6)
axz.set_xlim(-1.2e-6, 6e-6); axz.set_ylim(-0.342620, -0.342330)
axz.set_xticks([0, 2.5e-6, 5e-6]); axz.set_xticklabels(["0", "2.5e-6", "5e-6"], fontsize=7)
axz.tick_params(labelsize=7)
axz.set_xlabel(r"$\varepsilon_b$", fontsize=8)
axz.set_title(r"(b)  the limits: spread $3.93\times10^{-5}$", fontsize=9, loc="left")
axz.annotate(f"reference\n{REF}", (5.6e-6, REF), fontsize=6.5, color="#2a9d4a",
             ha="right", va="bottom")
axz.annotate("quadratic  $-0.34240048$", (5.6e-6, quad), fontsize=6.5, color="#1f6fb4",
             ha="right", va="top")
fig.savefig(OUT/"fig1_ladder.pdf"); fig.savefig(OUT/"fig1_ladder.png"); plt.close(fig)

# ---- Fig 2: the free residual separates the true root from the ghost --------
gh_deg = np.array([16, 24, 28]); gh_h = np.array([2.3856, 0.99419, 0.88016])
gr_h = 1.0605e-3
fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.semilogy(gh_deg, gh_h, "o-", ms=5, lw=1.2, color="#c0392b", label="false root (Sec. 7)")
ax.semilogy(gh_deg, np.full(3, gr_h), "s--", ms=5, lw=1.2, color="#1f6fb4",
            label="converged profile")
ax.fill_between([15, 29], 1e-4, 1e-2, color="#1f6fb4", alpha=.06)
ax.annotate("", xy=(21, 0.99), xytext=(21, 1.06e-3),
            arrowprops=dict(arrowstyle="<->", lw=.8, color="#444"))
ax.annotate("three orders of magnitude,\non a functional the solve\nis answerable to nothing for",
            (21.4, 3.2e-2), fontsize=7.5, ha="left", va="center", color="#444")
ax.set_xlim(15, 29); ax.set_ylim(3e-4, 12)
ax.set_xlabel("corner panel degree"); ax.set_ylabel(r"$|h_{\mathrm{id}}| = |c_\ell - 2\Theta_{xx}/W_x|$")
ax.set_title("The free residual, on a state that passed every other test", fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)
fig.savefig(OUT/"fig2_free_residual.pdf"); fig.savefig(OUT/"fig2_free_residual.png"); plt.close(fig)

# ---- Fig 3: the false root drifts, the true one does not -------------------
fig, ax = plt.subplots(figsize=(5.0, 3.4))
gh_a = np.array([-0.42172919, -0.42554621, -0.43083651])
ax.plot(gh_deg, gh_a, "o-", ms=5, lw=1.2, color="#c0392b", label="false root")
for y, lab, c in ((-0.4168236, r"$\alpha_1$", "#666"), (-0.4439811, r"$\alpha_2$", "#666"),
                  (-0.4578230, r"$\alpha_3$", "#666")):
    ax.axhline(y, lw=.7, ls=":", color=c)
    ax.annotate(lab, (28.7, y), fontsize=7.5, color=c, va="center")
for i in range(2):
    ax.annotate(f"{gh_a[i+1]-gh_a[i]:+.1e}", (gh_deg[i]+4, (gh_a[i]+gh_a[i+1])/2),
                fontsize=7, color="#c0392b", ha="center")
ax.set_xlim(15, 29.6); ax.set_xlabel("corner panel degree"); ax.set_ylabel(r"$\alpha$")
ax.set_title("Refinement moves it away from $\\alpha_1$, with growing steps", fontsize=9)
ax.legend(fontsize=7.5, frameon=False, loc="lower left")
fig.savefig(OUT/"fig3_drift.pdf"); fig.savefig(OUT/"fig3_drift.png"); plt.close(fig)

print("figures written:")
for f_ in sorted(OUT.glob("*")): print(f"   {f_.name:28s} {f_.stat().st_size/1024:6.1f} KB")
