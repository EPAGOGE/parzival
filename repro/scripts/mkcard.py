"""1200x630 Open Graph card. This image is what replaces GitHub's octocat in a
link preview on Slack, iMessage, X, LinkedIn."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pathlib

OUT = pathlib.Path("/Users/epagogellc/parzival/docs"); OUT.mkdir(exist_ok=True)
BG, FG, DIM, ACC, GRN = "#0d1117", "#e6edf3", "#9198a1", "#2f81f7", "#3fb950"
fig = plt.figure(figsize=(12, 6.3), dpi=100); fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.set_xlim(0, 12); ax.set_ylim(0, 6.3)

ax.text(0.75, 5.62, "E P A G O G E", fontsize=22, color=FG, family="serif", weight="bold")
ax.plot([0.78, 3.30], [5.40, 5.40], lw=2.2, color=ACC)

ax.text(0.75, 4.72, "The 2D Boussinesq corner blowup profile",
        fontsize=30, color=FG, family="serif", va="top")
ax.text(0.75, 3.86, "An independent computation: the scaling exponent, an eigenvalue-free\n"
                    "stability certificate, and a free-residual test that catches a false root",
        fontsize=15, color=DIM, family="serif", va="top", linespacing=1.6)

ax.add_patch(plt.Rectangle((0.75, 1.00), 4.75, 1.20, facecolor="#161b22",
                           edgecolor="#30363d", lw=1.2))
ax.text(0.98, 1.84, "scaling exponent", fontsize=11, color=DIM, family="serif")
ax.text(0.98, 1.26, r"$\alpha = -0.34240 \pm 4.4\times10^{-5}$", fontsize=20,
        color=ACC, family="serif")
for i, t in enumerate(["third independent method",
                       "every number logged",
                       "retractions included"]):
    ax.text(5.95, 2.02 - i*0.42, "•  " + t, fontsize=12.5, color=DIM, family="serif")

# real data, bottom-right, with the reference line actually in frame
e = np.array([1e-4, 5e-5, 2.5e-5, 1e-5, 5e-6])
a = np.array([-0.34541032, -0.34386167, -0.34312079, -0.34268591, -0.34254247])
sub = fig.add_axes([0.755, 0.165, 0.195, 0.235]); sub.set_facecolor(BG)
sub.plot(e, a, "-o", ms=3.5, lw=1.4, color=ACC)
sub.axhline(-0.34240009, lw=1.1, color=GRN)
sub.set_ylim(-0.3456, -0.34225)          # reference now inside the view
sub.set_xlim(-4e-6, 1.06e-4)
for s in sub.spines.values(): s.set_color("#30363d")
sub.set_xticks([]); sub.set_yticks([])
sub.text(0.05, 0.72, "reference", transform=sub.transAxes, fontsize=7.5, color=GRN)
sub.text(0.04, 0.06, r"$\varepsilon_b \to 0$", transform=sub.transAxes, fontsize=8.5, color="#484f58")

fig.savefig(OUT / "og-card.png", facecolor=BG)
print("og-card.png written")

# square favicon: just the wordmark initial on the accent rule
f2 = plt.figure(figsize=(2, 2), dpi=128); f2.patch.set_facecolor(BG)
a2 = f2.add_axes([0, 0, 1, 1]); a2.set_axis_off(); a2.set_xlim(0, 1); a2.set_ylim(0, 1)
a2.text(.5, .44, "E", fontsize=95, color=FG, family="serif", weight="bold",
        ha="center", va="center")
a2.plot([.22, .78], [.20, .20], lw=6, color=ACC)
f2.savefig(OUT / "favicon.png", facecolor=BG)
print("favicon.png written")
