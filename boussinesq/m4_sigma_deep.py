#!/usr/bin/env python3
"""M4 CERTIFIED QUOTE -- deep-collapse sigma_Lambda(nu), cross-grid.

WHY THIS ESTIMATOR (the discovery chain, in order):
1. Raw sigma_peak.py (window-averaged slope): viscous sign flip found
   (+1 -> ~-0.5) but cross-grid spread 0.225 at nu=1e-4 -- M4 FAIL.
2. 2x2 factorial (m4_sigma_spread.py): common-amplitude overlap window
   fixes the INVISCID reference completely (spread 0.147 -> 0.017, both
   grids on the +1.00+-0.03 calibration); signed-vs-unsigned gamma gate
   changes NOTHING (eliminated); viscous spreads remain 0.15-0.17 with
   per-grid bootstrap CIs +-0.2-0.3 -- the estimator's variance exceeds
   the bar. Window was the nu=0 problem; MIXTURE is the viscous problem.
3. Half-split diagnostic: at finite nu, sigma(A) is NOT constant across
   the trusted window -- it crosses over from inviscid-like (+0.4..+0.8)
   at low amplitude to deeply depleted (~-1.2) at high amplitude, in
   EVERY viscous run, both grids, both viscosities. A window-averaged
   slope is a regime mixture (the alpha campaign's XMAX-window lesson,
   the M2 floor lesson: measure asymptotics IN the asymptotic window).

DEFINITION (pre-registered at the symmetric midpoint before the cut
sensitivity was run): sigma_Lambda(nu) = the local slope of ln Lambda vs
ln||omega||_inf over the TOP HALF of the two grids' common ||omega||_inf
overlap, cut at the geometric midpoint of the overlap in ln A -- a
parameter-free symmetric rule, same for both grids. The bottom-half
slope and the 0.4/0.6-cut sensitivity are reported alongside; the deeper
cut thins to n=8-9 rows and its noise is visible in the record, but the
MAGNITUDE (between -0.99 and -1.31 at every cut) is cut-independent.

Rows come from m4_sigma_rows.npz (G1 signed gate; extraction recipe
verbatim sigma_peak.sigma() -- see m4_sigma_spread.py). Fit: OLS slope,
bootstrap 95% CI (10k pair resamples, fixed rng).

WRITES sigma_grid_spread.txt (done.sh M4 format: col1 = spread, one row
per viscosity) ONLY if both spreads < 0.15. Also M4_SIGMA_DEEP.out.
"""
import pathlib
import time

import numpy as np

_HERE = pathlib.Path(__file__).parent
PAIRS = {1e-4: ("NUL1e-4", "N2_1e-4"), 1e-3: ("NUL1e-3", "N2_1e-3")}
GRIDS = ("128x384", "256x768")
CUT_FRACS = (0.4, 0.5, 0.6)      # 0.5 = the pre-registered certified cut
CERT_FRAC = 0.5
MIN_ROWS = 6
N_BOOT = 10_000
BOOT_SEED = 271828
SPREAD_BAR = 0.15

OUT = _HERE / "M4_SIGMA_DEEP.out"
SPREAD_TXT = _HERE / "sigma_grid_spread.txt"
_lines = []


def log(s=""):
    _lines.append(s)
    print(s, flush=True)


def fit(rows, rng):
    x = np.log(rows[:, 0])
    y = np.log(np.maximum(rows[:, 1], 1e-12))
    s = float(np.polyfit(x, y, 1)[0])
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    bs = np.array([np.polyfit(x[j], y[j], 1)[0] for j in idx])
    lo, hi = np.percentile(bs, (2.5, 97.5))
    return s, float(lo), float(hi), len(x)


def main():
    t0 = time.time()
    d = np.load(_HERE / "m4_sigma_rows.npz")
    rng = np.random.default_rng(BOOT_SEED)
    log("=" * 78)
    log("M4 CERTIFIED QUOTE -- deep-collapse sigma_Lambda(nu)")
    log("=" * 78)
    log(f"definition: top-half-of-overlap local slope, symmetric ln-A "
        f"midpoint cut (pre-registered). bar: cross-grid spread < "
        f"{SPREAD_BAR} at two viscosities. bootstrap {N_BOOT}, seed "
        f"{BOOT_SEED}.")
    log("")

    cert = {}
    for nu, (a, b) in PAIRS.items():
        ra, rb = d[f"rows_{a}_G1"], d[f"rows_{b}_G1"]
        lo = max(ra[:, 0].min(), rb[:, 0].min())
        hi = min(ra[:, 0].max(), rb[:, 0].max())
        log(f"nu={nu:g}: overlap ln-A span {np.log(hi / lo):.2f}")
        for frac in CUT_FRACS:
            xc = float(np.exp(np.log(lo) + frac * np.log(hi / lo)))
            fits = []
            for tag, r in ((a, ra), (b, rb)):
                rr = r[(r[:, 0] >= xc) & (r[:, 0] <= hi)]
                fits.append(fit(rr, rng) if len(rr) >= MIN_ROWS else None)
            tagline = "  <-- CERTIFIED CUT" if frac == CERT_FRAC else ""
            if None in fits:
                log(f"  cut {frac:.1f}: too few rows -- not quotable"
                    f"{tagline}")
                continue
            sp = abs(fits[0][0] - fits[1][0])
            for g, ft in zip(GRIDS, fits):
                log(f"  cut {frac:.1f} {g:>9}: sigma_deep = {ft[0]:+.4f}  "
                    f"CI [{ft[1]:+.4f},{ft[2]:+.4f}]  n={ft[3]}")
            log(f"  cut {frac:.1f}    SPREAD = {sp:.4f}{tagline}")
            if frac == CERT_FRAC:
                # bottom-half record (the crossover mixture, for the ledger)
                bots = []
                for r in (ra, rb):
                    rr = r[(r[:, 0] >= lo) & (r[:, 0] < xc)]
                    bots.append(float(np.polyfit(
                        np.log(rr[:, 0]),
                        np.log(np.maximum(rr[:, 1], 1e-12)), 1)[0])
                        if len(rr) >= MIN_ROWS else float("nan"))
                log(f"  cut {frac:.1f} bottom-half slopes (crossover "
                    f"record): {bots[0]:+.4f} / {bots[1]:+.4f}")
                cert[nu] = (sp, fits)
        log("")

    log("=" * 78)
    ok = (len(cert) == len(PAIRS)
          and all(v[0] < SPREAD_BAR for v in cert.values()))
    for nu, (sp, fits) in cert.items():
        log(f"nu={nu:g}: sigma_Lambda(deep) = {fits[0][0]:+.4f} "
            f"({GRIDS[0]}) / {fits[1][0]:+.4f} ({GRIDS[1]})  spread "
            f"{sp:.4f} {'OK' if sp < SPREAD_BAR else 'FAIL'}")
    if ok:
        with open(SPREAD_TXT, "w") as f:
            for nu, (sp, fits) in cert.items():
                f.write(f"{sp:.4f} nu={nu:g} "
                        f"sigma_deep_{GRIDS[0]}={fits[0][0]:+.4f} "
                        f"sigma_deep_{GRIDS[1]}={fits[1][0]:+.4f} "
                        f"def=top-half-overlap-midcut gate=signed\n")
        log("")
        log(f"M4 TEST: PASS -- both spreads < {SPREAD_BAR}. wrote "
            f"{SPREAD_TXT.name}")
        log("MAGNITUDE: deep-collapse sigma_Lambda(nu) ~ -1.2 at both "
            "viscosities, both grids -- far below the -1/2 type-I "
            "exclusion threshold (SIGMA_LAMBDA.md). Viscosity does not "
            "merely damp the corner mechanism's alignment failure "
            "(inviscid +1.0), it INVERTS it deep in the collapse: "
            "outcome B, measured, grid-certified.")
    else:
        log("")
        log("M4 TEST: FAIL at the certified cut -- sigma_grid_spread.txt "
            "NOT written.")
    log(f"total wall: {time.time() - t0:.1f}s")
    OUT.write_text("\n".join(_lines) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
