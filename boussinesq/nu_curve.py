#!/usr/bin/env python3
"""NU-CURVE ANALYSIS -- sigma_Lambda(nu) at five viscosities + A_c(nu).

Implements NU_CURVE_SPEC.md's pre-registered analysis VERBATIM. Written
BEFORE any C-tag run completed (2026-08-02): the estimator cannot have been
tuned to the data because the data did not exist.

1. Per nu: the certified deep-collapse estimator, unchanged (G1 signed
   circulation-drift gate, sigma_peak row recipe via
   m4_sigma_spread.extract_rows, cross-grid overlap, top half, symmetric
   midpoint cut, OLS + 10k pair bootstrap, rng 271828). Spread bar 0.15
   per point.
2. A_c(nu): per run, lower-window line (rows below the cross-grid overlap
   midpoint) vs upper-window line (rows above); A_c = intersection.
   UNANSWERABLE if slope gap < 0.5 (near-parallel), reported not forced.
3. Curve: sigma_deep(nu) table + nu=0 matched anchor. ln A_c vs ln nu fit
   ONLY if >= 4 answerable points.
4. Replication: C-tags at 1e-3/1e-4 vs certified M4 values; agree within
   max(spread, CI half-width) = replicated, else a tension is due.

Outputs: NU_CURVE.out, nu_curve_data.npz. No existing file modified.
"""
import pathlib
import time

import numpy as np

from m4_sigma_spread import extract_rows, fit_sigma

_HERE = pathlib.Path(__file__).parent
NU_TAGS = ["1e-3", "3e-4", "1e-4", "3e-5", "1e-5"]   # literal run tags
NUS = [float(t) for t in NU_TAGS]
PAIRS = {float(t): (f"C128_{t}", f"C256_{t}") for t in NU_TAGS}
GRIDS = ("128x384", "256x768")
SPREAD_BAR = 0.15
SLOPE_GAP_MIN = 0.5
BOOT_SEED = 271828
M4_CERT = {1e-4: (-1.1164, -1.2371), 1e-3: (-1.2538, -1.2854)}
M4_HALFW = {1e-4: (0.304, 0.137), 1e-3: (0.216, 0.158)}
NU0_ANCHOR = (+0.5894, +0.5848)   # M4_SIGMA_DEEP_NU0.out

OUT = _HERE / "NU_CURVE.out"
NPZ = _HERE / "nu_curve_data.npz"
_lines = []


def log(s=""):
    _lines.append(s)
    print(s, flush=True)


def a_c_of(rows, xc):
    """Pre-registered A_c: intersection of lower/upper window lines."""
    lo_rows = rows[rows[:, 0] < xc]
    hi_rows = rows[rows[:, 0] >= xc]
    if len(lo_rows) < 6 or len(hi_rows) < 6:
        return None, "too few rows in a window"
    a1, b1 = np.polyfit(np.log(lo_rows[:, 0]),
                        np.log(np.maximum(lo_rows[:, 1], 1e-12)), 1)
    a2, b2 = np.polyfit(np.log(hi_rows[:, 0]),
                        np.log(np.maximum(hi_rows[:, 1], 1e-12)), 1)
    if abs(a1 - a2) < SLOPE_GAP_MIN:
        return None, f"near-parallel (slope gap {abs(a1 - a2):.2f} < {SLOPE_GAP_MIN})"
    return float(np.exp((b2 - b1) / (a1 - a2))), None


def main():
    t0 = time.time()
    log("=" * 78)
    log("NU-CURVE -- sigma_Lambda(nu) + A_c(nu), pre-registered analysis")
    log("=" * 78)
    log(f"spec: NU_CURVE_SPEC.md  estimator: certified (unchanged)  "
        f"bar: {SPREAD_BAR}/point  bootstrap rng {BOOT_SEED}")
    log("")

    rng = np.random.default_rng(BOOT_SEED)
    curve = {}
    acs = {}
    save = {}
    for nu in NUS:
        a, b = PAIRS[nu]
        log("-" * 78)
        log(f"nu = {nu:g}   ({a} / {b})")
        try:
            ra = extract_rows(a, signed=True)
            rb = extract_rows(b, signed=True)
        except FileNotFoundError as e:
            log(f"  MISSING RUN: {e} -- skipped")
            continue
        if len(ra) < 6 or len(rb) < 6:
            log(f"  too few gated rows ({len(ra)}/{len(rb)}) -- UNANSWERABLE")
            continue
        save[f"rows_{a}"] = ra
        save[f"rows_{b}"] = rb
        lo = max(ra[:, 0].min(), rb[:, 0].min())
        hi = min(ra[:, 0].max(), rb[:, 0].max())
        xc = float(np.exp(0.5 * (np.log(lo) + np.log(hi))))
        log(f"  rows {len(ra)}/{len(rb)}  overlap lnA span "
            f"{np.log(hi / lo):.2f}  midpoint A={xc:.5g}")
        fits = []
        for tag, r in ((a, ra), (b, rb)):
            top = r[(r[:, 0] >= xc) & (r[:, 0] <= hi)]
            fits.append(fit_sigma(top, rng) if len(top) >= 6 else None)
        if None in fits:
            log("  too few top-half rows -- sigma_deep UNANSWERABLE")
        else:
            sp = abs(fits[0][0] - fits[1][0])
            for g, ft in zip(GRIDS, fits):
                log(f"  sigma_deep {g}: {ft[0]:+.4f}  "
                    f"CI [{ft[1]:+.4f},{ft[2]:+.4f}]  n={ft[3]}")
            log(f"  SPREAD = {sp:.4f}  "
                f"{'< 0.15 OK' if sp < SPREAD_BAR else '>= 0.15 NOT CERTIFIED'}")
            curve[nu] = (fits, sp)
        pair_ac = []
        for tag, r in ((a, ra), (b, rb)):
            rr = r[(r[:, 0] >= lo) & (r[:, 0] <= hi)]
            ac, why = a_c_of(rr, xc)
            if ac is None:
                log(f"  A_c {tag}: UNANSWERABLE ({why})")
            else:
                log(f"  A_c {tag}: {ac:.5g}")
                pair_ac.append(ac)
        if len(pair_ac) == 2:
            acs[nu] = tuple(pair_ac)

    log("")
    log("=" * 78)
    log("CURVE TABLE (sigma_deep per grid; nu=0 anchor from "
        "M4_SIGMA_DEEP_NU0.out)")
    log("=" * 78)
    log(f"{'nu':>8} {'128x384':>10} {'256x768':>10} {'spread':>8} {'bar':>6}")
    log(f"{'0':>8} {NU0_ANCHOR[0]:>+10.4f} {NU0_ANCHOR[1]:>+10.4f} "
        f"{abs(NU0_ANCHOR[0] - NU0_ANCHOR[1]):>8.4f} {'':>6}")
    for nu in sorted(curve, reverse=True):
        fits, sp = curve[nu]
        log(f"{nu:>8g} {fits[0][0]:>+10.4f} {fits[1][0]:>+10.4f} "
            f"{sp:>8.4f} {'OK' if sp < SPREAD_BAR else 'OVER':>6}")

    log("")
    log("REPLICATION CHECK vs certified M4 (new cadence, independent runs):")
    for nu, cert_vals in M4_CERT.items():
        if nu not in curve:
            log(f"  nu={nu:g}: C-run not available")
            continue
        fits, _sp = curve[nu]
        for i, g in enumerate(GRIDS):
            new = fits[i][0]
            old = cert_vals[i]
            tol = max(abs(fits[i][2] - fits[i][1]) / 2, M4_HALFW[nu][i])
            ok = abs(new - old) <= tol
            log(f"  nu={nu:g} {g}: new {new:+.4f} vs cert {old:+.4f}  "
                f"|delta|={abs(new - old):.4f}  tol={tol:.3f}  "
                f"{'REPLICATED' if ok else 'DISAGREES -- mint a tension'}")

    if acs:
        log("")
        log("A_c(nu) (both grids answerable only):")
        for nu in sorted(acs, reverse=True):
            log(f"  nu={nu:g}: A_c = {acs[nu][0]:.5g} / {acs[nu][1]:.5g}")
        if len(acs) >= 4:
            lnu = np.log(sorted(acs))
            lac = np.log([np.mean(acs[n]) for n in sorted(acs)])
            sl = np.polyfit(lnu, lac, 1)[0]
            log(f"  ln A_c vs ln nu slope = {sl:+.3f}  "
                f"({len(acs)} points, pre-registered >= 4 met)")
        else:
            log(f"  {len(acs)} answerable points < 4 -- no scaling fit "
                f"(pre-registered rule), table only")

    np.savez(NPZ, **save, nus=np.array(sorted(curve)),
             boot_seed=BOOT_SEED)
    log("")
    log(f"data saved: {NPZ.name}")
    log(f"total wall: {time.time() - t0:.1f}s")
    OUT.write_text("\n".join(_lines) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
