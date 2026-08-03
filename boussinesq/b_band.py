#!/usr/bin/env python3
"""B-TEST ANALYSIS -- EJA #77 discriminating test: vary A0, watch the band.

Pre-registered in NU_CURVE_SPEC.md (B-TEST section) BEFORE the runs
existed. Band estimator is SPLIT-FREE (the chord A_c died by split
tracking): sliding-window local slope sigma_loc over k consecutive rows in
ln A; band center A* = first zero crossing from above (linear interp
between window centers). k = 11/15/19; k-drift beyond cross-grid spread =
unanswerable.

Hypotheses (fixed): H_IC A*(A0)/A*(100) = A0/100; H_intrinsic ratio = 1.
Verdict needs BOTH grids agreeing at BOTH A0 values; intermediate scaling
reported as measured power, not forced.

Baseline A0=100: the C-tag nu=1e-4 rows already saved in
nu_curve_data.npz. B-tags extracted fresh (G1 gate, verbatim recipe).
Outputs: B_BAND.out, b_band_data.npz.
"""
import pathlib
import time

import numpy as np

from m4_sigma_spread import extract_rows

_HERE = pathlib.Path(__file__).parent
KS = (11, 15, 19)
CASES = {50: ("B128_A50", "B256_A50"),
         100: ("C128_1e-4", "C256_1e-4"),      # baseline, from npz
         200: ("B128_A200", "B256_A200")}
GRIDS = ("128x384", "256x768")

OUT = _HERE / "B_BAND.out"
NPZ = _HERE / "b_band_data.npz"
_lines = []


def log(s=""):
    _lines.append(s)
    print(s, flush=True)


def local_slope_zero(rows, k):
    """First zero crossing (from above) of the k-row sliding local slope
    in ln A. Returns A* or None."""
    order = np.argsort(rows[:, 0])
    x = np.log(rows[order, 0])
    y = np.log(np.maximum(rows[order, 1], 1e-12))
    if len(x) < k + 2:
        return None
    centers, slopes = [], []
    for i in range(len(x) - k + 1):
        centers.append(np.mean(x[i:i + k]))
        slopes.append(np.polyfit(x[i:i + k], y[i:i + k], 1)[0])
    centers, slopes = np.array(centers), np.array(slopes)
    for i in range(len(slopes) - 1):
        if slopes[i] > 0 and slopes[i + 1] <= 0:
            t = slopes[i] / (slopes[i] - slopes[i + 1])
            return float(np.exp(centers[i] + t * (centers[i + 1] - centers[i])))
    return None


def main():
    t0 = time.time()
    log("=" * 78)
    log("B-TEST -- vary A0 at nu=1e-4, watch the crossover band (EJA #77)")
    log("=" * 78)
    log("estimator: sliding-window local-slope zero crossing, k=11/15/19 "
        "(split-free; k-drift is the invariance check). "
        "H_IC: A* scales with A0. H_intrinsic: A* fixed.")
    log("")

    base = np.load(_HERE / "nu_curve_data.npz")
    stars = {}          # (A0, grid) -> {k: A*}
    save = {}
    for A0, tags in CASES.items():
        log("-" * 78)
        log(f"A0 = {A0}   ({tags[0]} / {tags[1]})")
        for gi, tag in enumerate(tags):
            if tag.startswith("C"):
                rows = base[f"rows_{tag}"]
            else:
                try:
                    rows = extract_rows(tag, signed=True)
                except FileNotFoundError:
                    log(f"  {tag}: MISSING -- skipped")
                    continue
            save[f"rows_{tag}"] = rows
            if len(rows) < max(KS) + 2:
                log(f"  {tag}: only {len(rows)} rows -- unanswerable")
                continue
            amin, amax = rows[:, 0].min(), rows[:, 0].max()
            ks = {}
            for k in KS:
                ks[k] = local_slope_zero(rows, k)
            stars[(A0, gi)] = ks
            vals = " ".join(f"k={k}: " + (f"{v:.0f}" if v else "none")
                            for k, v in ks.items())
            log(f"  {tag}: rows={len(rows)}  A-range [{amin:.0f}, "
                f"{amax:.0f}]  A* {{{vals}}}")

    log("")
    log("=" * 78)
    log("DISCRIMINATION TABLE (A* at k=15; k-spread as invariance check)")
    log("=" * 78)
    log(f"{'A0':>5} {'grid':>9} {'A*(k=15)':>9} {'k-spread':>9} "
        f"{'ratio vs A0=100':>16} {'H_IC predicts':>14}")
    ratios = []
    for A0 in (50, 100, 200):
        for gi, g in enumerate(GRIDS):
            ks = stars.get((A0, gi))
            if not ks or ks[15] is None:
                log(f"{A0:>5} {g:>9} {'--':>9}")
                continue
            valid = [v for v in ks.values() if v]
            kspread = (max(valid) - min(valid)) / min(valid) if len(valid) > 1 else float("nan")
            b = stars.get((100, gi), {}).get(15)
            ratio = ks[15] / b if b else float("nan")
            if A0 != 100 and b:
                ratios.append((A0, ratio))
            log(f"{A0:>5} {g:>9} {ks[15]:>9.0f} {kspread:>8.1%} "
                f"{ratio:>16.2f} {A0/100:>14.2f}")

    log("")
    if len(ratios) == 4:
        lr = np.array([[np.log(a / 100), np.log(r)] for a, r in ratios])
        power = float(np.polyfit(lr[:, 0], lr[:, 1], 1)[0])
        log(f"measured scaling power: A* ~ A0^{power:+.2f}   "
            f"(H_IC predicts +1, H_intrinsic predicts 0)")
        near_ic = all(abs(r - a / 100) / (a / 100) < 0.35 for a, r in ratios)
        near_in = all(abs(r - 1.0) < 0.35 for a, r in ratios)
        if near_ic and not near_in:
            log("VERDICT: H_IC -- the band tracks the IC amplitude (Euler "
                "heritage; the amplitude symmetry carries it).")
        elif near_in and not near_ic:
            log("VERDICT: H_intrinsic -- the band does NOT move with A0; "
                "it is set by something internal (r0 next, EJA #77b).")
        else:
            log("VERDICT: NEITHER clean hypothesis -- intermediate/mixed "
                "scaling; reported as measured, not forced.")
    else:
        log(f"VERDICT: UNANSWERABLE -- only {len(ratios)}/4 ratio "
            f"measurements available.")

    np.savez(NPZ, **save)
    log("")
    log(f"data saved: {NPZ.name}")
    log(f"total wall: {time.time() - t0:.1f}s")
    OUT.write_text("\n".join(_lines) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
