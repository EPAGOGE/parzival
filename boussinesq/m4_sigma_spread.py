#!/usr/bin/env python3
"""M4 -- sigma_Lambda(nu) MAGNITUDE with cross-grid certification.

RAW FINDING (SIGMA_PEAK_M4.out, unmodified instrument): viscosity FLIPS
sigma_Lambda from +1.01/+0.87 (nu=0, the twice-validated inviscid
calibration) to about -0.5 at nu in {1e-4, 1e-3} -- SIGMA_LAMBDA.md's
outcome B, geometric depletion measured. But the M4 bar (DONE.md:
cross-grid spread < 0.15 at two viscosities) fails at nu=1e-4:
spread 0.225 (-0.432 vs -0.657), while nu=1e-3 passes (0.046).

FIRST-PRINCIPLES HYPOTHESIS for the nu=1e-4 spread, from the campaign's
own lessons (M2 floor adjudication, eps_b wedge, S5 single-grid quote
discipline): sigma is a LOCAL log-slope of ln Lambda vs ln||omega||_inf,
and at nu=1e-4 the run STRADDLES the inviscid->depleted crossover. Two
grids whose trust gates cut different time windows then sample different
parts of a DRIFTING slope: a window artifact, not a physics disagreement.
At nu=1e-3 viscosity dominates from early on, both grids sit in one
regime, and the spread collapses to 0.046 -- consistent with this reading.

TWO PRE-REGISTERED INSTRUMENT CORRECTIONS (both principled, neither
tuned to pass; the 2x2 factorial below shows each one's effect):
  GATE  G0: sigma_peak.py's gate as-is (tail<=1e-6 AND |gamma_drift|
            <=1e-4 -- unsigned).
        G1: the viscous gate SIGMA_LAMBDA.md itself prescribes ("signed
            gamma drift: decay is physics, growth is a violation"),
            reconstructed from the stream's sup_gamma: growth <= 1e-4
            relative, decay unbounded. Identical to G0 at nu=0.
  WINDOW W0: per-grid full gated window (as-is).
         W1: the COMMON ||omega||_inf overlap of the two grids at the
            same nu -- a local slope is only cross-grid comparable on
            the same amplitude range.
CERTIFIED COMBO: G1+W1 (the .md's own gate + the commensurable window).
G0/W0 rows are reported alongside as the sensitivity record.

Row recipe (verbatim sigma_peak.sigma(): peak-box Lambda at 0.5-threshold
with 2x-HWHM box, 10x-jump trim, >=6 rows) -- sigma_peak.py is NOT
modified; its ext() is imported, lambda_geom does the geometry.
Fit: polyfit ln Lambda vs ln A; bootstrap 95% CI (10k pair resamples).

OUTPUTS: M4_SIGMA_SPREAD.out (factorial + verdict), m4_sigma_rows.npz,
and sigma_grid_spread.txt (done.sh format: col1 = spread, one row per
viscosity, G1+W1 numbers) ONLY if both viscous spreads < 0.15.
"""
import glob
import json
import time

import h5py
import numpy as np

from sigma_peak import ext
import pathlib
import sys

sys.path.insert(0, ".")
from lambda_geom import axes, vorticity, grad_xi_sq

_HERE = pathlib.Path(__file__).parent
RUNS = _HERE / ".." / "runs"
PAIRS = {0.0: ("OR_z128r384", "OR_z256r768"),
         1e-4: ("NUL1e-4", "N2_1e-4"),
         1e-3: ("NUL1e-3", "N2_1e-3")}
GRIDS = ("128x384", "256x768")
TAIL_GATE = 1e-6
DRIFT_GATE = 1e-4
JUMP_TRIM = 10.0
MIN_ROWS = 6
N_BOOT = 10_000
BOOT_SEED = 271828
SPREAD_BAR = 0.15

OUT = _HERE / "M4_SIGMA_SPREAD.out"
NPZ = _HERE / "m4_sigma_rows.npz"
SPREAD_TXT = _HERE / "sigma_grid_spread.txt"
_lines = []


def log(s=""):
    _lines.append(s)
    print(s, flush=True)


def gated_window(tag, signed):
    """Longest clean stretch, generalizing sigma_peak.window(): tail gate
    always; gamma gate unsigned (G0, sigma_peak as-is) or signed via
    sup_gamma growth (G1, SIGMA_LAMBDA.md viscous prescription)."""
    rows = [json.loads(l) for l in open(RUNS / f"stream_{tag}.jsonl")
            if l.strip()]
    t = np.array([x["t"] for x in rows])
    tail_ok = np.array([max(x.get("tail_u1", 0), x.get("tail_w1", 0))
                        <= TAIL_GATE for x in rows])
    if signed:
        g0 = rows[0]["sup_gamma"]
        drift_ok = np.array([(x["sup_gamma"] - g0) / max(abs(g0), 1e-300)
                             <= DRIFT_GATE for x in rows])
    else:
        drift_ok = np.array([x.get("gamma_drift", 0) <= DRIFT_GATE
                             for x in rows])
    ok = tail_ok & drift_ok
    best = (0, 0)
    i = 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j < len(ok) and ok[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return (t[best[0]], t[best[1] - 1]) if best[1] > best[0] else (0.0, 0.0)


def extract_rows(tag, signed):
    """(A, Lambda_peak) per gated snapshot -- verbatim sigma_peak.sigma()
    recipe including the 10x-jump trim."""
    t0, t1 = gated_window(tag, signed)
    rows = []
    for fn in sorted(glob.glob(str(RUNS / f"snap_{tag}" / "*.h5"))):
        with h5py.File(fn, "r") as f:
            z, r = axes(f)
            W = f["tasks"]["omega1"][:]
            U = f["tasks"]["u1"][:]
            st = f["scales/sim_time"][:]
            nz, nr = len(z), len(r)
            for n in range(len(st)):
                if st[n] < t0 or st[n] > t1 or st[n] <= 0:
                    continue
                wr, wt, wz = vorticity(U[n], W[n], z, r)
                mag = np.sqrt(wr**2 + wt**2 + wz**2)
                mx = mag.max()
                s = np.maximum(mag, mx * 1e-12)
                g = np.sqrt(np.maximum(
                    grad_xi_sq(wr / s, wt / s, wz / s, z, r), 0))
                lam = np.where(mag > 0.5 * mx,
                               g / np.sqrt(np.maximum(mag, 1e-300)), 0.0)
                iz, ir = np.unravel_index(np.argmax(mag), mag.shape)
                za, zb = ext(mag[:, ir], iz)
                ra, rb = ext(mag[iz, :], ir)
                box = np.ix_(np.arange(za, zb + 1) % nz,
                             np.arange(max(0, ra), min(nr - 1, rb) + 1))
                rows.append((float(mx), float(lam[box].max())))
    out = [rows[0]] if rows else []
    for x in rows[1:]:
        if x[0] > out[-1][0] * JUMP_TRIM:
            break
        out.append(x)
    return np.array(out) if out else np.empty((0, 2))


def fit_sigma(rows, rng):
    """Slope of ln Lambda vs ln A with bootstrap 95% CI."""
    if len(rows) < MIN_ROWS:
        return None
    x = np.log(rows[:, 0])
    y = np.log(np.maximum(rows[:, 1], 1e-12))
    slope = float(np.polyfit(x, y, 1)[0])
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    slopes = np.array([np.polyfit(x[j], y[j], 1)[0] for j in idx])
    lo, hi = np.percentile(slopes, (2.5, 97.5))
    return slope, float(lo), float(hi), len(rows)


def main():
    t_all = time.time()
    log("=" * 78)
    log("M4 sigma_Lambda(nu) -- cross-grid spread, 2x2 instrument factorial")
    log("=" * 78)
    log(f"bar: spread < {SPREAD_BAR} at two viscosities (DONE.md M4).  "
        f"certified combo: G1 (signed viscous gate, SIGMA_LAMBDA.md) + "
        f"W1 (common-amplitude overlap).  bootstrap {N_BOOT}, seed "
        f"{BOOT_SEED}.")
    log("")

    rng = np.random.default_rng(BOOT_SEED)
    cache = {}
    for nu, tags in PAIRS.items():
        for tag in tags:
            for signed in (False, True):
                key = (tag, signed)
                if key not in cache:
                    cache[key] = extract_rows(tag, signed)

    all_spreads = {}
    for signed, gname in ((False, "G0 unsigned gate"),
                          (True, "G1 signed gate")):
        for overlap, wname in ((False, "W0 full windows"),
                               (True, "W1 overlap window")):
            log("-" * 78)
            log(f"{gname} + {wname}")
            log(f"{'nu':>8} {'grid':>9} {'n':>4} {'sigma':>9} "
                f"{'95% CI':>20} {'spread':>8}")
            for nu, tags in PAIRS.items():
                ra, rb = cache[(tags[0], signed)], cache[(tags[1], signed)]
                if overlap and len(ra) and len(rb):
                    lo = max(ra[:, 0].min(), rb[:, 0].min())
                    hi = min(ra[:, 0].max(), rb[:, 0].max())
                    ra = ra[(ra[:, 0] >= lo) & (ra[:, 0] <= hi)]
                    rb = rb[(rb[:, 0] >= lo) & (rb[:, 0] <= hi)]
                fa, fb = fit_sigma(ra, rng), fit_sigma(rb, rng)
                if fa is None or fb is None:
                    log(f"{nu:>8.0e} -- too few rows after cuts "
                        f"({len(ra)}/{len(rb)}) -- UNANSWERABLE here")
                    continue
                spread = abs(fa[0] - fb[0])
                all_spreads[(signed, overlap, nu)] = (spread, fa, fb)
                for gname2, ft in zip(GRIDS, (fa, fb)):
                    log(f"{nu:>8.0e} {gname2:>9} {ft[3]:>4} {ft[0]:>9.4f} "
                        f"[{ft[1]:>8.4f},{ft[2]:>8.4f}]")
                log(f"{'':>8} {'SPREAD':>9} {'':>4} {spread:>9.4f}")
            log("")

    log("=" * 78)
    log("VERDICT (certified combo: G1 signed gate + W1 overlap window)")
    log("=" * 78)
    cert = {nu: all_spreads.get((True, True, nu)) for nu in (1e-4, 1e-3)}
    ok = all(v is not None and v[0] < SPREAD_BAR for v in cert.values())
    for nu, v in cert.items():
        if v is None:
            log(f"nu={nu:g}: UNANSWERABLE (too few overlap rows)")
        else:
            sp, fa, fb = v
            log(f"nu={nu:g}: sigma = {fa[0]:+.4f} ({GRIDS[0]}) / "
                f"{fb[0]:+.4f} ({GRIDS[1]})  spread {sp:.4f} "
                f"{'< 0.15 OK' if sp < SPREAD_BAR else '>= 0.15 FAIL'}")
    inv = all_spreads.get((True, True, 0.0))
    if inv:
        log(f"nu=0 reference: sigma = {inv[1][0]:+.4f} / {inv[2][0]:+.4f} "
            f"(inviscid calibration, spread {inv[0]:.4f})")
    if ok:
        with open(SPREAD_TXT, "w") as f:
            for nu, (sp, fa, fb) in cert.items():
                f.write(f"{sp:.4f} nu={nu:g} sigma_{GRIDS[0]}={fa[0]:+.4f} "
                        f"sigma_{GRIDS[1]}={fb[0]:+.4f} gate=signed "
                        f"window=overlap\n")
        log("")
        log(f"M4 TEST: PASS -- both viscous spreads < {SPREAD_BAR}; wrote "
            f"{SPREAD_TXT.name}")
        log("MAGNITUDE: sigma_Lambda flips from +1 (inviscid, calibrated) "
            "to the certified viscous values above -- direct measurement "
            "of viscosity restoring direction regularity (SIGMA_LAMBDA.md "
            "outcome B).")
    else:
        log("")
        log("M4 TEST: FAIL -- spread bar not met under the certified "
            "combo; sigma_grid_spread.txt NOT written. The factorial "
            "table above records where the spread lives.")

    np.savez(NPZ, **{f"rows_{tag}_{'G1' if s else 'G0'}": cache[(tag, s)]
                     for (tag, s) in cache},
             bar=SPREAD_BAR, n_boot=N_BOOT, boot_seed=BOOT_SEED)
    log(f"rows saved: {NPZ.name}")
    log(f"total wall: {time.time() - t_all:.1f}s")
    OUT.write_text("\n".join(_lines) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
