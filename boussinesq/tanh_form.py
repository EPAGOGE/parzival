#!/usr/bin/env python3
"""#78(b) -- sigma(A) functional form through the crossover.

Pre-registered (NU_CURVE_SPEC.md, CROSS-TERM + FORM section) before any
fit was run. Model: slope blends between two asymptotes via tanh,

    y(x) = c + (s1+s2)/2 (x-x0) + (s2-s1)/2 * w * ln cosh((x-x0)/w),
    x = ln A, y = ln Lambda;  dy/dx = (s1+s2)/2 + (s2-s1)/2 tanh((x-x0)/w).

Per run: center exp(x0)/A0, width w (e-folds), slopes s1 (low-A) and s2
(deep). Consistency: fit-implied slope zero crossing vs measured A*
(b_band zero-crossing estimator); s2 vs certified deep values; MODEL
INADEQUATE if fit residual RMS > 1.5x the deep-window linear residual.

Runs: any tag given on the command line, else the six existing band runs.
Outputs: TANH_FORM.out (appended per invocation set), tanh_form_data.npz.
"""
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import curve_fit

from b_band import local_slope_zero
from m4_sigma_spread import extract_rows

_HERE = pathlib.Path(__file__).parent
DEFAULT = [("B128_A50", 50), ("B256_A50", 50),
           ("C128_1e-4", 100), ("C256_1e-4", 100),
           ("B128_A200", 200), ("B256_A200", 200)]
OUT = _HERE / "TANH_FORM.out"
NPZ = _HERE / "tanh_form_data.npz"
_lines = []


def log(s=""):
    _lines.append(s)
    print(s, flush=True)


def model(x, c, x0, w, s1, s2):
    return (c + 0.5 * (s1 + s2) * (x - x0)
            + 0.5 * (s2 - s1) * w * np.log(np.cosh((x - x0) / w)))


def fit_run(tag, A0):
    if tag.startswith("C"):
        rows = np.load(_HERE / "nu_curve_data.npz")[f"rows_{tag}"]
    else:
        rows = extract_rows(tag, signed=True)
    order = np.argsort(rows[:, 0])
    x = np.log(rows[order, 0])
    y = np.log(np.maximum(rows[order, 1], 1e-12))
    # starts: center at mid-range, width 0.5 e-fold, slopes +1/-1.2
    p0 = [y[len(y) // 2], 0.5 * (x.min() + x.max()), 0.5, 1.0, -1.2]
    p, _cov = curve_fit(model, x, y, p0=p0, maxfev=50000)
    c, x0, w, s1, s2 = p
    w = abs(w)
    resid = float(np.std(y - model(x, *p)))
    # deep-window linear residual for the adequacy bar
    xc = 0.5 * (x.min() + x.max())
    deep = x >= xc
    lin = np.polyfit(x[deep], y[deep], 1)
    resid_deep = float(np.std(y[deep] - np.polyval(lin, x[deep])))
    # fit-implied slope zero crossing
    ratio = -(s1 + s2) / (s2 - s1)
    x_zc = x0 + w * np.arctanh(ratio) if abs(ratio) < 1 else float("nan")
    a_star = local_slope_zero(rows, 15)
    return dict(tag=tag, A0=A0, center=float(np.exp(x0)), w=float(w),
                s1=float(s1), s2=float(s2), resid=resid,
                resid_deep=resid_deep, a_zc=float(np.exp(x_zc)),
                a_star=a_star, n=len(x), rows=rows)


def main():
    t0 = time.time()
    cases = ([(t, int(a)) for t, a in
              (arg.split(":") for arg in sys.argv[1:])] or DEFAULT)
    log("=" * 78)
    log("TANH FORM -- sigma(A) through the crossover (#78b, pre-registered)")
    log("=" * 78)
    log(f"{'tag':<12} {'n':>3} {'ctr/A0':>7} {'w':>6} {'s1':>7} {'s2':>7} "
        f"{'zc/A0':>6} {'A*/A0':>6} {'resid':>6} {'adequate':>9}")
    save = {}
    oks = []
    for tag, A0 in cases:
        try:
            r = fit_run(tag, A0)
        except (FileNotFoundError, RuntimeError) as e:
            log(f"{tag:<12} FAILED: {e}")
            continue
        adequate = r["resid"] <= 1.5 * max(r["resid_deep"], 1e-9)
        oks.append((r, adequate))
        save[f"fit_{tag}"] = np.array([r["center"], r["w"], r["s1"],
                                       r["s2"], r["resid"]])
        log(f"{r['tag']:<12} {r['n']:>3} {r['center'] / A0:>7.1f} "
            f"{r['w']:>6.2f} {r['s1']:>+7.2f} {r['s2']:>+7.2f} "
            f"{r['a_zc'] / A0:>6.1f} "
            f"{(r['a_star'] / A0) if r['a_star'] else float('nan'):>6.1f} "
            f"{r['resid']:>6.3f} "
            f"{'yes' if adequate else 'NO -- MODEL INADEQUATE':>9}")
    good = [r for r, ok in oks if ok]
    log("")
    if good:
        ctrs = [r["center"] / r["A0"] for r in good]
        ws = [r["w"] for r in good]
        s2s = [r["s2"] for r in good]
        log(f"across {len(good)} adequate fits: center/A0 = "
            f"{np.mean(ctrs):.1f} +- {np.std(ctrs):.1f}   width = "
            f"{np.mean(ws):.2f} +- {np.std(ws):.2f} e-folds   s2 = "
            f"{np.mean(s2s):+.2f} +- {np.std(s2s):.2f}")
        log("consistency: fit-implied zero crossing vs measured A* per row "
            "above (zc/A0 vs A*/A0).")
    inad = [r for r, ok in oks if not ok]
    if inad:
        log(f"MODEL INADEQUATE on {len(inad)} run(s): "
            f"{[r['tag'] for r in inad]} -- reported, not forced.")
    np.savez(NPZ, **save)
    log(f"total wall: {time.time() - t0:.1f}s")
    prev = OUT.read_text() if OUT.exists() else ""
    OUT.write_text(prev + "\n".join(_lines) + "\n\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
