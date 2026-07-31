#!/usr/bin/env python
"""Rung 1b: validate the edge-branch pole before believing it.
(1) spectral tail of the tip states at N=128, (2) re-converge the tip at
N=256 via Fourier interpolation, (3) extend toward a=1 at N=256,
(4) refit the pole with a_c free vs pinned at 1."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from swarm_m1 import build_mats, macro_step_np, make_ic

NU = 1.0


def make_ops(n: int) -> dict:
    Hm, Lm, Dm, Gm = build_mats(n)
    return {"H": Hm, "L": Lm, "D": Dm, "G": Gm, "n": n}


def F(w, a, o):
    u, ux, wx = o["G"] @ w, o["H"] @ w, o["D"] @ w
    return ux * w - a * u * wx - NU * (o["L"] @ w)


def jac(w, a, o):
    u, ux, wx = o["G"] @ w, o["H"] @ w, o["D"] @ w
    return (np.diag(w) @ o["H"] + np.diag(ux)
            - a * (np.diag(wx) @ o["G"] + np.diag(u) @ o["D"]) - NU * o["L"])


def newton(w0, a, ref, o, c0=0.0, tol=1e-10, itmax=80):
    n = o["n"]
    w, c = w0.copy(), c0
    phase, mean_row = o["D"] @ ref, np.ones(n) / n
    res = np.inf
    for _ in range(itmax):
        wx = o["D"] @ w
        Fv = F(w, a, o) + c * wx
        res = float(np.max(np.abs(Fv)))
        if res < tol:
            return w, res, c
        aug = np.vstack([np.hstack([jac(w, a, o) + c * o["D"], wx[:, None]]),
                         np.append(phase, 0.0)[None, :],
                         np.append(mean_row, 0.0)[None, :]])
        rhs = np.concatenate([-Fv, [-phase @ (w - ref)], [-w.mean()]])
        dz, *_ = np.linalg.lstsq(aug, rhs, rcond=None)
        w, c = w + dz[:-1], c + float(dz[-1])
    return (w, res, c) if res < 1e-8 else (None, res, c)


def tail_frac(w: np.ndarray) -> float:
    sp = np.abs(np.fft.rfft(w))
    return float((sp[3 * len(sp) // 4:] ** 2).sum() / (sp ** 2).sum())


def interp2x(w: np.ndarray, n2: int) -> np.ndarray:
    wh = np.fft.rfft(w)
    pad = np.zeros(n2 // 2 + 1, complex)
    pad[:len(wh)] = wh
    return np.fft.irfft(pad, n=n2) * (n2 / len(w))


def main() -> None:
    o128, o256 = make_ops(128), make_ops(256)

    # regenerate the a=0.93 seed from the known threshold (bisected in rung1)
    amid = 18.13614
    w = make_ic(np.array([amid]), 128, "cos2")
    t = np.zeros(1)
    best, best_rate, sup_prev, t_prev = None, np.inf, amid, 0.0
    mats = {k: o128[k] for k in "HLDG"}
    for _ in range(150000):
        w, t = macro_step_np(w, t, mats, NU, "gclm", a=0.93)
        sup, tn = float(np.max(np.abs(w))), float(t[0])
        if sup > 1e3 or (sup < 0.1 * amid and tn > 0.5):
            break
        if tn > 1.0 and tn - t_prev > 1e-6:
            rate = abs(np.log(sup) - np.log(sup_prev)) / (tn - t_prev)
            if rate < best_rate:
                best_rate, best = rate, w[0].copy()
        sup_prev, t_prev = sup, tn
    ws, _, cs = newton(best, 0.93, best, o128)
    print(f"seed reconverged a=0.93: sup={np.max(np.abs(ws)):.4f}")

    # walk the branch up at N=128, recording tail; keep profiles
    ups = list(np.arange(0.935, 0.9701, 0.005)) + \
          list(np.arange(0.972, 0.99001, 0.002))
    wc, cc = ws, cs
    tip = {}
    for at in ups:
        wc, _, cc = newton(wc, float(at), wc, o128, c0=cc)
        if wc is None:
            print(f"stopped at {at}")
            return
        tip[round(float(at), 4)] = (wc.copy(), cc)
    for at in (0.97, 0.98, 0.99):
        wv, cv = tip[at]
        print(f"N=128 a={at}: sup={np.max(np.abs(wv)):8.3f} c={cv:.4f} "
              f"tail={tail_frac(wv):.2e}")

    # N=256 re-convergence of the tip + extension toward a=1
    w2 = interp2x(tip[0.99][0], 256)
    w2, res2, c2 = newton(w2, 0.99, w2, o256, c0=tip[0.99][1])
    print(f"N=256 a=0.99: sup={np.max(np.abs(w2)):.4f} c={c2:.4f} "
          f"res={res2:.1e} tail={tail_frac(w2):.2e} "
          f"(N=128 sup={np.max(np.abs(tip[0.99][0])):.4f})")

    ext = [(0.99, float(np.max(np.abs(w2))), c2)]
    wc2, cc2 = w2, c2
    for at in [0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997]:
        wn, resn, cn = newton(wc2, at, wc2, o256, c0=cc2)
        if wn is None:
            print(f"N=256 branch stopped at a={at} (res={resn:.1e})")
            break
        sup = float(np.max(np.abs(wn)))
        print(f"N=256 a={at}: sup={sup:9.3f} c={cn:8.4f} tail={tail_frac(wn):.2e}")
        ext.append((at, sup, cn))
        wc2, cc2 = wn, cn
        if sup > 3000:
            break

    # pole fit on N=256 extension + N=128 branch points a>=0.95
    pts = [(a, tip[a][0]) for a in tip if a >= 0.95]
    aa = np.array([p[0] for p in pts] + [e[0] for e in ext[1:]])
    ss = np.array([float(np.max(np.abs(p[1]))) for p in pts] +
                  [e[1] for e in ext[1:]])
    best_fit = None
    for ac in np.arange(max(aa) + 0.0003, 1.0301, 0.0002):
        x, y = np.log(ac - aa), np.log(ss)
        A_ = np.vstack([x, np.ones_like(x)]).T
        coef, resid, *_ = np.linalg.lstsq(A_, y, rcond=None)
        r = float(resid[0]) if len(resid) else 0.0
        if best_fit is None or r < best_fit[0]:
            best_fit = (r, float(ac), float(-coef[0]), float(np.exp(coef[1])))
    _, ac, p, cfit = best_fit
    x1 = np.log(1.0 - aa)
    A1 = np.vstack([x1, np.ones_like(x1)]).T
    coef1, resid1, *_ = np.linalg.lstsq(A1, np.log(ss), rcond=None)
    print(f"\npole fit (free a_c):   a_c={ac:.4f} p={p:.3f} C={cfit:.3f}")
    print(f"pole fit (a_c=1 pinned): p={-coef1[0]:.3f} C={np.exp(coef1[1]):.3f} "
          f"resid={float(resid1[0]) if len(resid1) else 0:.4f}")
    json.dump({"ext_256": ext, "fit_free": [ac, p, cfit],
               "fit_pinned1": [float(-coef1[0]), float(np.exp(coef1[1]))]},
              open(pathlib.Path(__file__).parent / "runs" / "rung1b.json", "w"),
              indent=2)


if __name__ == "__main__":
    main()
