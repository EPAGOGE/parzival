#!/usr/bin/env python
"""bq2 with the well-conditioned Shen-Galerkin Poisson solve swapped in.

Overrides BQ2.solve_psi (dense Chebyshev collocation, cond ~1e17) with the
Shen Chebyshev-Galerkin Helmholtz solve (cond ~1e3, gate-matched to 1e-13,
helmholtz.py). Everything else -- Fourier x, dealiasing, RK4, diagnostics --
is inherited unchanged. Tests whether the theta^2-conserved window (capped at
t~2.15 with the dense solve) EXTENDS once the Poisson solve stops losing ~10
digits per step near the steepening front.

Run: bq2_fast.py --gate            (solve_psi fast vs dense to roundoff)
     bq2_fast.py --probe <nx> <ny> (theta^2-window probe, well-conditioned)
"""
from __future__ import annotations

import sys
import time
import json
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bq2 import BQ2, theta0, LY
from helmholtz import ShenHelmholtz, vals_to_coeffs, coeffs_to_vals


class BQ2Fast(BQ2):
    def __init__(self, nx: int, ny: int):
        super().__init__(nx, ny)
        s = (LY / 2.0) ** 2                 # rescale (2/LY)^2 D2 - k^2 -> D2 - a
        self.rescale = s
        alphas = np.array([s * kk ** 2 for kk in self.kx])
        self.shen = ShenHelmholtz(ny + 1, alphas)
        self.alphas = alphas

    def solve_psi(self, w: np.ndarray) -> np.ndarray:
        wh = np.fft.rfft(w, axis=0) * self.xmask
        ph = np.empty_like(wh)
        for k in range(len(self.kx)):
            a = float(self.alphas[k])
            fr = vals_to_coeffs(self.rescale * wh[k].real)
            fi = vals_to_coeffs(self.rescale * wh[k].imag)
            ur = coeffs_to_vals(self.shen.solve_coeffs(fr, a))
            ui = coeffs_to_vals(self.shen.solve_coeffs(fi, a))
            ph[k] = ur + 1j * ui
        return np.fft.irfft(ph, n=self.nx, axis=0)


def gate() -> None:
    print("=== bq2_fast solve_psi: Shen vs dense collocation ===")
    for nx, ny in ((64, 48), (128, 96)):
        dense = BQ2(nx, ny)
        fast = BQ2Fast(nx, ny)
        rng = np.random.default_rng(1)
        w = theta0(dense, 3.0) + 0.1 * rng.standard_normal((nx, ny + 1))
        w = dense.filt(w)
        pd = dense.solve_psi(w)
        pf = fast.solve_psi(w)
        # residual of each against the actual PDE Lap(psi)=w (the real test)
        def lap(p):
            return fast.dx(fast.dx(p)) + p @ (fast.Dy @ fast.Dy).T
        rd = np.abs(lap(pd) - w).max()
        rf = np.abs(lap(pf) - w).max()
        diff = np.abs(pd - pf).max() / max(np.abs(pd).max(), 1e-30)
        print(f"  N={nx}x{ny}: |psi_dense-psi_fast|/|psi| {diff:.2e} | "
              f"Poisson residual dense {rd:.2e} fast {rf:.2e}")
        assert diff < 1e-8, f"GATE FAIL: fast != dense ({diff:.2e})"
    print("bq2_fast gate PASS (fast solve == dense to <1e-8, same BVP)")


def probe(nx: int, ny: int, A: float = 4.0) -> None:
    eng = BQ2Fast(nx, ny)
    th = theta0(eng, A)
    w = np.zeros_like(th)
    th2_0 = eng.integ(th ** 2)
    t, si, t0 = 0.0, 0, time.time()
    ser = {"t": [], "sup_w": [], "sup_gth": [], "th2_drift": []}
    while si < 200000:
        w, th, dt, aux = eng.step(w, th)
        t += dt; si += 1
        if si % 25 == 0 or si == 1:
            d = abs(eng.integ(th ** 2) - th2_0) / max(th2_0, 1e-300)
            ser["t"].append(t); ser["sup_w"].append(float(np.abs(w).max()))
            ser["sup_gth"].append(aux["sup_gth"]); ser["th2_drift"].append(d)
            if si % 500 == 0:
                print(f"  t={t:.4f} step={si} sup|gth|={aux['sup_gth']:.3e} "
                      f"th2_drift={d:.2e} dt={dt:.1e} ({time.time()-t0:.0f}s)",
                      flush=True)
            if d > 3e-3:
                print(f"  theta^2 break at t={t:.4f} (drift {d:.2e})", flush=True)
                break
        if dt < 1e-9:
            print(f"  dt exhausted at t={t:.4f}", flush=True); break
    tt = np.array(ser["t"]); gg = np.array(ser["sup_gth"])
    ww = np.array(ser["sup_w"]); dd = np.array(ser["th2_drift"])
    trust = dd < 1e-3
    tb = float(tt[trust][-1]) if trust.any() else 0.0
    out = {"nx": nx, "ny": ny, "A": A, "solver": "shen", "t_trust_end": tb,
           "gth_ratio": float(gg[trust][-1] / gg[trust][0]) if trust.sum() > 1 else 1.0,
           "supw_trust_end": float(ww[trust][-1]) if trust.any() else 0.0}
    if trust.sum() >= 6:
        tg, ggt = tt[trust], gg[trust]
        q = len(tg) // 3
        out["accel"] = float(np.polyfit(tg[-q:], np.log(ggt[-q:]), 1)[0]
                             / np.polyfit(tg[-2 * q:-q], np.log(ggt[-2 * q:-q]), 1)[0])
    pathlib.Path(f"../runs/bq2fast_N{nx}.json").write_text(json.dumps(out, indent=2))
    print(f"[SHEN] N={nx}x{ny} A={A:g}: window t<={tb:.3f} | gth x{out['gth_ratio']:.1f} "
          f"| accel={out.get('accel', 0):.2f} | sup|w|_end={out['supw_trust_end']:.2f} "
          f"| {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if "--gate" in sys.argv:
        gate()
    if "--probe" in sys.argv:
        i = sys.argv.index("--probe")
        probe(int(sys.argv[i + 1]), int(sys.argv[i + 2]),
              float(sys.argv[i + 3]) if len(sys.argv) > i + 3 else 4.0)
