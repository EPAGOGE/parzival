#!/usr/bin/env python
"""Era-B2 wall engine: 2D inviscid Boussinesq in the PROVEN geometry class.

x1 in [0, 2pi) periodic (rFFT; theorem symmetry theta even / w odd about
x1=0 carried by the IC and monitored, not hard-coded), y in [0, Ly] with a
GENUINE wall at y=0: Chebyshev collocation, no-flow psi=0 at y=0 and y=Ly.
theta and w are FREE on the wall (advected there by u1 only) -- the feature
the Chen-Hou mechanism requires and the B1 parity box excluded.

Conventions carried verbatim from the gate-verified bq.py:
  u1 = -psi_y? NO -- as in bq.py: u = (-psi_x2, psi_x1), Lap psi = w,
  torque +theta_x1, i.e. w_t + u.grad w = theta_x1, theta_t + u.grad theta = 0.

Bring-up trust level (era B2-bringup): operator/Poisson gates at machine
precision; budget conservation is a RESOLUTION MONITOR here (collocation
advection is not exactly conservative), tolerance-gated, not roundoff-gated.
Dispersion gate deferred (usage-constrained round; pre-registered next).
IC: Luo-Hou style (w0 = 0, wall-concentrated even theta, quadratic from the
axis along the wall). Oracle exponents pre-registered in FORMULATION.md
Corrections sec. 4.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve

DT_MAX, DT_MIN = 2e-3, 1e-9
C_ADV, C_WALL = 1.2, 1.2
TAIL_LOW, TAIL_EXHAUST = 1e-4, 1e-2
LY = math.pi


def cheb(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Trefethen Chebyshev differentiation matrix on n+1 Lobatto points."""
    if n == 0:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.hstack([2.0, np.ones(n - 1), 2.0]) * (-1.0) ** np.arange(n + 1)
    X = np.tile(x, (n + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def clencurt(n: int) -> np.ndarray:
    """Clenshaw-Curtis quadrature weights on n+1 Lobatto points, [-1,1]."""
    th = np.pi * np.arange(n + 1) / n
    w = np.zeros(n + 1)
    v = np.ones(n - 1)
    for k in range(1, n // 2 + 1):
        fac = 2.0 if 2 * k < n else 1.0
        v -= fac * np.cos(2 * k * th[1:-1]) / (4 * k * k - 1)
    w[0] = w[-1] = 1.0 / (n * n - (n % 2 == 0))
    w[1:-1] = 2.0 * v / n
    return w


class BQ2:
    def __init__(self, nx: int, ny: int):
        self.nx, self.ny = nx, ny
        self.kx = np.fft.rfftfreq(nx, 1.0 / nx)          # 0..nx/2
        self.kc = (2 * (nx // 2)) // 3                   # 2/3 rule in x1
        self.xmask = (self.kx <= self.kc)[:, None]
        Dc, xc = cheb(ny)
        self.y = LY * (1.0 + xc) / 2.0                   # y[0]=Ly ... y[ny]=0
        self.Dy = (2.0 / LY) * Dc
        self.wq = clencurt(ny) * (LY / 2.0)              # integral weights in y
        self.dy_min = float(self.y[-2] - self.y[-1])
        k2 = self.kx ** 2
        D2 = self.Dy @ self.Dy
        self.lus = []
        for k in range(len(self.kx)):
            A = D2 - k2[k] * np.eye(ny + 1)
            A[0, :] = 0.0; A[0, 0] = 1.0                 # psi=0 at y=Ly
            A[-1, :] = 0.0; A[-1, -1] = 1.0              # psi=0 at wall
            self.lus.append(lu_factor(A))

    def dx(self, f: np.ndarray) -> np.ndarray:
        return np.fft.irfft(1j * self.kx[:, None] * np.fft.rfft(f, axis=0)
                            * self.xmask, n=self.nx, axis=0)

    def filt(self, f: np.ndarray) -> np.ndarray:
        return np.fft.irfft(np.fft.rfft(f, axis=0) * self.xmask,
                            n=self.nx, axis=0)

    def solve_psi(self, w: np.ndarray) -> np.ndarray:
        wh = np.fft.rfft(w, axis=0) * self.xmask
        ph = np.empty_like(wh)
        for k in range(len(self.kx)):
            rhs = wh[k].copy()
            rhs[0] = 0.0; rhs[-1] = 0.0
            ph[k] = lu_solve(self.lus[k], rhs.real) + \
                1j * lu_solve(self.lus[k], rhs.imag)
        return np.fft.irfft(ph, n=self.nx, axis=0)

    def rhs(self, w: np.ndarray, th: np.ndarray):
        psi = self.solve_psi(w)
        u1 = -(psi @ self.Dy.T)                          # -psi_y
        u2 = self.dx(psi)                                # +psi_x1
        thx = self.dx(th)
        adv_w = self.filt(u1 * self.dx(w) + u2 * (w @ self.Dy.T))
        adv_t = self.filt(u1 * thx + u2 * (th @ self.Dy.T))
        aux = {"sup_u1": float(np.abs(u1).max()),
               "sup_u2": float(np.abs(u2).max()),
               "sup_gth": float(np.hypot(thx, th @ self.Dy.T).max()),
               "sup_tx1": float(np.abs(thx).max())}
        return thx - adv_w, -adv_t, aux

    def dt_of(self, aux: dict) -> float:
        dt = DT_MAX
        if aux["sup_u1"] > 0:
            dt = min(dt, C_ADV / (self.kc * aux["sup_u1"]))
        if aux["sup_u2"] > 0:
            dt = min(dt, C_WALL * self.dy_min / aux["sup_u2"])
        return dt

    def step(self, w, th):
        k1w, k1t, aux = self.rhs(w, th)
        dt = self.dt_of(aux)
        k2w, k2t, _ = self.rhs(w + 0.5 * dt * k1w, th + 0.5 * dt * k1t)
        k3w, k3t, _ = self.rhs(w + 0.5 * dt * k2w, th + 0.5 * dt * k2t)
        k4w, k4t, _ = self.rhs(w + dt * k3w, th + dt * k3t)
        w = self.filt(w + (dt / 6) * (k1w + 2 * k2w + 2 * k3w + k4w))
        th = self.filt(th + (dt / 6) * (k1t + 2 * k2t + 2 * k3t + k4t))
        return w, th, dt, aux

    def integ(self, f: np.ndarray) -> float:
        return float((f.mean(axis=0) * self.wq).sum() * 2 * np.pi)

    def tail_x(self, f: np.ndarray) -> float:
        sp = np.abs(np.fft.rfft(f, axis=0)) ** 2
        hi = sp[3 * len(self.kx) // 4:].sum()
        return float(hi / max(sp.sum(), 1e-300))


def theta0(eng: BQ2, A: float) -> np.ndarray:
    """Luo-Hou style: even in x1 about 0, vanishing quadratically on the axis
    x1=0, NONZERO on the wall, concentrated near it. w0 = 0."""
    x = 2 * np.pi * np.arange(eng.nx) / eng.nx
    prof_x = np.sin(x / 2) ** 2                          # even, ~x^2/4 at axis
    prof_y = np.exp(-30.0 * (eng.y / LY) ** 4)          # wall-concentrated
    return A * prof_x[:, None] * prof_y[None, :]


def gates() -> None:
    eng = BQ2(64, 48)
    f = np.cos(3 * 2 * np.pi * np.arange(64) / 64)[:, None] * \
        np.exp(-eng.y / LY)[None, :]
    dfx = -3 * np.sin(3 * 2 * np.pi * np.arange(64) / 64)[:, None] * \
        np.exp(-eng.y / LY)[None, :]
    e1 = float(np.abs(eng.dx(f) - dfx).max())
    dfy = f * (-1.0 / LY)
    e2 = float(np.abs(f @ eng.Dy.T - dfy).max())
    psi_m = np.sin(2 * np.pi * np.arange(64) / 64)[:, None] * \
        (eng.y ** 2 * (LY - eng.y) ** 2)[None, :]
    w_m = eng.dx(eng.dx(psi_m)) + psi_m @ (eng.Dy @ eng.Dy).T
    e3 = float(np.abs(eng.solve_psi(w_m) - psi_m).max())
    e4 = abs(eng.integ(np.ones((64, 49))) - 2 * np.pi * LY)
    print(f"G1 d/dx1 {e1:.2e} | d/dy {e2:.2e} | Poisson {e3:.2e} | "
          f"quad {e4:.2e}")
    assert max(e1, e2, e3, e4) < 1e-9, "B2 GATE FAIL"
    print("B2 operator gates PASS")


def scenario(A: float, nx: int, ny: int, tmax: float, out: str) -> None:
    eng = BQ2(nx, ny)
    th = theta0(eng, A)
    w = np.zeros_like(th)
    th2_0 = eng.integ(th ** 2)
    t, wall0, status, low_t = 0.0, time.time(), "completed", None
    series = {k: [] for k in ("t", "dt", "sup_w", "sup_gth", "th_wall_max",
                              "th2_drift", "E", "buoy_work", "tail_w",
                              "tail_th", "sym_drift", "itgth")}
    itgth, gth_prev, dt_prev, si = 0.0, None, None, 0
    while t < tmax and si < 400000:
        w, th, dt, aux = eng.step(w, th)
        t += dt
        si += 1
        if gth_prev is not None:
            itgth += 0.5 * (gth_prev + aux["sup_gth"]) * dt_prev
        gth_prev, dt_prev = aux["sup_gth"], dt
        if si % 10 == 0 or t >= tmax:
            psi = eng.solve_psi(w)
            u2 = eng.dx(psi)
            sym = float(np.abs(th - th[::-1] if False else
                               th - np.roll(th[::-1], 1, axis=0)).max())
            row = {"t": t, "dt": dt, "sup_w": float(np.abs(w).max()),
                   "sup_gth": aux["sup_gth"],
                   "th_wall_max": float(np.abs(th[:, -1]).max()),
                   "th2_drift": abs(eng.integ(th ** 2) - th2_0)
                   / max(th2_0, 1e-300),
                   "E": eng.integ(u2 ** 2), "buoy_work": eng.integ(th * u2),
                   "tail_w": eng.tail_x(w), "tail_th": eng.tail_x(th),
                   "sym_drift": sym, "itgth": itgth}
            for k, v in row.items():
                series[k].append(v)
            tails = max(row["tail_w"], row["tail_th"])
            if tails > TAIL_LOW and low_t is None:
                low_t = t
            if tails > TAIL_EXHAUST:
                status = "tail_exhausted"
                break
        if dt < DT_MIN:
            status = "dt_exhausted"
            break
    fin = {k: series[k][-1] for k in series}
    fin["status"], fin["low_trust_since_t"] = status, low_t
    pathlib.Path(out).write_text(json.dumps(
        {"header": {"engine": "bq2 era B2-bringup", "nx": nx, "ny": ny,
                    "A": A, "Ly": LY, "steps": si,
                    "wall_s": round(time.time() - wall0, 1)},
         "series": series, "final": fin}, allow_nan=False))
    print(f"bq2 A={A:g} {nx}x{ny} t={t:.3f} status={status} | "
          f"sup|w| {fin['sup_w']:.3f} sup|gth| {fin['sup_gth']:.2f} "
          f"th_wall {fin['th_wall_max']:.3f} | th2 drift "
          f"{fin['th2_drift']:.2e} | int gth {itgth:.2f} | "
          f"sym {fin['sym_drift']:.2e} | {si} steps "
          f"{(time.time()-wall0):.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--scenario", action="store_true")
    ap.add_argument("--A", type=float, default=1.0)
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=96)
    ap.add_argument("--tmax", type=float, default=3.0)
    ap.add_argument("--out", default="runs/bq2_scenario.json")
    args = ap.parse_args()
    if args.gates:
        gates()
    if args.scenario:
        scenario(args.A, args.nx, args.ny, args.tmax, args.out)
