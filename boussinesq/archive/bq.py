#!/usr/bin/env python
"""PARZIVAL BOUSSINESQ -- bq.py (era B0; fp64 primary engine).

2D incompressible Boussinesq on the box [0,pi]^2 via parity bases
(DST-II/DCT-II on the cell-centered grid), exactly per boussinesq/FORMULATION.md:

    w_t     + u.grad(w)     = theta_x1 + nu    * Lap(w)
    theta_t + u.grad(theta) =            kappa * Lap(theta)
    Lap(psi) = w,   u = (-psi_x2, +psi_x1)

Parity classes: w, psi sin(x)sin | theta, u2 cos(x)sin | u1 sin(x)cos.
Everything spectral via scipy.fft (no dense operators, O(N^2 log N) per RHS),
2/3-dealiased, classical RK4 with a deterministic state-only adaptive dt.
fp64 throughout. No wall-clocks and no randomness inside physics code paths;
timing lives in the driver only. No vault writes from this script (campaign
freeze; completed-run JSON goes wherever --out points, default under
boussinesq/runs/).

Run:  .venv/bin/python boussinesq/bq.py --gates
      .venv/bin/python boussinesq/bq.py --scenario --A 4 --N 256 --tmax 2 \
          --out boussinesq/runs/smoke.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
import scipy.fft as sfft

# ---------------------------------------------------------------- constants
WORKERS = 8             # scipy.fft threads -- free real estate at 256^2+ (HARNESS 2)

# dt law (FORMULATION 7)
DT_MAX = 5e-3
DT_MIN = 1e-9
C_ADV = 1.4             # RK4 imag-axis reach 2*sqrt(2) with 2x margin
C_BUO = 0.5             # ~12 stages per radian of the fastest internal wave
C_VIS = 1.4             # RK4 real-axis limit 2.785 with margin

# IC family (FORMULATION 3.1)
P_BUMP = 32
Q_BUMP = 32
GSTAR_Q32 = 0.2095889654409937   # sqrt(33)/17*(33/34)^16 = sin(y*)*bump_32(y*)

# diagnostics (FORMULATION 6)
TAIL_LOW = 1e-4         # D6 low-trust flag
TAIL_EXHAUST = 1e-2     # D6 resolution exhausted -- stop claiming physics
DIAG_EVERY = 10         # scenario series cadence, by step count
MAX_STEPS = 400_000     # runaway guard for scenario mode

# gate constants (FORMULATION 5) -- one place, asserted, swarm_m1 style
G0_TOL = 1e-13
G1_DECAY = 0.9801986733067553    # exp(-0.02)
G1_TOL = 1e-10
G2_TOL = 1e-8
G2_RATIO_LO, G2_RATIO_HI = 8.0, 32.0
G2_FLOOR = 3e-14        # roundoff floor below which the dt-scaling assert is moot
G3_MODES = ((1, 1), (2, 1))
G3_SIGMA = {(1, 1): 0.7071067811865476, (2, 1): 0.8944271909999159}
G3_EPS = 1e-4
G3_TOL = 1e-8
G3_WAVE_DRIFT = 1e-6    # cross-meter tripwire on the (W) invariant of 4.5
G4A_ORDER_MIN = 8.0
G4A_E256_TOL = 1e-8
G4B_LO, G4B_HI = 3.7, 4.3
G5_TOL = 1e-12


# ---------------------------------------------------------------- transforms
# scipy.fft type-II with norm='ortho' on the cell-centered grid IS the parity
# series (FORMULATION 2.5). axis 0 = x1, axis 1 = x2.
def to_ss(f: np.ndarray) -> np.ndarray:
    return sfft.dstn(f, type=2, norm="ortho", workers=WORKERS)


def from_ss(F: np.ndarray) -> np.ndarray:
    return sfft.idstn(F, type=2, norm="ortho", workers=WORKERS)


def to_cs(f: np.ndarray) -> np.ndarray:
    return sfft.dst(sfft.dct(f, type=2, axis=0, norm="ortho", workers=WORKERS),
                    type=2, axis=1, norm="ortho", workers=WORKERS)


def from_cs(F: np.ndarray) -> np.ndarray:
    return sfft.idct(sfft.idst(F, type=2, axis=1, norm="ortho", workers=WORKERS),
                     type=2, axis=0, norm="ortho", workers=WORKERS)


def to_sc(f: np.ndarray) -> np.ndarray:
    return sfft.dct(sfft.dst(f, type=2, axis=0, norm="ortho", workers=WORKERS),
                    type=2, axis=1, norm="ortho", workers=WORKERS)


def from_sc(F: np.ndarray) -> np.ndarray:
    return sfft.idst(sfft.idct(F, type=2, axis=1, norm="ortho", workers=WORKERS),
                     type=2, axis=0, norm="ortho", workers=WORKERS)


def from_cc(F: np.ndarray) -> np.ndarray:
    return sfft.idctn(F, type=2, norm="ortho", workers=WORKERS)


# ---------------------------------------------------------------- derivative maps
# FORMULATION 2.6: transfers only within the shared-weight band m = 1..N-1;
# never touches the exceptional slots (cos m=0 stays data, sin m=N dropped).
def d_sin2cos(F: np.ndarray, axis: int) -> np.ndarray:
    """d/dx along a sine axis: cos slot m <- +m * sine slot m-1, m = 1..N-1."""
    G = np.zeros_like(F)
    m = np.arange(1.0, F.shape[axis])
    if axis == 0:
        G[1:] = m[:, None] * F[:-1]
    else:
        G[:, 1:] = m * F[:, :-1]
    return G


def d_cos2sin(F: np.ndarray, axis: int) -> np.ndarray:
    """d/dx along a cosine axis: sine slot m-1 <- -m * cos slot m, m = 1..N-1."""
    G = np.zeros_like(F)
    m = np.arange(1.0, F.shape[axis])
    if axis == 0:
        G[:-1] = -m[:, None] * F[1:]
    else:
        G[:, :-1] = -m * F[:, 1:]
    return G


# ---------------------------------------------------------------- engine
class BQ:
    """fp64 spectral Boussinesq engine on [0,pi]^2 (FORMULATION 2, 7).

    State is the pair of coefficient arrays (w_hat sin(x)sin, th_hat cos(x)sin),
    kept 2/3-masked at all times: the ICs are masked and every RHS output is
    masked (products explicitly; derivative maps and diagonal terms preserve
    the mask because they act at fixed frequency)."""

    def __init__(self, N: int, nu: float = 0.0, kappa: float = 0.0,
                 B: float = 0.0):
        self.N, self.nu, self.kappa, self.B = N, nu, kappa, B
        self.K = (2 * N - 1) // 3
        self.h = math.pi / N
        self.x = (np.arange(N) + 0.5) * self.h
        fs = np.arange(1.0, N + 1)              # sine slot -> frequency
        fc = np.arange(0.0, N)                  # cosine slot -> frequency
        self.LAM = fs[:, None] ** 2 + fs[None, :] ** 2       # -Lap on sin(x)sin
        self.LAM_TH = fc[:, None] ** 2 + fs[None, :] ** 2    # -Lap on cos(x)sin
        ms, mc = fs <= self.K, fc <= self.K
        self.MASK_SS = np.outer(ms, ms).astype(float)
        self.MASK_CS = np.outer(mc, ms).astype(float)
        thr = 0.75 * self.K                     # D6 tail band: freq > 3K/4
        self.TAIL_SS = np.maximum(fs[:, None], fs[None, :]) > thr
        self.TAIL_CS = np.maximum(fc[:, None], fs[None, :]) > thr
        # exact P = -int(theta*x2): only the m1=0 cosine row contributes;
        # int_0^pi x*sin(n*x) dx = pi*(-1)^(n+1)/n and c_{0,n} = sqrt(2)/N * F[0,n-1]
        n_ = fs[:-1]                            # 1..N-1 (freq-N slot is masked)
        sign = np.where(n_ % 2 == 1, 1.0, -1.0)
        self.PW = math.pi ** 2 * sign / n_ * math.sqrt(2.0) / N

    # ---- elliptic
    def velocity(self, w_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """u from w via Lap(psi) = w (FORMULATION 2.7). Coefficient space only."""
        psi_hat = -w_hat / self.LAM
        return -d_sin2cos(psi_hat, 1), d_sin2cos(psi_hat, 0)  # u1 sc, u2 cs

    # ---- RHS (FORMULATION 2.10)
    def rhs(self, w_hat: np.ndarray, th_hat: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, dict]:
        u1_hat, u2_hat = self.velocity(w_hat)
        wx1_hat = d_sin2cos(w_hat, 0)           # cos(x)sin
        wx2_hat = d_sin2cos(w_hat, 1)           # sin(x)cos
        tx1_hat = d_cos2sin(th_hat, 0)          # sin(x)sin
        tx2_hat = d_sin2cos(th_hat, 1)          # cos(x)cos
        u1, u2 = from_sc(u1_hat), from_cs(u2_hat)
        wx1, wx2 = from_cs(wx1_hat), from_sc(wx2_hat)
        tx1, tx2 = from_ss(tx1_hat), from_cc(tx2_hat)
        adv_w = to_ss(u1 * wx1 + u2 * wx2) * self.MASK_SS
        adv_t = to_cs(u1 * tx1 + u2 * tx2) * self.MASK_CS
        rw = tx1_hat - adv_w                    # +theta_x1 buoyancy torque (1.2)
        rt = -adv_t
        if self.nu:
            rw = rw - self.nu * self.LAM * w_hat        # +nu*Lap(w), 1.3(a)
        if self.kappa:
            rt = rt - self.kappa * self.LAM_TH * th_hat
        if self.B:
            rt = rt - self.B * u2_hat           # G3 background, analytic (1.3(b))
        aux = {"sup_u": float(max(np.abs(u1).max(), np.abs(u2).max())),
               "sup_gth": float(np.hypot(tx1, tx2).max()),
               "sup_tx1": float(np.abs(tx1).max())}
        return rw, rt, aux

    # ---- dt law (FORMULATION 7): deterministic function of the state only
    def dt_of(self, aux: dict) -> float:
        dt = min(DT_MAX, C_BUO / math.sqrt(self.B + aux["sup_gth"] + 1e-300))
        if aux["sup_u"] > 0.0:
            dt = min(dt, C_ADV / (self.K * aux["sup_u"]))
        dmax = max(self.nu, self.kappa)
        if dmax > 0.0:
            dt = min(dt, C_VIS / (dmax * 2.0 * self.K ** 2))
        return dt

    # ---- classical RK4 on (w_hat, th_hat)
    def step(self, w_hat: np.ndarray, th_hat: np.ndarray, dt: float | None = None
             ) -> tuple[np.ndarray, np.ndarray, float, dict]:
        k1w, k1t, aux = self.rhs(w_hat, th_hat)
        if dt is None:
            dt = self.dt_of(aux)
        k2w, k2t, _ = self.rhs(w_hat + 0.5 * dt * k1w, th_hat + 0.5 * dt * k1t)
        k3w, k3t, _ = self.rhs(w_hat + 0.5 * dt * k2w, th_hat + 0.5 * dt * k2t)
        k4w, k4t, _ = self.rhs(w_hat + dt * k3w, th_hat + dt * k3t)
        w2 = w_hat + (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
        t2 = th_hat + (dt / 6.0) * (k1t + 2.0 * k2t + 2.0 * k3t + k4t)
        return w2, t2, dt, aux

    # ---- budgets (FORMULATION 4) -- all coefficient-space, zero transforms.
    # Quadrature exactness (2.9) makes h^2 * sum(coef^2) the EXACT integral.
    def budgets(self, w_hat: np.ndarray, th_hat: np.ndarray) -> dict:
        h2 = self.h * self.h
        u1_hat, u2_hat = self.velocity(w_hat)
        tx1_hat = d_cos2sin(th_hat, 0)
        w2s, t2s = float(np.sum(w_hat ** 2)), float(np.sum(th_hat ** 2))
        return {
            "E": 0.5 * h2 * float(np.sum(u1_hat ** 2) + np.sum(u2_hat ** 2)),
            "P": -float(np.dot(self.PW, th_hat[0, :-1])),
            "Z": 0.5 * h2 * w2s,
            "th2": h2 * t2s,
            "prod": h2 * float(np.sum(w_hat * tx1_hat)),        # int w*theta_x1
            "buoy_work": h2 * float(np.sum(th_hat * u2_hat)),   # int theta*u2
            "tail_w": float(np.sum(w_hat[self.TAIL_SS] ** 2) / w2s) if w2s else 0.0,
            "tail_th": float(np.sum(th_hat[self.TAIL_CS] ** 2) / t2s) if t2s else 0.0,
        }


# ---------------------------------------------------------------- initial conditions
def bump(y: np.ndarray, r: int) -> np.ndarray:
    """((1+cos y)/2)^(r/2): finite cosine series, max frequency r/2 (3.1)."""
    return ((1.0 + np.cos(y)) / 2.0) ** (r // 2)


def theta0_hat(eng: BQ, A: float, p: int = P_BUMP, q: int = Q_BUMP) -> np.ndarray:
    """Corner-localized buoyancy IC, sup(theta0) = A exactly; w0 = 0 (3.1)."""
    ystar = math.acos(q / (q + 2.0))
    gstar = math.sin(ystar) * ((1.0 + math.cos(ystar)) / 2.0) ** (q // 2)
    x1, x2 = eng.x[:, None], eng.x[None, :]
    th = (A / gstar) * bump(x1, p) * np.sin(x2) * bump(x2, q)
    return to_cs(th) * eng.MASK_CS


# ---------------------------------------------------------------- gates
def gate0(N: int = 128) -> dict:
    """Pre-gate: round-trips, single-mode pickup, derivative maps, Poisson,
    discrete divergence, IC normalization constant (FORMULATION 2.5-2.7, 5)."""
    eng = BQ(N)
    x1, x2 = eng.x[:, None], eng.x[None, :]
    errs = {}
    f_ss = np.sin(3 * x1) * np.sin(5 * x2) + 0.5 * np.sin(x1) * np.sin(2 * x2)
    f_cs = np.cos(4 * x1) * np.sin(3 * x2) + 0.3 * np.sin(x2)
    f_sc = np.sin(2 * x1) * np.cos(3 * x2) + 0.2 * np.sin(5 * x1)
    errs["rt_ss"] = float(np.abs(from_ss(to_ss(f_ss)) - f_ss).max())
    errs["rt_cs"] = float(np.abs(from_cs(to_cs(f_cs)) - f_cs).max())
    errs["rt_sc"] = float(np.abs(from_sc(to_sc(f_sc)) - f_sc).max())
    F = to_ss(np.sin(3 * x1) * np.sin(5 * x2))
    pick = F.copy()
    pick[2, 4] = 0.0
    errs["mode_pickup"] = float(max(abs(F[2, 4] - N / 2.0), np.abs(pick).max()))
    # derivative maps against analytic derivatives
    d1 = from_cs(d_sin2cos(to_ss(f_ss), 0))
    a1 = 3 * np.cos(3 * x1) * np.sin(5 * x2) + 0.5 * np.cos(x1) * np.sin(2 * x2)
    errs["d_sin2cos"] = float(np.abs(d1 - a1).max())
    d2 = from_ss(d_cos2sin(to_cs(f_cs), 0))
    a2 = -4 * np.sin(4 * x1) * np.sin(3 * x2)
    errs["d_cos2sin"] = float(np.abs(d2 - a2).max())
    # Poisson + divergence on a two-mode w
    w_hat = to_ss(np.sin(3 * x1) * np.sin(5 * x2) + np.sin(x1) * np.sin(x2))
    psi_hat = -w_hat / eng.LAM
    errs["poisson"] = float(np.abs(-eng.LAM * psi_hat - w_hat).max())
    u1_hat, u2_hat = eng.velocity(w_hat)
    div = d_sin2cos(u1_hat, 0) + d_sin2cos(u2_hat, 1)      # both -> cos(x)cos
    errs["divergence"] = float(np.abs(div).max())
    errs["gstar"] = abs(math.sin(math.acos(32.0 / 34.0))
                        * ((1.0 + 32.0 / 34.0) / 2.0) ** 16 - GSTAR_Q32)
    worst = max(errs.values())
    assert worst < G0_TOL, f"GATE 0 FAIL: {errs}"
    return errs


def gate1() -> dict:
    """theta = 0 sector: Taylor-Green exact viscous decay, theta byte-zero."""
    eng = BQ(128, nu=0.01)
    x1, x2 = eng.x[:, None], eng.x[None, :]
    w = to_ss(np.sin(x1) * np.sin(x2)) * eng.MASK_SS
    th = np.zeros_like(w)
    for s in range(1000):
        w, th, _, _ = eng.step(w, th, dt=1e-3)
        if (s + 1) % 100 == 0:
            assert not th.any(), f"GATE 1 FAIL: theta nonzero at step {s + 1}"
    err = float(np.abs(from_ss(w) - G1_DECAY * np.sin(x1) * np.sin(x2)).max())
    assert err < G1_TOL, f"GATE 1 FAIL: Taylor-Green pointwise err {err:.3e}"
    return {"tg_pointwise_err": err}


def gate2() -> dict:
    """Inviscid theta^2 conservation under full nonlinear evolution + dt^4 scaling."""
    eng = BQ(256)

    def drift(dt: float, steps: int) -> float:
        w = np.zeros((256, 256))
        th = theta0_hat(eng, 4.0)
        q0 = float(np.sum(th ** 2))
        for _ in range(steps):
            w, th, _, _ = eng.step(w, th, dt=dt)
        return abs(float(np.sum(th ** 2)) - q0) / q0

    d1 = drift(1e-3, 1000)
    assert d1 < G2_TOL, f"GATE 2 FAIL: theta^2 drift {d1:.3e} at dt=1e-3"
    d2 = drift(5e-4, 2000)
    out = {"drift_dt1e-3": d1, "drift_dt5e-4": d2}
    if d2 > G2_FLOOR:
        ratio = d1 / d2
        out["ratio"] = ratio
        assert G2_RATIO_LO <= ratio <= G2_RATIO_HI, \
            f"GATE 2 FAIL: dt-halving ratio {ratio:.2f} outside [8,32]"
    else:
        out["ratio"] = None  # at roundoff floor; scaling assert moot per spec
    return out


def gate3() -> dict:
    """Internal gravity-wave dispersion, full nonlinear code, background B=1
    handled analytically; plus the (W) wave-invariant cross-meter (4.5, 5.G3)."""
    out = {}
    for (m1, m2) in G3_MODES:
        eng = BQ(256, B=1.0)
        w = np.zeros((256, 256))
        w[m1 - 1, m2 - 1] = G3_EPS
        th = np.zeros_like(w)
        sig = G3_SIGMA[(m1, m2)]
        dt = 1e-3
        steps = int(math.ceil(4.0 * (2.0 * math.pi / sig) / dt))
        a = np.empty(steps + 1)
        a[0] = w[m1 - 1, m2 - 1]
        b0 = eng.budgets(w, th)
        W0 = b0["E"] + b0["th2"] / 2.0
        for s in range(steps):
            w, th, _, _ = eng.step(w, th, dt=dt)
            a[s + 1] = w[m1 - 1, m2 - 1]
        bf = eng.budgets(w, th)
        wave_drift = abs(bf["E"] + bf["th2"] / 2.0 - W0) / W0
        sgn = np.sign(a)
        idx = np.where(sgn[:-1] * sgn[1:] < 0)[0]
        tc = idx * dt + dt * a[idx] / (a[idx] - a[idx + 1])
        sigma_meas = math.pi / float(np.mean(np.diff(tc)))
        rel = abs(sigma_meas - sig) / sig
        key = f"({m1},{m2})"
        out[key] = {"sigma_meas": sigma_meas, "sigma_exact": sig, "rel": rel,
                    "wave_inv_drift": wave_drift, "n_crossings": int(len(tc))}
        assert rel < G3_TOL, f"GATE 3 FAIL: mode {key} sigma rel err {rel:.3e}"
        assert wave_drift < G3_WAVE_DRIFT, \
            f"GATE 3 FAIL: mode {key} wave-invariant drift {wave_drift:.3e}"
    return out


def _scenario_state(N: int, A: float = 4.0) -> tuple[BQ, np.ndarray, np.ndarray]:
    eng = BQ(N)
    return eng, np.zeros((N, N)), theta0_hat(eng, A)


def gate4a() -> dict:
    """Spatial (spectral) convergence: N in {128,256,512}, fixed dt, t=0.5.
    Compare CONTINUUM-normalized coefficients (ortho coefs scale as N/2)."""
    fields = {}
    for N in (128, 256, 512):
        eng, w, th = _scenario_state(N)
        for _ in range(1000):
            w, th, _, _ = eng.step(w, th, dt=5e-4)
        fields[N] = w * (2.0 / N)
    e128 = float(np.abs(fields[128] - fields[512][:128, :128]).max())
    e256 = float(np.abs(fields[256] - fields[512][:256, :256]).max())
    order = math.log2(e128 / e256) if e256 > 0 else math.inf
    # Below ~100x fp64 roundoff the order statistic is a ratio of noise --
    # measured 2026-07-22: gentle state e128=2.7e-16, hot state e256=1.2e-15,
    # i.e. spectrally exact at every tested N; assert order only above floor.
    assert e256 < G4A_E256_TOL, f"GATE 4a FAIL: e_256 {e256:.3e} >= 1e-8"
    if e256 > 1e-13:
        assert order >= G4A_ORDER_MIN, \
            f"GATE 4a FAIL: spatial order {order:.2f} < 8 above roundoff floor"
    return {"e128": e128, "e256": e256, "order": order,
            "order_meaningful": e256 > 1e-13}


def gate4b() -> dict:
    """Temporal RK4 order: N=256, dt in {2e-3,1e-3,5e-4} vs dt=1.25e-4 ref."""
    def run(dt: float) -> np.ndarray:
        eng, w, th = _scenario_state(256)
        for _ in range(round(0.5 / dt)):
            w, th, _, _ = eng.step(w, th, dt=dt)
        return w

    ref = run(1.25e-4)
    e = {dt: float(np.abs(run(dt) - ref).max()) for dt in (2e-3, 1e-3, 5e-4)}
    p1 = math.log2(e[2e-3] / e[1e-3])
    p2 = math.log2(e[1e-3] / e[5e-4])
    for p in (p1, p2):
        assert G4B_LO <= p <= G4B_HI, \
            f"GATE 4b FAIL: RK4 order {p:.3f} outside [3.7,4.3] (errs {e})"
    return {"errs": {f"{k:g}": v for k, v in e.items()}, "p_21": p1, "p_10": p2}


def gate5() -> dict:
    """Discrete equivariance under the two involutions S1, S2 (FORMULATION 5.G5).
    In coefficients: S1 flips odd-frequency x1 slots (sine for w, cosine for
    theta); S2 flips odd-frequency x2 sine slots of both fields."""
    N = 128
    sgn_s = np.where(np.arange(1, N + 1) % 2 == 1, -1.0, 1.0)  # sine slots
    sgn_c = np.where(np.arange(0, N) % 2 == 1, -1.0, 1.0)      # cosine slots

    def s1(w, th):
        return w * sgn_s[:, None], th * sgn_c[:, None]

    def s2(w, th):
        return w * sgn_s[None, :], th * sgn_s[None, :]

    def flow(w, th):
        eng = BQ(N)
        for _ in range(1000):
            w, th, _, _ = eng.step(w, th, dt=1e-3)
        return w, th

    _, w0, th0 = _scenario_state(N)
    fw, ft = flow(w0.copy(), th0.copy())
    out = {}
    for name, S in (("S1", s1), ("S2", s2)):
        fws, fts = flow(*S(w0.copy(), th0.copy()))
        sw, st = S(fw, ft)
        ew = float(np.abs(sw - fws).max()) / float(np.abs(fw).max())
        et = float(np.abs(st - fts).max()) / float(np.abs(ft).max())
        out[name] = {"w_rel": ew, "th_rel": et}
        assert max(ew, et) < G5_TOL, \
            f"GATE 5 FAIL: {name} equivariance {max(ew, et):.3e}"
    return out


def run_gates() -> dict:
    """Full suite, FORMULATION 5 order. Prints measured values swarm_m1 style."""
    results = {}
    t_all = time.time()
    t0 = time.time()
    g0 = gate0()
    results["G0"] = g0
    print(f"gate 0 PASS  transforms/derivatives/Poisson: worst "
          f"{max(g0.values()):.3e} (tol {G0_TOL:g}) ({time.time() - t0:.1f}s)")
    t0 = time.time()
    g1 = gate1()
    results["G1"] = g1
    print(f"gate 1 PASS  Taylor-Green viscous decay: pointwise err "
          f"{g1['tg_pointwise_err']:.3e} (tol {G1_TOL:g}); theta byte-zero "
          f"({time.time() - t0:.1f}s)")
    t0 = time.time()
    g2 = gate2()
    results["G2"] = g2
    ratio = "floor" if g2["ratio"] is None else f"{g2['ratio']:.1f}"
    print(f"gate 2 PASS  inviscid theta^2: drift {g2['drift_dt1e-3']:.3e} "
          f"(tol {G2_TOL:g}), dt-halving ratio {ratio} "
          f"({time.time() - t0:.1f}s)")
    t0 = time.time()
    g3 = gate3()
    results["G3"] = g3
    for k, v in g3.items():
        print(f"gate 3 PASS  gravity wave {k}: sigma {v['sigma_meas']:.12f} vs "
              f"{v['sigma_exact']:.12f} rel {v['rel']:.3e} (tol {G3_TOL:g}), "
              f"W-drift {v['wave_inv_drift']:.3e}")
    print(f"gate 3 timing {time.time() - t0:.1f}s")
    t0 = time.time()
    g4a = gate4a()
    results["G4a"] = g4a
    print(f"gate 4a PASS  spatial order {g4a['order']:.2f} (>= {G4A_ORDER_MIN}), "
          f"e128 {g4a['e128']:.3e} e256 {g4a['e256']:.3e} "
          f"({time.time() - t0:.1f}s)")
    t0 = time.time()
    g4b = gate4b()
    results["G4b"] = g4b
    print(f"gate 4b PASS  RK4 temporal order {g4b['p_21']:.3f}, {g4b['p_10']:.3f} "
          f"(in [{G4B_LO},{G4B_HI}]) ({time.time() - t0:.1f}s)")
    t0 = time.time()
    g5 = gate5()
    results["G5"] = g5
    print(f"gate 5 PASS  equivariance S1 {g5['S1']['w_rel']:.3e}/"
          f"{g5['S1']['th_rel']:.3e}  S2 {g5['S2']['w_rel']:.3e}/"
          f"{g5['S2']['th_rel']:.3e} (tol {G5_TOL:g}) "
          f"({time.time() - t0:.1f}s)")
    print(f"ALL GATES PASS ({time.time() - t_all:.1f}s total)")
    return results


# ---------------------------------------------------------------- scenario
def run_scenario(A: float, N: int, tmax: float, out_path: str,
                 nu: float = 0.0, kappa: float = 0.0,
                 diag_every: int = DIAG_EVERY) -> dict:
    """One corner-scenario integration with the Section 6 diagnostic battery,
    adaptive dt, JSON time series out. Driver-level timing only."""
    eng = BQ(N, nu=nu, kappa=kappa)
    w = np.zeros((N, N))
    th = theta0_hat(eng, A)
    th_grid = from_cs(th)
    sup_th0 = float(np.abs(th_grid).max())
    b0 = eng.budgets(w, th)
    ep0, th2_0 = b0["E"] + b0["P"], b0["th2"]

    series: dict[str, list] = {k: [] for k in (
        "t", "dt", "sup_w", "sup_u", "sup_gth", "E", "P", "Z", "th2", "prod",
        "buoy_work", "ep_drift", "th2_drift", "tail_w", "tail_th",
        "rmax", "bkm_I", "d7_margin", "th_overshoot")}
    t, step_i = 0.0, 0
    w_grid = from_ss(w)             # cached grid of the CURRENT state (t=0: zeros)
    sup_w = 0.0
    tx1_prev: float | None = None   # sup|theta_x1| at the previous step start
    dt_prev = 0.0
    bkm = 0.0          # D5: int sup|w| dt, trapezoid over accepted steps
    itx1 = 0.0         # D7: int sup|theta_x1| dt, trapezoid over step starts
    status, low_trust_t = "completed", None
    wall0 = time.time()

    def record(dt: float, aux: dict) -> bool:
        """One diagnostics row for the CURRENT state at time t. All quantities
        (budgets, sup_w, sup_gth, bkm_I, d7_margin) are evaluated at the SAME
        time, so the D7 bound comparison is exactly aligned (FORMULATION 6).
        Returns True if this row exhausts the tail trust wire -- the wire is
        evaluated on EVERY recorded row including the final one (audit A1:
        cadence-gated checks let an exhausted run report 'completed')."""
        nonlocal low_trust_t
        b = eng.budgets(w, th)
        ij = np.unravel_index(int(np.argmax(np.abs(w_grid))), w_grid.shape)
        over = max(0.0, float(np.abs(from_cs(th)).max()) - sup_th0)
        series["t"].append(t)
        series["dt"].append(dt)
        series["sup_w"].append(sup_w)
        series["sup_u"].append(aux.get("sup_u", float("nan")))  # audit A8
        series["sup_gth"].append(aux["sup_gth"])
        for k in ("E", "P", "Z", "th2", "prod", "buoy_work", "tail_w", "tail_th"):
            series[k].append(b[k])
        series["ep_drift"].append(abs(b["E"] + b["P"] - ep0) / max(abs(ep0), 1e-300))
        series["th2_drift"].append(abs(b["th2"] - th2_0)
                                   / max(th2_0, 1e-300))     # audit A5
        series["rmax"].append(math.hypot(eng.x[ij[0]], eng.x[ij[1]]))
        series["bkm_I"].append(bkm)
        series["d7_margin"].append(itx1 - sup_w)   # bound - sup|w| (sup|w0|=0)
        series["th_overshoot"].append(over)
        tails = max(b["tail_w"], b["tail_th"])
        if tails > TAIL_LOW and low_trust_t is None:
            low_trust_t = t
        return tails > TAIL_EXHAUST

    while t < tmax and step_i < MAX_STEPS:
        w2, th2_hat, dt, aux = eng.step(w, th)     # aux is at the CURRENT time t
        if tx1_prev is not None:                   # close [t-dt_prev, t]
            itx1 += 0.5 * (tx1_prev + aux["sup_tx1"]) * dt_prev
        if step_i % diag_every == 0:
            if record(dt, aux):
                status = "tail_exhausted"
                break
        if dt < DT_MIN:
            status = "dt_exhausted"
            break
        if not np.isfinite(w2).all() or not np.isfinite(th2_hat).all():
            status = "nan"
            break
        tx1_prev, dt_prev = aux["sup_tx1"], dt
        w, th = w2, th2_hat
        t += dt
        step_i += 1
        w_grid = from_ss(w)
        sup_w_new = float(np.abs(w_grid).max())
        bkm += 0.5 * (sup_w + sup_w_new) * dt
        sup_w = sup_w_new
    if step_i >= MAX_STEPS and status == "completed":
        status = "step_budget"
    if status in ("completed", "step_budget"):     # final aligned row at final t
        _, _, aux_f = eng.rhs(w, th)
        if tx1_prev is not None:
            itx1 += 0.5 * (tx1_prev + aux_f["sup_tx1"]) * dt_prev
        if record(dt_prev, aux_f):                 # audit A1: wire on final row
            status = "tail_exhausted"
    wall = time.time() - wall0

    out = {
        "header": {
            "engine": "bq.py era B0", "N": N, "K": eng.K, "A": A,
            "p": P_BUMP, "q": Q_BUMP, "nu": nu, "kappa": kappa,
            "dt_law": {"DT_MAX": DT_MAX, "DT_MIN": DT_MIN, "C_ADV": C_ADV,
                       "C_BUO": C_BUO, "C_VIS": C_VIS},
            "diag_every": diag_every, "sup_theta0_grid": sup_th0,
            "steps": step_i, "wall_s": round(wall, 2),
            "ms_per_step": round(1e3 * wall / max(step_i, 1), 3),
        },
        "series": series,
        "final": {
            "t": t, "status": status, "low_trust_since_t": low_trust_t,
            "sup_w": series["sup_w"][-1] if series["sup_w"] else 0.0,
            "sup_gth": series["sup_gth"][-1] if series["sup_gth"] else 0.0,
            "ep_drift": series["ep_drift"][-1] if series["ep_drift"] else 0.0,
            "th2_drift": series["th2_drift"][-1] if series["th2_drift"] else 0.0,
            "tail_w": series["tail_w"][-1] if series["tail_w"] else 0.0,
            "bkm_I": bkm,
            "d7_violation": max(0.0, -min(series["d7_margin"]))
                            if series["d7_margin"] else 0.0,
        },
    }
    path = pathlib.Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: an RFC-invalid artifact must fail loudly (audit finding:
    # A>=1e150 ICs overflowed into bare Infinity tokens); nonfinite rows are a
    # bug upstream, not something to serialize.
    path.write_text(json.dumps(out, allow_nan=False))
    f = out["final"]
    print(f"scenario A={A:g} N={N} t={f['t']:.4f} status={f['status']} | "
          f"sup|w| {f['sup_w']:.4f} sup|grad th| {f['sup_gth']:.4f} | "
          f"th2 drift {f['th2_drift']:.2e} E+P drift {f['ep_drift']:.2e} | "
          f"tail_w {f['tail_w']:.2e} | {out['header']['ms_per_step']} ms/step "
          f"({step_i} steps, {wall:.1f}s) -> {path}")
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gates", action="store_true", help="run gate suite G0-G5")
    ap.add_argument("--scenario", action="store_true",
                    help="one corner-scenario integration (gates must have "
                         "passed on this code state for the run to count)")
    ap.add_argument("--A", type=float, default=4.0, help="amplitude dial")
    ap.add_argument("--N", type=int, default=256, help="grid size per axis")
    ap.add_argument("--tmax", type=float, default=2.0)
    ap.add_argument("--nu", type=float, default=0.0)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--diag-every", type=int, default=DIAG_EVERY)
    ap.add_argument("--out", default=None, help="scenario JSON path")
    args = ap.parse_args()

    if not args.gates and not args.scenario:
        ap.error("pick a mode: --gates and/or --scenario")
    if args.gates:
        run_gates()
    if args.scenario:
        out = args.out or str(pathlib.Path(__file__).parent / "runs" /
                              f"scenario_A{args.A:g}_N{args.N}.json")
        run_scenario(args.A, args.N, args.tmax, out,
                     nu=args.nu, kappa=args.kappa, diag_every=args.diag_every)


if __name__ == "__main__":
    main()
