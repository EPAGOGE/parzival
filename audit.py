#!/usr/bin/env python
"""Run-time hardcheck instruments for the swarm engine (tier-Q machinery).

Validated 2026-07-22 (fp64, N=128, gCLM a=0.93):

M2 aliasing meter -- the continuum enstrophy budget
      <w, rhs(w)> = (1 + a/2) * Sum[u_x w^2] - nu * <w, Lambda w>
  holds discretely at 1.4e-16 on smooth states, 5.8e-12 on the edge state,
  and degrades to 1.8e-2 on a sup~400 blowup transient. Its residual is a
  LIVE per-lane resolution meter: the discrete violation IS the aliasing
  error of the triple product. Catches under-resolution mid-flight, not at
  fate time.

M1 step tripwire -- energy change across one RK4 macro step vs Simpson
  quadrature of <w, rhs> over the stage states. Scales ~dt^2 (calibrate the
  threshold per era/dt); a hardware bit-flip or NaN precursor jumps it by
  orders of magnitude. Statistical SDC detection for the hot loop.

Tier-Q cross-method auditor -- scipy BDF (implicit multistep + adaptive
  error control + analytic Jacobian: an INDEPENDENT integrator family)
  re-resolves sampled lanes. Validated: reproduces the a=0.93 edge state to
  8.4e-5 over t=5 (under lam_u ~ 1.1 amplification) and agrees on
  near-threshold fates 0.07 apart in A. ~1 s per lane on CPU.

Wiring plan (POST-campaign, revs the meter era): swarm_m1 gains --audit K,
evaluating M1+M2 on every lane every K macro steps (device-side, a few
matmuls), auto-flagging violators into the ledger's lowtrust channel with a
distinct 'corrupt' counter; the BDF auditor spot-checks ~1% of resolved
boundary-cell fates per canary. Never a gradient/steering target -- monitors
only (the probe-is-not-the-loss law).
"""
from __future__ import annotations

import numpy as np


def m2_alias_residual(w: np.ndarray, a: float, nu: float, ops: dict) -> float:
    """Relative violation of the continuum enstrophy budget = live aliasing
    error. ops: {"H","L","D","G"} dense operators (rows = grid points)."""
    u, ux = ops["G"] @ w, ops["H"] @ w
    wx = ops["D"] @ w
    lhs = float(w @ (ux * w - a * u * wx - nu * (ops["L"] @ w)))
    rhs = float((1 + a / 2) * np.sum(ux * w ** 2) - nu * (w @ (ops["L"] @ w)))
    scale = abs(float(np.sum(np.abs(ux) * w ** 2))) + 1e-30
    return abs(lhs - rhs) / scale


def m1_step_residual(w0: np.ndarray, w1: np.ndarray, stages: tuple, dt: float,
                     rhs_fn) -> float:
    """Relative mismatch between the realized energy change across a macro
    step and the Simpson quadrature of <w, rhs>. stages = (k1, k2) from the
    RK4 evaluation (reuse, don't recompute)."""
    k1, k2 = stages
    dE = 0.5 * float(w1 @ w1 - w0 @ w0)
    mid = w0 + 0.5 * dt * k2
    q = (dt / 6) * (float(w0 @ k1) + 4 * float(mid @ rhs_fn(mid))
                    + float(w1 @ rhs_fn(w1)))
    return abs(dE - q) / (abs(dE) + 1e-30)


def bdf_refate(w0: np.ndarray, a: float, nu: float, ops: dict,
               a0_amp: float, t_max: float = 12.0,
               rtol: float = 1e-8) -> tuple[int, float]:
    """Independent-family fate check: scipy BDF with analytic Jacobian.
    Returns (fate, t_end); fate 1=blowup, 0=decay, -1=unresolved."""
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        u, ux, yx = ops["G"] @ y, ops["H"] @ y, ops["D"] @ y
        return ux * y - a * u * yx - nu * (ops["L"] @ y)

    def jac(t, y):
        u, ux, yx = ops["G"] @ y, ops["H"] @ y, ops["D"] @ y
        return (np.diag(y) @ ops["H"] + np.diag(ux)
                - a * (np.diag(yx) @ ops["G"] + np.diag(u) @ ops["D"])
                - nu * ops["L"])

    hit = lambda t, y: 1e3 - np.max(np.abs(y))
    hit.terminal, hit.direction = True, -1
    low = lambda t, y: np.max(np.abs(y)) - 0.1 * a0_amp
    low.terminal, low.direction = True, -1
    sol = solve_ivp(rhs, (0.0, t_max), w0, method="BDF", jac=jac,
                    rtol=rtol, atol=rtol * 1e-2, events=(hit, low))
    if sol.t_events[0].size:
        return 1, float(sol.t_events[0][0])
    if sol.t_events[1].size:
        return 0, float(sol.t_events[1][0])
    return -1, float(sol.t[-1])
