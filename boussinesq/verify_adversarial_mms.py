#!/usr/bin/env python
"""Adversarial manufactured-solution residual test for bq.py (independent verifier).

Written by an external verifier, NOT by bq.py's authors. The reference path
shares nothing with the engine implementation:

  - analytic fields and every analytic derivative come from explicit mode
    lists evaluated with np.sin/np.cos only (no scipy.fft, no bq derivative
    maps, no bq transforms);
  - the engine INPUT coefficients are placed directly into slots from
    first-principles orthonormal weights (sqrt(2/N) generic, sqrt(1/N) for
    cosine m=0), which are themselves PROVEN on the fly by asserting the
    hand-built basis matrices are orthogonal;
  - the engine OUTPUT is compared in coefficient space against my own
    O(N^3) matrix-quadrature projection of the analytic right-hand side.

Coverage: transforms + slot->frequency map, Poisson solve, velocity signs,
all four derivative routes, buoyancy-torque sign (+theta_x1), both diffusion
signs, the -B*u2 background term, Parseval budgets (E, th2, prod, buoy_work),
the analytic P weight vector, and dealias-mask semantics at the band edge
(modes at freq 40 of K=42 whose products alias on the N=64 grid: the alias
images must land above K and be killed, leaving in-band content exact).

Everything with nu=0.013, kappa=0.007, B=0.6 simultaneously nonzero.
"""
import math
import sys

import numpy as np

sys.path.insert(0, "/Users/epagogellc/parzival/boussinesq")
from bq import BQ

N = 64
NU, KAPPA, B = 0.013, 0.007, 0.6
TOL = 1e-12          # relative, against the max magnitude of each reference

eng = BQ(N, nu=NU, kappa=KAPPA, B=B)
h = math.pi / N
x = (np.arange(N) + 0.5) * h
X1, X2 = x[:, None], x[None, :]

# ---------------------------------------------------------------- my own bases
S = np.sin(np.outer(np.arange(1, N + 1), x))     # sine  slot k -> frequency k+1
C = np.cos(np.outer(np.arange(0, N), x))         # cosine slot k -> frequency k
ws = np.full(N, math.sqrt(2.0 / N)); ws[N - 1] = math.sqrt(1.0 / N)
wc = np.full(N, math.sqrt(2.0 / N)); wc[0] = math.sqrt(1.0 / N)
BS = ws[:, None] * S
BC = wc[:, None] * C
# prove the weight table (FORMULATION 2.5) from first principles:
assert np.abs(BS @ BS.T - np.eye(N)).max() < 1e-12, "my sine basis not orthonormal"
assert np.abs(BC @ BC.T - np.eye(N)).max() < 1e-12, "my cosine basis not orthonormal"

def proj_ss(f): return BS @ f @ BS.T             # grid -> sin(x)sin coefficients
def proj_cs(f): return BC @ f @ BS.T             # grid -> cos(x)sin coefficients

K = (2 * N - 1) // 3                             # my own dealias band, re-derived
fs, fc = np.arange(1, N + 1), np.arange(0, N)
MY_MASK_SS = np.outer(fs <= K, fs <= K).astype(float)
MY_MASK_CS = np.outer(fc <= K, fs <= K).astype(float)
assert K == eng.K, f"band mismatch: mine {K} vs engine {eng.K}"

# ---------------------------------------------------------------- manufactured state
# psi modes (m, n, a): psi = sum a*sin(m x1)*sin(n x2)
psi_modes = [(1, 2, 0.70), (3, 1, -0.40), (2, 5, 0.23), (1, 40, 0.011)]
# theta modes (m, n, b): theta = sum b*cos(m x1)*sin(n x2); m=0 exercised on
# purpose; (1,2) overlaps a psi mode so prod and buoy_work are nontrivial
th_modes = [(0, 4, 0.55), (2, 1, -0.83), (5, 3, 0.31), (40, 2, -0.009),
            (1, 2, 0.12)]

Z = np.zeros((N, N))
w_g = Z.copy(); u1_g = Z.copy(); u2_g = Z.copy()
wx1_g = Z.copy(); wx2_g = Z.copy(); lapw_g = Z.copy()
for m, n, a in psi_modes:
    sm, cm = np.sin(m * X1), np.cos(m * X1)
    sn, cn = np.sin(n * X2), np.cos(n * X2)
    k2 = m * m + n * n
    w_g    += -k2 * a * sm * sn
    u1_g   += -a * n * sm * cn
    u2_g   += +a * m * cm * sn
    wx1_g  += -k2 * a * m * cm * sn
    wx2_g  += -k2 * a * n * sm * cn
    lapw_g += k2 * k2 * a * sm * sn

th_g = Z.copy(); tx1_g = Z.copy(); tx2_g = Z.copy(); lapt_g = Z.copy()
for m, n, b in th_modes:
    sm, cm = np.sin(m * X1), np.cos(m * X1)
    sn, cn = np.sin(n * X2), np.cos(n * X2)
    k2 = m * m + n * n
    th_g   += b * cm * sn
    tx1_g  += -b * m * sm * sn
    tx2_g  += b * n * cm * cn
    lapt_g += -k2 * b * cm * sn

# analytic right-hand sides on the grid (continuum truth, pre-truncation)
rw_g = tx1_g - (u1_g * wx1_g + u2_g * wx2_g) + NU * lapw_g
rt_g = -(u1_g * tx1_g + u2_g * tx2_g) + KAPPA * lapt_g - B * u2_g

# ---------------------------------------------------------------- engine input
# placed DIRECTLY from my weights -- no engine transform in the input path
w_hat = np.zeros((N, N))
for m, n, a in psi_modes:
    w_hat[m - 1, n - 1] = -(m * m + n * n) * a / (ws[m - 1] * ws[n - 1])
th_hat = np.zeros((N, N))
for m, n, b in th_modes:
    th_hat[m, n - 1] = b / (wc[m] * ws[n - 1])

# input self-check: my placement against my own quadrature projection
assert np.abs(proj_ss(w_g) - w_hat).max() < 1e-10 * np.abs(w_hat).max()
assert np.abs(proj_cs(th_g) - th_hat).max() < 1e-10 * np.abs(th_hat).max()

# ---------------------------------------------------------------- the test
rw_eng, rt_eng, aux = eng.rhs(w_hat, th_hat)

# Galerkin reference: project the analytic RHS with MY matrices, mask with MY
# mask. In-band alias-freedom (3K < 2N) makes this exact for freq <= K.
rw_ref = proj_ss(rw_g) * MY_MASK_SS
rt_ref = proj_cs(rt_g) * MY_MASK_CS

res_w = np.abs(rw_eng - rw_ref).max() / np.abs(rw_ref).max()
res_t = np.abs(rt_eng - rt_ref).max() / np.abs(rt_ref).max()
print(f"MMS residual  w-equation: {res_w:.3e}   (scale {np.abs(rw_ref).max():.3e})")
print(f"MMS residual  theta-eq  : {res_t:.3e}   (scale {np.abs(rt_ref).max():.3e})")
assert res_w < TOL, f"FAIL: w-equation residual {res_w:.3e}"
assert res_t < TOL, f"FAIL: theta-equation residual {res_t:.3e}"

# ---------------------------------------------------------------- budgets
bud = eng.budgets(w_hat, th_hat)
h2 = h * h
ref = {
    "E":         0.5 * h2 * float(np.sum(u1_g ** 2 + u2_g ** 2)),
    "Z":         0.5 * h2 * float(np.sum(w_g ** 2)),
    "th2":       h2 * float(np.sum(th_g ** 2)),
    "prod":      h2 * float(np.sum(w_g * tx1_g)),
    "buoy_work": h2 * float(np.sum(th_g * u2_g)),
    # P = -int(theta*x2): closed form, only m=0 modes survive the x1 integral
    "P": -sum(b * math.pi * (math.pi * (-1) ** (n + 1) / n)
              for m, n, b in th_modes if m == 0),
}
for k, v in ref.items():
    got = bud[k]
    rel = abs(got - v) / max(abs(v), 1e-300)
    print(f"budget {k:9s}: engine {got:+.15e}  ref {v:+.15e}  rel {rel:.3e}")
    assert rel < 1e-12, f"FAIL: budget {k} rel err {rel:.3e}"

# aux sanity: sup|grad theta| against my analytic gradient
sg_ref = float(np.hypot(tx1_g, tx2_g).max())
rel = abs(aux["sup_gth"] - sg_ref) / sg_ref
print(f"aux sup_gth   : engine {aux['sup_gth']:.15e}  ref {sg_ref:.15e}  rel {rel:.3e}")
assert rel < 1e-12

print("ALL MMS CHECKS PASS")
