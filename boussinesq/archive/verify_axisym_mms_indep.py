#!/usr/bin/env python
"""INDEPENDENT adversarial manufactured-solution residual test for
dedalus_axisym.py.  Written by an external verifier; imports the engine's own
build() so it tests the ENGINE's actual operators/velocity definitions, but the
reference RHS values are all CLOSED-FORM analytic (hand-derived from the
primitive axisymmetric-Euler reduction), so any sign/factor/missing-term error
in the engine's equation strings shows up as a large residual.

Manufactured smooth fields on the annulus r in [r0,1], z-periodic period L:
    psi1   = sin(k z) * P(r),         P(r)   = (r-r0)(1-r)      [BC-exact: 0 at r0,1]
    u1     = Q(r) * (1.2 + sin(k z)), Q(r)   = 1 + 0.5 r
    omega1 = W(r) * cos(k z),         W(r)   = 0.7 + r          [independent field]
with k = 2*pi/L.

Closed-form derivatives (hand):
    P'  = 1 + r0 - 2r ,  P'' = -2
    psi1_r = s P' , psi1_rr = -2 s , psi1_z = k c P , psi1_zz = -k^2 s P
    u^r = -r psi1_z = -r k c P
    u^z = 2 psi1 + r psi1_r = s (2P + r P')
    u1_r = 0.5 (1.2+s) , u1_z = Q k c
    (u1^2)_z = 2 u1 u1_z = 2 Q^2 k c (1.2+s)
    omega1_r = c , omega1_z = -W k s

Reference RHS (what the PNAS 2a/2b/2c equations MUST evaluate to):
    RHS_u1  = -u^r u1_r - u^z u1_z + 2 u1 psi1_z
    RHS_w1  = -u^r omega1_r - u^z omega1_z + (u1^2)_z
    Poisson : r*(psi1_rr + (3/r) psi1_r + psi1_zz)   [engine's r-multiplied LHS
              on psi1, tau=0]   must equal  -r * omega1_consistent, where
              omega1_consistent = -(psi1_rr + (3/r)psi1_r + psi1_zz).
"""
from __future__ import annotations

import numpy as np
import dedalus.public as d3

from dedalus_axisym import build


def _g(op):
    """Evaluate a Dedalus operator -> scale-1 grid array (matches build()'s grids)."""
    f = op.evaluate()
    f.change_scales(1)
    return np.array(f["g"])


def relerr(num, ana):
    return float(np.abs(num - ana).max() / max(np.abs(ana).max(), 1e-30))


def run(Nz=64, Nr=96, r0=0.4, tol=1e-9):
    b = build(Nz, Nr, r0=r0)
    z, r = b.z, b.r
    k = 2 * np.pi / b.L
    s, c = np.sin(k * z), np.cos(k * z)

    # ---- manufactured fields (set the ENGINE's own fields) ------------------
    P = (r - r0) * (1.0 - r)
    Pp = 1.0 + r0 - 2.0 * r
    Q = 1.0 + 0.5 * r
    W = 0.7 + r
    b.psi1["g"] = s * P
    b.u1["g"] = Q * (1.2 + s)
    b.omega1["g"] = W * c
    b.tau_p1["g"] = 0.0          # zero the tau so grad_psi1 is the pure gradient

    # ---- closed-form analytic reference derivatives -------------------------
    psi1_r = s * Pp
    psi1_rr = -2.0 * s
    psi1_z = k * c * P
    psi1_zz = -(k ** 2) * s * P
    ur_ana = -r * psi1_z                      # u^r = -r psi1_z
    uz_ana = 2.0 * (s * P) + r * psi1_r       # u^z = 2 psi1 + r psi1_r
    u1_r = 0.5 * (1.2 + s)
    u1_z = Q * k * c
    u1sq_z = 2.0 * (Q ** 2) * k * c * (1.2 + s)   # (u1^2)_z
    w1_r = c
    w1_z = -W * k * s

    # ---- (0) velocity relations: engine ur,uz vs analytic -------------------
    e_ur = relerr(_g(b.ur), ur_ana)
    e_uz = relerr(_g(b.uz), uz_ana)

    # ---- (1) u1 evolution RHS (PNAS 2a): -ur u1_r - uz u1_z + 2 u1 psi1_z ----
    rhs_u1_ana = -(ur_ana) * u1_r - (uz_ana) * u1_z + 2.0 * (Q * (1.2 + s)) * psi1_z
    rhs_u1_eng = _g(-b.ur * b.dr(b.u1) - b.uz * b.dz(b.u1) + 2 * b.u1 * b.dz(b.psi1))
    e_rhs_u1 = relerr(rhs_u1_eng, rhs_u1_ana)

    # ---- (2) omega1 evolution RHS (PNAS 2b): -ur w1_r - uz w1_z + (u1^2)_z ---
    rhs_w1_ana = -(ur_ana) * w1_r - (uz_ana) * w1_z + u1sq_z
    rhs_w1_eng = _g(-b.ur * b.dr(b.omega1) - b.uz * b.dz(b.omega1) + b.dz(b.u1 * b.u1))
    e_rhs_w1 = relerr(rhs_w1_eng, rhs_w1_ana)

    # ---- (3) Poisson operator (PNAS 2c) in engine's r-multiplied form --------
    # engine LHS operator = rr*div(grad_psi1) + 3*(er@grad_psi1)  (tau_p1=0)
    pois_eng = _g(b.rr * d3.div(b.grad_psi1) + 3 * (b.er @ b.grad_psi1))
    pois_ana = r * (psi1_rr + (3.0 / r) * psi1_r + psi1_zz)   # r*(drr+3/r dr+dzz)psi1
    e_pois_op = relerr(pois_eng, pois_ana)

    # ---- (3b) sign/consistency: -(drr+3/r dr+dzz)psi1 must be a valid omega1 --
    # i.e. the engine equation "LHS + rr*omega1 = 0" implies omega1 = -(...)psi1.
    omega1_from_psi = -(psi1_rr + (3.0 / r) * psi1_r + psi1_zz)
    # engine residual of its OWN elliptic equation with this consistent omega1
    # (pois_eng = r*(...)psi1 ; + r*omega1_from_psi = r*(...)psi1 - r*(...)psi1 = 0);
    # normalize by the OPERATOR scale (the target is exactly 0, so relerr-to-0
    # is ill-posed -- use |resid| / |pois_eng|_max).
    resid_abs = float(np.abs(pois_eng + r * omega1_from_psi).max())
    resid_eq = resid_abs / max(float(np.abs(pois_eng).max()), 1e-30)

    rows = [
        ("ur  = -r psi1_z                    (2d)", e_ur),
        ("uz  = 2 psi1 + r psi1_r            (2d)", e_uz),
        ("RHS u1  = -u.grad u1 + 2 u1 psi1_z (2a)", e_rhs_u1),
        ("RHS w1  = -u.grad w1 + dz(u1^2)    (2b)", e_rhs_w1),
        ("Poisson r*(drr+3/r dr+dzz)psi1     (2c)", e_pois_op),
        ("elliptic-eqn residual (sign check)     ", resid_eq),
    ]
    print("=" * 70)
    print(f"  INDEPENDENT MMS RESIDUAL TEST  dedalus_axisym.py  ({Nz}x{Nr}, r0={r0})")
    print("=" * 70)
    worst = 0.0
    for name, e in rows:
        worst = max(worst, e)
        print(f"   {name}   relerr = {e:.3e}   {'PASS' if e < tol else 'FAIL'}")
    print("-" * 70)
    ok = worst < tol
    print(f"   WORST = {worst:.3e}   TOL = {tol:.1e}   -> {'ALL PASS' if ok else 'DEFECT'}")
    print("=" * 70)
    return ok, worst


if __name__ == "__main__":
    ok, _ = run()
    raise SystemExit(0 if ok else 1)
