#!/usr/bin/env python
"""3D AXISYMMETRIC EULER (with swirl) corner-blowup engine, in DEDALUS.

Luo-Hou / Chen-Hou formulation -- the transformed axisymmetric Euler system in
the rescaled variables u1 = u^theta/r, omega1 = omega^theta/r, psi1 = psi^theta/r
(psi = vector potential, u = curl(psi)).  Verified against the primary sources
(Luo & Hou PNAS 111(36) 2014 eqs 2a-2d/3a-3b/6; Luo & Hou SIAM MMS 12(4) 2014
eqs 2.1-2.2/Table 1; cross-checked vs the unified Boussinesq-Euler reduction).

    STATE (prognostic):  u1(z,r,t)  [swirl / angular-velocity variable]
                         omega1(z,r,t)  [meridional vorticity variable]
    DIAGNOSTIC (slaved each stage):  psi1(z,r,t)  from the Poisson constraint

    VEL   u^r = - r psi1_z ,          u^z = 2 psi1 + r psi1_r      (PNAS 2d)
    EVOLVE  u1_t   = - u^r u1_r - u^z u1_z + 2 u1 psi1_z           (PNAS 2a)
            omega1_t = - u^r omega1_r - u^z omega1_z + dz(u1^2)    (PNAS 2b)
    ELLIPTIC  -(drr + (3/r) dr + dzz) psi1 = omega1               (PNAS 2c)
    IC   u1 = A exp(-30(1-r^2)) sin(2 pi z / L),  omega1 = psi1 = 0  (PNAS 3a)
         L = 1/6 (z period);  A = 100 reproduces the paper exactly.
    TARGET  ring singularity at the WALL CORNER (r,z) = (1,0), ts ~= 0.0035056.

GEOMETRY -- ANNULUS AWAY FROM THE AXIS (a localization, NOT a shortcut).
The singularity of this flow forms on the SOLID BOUNDARY, at the corner ring
(r=1, z=0); the field is a thin swirling tube pinned against r=1 (the IC decays
like exp(-30(1-r^2)), i.e. ~1e-11 already at r=0.4).  We therefore solve on the
annulus r in [r0, 1] with r0 ~ 0.3-0.5, which EXCLUDES the coordinate axis r=0
and hence the singular (3/r) term entirely -- on [r0,1] the coefficient 1/r is
smooth (bounded, analytic) and Dedalus represents it exactly.  This is precisely
the near-wall localization Luo-Hou / Chen-Hou exploit; the axis plays no role in
the corner blow-up.  At r=r0 we impose the homogeneous no-flow condition psi1=0
(the fields are exponentially small there for the whole approach to ts), and at
r=1 the physical no-flow wall psi1=0 (=> u^r = -r psi1_z = 0 on the wall).

CONDITIONING.  The elliptic equation is solved in r-multiplied form
    (r drr + 3 dr + r dzz) psi1 = - r omega1 ,
whose only non-constant coefficient is r itself (a degree-1 polynomial, exact in
Chebyshev) -- no 1/r series truncation anywhere.  A first-order gradient
reduction with a rank-1 tau (er*Lift) plus a scalar tau enforces the two
Dirichlet conditions (well-conditioned sparse tau solve; gate G2 verifies a
manufactured recovery to ~1e-15).

NUMERICS.  z: RealFourier (periodic, period L; carries the z-parity of the IC).
r: ChebyshevT on [r0,1] (wall-clustered, resolves the collapsing corner layer).
IMEX RK443, full 3/2 dealiasing, a Hou-Li exponential spectral filter each step
(copied from the validated dedalus_bsq.py), and a tight adaptive CFL.  fp64.

CROSS-CHECK (vs the validated 2D Boussinesq engine dedalus_bsq.py).  Under
theta := (r^2 u1)^2 (squared angular momentum) the system maps, near r=1, to
Dtheta/Dt=0, Domega1/Dt = (1/r^4) dz(theta) ~ dz(theta), -Delta psi1 = omega1 --
the flat-Boussinesq limit.  So theta plays the role of the Boussinesq buoyancy b
and omega1 the role of the Boussinesq vorticity; their blow-up rates must
coincide: omega1 ~ (ts-t)^-1, sup|grad theta| ~ (ts-t)^-2, theta bounded
(materially conserved).  The physical peak vorticity target is the robust BKM
headline ||omega||_inf = sup|omega^theta| = sup|r omega1| ~ (ts-t)^-2.46.

HARD RULES honoured: new file only, fp64, no wall-clock / randomness in physics.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from types import SimpleNamespace

import numpy as np
import dedalus.public as d3

# ---- pinned physical constants (Luo-Hou / Chen-Hou) --------------------------
L_PERIOD = 1.0 / 6.0          # z period (PNAS/MMS: L = 1/6)
IC_DECAY = 30.0               # exp(-30(1-r^2)^IC_POWER)  (PNAS eq 3a)
IC_POWER = 4.0                # power on (1-r^2). PNAS eq 3a is the FOURTH power.
                              # WAS 1.0, which contradicted the paper this file
                              # implements and silently confounded two ladders run
                              # with and without --ic-power 4. The paper-faithful
                              # value is now the DEFAULT; runs record it under
                              # res['ic'] so an artifact can never again be unable
                              # to say what IC produced it.
                              # power; the first power concentrates all swirl
                              # into r in [0.983,1] instead of [0.76,1] -- a
                              # 25x thinner layer, i.e. a different problem.
IC_AMP = 100.0               # u1^0 prefactor (PNAS eq 3a)
TS_REF = 0.0035056           # pre-registered singularity time (MMS Table 1)
TRUST_DRIFT = 1e-3           # Casimir-drift trust boundary (matches bsq engine)


# ---- Hou-Li spectral filter (verbatim pattern from dedalus_bsq.py) -----------
def make_filter(field, alpha=36.0, order=36, cutoff=0.65):
    """Hou-Li exponential filter (J.Comput.Phys 2007): ~1 up to cutoff*k_max then
    a very high-order rolloff that annihilates the aliasing-contaminated top
    modes while preserving the physical spectrum to roundoff."""
    csh = field["c"].shape
    iz = np.arange(csh[0]) / max(csh[0] - 1, 1)
    ir = np.arange(csh[1]) / max(csh[1] - 1, 1)
    IZ, IR = np.meshgrid(iz, ir, indexing="ij")
    kk = np.sqrt((IZ ** 2 + IR ** 2) / 2.0)
    return np.exp(-alpha * (np.maximum(kk - cutoff, 0.0) / (1 - cutoff)) ** order)


def tail_fraction(field, frac=2.0 / 3.0):
    """Fraction of spectral energy above `frac`*k_max (resolution health)."""
    c = np.abs(field["c"]) ** 2
    iz = np.arange(c.shape[0]) / max(c.shape[0] - 1, 1)
    ir = np.arange(c.shape[1]) / max(c.shape[1] - 1, 1)
    IZ, IR = np.meshgrid(iz, ir, indexing="ij")
    kk = np.sqrt((IZ ** 2 + IR ** 2) / 2.0)
    tot = float(c.sum())
    return float(c[kk > frac].sum() / tot) if tot > 0 else 0.0


def _scalar(op):
    """Global scalar from a d3 integral. Under MPI the scalar lives on ONE rank and
    the others see an empty array -> 0.0. Allreduce-SUM makes it correct everywhere.
    Without this, rank 0 (which owns the stream file) recorded Casimir drift = 0
    forever and the trust check could never trip -- a silent loss of the referee."""
    v = op.evaluate()["g"]
    loc = float(v.flat[0]) if np.size(v) else 0.0
    from mpi4py import MPI as _M
    return float(_M.COMM_WORLD.allreduce(loc, op=_M.SUM))


def _gval(op):
    """Evaluate an operator and return its grid data on the scale-1 grid, so it
    is directly comparable with analytic arrays built from the scale-1 grids."""
    f = op.evaluate()
    f.change_scales(1)
    return f["g"]


# ---- shared build: bases, fields, operators (used by gates AND scenario) -----
def build(Nz, Nr, r0=0.4, L=L_PERIOD, dealias=3.0 / 2.0):
    """Construct the (z: RealFourier) x (r: ChebyshevT annulus [r0,1]) discretely:
    fields u1, omega1, psi1, taus, the r-NCC, and the derived velocity operators.
    Returns a namespace; the IVP/LBVP are assembled from it on demand."""
    coords = d3.CartesianCoordinates("z", "r")          # r (Chebyshev) LAST axis
    dist = d3.Distributor(coords, dtype=np.float64)
    zb = d3.RealFourier(coords["z"], size=Nz, bounds=(0, L), dealias=dealias)
    rb = d3.ChebyshevT(coords["r"], size=Nr, bounds=(r0, 1.0), dealias=dealias)
    z = dist.local_grid(zb)                              # shape (Nz, 1)
    r = dist.local_grid(rb)                              # shape (1, Nr)
    ez, er = coords.unit_vector_fields(dist)             # order = (z, r)

    psi1 = dist.Field(name="psi1", bases=(zb, rb))
    u1 = dist.Field(name="u1", bases=(zb, rb))
    omega1 = dist.Field(name="omega1", bases=(zb, rb))
    tau_p1 = dist.Field(name="tau_p1", bases=zb)         # gradient-reduction tau
    tau_p2 = dist.Field(name="tau_p2", bases=zb)         # 2nd Dirichlet tau
    rr = dist.Field(name="rr", bases=rb)                 # r NCC (degree-1, exact)
    rr["g"] = r

    dr = lambda F: d3.Differentiate(F, coords["r"])
    dz = lambda F: d3.Differentiate(F, coords["z"])
    lift = lambda F: d3.Lift(F, rb.derivative_basis(1), -1)
    grad_psi1 = d3.grad(psi1) + er * lift(tau_p1)        # first-order reduction
    # velocity recovery (PNAS eq 2d): incompressible meridional field
    ur = -rr * dz(psi1)                                  # u^r = -r psi1_z
    uz = 2 * psi1 + rr * dr(psi1)                        # u^z = 2 psi1 + r psi1_r
    u_vec = uz * ez + ur * er

    return SimpleNamespace(**locals())


def _poisson_lhs():
    """The r-multiplied elliptic operator string, shared by IVP and LBVP.
    rr*div(grad_psi1) + 3*(er@grad_psi1) == (r drr + 3 dr + r dzz) psi1 (+taus)."""
    return "rr*div(grad_psi1) + 3*(er@grad_psi1)"


def make_ivp(b, stepper="RK443"):
    ns = vars(b)
    problem = d3.IVP([b.psi1, b.u1, b.omega1, b.tau_p1, b.tau_p2], namespace=ns)
    # elliptic constraint (algebraic; no dt): (r drr+3dr+r dzz)psi1 = -r omega1
    problem.add_equation(_poisson_lhs() + " + lift(tau_p2) + rr*omega1 = 0")
    problem.add_equation("psi1(r=1) = 0")               # solid wall no-flow
    problem.add_equation("psi1(r=r0) = 0")              # annulus far-field no-flow
    # advected fields: inviscid -> LHS is dt only, all spatial terms explicit RHS
    problem.add_equation("dt(u1) = -ur*dr(u1) - uz*dz(u1) + 2*u1*dz(psi1)")
    problem.add_equation("dt(omega1) = -ur*dr(omega1) - uz*dz(omega1) + dz(u1*u1)")
    solver = problem.build_solver(getattr(d3, stepper))
    return problem, solver


def set_ic(b, A=IC_AMP, power=IC_POWER, zpow=1, wamp=0.0, wpow=3):
    """Pinned IC (PNAS eq 3a): pure swirl, meridionally at rest."""
    # DEGENERACY LADDER. u1 is ODD about z=0 (gate G5, 1.4e-14), and the Boussinesq
    # analogue is theta <-> u1^2, so theta ~ z^(2*zpow) and only EVEN-ODD products are
    # reachable:  zpow=1 -> s=2 (Luo-Hou);  zpow=3 -> s=6;  zpow=5 -> s=10.
    # ** s = 4 is FORBIDDEN in axisymmetry ** -- s = 2 (mod 4) always -- so Liu's s=4
    # Boussinesq case has NO axisymmetric counterpart. The Boussinesq<->axisym analogy
    # does not preserve the degeneracy ladder; it is quoted for s=2, where it is fine.
    b.u1["g"] = (A * np.exp(-IC_DECAY * (1.0 - b.r ** 2) ** power)
                 * np.sin(2 * np.pi * b.z / b.L) ** zpow)
    # DEGENERATE VORTICITY (wamp != 0). The Luo-Hou IC is meridionally at rest
    # (omega1 = psi1 = 0), and the s=6 test above degenerated only theta (via u1) --
    # which turned out to be INERT: the forced -1 law still held with R^2 = 0.9993 and
    # t_s matched the s=2 value to 0.7%. But Liu (Sec 3.5) and Chen-Huang-Li
    # (arXiv:2604.01868) both degenerate the VORTICITY as well:
    #   Liu, Boussinesq:  w = sin^3(pi x1) (1 - x2)^3          -> odd, cubic
    #   CHL, Boussinesq:  Omega = 10 x1^9/(x1^10 + x2^10 + 16) -> odd, 9th order
    # omega1 must be ODD about z=0 (gate G5), so wpow must be ODD. Same wall-localised
    # radial factor as u1. psi1 is a solved variable (the elliptic constraint fixes it
    # from omega1 on the first step), so leaving it 0 here is only a starting value.
    b.omega1["g"] = (wamp * np.exp(-IC_DECAY * (1.0 - b.r ** 2) ** power)
                     * np.sin(2 * np.pi * b.z / b.L) ** wpow)
    b.psi1["g"] = 0.0


# =============================================================================
# GATE SUITE  (run with --gates; every gate must pass at its documented tol)
# =============================================================================
def g1_operators(Nz=64, Nr=96, tol=1e-11):
    """G1: the spectral derivative primitives the engine actually applies.

    The engine only ever takes FIRST derivatives explicitly (velocities u^r,u^z;
    the advection u.grad; the vorticity source dz(u1^2)) plus the Fourier dzz --
    all well-conditioned; these are gated to ~1e-11.  It never forms a raw
    Chebyshev drr: the elliptic operator is applied only through the well-
    conditioned first-order tau reduction, which G2 inverts to ~1e-15.  Raw
    Chebyshev drr is reported for information only -- it is O(N^4)-conditioning
    limited (~eps*N^4) and irrelevant to the solver."""
    b = build(Nz, Nr)
    z, r, k = b.z, b.r, 2 * np.pi / b.L
    Pr, Qr = 0.3 + 0.7 * r + 0.5 * r ** 2, 1.0 - 0.4 * r ** 3
    Pr1, Qr1 = 0.7 + 1.0 * r, -1.2 * r ** 2
    Pr2, Qr2 = 1.0 + 0.0 * r, -2.4 * r
    s, c = np.sin(k * z), np.cos(k * z)
    f = b.dist.Field(bases=(b.zb, b.rb))
    f["g"] = Pr * s + Qr * c
    ana = {"dr": Pr1 * s + Qr1 * c, "dz": k * (Pr * c - Qr * s),
           "dzz": -k * k * (Pr * s + Qr * c), "drr(info)": Pr2 * s + Qr2 * c}
    num = {"dr": _gval(b.dr(f)), "dz": _gval(b.dz(f)),
           "dzz": _gval(b.dz(b.dz(f))), "drr(info)": _gval(b.dr(b.dr(f)))}
    errs = {kk: float(np.abs(num[kk] - ana[kk]).max() / max(np.abs(ana[kk]).max(), 1e-30))
            for kk in ana}
    gated = max(errs["dr"], errs["dz"], errs["dzz"])   # engine-used primitives
    return gated < tol, gated, errs


def g2_poisson(Nz=64, Nr=96, r0=0.4, tol=1e-11):
    """G2: -(drr+3/r dr+dzz)psi1 = omega1 recovers a manufactured psi1 (BC-exact)
    from an ANALYTIC source -> ~1e-11 (independent of G1's operator check)."""
    b = build(Nz, Nr, r0=r0)
    z, r, k = b.z, b.r, 2 * np.pi / b.L
    # psi1_m = sin(kz) * (r-r0)(1-r)(r+0.5): zero at r=r0 and r=1, smooth
    g = (r - r0) * (1.0 - r) * (r + 0.5)
    gp = -3.0 * r ** 2 + (1.0 + 2.0 * r0) * r + (0.5 - 0.5 * r0)
    gpp = -6.0 * r + (1.0 + 2.0 * r0)
    s = np.sin(k * z)
    psi_m = s * g
    # source: omega1 = -(drr + (3/r)dr + dzz) psi1_m  (analytic; 1/r fine on grid)
    src = -(s * gpp + (3.0 / r) * s * gp + (-k * k) * s * g)
    b.omega1["g"] = src
    ns = vars(b)
    prob = d3.LBVP([b.psi1, b.tau_p1, b.tau_p2], namespace=ns)
    prob.add_equation(_poisson_lhs() + " + lift(tau_p2) = -rr*omega1")
    prob.add_equation("psi1(r=1) = 0")
    prob.add_equation("psi1(r=r0) = 0")
    solver = prob.build_solver()
    solver.solve()
    err = float(np.abs(b.psi1["g"] - psi_m).max() / max(np.abs(psi_m).max(), 1e-30))
    return err < tol, err


def g3_material(Nz=64, Nr=96, tol=1e-11):
    """G3: angular momentum m = r^2 u1 is materially conserved -- D(r^2 u1)/Dt = 0
    holds POINTWISE for any smooth (u1, psi1) given u^r=-r psi1_z, u^z=2psi1+r psi1_r.
    This is the exact form of 'swirl is transported'; check the cancellation to
    spectral accuracy on a manufactured smooth field."""
    b = build(Nz, Nr)
    z, r, k = b.z, b.r, 2 * np.pi / b.L
    b.psi1["g"] = np.sin(k * z) * (r - 0.4) * (1.0 - r) * (1.5 + r)
    b.u1["g"] = (1.0 + 0.5 * r) * (1.2 + np.sin(k * z))
    u1g = b.u1["g"].copy()                             # scale-1 snapshot (evals rescale)
    m = b.dist.Field(bases=(b.zb, b.rb))
    m["g"] = (r ** 2) * u1g                            # angular momentum r^2 u1
    ur = _gval(b.ur)
    uz = _gval(b.uz)
    u1r = _gval(b.dr(b.u1))
    u1z = _gval(b.dz(b.u1))
    psz = _gval(b.dz(b.psi1))
    mr = _gval(b.dr(m))
    mz = _gval(b.dz(m))
    # Dm/Dt = r^2 * (du1/dt) + u^r dr(m) + u^z dz(m), with du1/dt the PNAS-2a RHS
    du1dt = -ur * u1r - uz * u1z + 2 * u1g * psz
    Dm = (r ** 2) * du1dt + ur * mr + uz * mz
    scale = max(np.abs((r ** 2) * du1dt).max(), np.abs(ur * mr).max(),
                np.abs(uz * mz).max(), 1e-30)
    err = float(np.abs(Dm).max() / scale)
    return err < tol, err


def _short_run(Nz, Nr, A, dt, nsteps, use_filter, r0=0.4, zpow=1,
               ic_power=IC_POWER, wamp=0.0, wpow=3):
    b = build(Nz, Nr, r0=r0)
    _, solver = make_ivp(b)
    set_ic(b, A=A, power=ic_power, zpow=zpow, wamp=wamp, wpow=wpow)
    filt = make_filter(b.omega1) if use_filter else None
    for _ in range(nsteps):
        solver.step(dt)
        if filt is not None:
            b.u1["c"] *= filt
            b.omega1["c"] *= filt
    return b


def g4_convergence(A=10.0, dt=1e-5, nsteps=20, tol=1e-6):
    """G4: resolution-doubling convergence -- a smooth, well-resolved evolution at
    (Nz,Nr) and (2Nz,2Nr) with IDENTICAL fixed dt (isolating spatial error, filter
    OFF) must agree on the Casimir integral to spectral accuracy."""
    def casimir(b):
        m = b.rr * b.rr * b.u1
        return _scalar(d3.integ(m * m * b.rr))          # int (r^2 u1)^2 r dz dr
    bc = _short_run(32, 48, A, dt, nsteps, use_filter=False)
    bf = _short_run(64, 96, A, dt, nsteps, use_filter=False)
    Cc, Cf = casimir(bc), casimir(bf)
    rel = abs(Cc - Cf) / max(abs(Cf), 1e-30)
    return rel < tol, rel, Cc, Cf


def g5_zsymmetry(Nz=48, Nr=64, A=100.0, dt=2e-6, nsteps=30, tol=1e-10, zpow=1,
                 wamp=0.0, wpow=3):
    """G5: z-parity of the IC (u1, omega1, psi1 all ODD about z=0) is preserved by
    the dynamics to roundoff.  On the uniform RealFourier grid z_j=jL/Nz an odd
    field obeys f[j] = -f[(Nz-j) mod Nz] and f[0]=0; check all three fields."""
    b = _short_run(Nz, Nr, A, dt, nsteps, use_filter=True, zpow=zpow,
                   wamp=wamp, wpow=wpow)
    worst = 0.0
    for fld in (b.u1, b.omega1, b.psi1):
        fld.change_scales(1)                            # evals leave fields dealiased
        g = fld["g"]
        refl = -g[(Nz - np.arange(Nz)) % Nz, :]
        amp = max(np.abs(g).max(), 1e-30)
        worst = max(worst, float(np.abs(g - refl).max() / amp))
    return worst < tol, worst


def run_gates():
    print("=" * 68)
    print("  dedalus_axisym.py  GATE SUITE  (fp64; annulus r in [r0,1])")
    print("=" * 68)
    rows = []
    ok1, w1, e1 = g1_operators()
    rows.append(("G1 spectral derivatives (rel)", ok1, w1, "<1e-11"))
    print(f"   G1 detail: {', '.join(f'{k}={v:.2e}' for k, v in e1.items())}")
    ok2, e2 = g2_poisson()
    rows.append(("G2 Poisson manufactured recovery", ok2, e2, "<1e-11"))
    ok3, e3 = g3_material()
    rows.append(("G3 D(r^2 u1)/Dt = 0 pointwise", ok3, e3, "<1e-11"))
    ok4, r4, Cc, Cf = g4_convergence()
    rows.append(("G4 res-doubling Casimir (rel)", ok4, r4, "<1e-6"))
    print(f"   G4 detail: C(32x48)={Cc:.12e}  C(64x96)={Cf:.12e}")
    ok5, w5 = g5_zsymmetry()
    rows.append(("G5 z-parity preserved (rel)", ok5, w5, "<1e-10"))
    print("-" * 68)
    print(f"   {'GATE':<36}{'VALUE':<14}{'TOL':<9}{'PASS'}")
    for name, ok, val, tol in rows:
        print(f"   {name:<36}{val:<14.3e}{tol:<9}{'PASS' if ok else 'FAIL'}")
    print("=" * 68)
    allok = all(r[1] for r in rows)
    print(f"   RESULT: {'ALL GATES PASS' if allok else 'FAILURE'}")
    return allok


# =============================================================================
# SCENARIO  (--scenario): the corner blow-up run + blow-up diagnostics
# =============================================================================
def run_scenario(A, Nz, Nr, tmax, out, r0=0.4, stepper="RK443", safety=0.2,
                 use_filter=True, initial_dt=1e-6, max_dt=1e-4, run_id=None,
                 checkpoint_wall=300.0, resume=None, ic_power=IC_POWER, zpow=1,
                 wamp=0.0, wpow=3, ckpt_sim_dt=None, ckpt_max_writes=2):
    b = build(Nz, Nr, r0=r0)
    _, solver = make_ivp(b, stepper=stepper)
    solver.stop_sim_time = tmax

    tag = run_id or f"axisym_N{Nz}x{Nr}_A{A:g}"
    ckpt_dir = pathlib.Path(out).parent / f"ckpt_{tag}"
    if resume:
        cand = sorted(ckpt_dir.glob("*.h5"))
        rp = resume if isinstance(resume, str) and resume.endswith(".h5") else (
            str(cand[-1]) if cand else None)
        if rp is None:
            raise SystemExit(f"--resume: no checkpoint in {ckpt_dir}")
        solver.load_state(rp, -1)
        print(f"  RESUMED from {rp} at t={solver.sim_time:.6f} it={solver.iteration}",
              flush=True)
        ck_mode = "append"
    else:
        set_ic(b, A=A, power=ic_power, zpow=zpow, wamp=wamp, wpow=wpow)
        ck_mode = "overwrite"
    # File-handler cadence only -- no effect on the physics or the stepping.
    # ckpt_sim_dt lets snapshots land at controlled SIM times (needed to fit the
    # self-similar spatial exponent, which requires many late snapshots, not two).
    _ck = dict(sim_dt=ckpt_sim_dt) if ckpt_sim_dt else dict(wall_dt=checkpoint_wall)
    checkpoints = solver.evaluator.add_file_handler(
        str(ckpt_dir), max_writes=ckpt_max_writes, mode=ck_mode, **_ck)
    checkpoints.add_tasks(solver.state, layout="c")

    filt = make_filter(b.omega1) if use_filter else None

    # ---- blow-up diagnostic operators (built once) --------------------------
    mom = b.rr * b.rr * b.u1                       # m = r^2 u1 (angular momentum)
    theta = mom * mom                              # theta = (r^2 u1)^2  (bounded)
    casimir_op = d3.integ(theta * b.rr)            # int theta r dz dr (Casimir)
    wphys_op = b.rr * b.omega1                     # omega^theta = r omega1
    gu2 = d3.grad(b.u1 * b.u1)
    sup_gu2_op = np.sqrt(gu2 @ gu2)                # |grad(u1^2)|
    gth = d3.grad(theta)
    sup_gth_op = np.sqrt(gth @ gth)                # |grad theta| (grad-theta rate)

    C0 = _scalar(casimir_op)

    # --- MPI CORRECTNESS (same bug class fixed in dedalus_bsq.py): the sup
    # diagnostics below are LOCAL grid maxima, so under `mpirun -n K` each rank
    # would report its own max and the fitted exponents would be UNDERSTATED.
    # _scalar/casimir uses d3 integrals which already reduce globally.
    from mpi4py import MPI as _MPI
    _comm = b.omega1.dist.comm
    _rank = _comm.rank
    gmax = lambda v: float(_comm.allreduce(float(v), op=_MPI.MAX))

    stream_p = pathlib.Path(out).parent / f"stream_{tag}.jsonl"
    control_p = pathlib.Path(out).parent / f"control_{tag}.json"
    if not resume and _rank == 0:
        stream_p.write_text("")
    _comm.Barrier()                      # no rank appends before the truncate

    CFL = d3.CFL(solver, initial_dt=initial_dt, cadence=1, safety=safety,
                 max_change=1.2, min_change=0.2, max_dt=max_dt, threshold=0.05)
    CFL.add_velocity(b.u_vec)

    ser = {"t": [], "sup_w1": [], "sup_wphys": [], "sup_gu2": [],
           "sup_gth": [], "cas_drift": [], "dt": []}
    t0, brk, stop_cmd = time.time(), None, None

    def record(dt):
        """Compute + append + stream the blow-up diagnostics; return Casimir drift.
        Note: tail_fraction reads ['c'] and diag ops read ['g'] at whatever scale
        the fields sit -- both are scale-invariant (energy fraction / max)."""
        sup_w1 = gmax(np.abs(b.omega1["g"]).max())
        sup_wphys = gmax(np.abs(wphys_op.evaluate()["g"]).max())
        sup_gu2 = gmax(np.abs(sup_gu2_op.evaluate()["g"]).max())
        sup_gth = gmax(np.abs(sup_gth_op.evaluate()["g"]).max())
        drift = abs(_scalar(casimir_op) - C0) / max(abs(C0), 1e-300)
        ser["t"].append(solver.sim_time)
        ser["sup_w1"].append(sup_w1); ser["sup_wphys"].append(sup_wphys)
        ser["sup_gu2"].append(sup_gu2); ser["sup_gth"].append(sup_gth)
        ser["cas_drift"].append(drift); ser["dt"].append(float(dt))
        row = {"t": solver.sim_time, "it": int(solver.iteration),
               "sup_w1": sup_w1, "sup_wphys": sup_wphys, "sup_gu2": sup_gu2,
               "sup_gth": sup_gth, "cas_drift": drift, "dt": float(dt),
               "tail_u1": tail_fraction(b.u1), "tail_w1": tail_fraction(b.omega1),
               "wall": round(time.time() - t0, 1)}
        if _rank == 0:                   # rank 0 owns the stream file
            with open(stream_p, "a") as f:
                f.write(json.dumps(row) + "\n")
        return drift, sup_w1, sup_wphys, sup_gth

    dt = float(initial_dt)
    while solver.proceed:
        dt = CFL.compute_timestep()
        solver.step(dt)
        if filt is not None:
            b.u1["c"] *= filt
            b.omega1["c"] *= filt
        if solver.iteration % 5 == 0:
            drift, sup_w1, sup_wphys, sup_gth = record(dt)
            if solver.iteration % 200 == 0:
                print(f"  t={solver.sim_time:.6f} it={solver.iteration} "
                      f"sup|w1|={sup_w1:.3e} sup|w^th|={sup_wphys:.3e} "
                      f"|grad th|={sup_gth:.3e} cas_drift={drift:.2e} "
                      f"dt={dt:.1e} ({time.time()-t0:.0f}s)", flush=True)
            if control_p.exists():
                try:
                    cmd = json.loads(control_p.read_text()); control_p.unlink()
                    if cmd.get("cmd") == "checkpoint":
                        solver.evaluator.evaluate_handlers(
                            [checkpoints], wall_time=0, sim_time=solver.sim_time,
                            iteration=solver.iteration, world_time=0, timestep=dt)
                        print(f"  [control] checkpoint forced t={solver.sim_time:.6f}",
                              flush=True)
                    elif cmd.get("cmd") == "extend":
                        solver.stop_sim_time = float(cmd["stop_time"])
                        print(f"  [control] stop_time -> {solver.stop_sim_time}",
                              flush=True)
                    elif cmd.get("cmd") == "stop":
                        stop_cmd = "control-stop"; break
                except Exception as exc:
                    print(f"  [control] bad command ignored: {exc}", flush=True)
            if drift > TRUST_DRIFT:
                brk = solver.sim_time
                print(f"  Casimir break at t={brk:.6f} (drift {drift:.2e})", flush=True)
                break

    # final flush: capture the true end state (the last step is rarely a %10 one)
    if not ser["t"] or ser["t"][-1] != solver.sim_time:
        record(dt)

    tt = np.array(ser["t"]); w1 = np.array(ser["sup_w1"])
    wp = np.array(ser["sup_wphys"]); dd = np.array(ser["cas_drift"])
    trust = dd < TRUST_DRIFT
    tb = float(tt[trust][-1]) if trust.any() else 0.0
    res = {
        "engine": "dedalus-axisym-euler", "form": "u1-omega1-psi1 (Luo-Hou)",
        "geometry": {"annulus_r0": r0, "r1": 1.0, "L": b.L, "corner": [1.0, 0.0]},
        "meter": {"stepper": stepper, "cfl_safety": safety,
                  "hou_li_filter": bool(use_filter), "dealias": 1.5,
                  "ts_ref": TS_REF},
        # PROVENANCE -- the artifact must record the IC that produced it. Omitting
        # these cost a real error: ic_power defaults to 1.0 but PNAS eq 3a is the
        # FOURTH power, and two ladders run months apart with different --ic-power
        # were compared as if identical. A run JSON that cannot say what IC it came
        # from is not a result, it is a number.
        "ic": {"ic_power": float(ic_power), "zpow": int(zpow), "wamp": float(wamp),
               "wpow": int(wpow), "A": float(A), "r0": float(r0),
               "ic_decay": IC_DECAY, "L": b.L,
               "lattice_ord_theta": 2 * int(zpow),
               "lattice_ord_omega1": (int(wpow) if wamp else 2 * int(zpow) - 1)},
        "Nz": Nz, "Nr": Nr, "A": A, "t_trust_end": tb, "break": brk,
        "stop_cmd": stop_cmd,
        "w1_ratio": float(w1[trust][-1] / w1[trust][0]) if trust.sum() > 1 and w1[trust][0] else 0.0,
        "sup_wphys_trust_end": float(wp[trust][-1]) if trust.any() else 0.0,
        "cas_drift_end": float(dd[-1]) if dd.size else 0.0,
        "wall_s": round(time.time() - t0, 1), "iters": int(solver.iteration),
        "sec_per_step": round((time.time() - t0) / max(solver.iteration, 1), 4),
        "series": {k: np.array(v).tolist() for k, v in ser.items()},
    }
    if _rank == 0:
        pathlib.Path(out).write_text(json.dumps(res, indent=2))
    print(f"[AXISYM] {Nz}x{Nr} A={A:g}: trust t<={tb:.6f} | sup|w1| x"
          f"{res['w1_ratio']:.1f} | cas_drift_end={res['cas_drift_end']:.2e} | "
          f"{res['sec_per_step']*1000:.1f} ms/step | {res['wall_s']:.0f}s -> {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="3D axisymmetric Euler corner-blowup engine (Dedalus)")
    ap.add_argument("--gates", action="store_true", help="run the G1-G5 gate suite")
    ap.add_argument("--scenario", action="store_true", help="run the corner blow-up")
    ap.add_argument("--A", type=float, default=IC_AMP, help="IC swirl amplitude (100=paper)")
    ap.add_argument("--ic-power", type=float, default=IC_POWER,
                    help="power on (1-r^2) in the IC; PNAS eq 3a is 4")
    ap.add_argument("--wamp", type=float, default=0.0,
                    help="amplitude of an initial omega1 (0 = Luo-Hou rest). ONE-SIDED: measured law ord_z w1(t>0) = min(ord_z w1(0), 2q-1), q=zpow -- so this can only make w1 LESS degenerate than forced, never more.")
    ap.add_argument("--wpow", type=int, default=3,
                    help="ODD power on sin for omega1 (Liu uses cubic; CHL 9th)")
    ap.add_argument("--zpow", type=int, default=1,
                    help="odd power on sin(2 pi z/L). u1 ~ z^zpow and theta ~ u1^2, "
                         "so s = 2*zpow: 1->s=2 (Luo-Hou), 3->s=6 (first degenerate "
                         "case), 5->s=10. s=4 is FORBIDDEN by odd parity.")
    ap.add_argument("--Nz", type=int, default=128)
    ap.add_argument("--Nr", type=int, default=384)
    ap.add_argument("--r0", type=float, default=0.4, help="annulus inner radius (axis excluded)")
    ap.add_argument("--tmax", type=float, default=0.003)
    ap.add_argument("--out", default="../runs/dedalus_axisym.json")
    ap.add_argument("--stepper", default="RK443")
    ap.add_argument("--safety", type=float, default=0.2)
    ap.add_argument("--initial-dt", type=float, default=1e-6)
    ap.add_argument("--max-dt", type=float, default=1e-4)
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--checkpoint-wall", type=float, default=300.0)
    ap.add_argument("--ckpt-sim-dt", type=float, default=None,
                    help="checkpoint every N sim-time units instead of wall time")
    ap.add_argument("--ckpt-max-writes", type=int, default=2)
    ap.add_argument("--resume", nargs="?", const=True, default=None)
    a = ap.parse_args()

    if a.gates:
        ok = run_gates()
        raise SystemExit(0 if ok else 1)
    elif a.scenario:
        run_scenario(a.A, a.Nz, a.Nr, a.tmax, a.out, r0=a.r0, stepper=a.stepper,
                     safety=a.safety, use_filter=not a.no_filter,
                     initial_dt=a.initial_dt, max_dt=a.max_dt, run_id=a.run_id,
                     checkpoint_wall=a.checkpoint_wall, resume=a.resume, ic_power=a.ic_power,
                 zpow=a.zpow, wamp=a.wamp, wpow=a.wpow,
                     ckpt_sim_dt=a.ckpt_sim_dt, ckpt_max_writes=a.ckpt_max_writes)
    else:
        ap.print_help()
