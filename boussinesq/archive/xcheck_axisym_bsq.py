#!/usr/bin/env python
"""CROSS-CHECK: axisymmetric-Euler engine  vs  validated 2D Boussinesq engine.

Correctness gate for the tier-one axisym instrument (dedalus_axisym.py): in the
away-from-axis limit it must reproduce the validated flat 2D Boussinesq engine
(dedalus_bsq.py).  Correspondence map (Luo-Hou reduction, docstring of the axisym
engine):

        buoyancy  b   <->  theta = (r^2 u1)^2        (both materially conserved)
        vorticity w   <->  omega1
        Boussinesq x  <->  axisym z   (periodic / baroclinic direction)
        Boussinesq z  <->  axisym r   (wall direction),   r = z_B + r0

Under this map the axisym system reduces, as r -> const (thin annulus, far from
the axis), to the flat Boussinesq system.  The EXACT residual terms are geometric
and O(1/r):

    axisym vorticity source   dz(u1^2) = (1/r^4) dz(theta)     [Boussinesq: dx(b)]
    axisym elliptic           -(drr + (3/r) dr + dzz) psi1 = omega1
    axisym velocity           u^r = -r psi1_z ,  u^z = 2 psi1 + r psi1_r

so the leading discrepancy between the two engines is the baroclinic-source factor
1/r^4 (plus the subleading (3/r) dr elliptic term and the r-rescale of velocity).
As the field is pushed toward the wall r -> 1 (far from the axis) every one of
these -> its Cartesian value and the two engines MUST track.  Toward the axis they
must diverge, and 1/r^4 predicts by how much.  This script measures exactly that.

METHOD (no shortcuts, fp64, no wall-clock/randomness in physics):
  * The axisym side is the REAL engine: dedalus_axisym.build / make_ivp, verbatim.
  * The Boussinesq side is a flat solver whose equations are line-for-line the
    validated dedalus_bsq.py formulation (inviscid vorticity-streamfunction,
    sparse-tau Poisson, skew velocity, Hou-Li filter option).
  * IDENTICAL grids: axisym (RealFourier z in [0,L]) x (ChebyshevT r in [r0,1])
    and Boussinesq (RealFourier x in [0,L]) x (ChebyshevT z in [0,1-r0]) are the
    SAME collocation points under x=z, z_B=r-r0.  The initial buoyancy is set from
    ONE closed form on both, so theta_axisym(t=0) == b_Boussinesq(t=0) to roundoff
    (verified at t=0 -- the first row of every table).
  * IDENTICAL numerics: same RK443, same fixed shared dt (CFL OFF, to isolate the
    spatial operators from any adaptive-dt confound), same filter setting.
  * The two solvers are stepped in LOCKSTEP; diagnostics are read at identical t.

Two experiments:
  A  --mode wallpinned : paper-form swirl exp(-beta(1-r^2)) sin, mass pinned at the
     wall r~1 -> the actual Luo-Hou near-wall regime -> the engines should TRACK.
  B  --mode centered   : a thin Gaussian swirl bump at radius r_c, swept inward.
     Shows the divergence grow toward the axis, dominated by the 1/r_c^4 source.

Outputs a JSON with both trajectories + a printed table (t, axisym vs Boussinesq
sup|omega|, sup|grad buoyancy|, and their ratio; vs the 1/r_c^4 prediction).

HARD RULES honoured: new file only, fp64, no wall-clock / randomness in physics.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

# import the REAL axisym engine (sibling module; run_dedalus.sh cd's into this dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dedalus.public as d3
import dedalus_axisym as ax


# ---- flat 2D Boussinesq (faithful to dedalus_bsq.py, inviscid) --------------
def build_bsq(Nx, Nz, Lx, Lz, dealias=3.0 / 2.0, stepper="RK443"):
    """Vorticity-streamfunction inviscid Boussinesq, verbatim in formulation with
    dedalus_bsq.build_and_run (nu=kappa=0). Domain [0,Lx] periodic x [0,Lz] wall."""
    coords = d3.CartesianCoordinates("x", "z")
    dist = d3.Distributor(coords, dtype=np.float64)
    xb = d3.RealFourier(coords["x"], size=Nx, bounds=(0, Lx), dealias=dealias)
    zb = d3.ChebyshevT(coords["z"], size=Nz, bounds=(0, Lz), dealias=dealias)
    x = dist.local_grid(xb)
    z = dist.local_grid(zb)
    ex, ez = coords.unit_vector_fields(dist)

    psi = dist.Field(name="psi", bases=(xb, zb))
    w = dist.Field(name="w", bases=(xb, zb))
    buoy = dist.Field(name="buoy", bases=(xb, zb))
    tau1 = dist.Field(name="tau1", bases=xb)
    tau2 = dist.Field(name="tau2", bases=xb)

    lift = lambda F: d3.Lift(F, zb.derivative_basis(1), -1)
    grad_psi = d3.grad(psi) + ez * lift(tau1)          # first-order reduction
    u = d3.skew(d3.grad(psi))                           # (-dz psi, dx psi)
    dx = lambda F: d3.Differentiate(F, coords["x"])

    problem = d3.IVP([psi, w, buoy, tau1, tau2], namespace=locals())
    problem.add_equation("div(grad_psi) + lift(tau2) - w = 0")   # lap(psi)=w
    problem.add_equation("psi(z=0) = 0")
    problem.add_equation("psi(z=Lz) = 0")
    problem.add_equation("dt(w) = dx(buoy) - u@grad(w)")         # inviscid
    problem.add_equation("dt(buoy) = -u@grad(buoy)")
    solver = problem.build_solver(getattr(d3, stepper))

    gradb = d3.grad(buoy)
    supgb_op = np.sqrt(gradb @ gradb)
    from types import SimpleNamespace
    return SimpleNamespace(**locals())


# ---- shared initial swirl u1(r,z); theta=(r^2 u1)^2 is the shared buoyancy ----
def u1_profile(r, z, mode, A, L, beta, rc, sigma):
    zc = np.sin(2 * np.pi * z / L)
    if mode == "wallpinned":
        radial = np.exp(-beta * (1.0 - r ** 2))        # paper form, pinned at r=1
    elif mode == "centered":
        radial = np.exp(-((r - rc) / sigma) ** 2)      # thin Gaussian at r=rc
    else:
        raise ValueError(mode)
    return A * radial * zc


def theta_closed(r, z, mode, A, L, beta, rc, sigma):
    """theta = (r^2 u1)^2 in closed form (used to set the Boussinesq buoyancy)."""
    u1 = u1_profile(r, z, mode, A, L, beta, rc, sigma)
    return (r ** 2 * u1) ** 2


def make_filter_1(field, alpha=36.0, order=36, cutoff=0.65):
    """Hou-Li filter, verbatim shape from both engines (coeff-space rolloff)."""
    csh = field["c"].shape
    i0 = np.arange(csh[0]) / max(csh[0] - 1, 1)
    i1 = np.arange(csh[1]) / max(csh[1] - 1, 1)
    I0, I1 = np.meshgrid(i0, i1, indexing="ij")
    kk = np.sqrt((I0 ** 2 + I1 ** 2) / 2.0)
    return np.exp(-alpha * (np.maximum(kk - cutoff, 0.0) / (1 - cutoff)) ** order)


def sup(op_or_field_g):
    return float(np.abs(op_or_field_g).max())


def _eval_sup(op):
    return float(np.abs(op.evaluate()["g"]).max())


def run(mode, Nz, Nr, r0, L, A, beta, rc, sigma, dt, T, cadence,
        use_filter, out):
    Lx = L                       # periodic direction identical
    Lz = 1.0 - r0                # wall direction: z_B = r - r0  ->  Lz = 1-r0

    # ---- build both engines at identical resolution -------------------------
    b = ax.build(Nz, Nr, r0=r0, L=L)          # REAL axisym engine
    _, ax_solver = ax.make_ivp(b)
    bs = build_bsq(Nz, Nr, Lx, Lz)            # faithful flat Boussinesq

    # ---- field-identical initial condition ----------------------------------
    # axisym: set the swirl u1; theta=(r^2 u1)^2 is derived by the engine.
    b.u1["g"] = u1_profile(b.r, b.z, mode, A, L, beta, rc, sigma)
    b.omega1["g"] = 0.0
    b.psi1["g"] = 0.0
    # Boussinesq: buoyancy = theta at the SAME physical point (r = z_B + r0).
    r_of_z = bs.z + r0
    bs.buoy["g"] = theta_closed(r_of_z, bs.x, mode, A, L, beta, rc, sigma)
    bs.w["g"] = 0.0

    # ---- diagnostic operators (built once) ----------------------------------
    mom = b.rr * b.rr * b.u1
    theta = mom * mom
    gth = d3.grad(theta)
    ax_sup_gth = np.sqrt(gth @ gth)          # sup|grad theta|  <->  sup|grad b|
    ax_uvec = b.u_vec

    filt_ax = make_filter_1(b.omega1) if use_filter else None
    filt_bs = make_filter_1(bs.buoy) if use_filter else None

    ser = {"t": [], "it": [],
           "ax_w": [], "bs_w": [], "ax_gb": [], "bs_gb": [],
           "ax_u": [], "bs_u": []}

    def record(it, t):
        ax_w = sup(b.omega1["g"])
        bs_w = sup(bs.w["g"])
        ax_gb = _eval_sup(ax_sup_gth)
        bs_gb = _eval_sup(bs.supgb_op)
        ax_u = _eval_sup(ax_uvec)
        bs_u = _eval_sup(bs.u)
        ser["t"].append(float(t)); ser["it"].append(int(it))
        ser["ax_w"].append(ax_w); ser["bs_w"].append(bs_w)
        ser["ax_gb"].append(ax_gb); ser["bs_gb"].append(bs_gb)
        ser["ax_u"].append(ax_u); ser["bs_u"].append(bs_u)

    # t=0 baseline: sup|grad theta| must equal sup|grad b| to roundoff (IC check)
    record(0, 0.0)

    nsteps = int(round(T / dt))
    for it in range(1, nsteps + 1):
        ax_solver.step(dt)
        bs.solver.step(dt)
        if use_filter:
            b.u1["c"] *= filt_ax; b.omega1["c"] *= filt_ax
            bs.w["c"] *= filt_bs; bs.buoy["c"] *= filt_bs
        if it % cadence == 0 or it == nsteps:
            record(it, it * dt)

    # ---- assemble result + table --------------------------------------------
    t = np.array(ser["t"])
    aw, bw = np.array(ser["ax_w"]), np.array(ser["bs_w"])
    ag, bg = np.array(ser["ax_gb"]), np.array(ser["bs_gb"])
    # relative deviation between engines on the two headline diagnostics
    def reldev(a, c):
        d = np.abs(a - c) / np.maximum(np.abs(c), 1e-300)
        return d
    dev_w = reldev(aw, bw)
    dev_g = reldev(ag, bg)
    ic_match = float(np.abs(ag[0] - bg[0]) / max(abs(bg[0]), 1e-300))
    src_pred = (1.0 / rc ** 4) if mode == "centered" else None

    res = {
        "cross_check": "axisym_euler_vs_flat_boussinesq",
        "map": "b<->theta=(r^2 u1)^2 ; w<->omega1 ; x<->z ; z_B=r-r0",
        "mode": mode, "Nz": Nz, "Nr": Nr, "r0": r0, "L": L, "A": A,
        "beta": beta, "rc": rc, "sigma": sigma,
        "dt": dt, "T": T, "cadence": cadence, "use_filter": bool(use_filter),
        "ic_grad_match_rel": ic_match,
        "src_factor_pred_1_over_rc4": src_pred,
        "w_ratio_end_ax_over_bs": float(aw[-1] / bw[-1]) if bw[-1] else None,
        "gb_ratio_end_ax_over_bs": float(ag[-1] / bg[-1]) if bg[-1] else None,
        "max_reldev_w": float(np.nanmax(dev_w[1:])) if len(dev_w) > 1 else 0.0,
        "max_reldev_gb": float(np.nanmax(dev_g[1:])) if len(dev_g) > 1 else 0.0,
        "series": ser,
    }
    pathlib.Path(out).write_text(json.dumps(res, indent=2))

    hdr = (f"\n=== CROSS-CHECK  mode={mode}  N={Nz}x{Nr}  r0={r0}  L={L:g}  "
           f"A={A:g}" + (f"  rc={rc:g} sigma={sigma:g}" if mode == "centered"
                         else f"  beta={beta:g}") + f"  dt={dt:g} filter={bool(use_filter)} ===")
    print(hdr)
    print(f"  IC buoyancy-gradient match (axisym vs bsq) at t=0: "
          f"rel={ic_match:.2e}  (should be ~1e-14)")
    if src_pred is not None:
        print(f"  predicted baroclinic-source amplification 1/rc^4 = {src_pred:.4f}")
    print(f"  {'t':>10} | {'ax sup|w|':>12} {'bs sup|w|':>12} {'w ax/bs':>9} | "
          f"{'ax|grad b|':>12} {'bs|grad b|':>12} {'gb ax/bs':>9}")
    print("  " + "-" * 92)
    for i in range(len(t)):
        rw = (aw[i] / bw[i]) if bw[i] else float("nan")
        rg = (ag[i] / bg[i]) if bg[i] else float("nan")
        print(f"  {t[i]:>10.6f} | {aw[i]:>12.5e} {bw[i]:>12.5e} {rw:>9.4f} | "
              f"{ag[i]:>12.5e} {bg[i]:>12.5e} {rg:>9.4f}")
    print("  " + "-" * 92)
    print(f"  end  w  ratio ax/bs = {res['w_ratio_end_ax_over_bs']}")
    print(f"  end |grad b| ratio  = {res['gb_ratio_end_ax_over_bs']}")
    if src_pred is not None:
        print(f"  (compare w-ratio to 1/rc^4 = {src_pred:.4f})")
    print(f"  max rel-dev over run:  w={res['max_reldev_w']:.3e}  "
          f"|grad b|={res['max_reldev_gb']:.3e}")
    print(f"  -> {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="axisym-Euler vs flat-Boussinesq cross-check")
    ap.add_argument("--mode", choices=["wallpinned", "centered"], default="wallpinned")
    ap.add_argument("--Nz", type=int, default=128, help="periodic modes (both engines)")
    ap.add_argument("--Nr", type=int, default=128, help="wall (Chebyshev) modes (both)")
    ap.add_argument("--r0", type=float, default=0.4, help="annulus inner radius")
    ap.add_argument("--L", type=float, default=1.0 / 6.0, help="periodic period (paper 1/6)")
    ap.add_argument("--A", type=float, default=1.0, help="swirl amplitude")
    ap.add_argument("--beta", type=float, default=30.0, help="wallpinned decay (paper 30)")
    ap.add_argument("--rc", type=float, default=0.9, help="centered-bump radius")
    ap.add_argument("--sigma", type=float, default=0.02, help="centered-bump width")
    ap.add_argument("--dt", type=float, default=2e-4)
    ap.add_argument("--T", type=float, default=0.02)
    ap.add_argument("--cadence", type=int, default=10)
    ap.add_argument("--filter", action="store_true", help="enable Hou-Li filter (both)")
    ap.add_argument("--out", default="../runs/xcheck.json")
    a = ap.parse_args()
    run(a.mode, a.Nz, a.Nr, a.r0, a.L, a.A, a.beta, a.rc, a.sigma,
        a.dt, a.T, a.cadence, a.filter, a.out)
