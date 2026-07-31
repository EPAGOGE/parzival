#!/usr/bin/env python3
"""LAMBDA: the scale-invariant geometric observable.

Correlating the five criticality constraints leaves exactly one candidate in 3D:
the vorticity DIRECTION field xi = omega/|omega|.

  * globally controlled, trivially and unbreakably -- |xi| == 1 on the sphere
  * criticality index sigma = 0 EXACTLY: a unit vector carries no scale
  * purely geometric, so Tao's averaging (which preserves energy and scaling
    but destroys geometry) cannot reproduce it
  * it is THE factor governing vortex stretching, since omega.grad(u) depends on
    the alignment of xi with the strain eigenvectors

||xi||_inf = 1 carries no information. The information is in xi's MODULUS OF
CONTINUITY, which is the Constantin-Fefferman(-Majda) object, and is also the
object the Kiselev-Nazarov-Volberg method controls for critical SQG. The
scale-invariant combination is

    Lambda(t) = sup over the high-vorticity region of  |grad xi| * |omega|^{-1/2}

Check the invariance: under u_s(x,t) = s u(sx, s^2 t) we get |omega| -> s^2|omega|
and |grad xi| -> s |grad xi|, so Lambda -> s/(s) = Lambda. Dimensionless, sigma=0.

WHY THIS WORKS ON DATA THAT COULD NOT MEASURE lambda. The exponent lambda is a
RATE and needs an asymptotic regime the uniform-grid march never reaches (three
independent confounds established that). Lambda is scale-FREE, so it is readable
at any instant without asymptotics. The march was being asked for the wrong kind
of quantity.

GEOMETRY. dedalus_axisym.py evolves the Hou-Luo rescaled variables on an annulus
r in [r0, 1] (no axis singularity, so 1/r is bounded):

    u^theta   = r * u1          omega^theta = r * omega1
    omega^r   = -dz(u^theta)    = -r dz(u1)
    omega^z   = (1/r) dr(r u^theta) = 2 u1 + r dr(u1)

For an AXISYMMETRIC vector field the gradient tensor in the orthonormal
cylindrical frame keeps the basis-rotation terms even though d/dtheta = 0:

    |grad A|^2 = (dr A_r)^2 + (A_theta/r)^2 + (dz A_r)^2
               + (dr A_th)^2 + (A_r/r)^2     + (dz A_th)^2
               + (dr A_z)^2  + 0             + (dz A_z)^2

Dropping the A_theta/r and A_r/r terms would silently make an axisymmetric
direction field look smoother than it is.
"""
from __future__ import annotations

import glob
import sys

import h5py
import numpy as np


def axes(f):
    z = r = None
    for k in f["scales"]:
        if k.startswith("z_hash"):
            z = f["scales"][k][:]
        if k.startswith("r_hash"):
            r = f["scales"][k][:]
    return z, r


def d_dz(A, z):
    """z is uniform and PERIODIC (RealFourier), so use a periodic difference."""
    dz = z[1] - z[0]
    return (np.roll(A, -1, axis=0) - np.roll(A, 1, axis=0)) / (2 * dz)


def d_dr(A, r):
    """r is a Chebyshev grid: non-uniform. np.gradient handles varying spacing."""
    return np.gradient(A, r, axis=1, edge_order=2)


def vorticity(u1, w1, z, r):
    R = r[None, :]
    w_r = -R * d_dz(u1, z)
    w_th = R * w1
    w_z = 2.0 * u1 + R * d_dr(u1, r)
    return w_r, w_th, w_z


def grad_xi_sq(xr, xth, xz, z, r):
    R = r[None, :]
    return (d_dr(xr, r) ** 2 + (xth / R) ** 2 + d_dz(xr, z) ** 2
            + d_dr(xth, r) ** 2 + (xr / R) ** 2 + d_dz(xth, z) ** 2
            + d_dr(xz, r) ** 2 + d_dz(xz, z) ** 2)


def lam(u1, w1, z, r, frac, sep_tol=0.02):
    """Lambda restricted to {|omega| > frac * sup|omega|}.

    TWO INDEPENDENT GATES, and they are not the same gate.

    RESOLUTION identity: enough cells across the structure. Checked by callers.

    STRUCTURE identity: the point ATTAINING Lambda must sit on the vorticity
    peak. On 2026-07-30 the resolution gate passed every point while the
    Lambda-attaining point silently detached from the peak at one decade of
    vorticity growth, and two runs then disagreed in SIGN on the slope. The
    structure was beautifully resolved. It was a DIFFERENT structure. This is
    the campaign's recorded 'wandering maximum' refusal recurring inside an
    observable built to be more careful than the one that first produced it, so
    the check is now mechanical rather than remembered.

    Returns (Lambda, sup|omega|, n_cells, separation). A separation above
    sep_tol means the row is measuring two objects and must not enter a fit.
    """
    w_r, w_th, w_z = vorticity(u1, w1, z, r)
    mag = np.sqrt(w_r ** 2 + w_th ** 2 + w_z ** 2)
    mx = mag.max()
    if mx <= 0:
        return np.nan, np.nan, 0, np.nan
    safe = np.maximum(mag, mx * 1e-12)
    xr, xth, xz = w_r / safe, w_th / safe, w_z / safe
    g = np.sqrt(np.maximum(grad_xi_sq(xr, xth, xz, z, r), 0.0))
    m = mag > frac * mx
    if m.sum() < 8:
        return np.nan, mx, int(m.sum()), np.nan
    ratio = np.where(m, g / np.sqrt(np.maximum(mag, 1e-300)), 0.0)
    pz, pr = np.unravel_index(np.argmax(mag), mag.shape)      # vorticity peak
    qz, qr = np.unravel_index(np.argmax(ratio), ratio.shape)  # Lambda point
    sep = float(np.hypot(z[pz] - z[qz], r[pr] - r[qr]))
    return float(ratio.max()), float(mx), int(m.sum()), sep


def run(dirs, fracs=(0.5, 0.2, 0.05)):
    for d in dirs:
        fs = sorted(glob.glob(f"{d}/*.h5"))
        if not fs:
            continue
        rows = []
        for fn in fs:
            with h5py.File(fn, "r") as f:
                if "omega1" not in f["tasks"] or "u1" not in f["tasks"]:
                    continue
                z, r = axes(f)
                if z is None or r is None:
                    continue
                W, U, st = f["tasks"]["omega1"], f["tasks"]["u1"], f["scales/sim_time"][:]
                for n in range(W.shape[0]):
                    if st[n] <= 0:
                        continue
                    u1, w1 = U[n], W[n]
                    vals = [lam(u1, w1, z, r, fc) for fc in fracs]
                    # structure-identity gate on the primary (tightest) fraction
                    sep = vals[0][3]
                    keep = np.isfinite(sep) and sep <= 0.02
                    rows.append((float(st[n]), vals[0][1],
                                 [v[0] if keep else np.nan for v in vals], sep))
        if len(rows) < 4:
            continue
        print(f"\n=== {d}  ({len(rows)} snapshots) ===")
        hdr = "".join(f"{'L(>'+str(fc)+')':>13}" for fc in fracs)
        print(f"{'t':>10}{'sup|omega|':>13}{hdr}")
        for t, mx, L, sep in rows:
            flag = "" if (np.isfinite(sep) and sep <= 0.02) else f"   <-STRUCTURE SPLIT sep={sep:.4f}"
            print(f"{t:10.5f}{mx:13.4e}" + "".join(f"{x:13.4f}" for x in L) + flag)
        t = np.array([x[0] for x in rows])
        mx = np.array([x[1] for x in rows])
        for i, fc in enumerate(fracs):
            L = np.array([x[2][i] for x in rows])
            ok = np.isfinite(L) & (mx > 0)
            if ok.sum() < 4:
                continue
            s = np.polyfit(np.log(mx[ok]), np.log(L[ok]), 1)[0]
            grow = L[ok][-1] / L[ok][0]
            print(f"  frac>{fc}:  Lambda {L[ok][0]:.3f} -> {L[ok][-1]:.3f} "
                  f"(x{grow:.2f}) while sup|omega| grew x{mx[ok][-1]/mx[ok][0]:.1f};"
                  f"  dlnL/dln|omega| = {s:+.4f}")
            print(f"            -> {'GROWING: alignment degrading, blowup-compatible' if s > 0.05 else ('BOUNDED/DEPLETING: geometric regularity' if s < -0.05 else 'FLAT: exactly marginal')}")


if __name__ == "__main__":
    ds = sys.argv[1:] or sorted(glob.glob("../runs/snap_*"))
    run(ds)
