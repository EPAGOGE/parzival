#!/usr/bin/env python3
"""MARCH IN RESCALED TIME s -- the fix for the e-folding wall.

THE PROBLEM. Marching in physical variables costs (T-t)^{-2 c_l} degrees of
freedom, ~10^5.84 per decade at c_l = 2.92. Every physical-frame run in this
campaign bought 1-2 e-foldings of vorticity growth and died. Distinguishing a
settling orbit from a wandering one needs ~20. No machine closes that gap;
Theorem 2a says so as a power law.

THE FIX. Solve in the self-similar corner frame, where the collapse is
stationary and one e-folding of amplitude is O(1) of rescaled time s. The grid
never has to follow the collapse because the collapse does not move.

WHY THIS IS A SMALL BUILD RATHER THAN A NEW SOLVER. polar_cornerreg.py already
solves this exact system in exactly these coordinates, with c_l and c_omega
carried AS UNKNOWNS in the state vector and closed by two gauge functionals.
Its residual rows RO and RB already contain the rescaling terms cl*(y.grad) and
cw*(.). Setting F(z) = 0 therefore already means d_s A = d_s B = 0: the steady
solver IS the fixed point of the rescaled dynamics. Marching is backward Euler
on rows that already exist.

    field rows :  (A - A_old)/ds  =  sgn * RO(z)      [same for B with RB]
    P rows     :  RP(z) = 0                (elliptic, slaved, no time term)
    gauge rows :  g1 = g2 = 0              (fix c_l, c_omega each step)

Jacobian: the existing one with I/ds added on the evolution rows. Nothing else
changes.

ROWS THAT MUST NOT GET THE MASS TERM. Inside RO/RB the solver overwrites
rT_pin (seed pins) and rT_c0 (corner parity partners) with algebraic
constraints. Those are not evolution equations; giving them a time derivative
would march a constraint. They are excluded explicitly below.

THE SIGN IS NOT ASSUMED. Whether d_s A = +RO or -RO is a convention question I
refuse to settle by derivation, because a silent sign error would produce a
confident wrong answer of exactly the kind this campaign has generated all day.
It carries a free residual instead: the alpha_0 branch is the ATTRACTING one,
so a perturbed fixed point must DECAY back under the correct sign and blow up
under the wrong one. calibrate() runs that test and reports which sign the
dynamics itself selects.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from polar_cornerreg import CornerRegSolver, converge


class SMarcher:
    def __init__(self, S, sign=+1.0):
        self.S = S
        self.sign = float(sign)
        self.n2 = S.Nx * S.Nb
        n2 = self.n2
        # evolution rows = the A and B blocks MINUS the algebraic overwrites
        alg = set(int(r) for r in list(S.rT_pin) + list(S.rT_c0))
        rows = [i for i in range(n2) if i not in alg]
        self.evo = np.array(rows + [n2 + i for i in rows], dtype=int)
        self.mask = np.zeros(len(S.pack(np.zeros((S.Nx, S.Nb)),
                                        np.zeros((S.Nx, S.Nb)),
                                        np.zeros((S.Nx, S.Nb)), 0.0, 0.0)), bool)
        self.mask[self.evo] = True

    def _F(self, z, z_old, ds):
        f = np.asarray(self.S.residual(z), float).copy()
        e = self.evo
        f[e] = (z[e] - z_old[e]) / ds - self.sign * f[e]
        return f

    def _J(self, z, ds):
        J = sp.csr_matrix(self.S.jacobian(z))
        n = J.shape[0]
        D = sp.diags(np.where(self.mask, -self.sign, 1.0))     # scale evo rows
        Jn = D @ J
        # add I/ds on the evolution rows only
        add = sp.diags(np.where(self.mask, 1.0 / ds, 0.0))
        return sp.csc_matrix(Jn + add)

    def step(self, z_old, ds, tol=1e-10, maxit=25):
        z = z_old.copy()
        for k in range(maxit):
            f = self._F(z, z_old, ds)
            r = float(np.max(np.abs(f)))
            if r < tol:
                return z, r, k
            try:
                dz = spla.splu(self._J(z, ds)).solve(-f)
            except Exception:
                return None, r, k
            lam = 1.0
            for _ in range(20):                                # damped linesearch
                zt = z + lam * dz
                rt = float(np.max(np.abs(self._F(zt, z_old, ds))))
                if rt < r:
                    break
                lam *= 0.5
            else:
                return z, r, k
            z = zt
        return z, float(np.max(np.abs(self._F(z, z_old, ds)))), maxit


def state_distance(z1, z2, n2):
    """Relative L2 distance between two profiles (fields only, not c_l/c_w)."""
    a, b = z1[:2 * n2], z2[:2 * n2]
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def calibrate(nsteps=12, ds=0.25, amp=1e-3, seed=0):
    """FREE RESIDUAL for the sign. Perturb the converged fixed point and march.
    alpha_0 is the attracting branch, so the correct sign must CONTRACT."""
    S, zf, r, info = converge(verbose=False)
    print(f"fixed point: alpha={info.get('alpha')}  residual={r:.2e}  "
          f"converged={info.get('converged')}")
    n2 = S.Nx * S.Nb
    rng = np.random.default_rng(seed)
    pert = np.zeros_like(zf)
    pert[:2 * n2] = rng.normal(0, 1, 2 * n2)
    pert *= amp * np.linalg.norm(zf[:2 * n2]) / max(np.linalg.norm(pert), 1e-300)
    out = {}
    for sgn in (+1.0, -1.0):
        M = SMarcher(S, sign=sgn)
        z = zf + pert
        d = [state_distance(z, zf, n2)]
        ok = True
        for i in range(nsteps):
            zn, rr, it = M.step(z, ds)
            if zn is None or not np.all(np.isfinite(zn)):
                ok = False
                break
            z = zn
            d.append(state_distance(z, zf, n2))
            if d[-1] > 1e3 * d[0]:
                break
        ratio = d[-1] / max(d[0], 1e-300)
        out[sgn] = (d, ratio, ok)
        print(f"  sign {sgn:+.0f}: |z-z*| {d[0]:.3e} -> {d[-1]:.3e} "
              f"(x{ratio:.3e}) over {len(d)-1} steps  "
              f"{'CONTRACTS' if ratio < 0.9 else 'EXPANDS' if ratio > 1.1 else 'neutral'}"
              f"{'' if ok else '  [diverged/failed]'}")
    good = [s for s, (d, ra, ok) in out.items() if ok and ra < 0.9]
    if len(good) == 1:
        print(f"\nSIGN SELECTED BY THE DYNAMICS: {good[0]:+.0f} "
              "(the attracting branch contracts, as it must)")
    else:
        print(f"\nSIGN NOT DETERMINED (contracting signs: {good}). "
              "Do not march until this is resolved.")
    return S, zf, out


if __name__ == "__main__":
    calibrate()
