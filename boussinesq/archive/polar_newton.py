"""
NEWTON on the corner-inclusive frame. The march cannot work HERE; Newton can.

WHY NEWTON AFTER ALL
--------------------
The recon established that Chen-Hou MARCH (`run_pertb.m` is RK2 on the dynamic rescaling
equations; no Jacobian anywhere), and section 10 recorded that. That is true of THEM and
it is still true. But it does not follow that marching is right HERE, and the measurement
that settles it is this:

    seed steadiness (corner frame), max|dOt|/|Ot| vs N
    N =  32     48     64     96    128
        1.12e-2 1.41e-2 1.49e-2 1.54e-2 1.55e-2      -- FLAT

Flat in N means the ~1.5% is NOT discretization error (that would fall spectrally on a
smooth profile) -- it is the INTERPOLATION FLOOR of the seed, and it is concentrated at
xi in [2,10] (r in [6, 2.2e4]), the transition region where every residual in this project
has ended up.

Chen-Hou's march converges because they start from THEIR OWN discrete representation and
relax it. We start from an interpolation of their answer onto a different grid, carrying
1.5% error in the most dynamically active region. Any operator with a growing direction
amplifies that instead of relaxing it -- which is exactly what every march here did, in
both frames, under three gauges, with and without filtering, across N = 24..144, and under
one- and two-tier substepping.

Newton does not care. It converges to a fixed point whether that point is stable or not,
which is the original reasoning in POLAR_SPEC line 119 -- correct as stated, even though
the inference "therefore Chen-Hou must use Newton" was wrong.

WHAT MAKES IT VIABLE NOW, AND NOT BEFORE
  - the frame contains r = 0, so the gauge is defined where it is defined (section 15-17)
  - the gauge reproduces c_l, c_w to 0.85% and alpha to 1.5e-5 (section 17)
  - the Jacobian is already built and validated -- it is the same dense operator the
    spectrum work uses
  - the starting point is within 1.5%, which is a good Newton seed even though it is a
    bad march seed

GUARDS, all of them earned the hard way in this project
  - PRINT ||field|| BESIDE ||F||. Newton converging to the ZERO field has already
    happened here once. Tell: the residual falling by exactly the damping factor each
    step. A small residual on a vanishing field is not a solution.
  - report the residual AT INIT before iterating.
  - damped step with a simple linesearch; reject steps that raise the residual.
  - watch c_l, c_w: they should approach the reference values, not wander.
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(HERE / fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class NewtonSolver:
    """Chen-Hou's actual system: c_l and c_w are UNKNOWNS, closed by the two CORNER
    conditions that their march conserves to 1e-11:

        Ot_xi(0, b=0)   = w_x(0)_ref     (amplitude)
        Bt_xixi(0, b=0) = th_xx(0)_ref   (length scale)

    A first attempt let the gauge compute c_l, c_w FROM the field instead. Newton then
    converged quadratically (residual 7.5e-3 -> 8.0e-7) but slid along the free scaling
    family to c_l = 1.513 against the reference 3.0065, with the field 28-45% smaller.
    The scaling directions must be REMOVED BY CONSTRAINT, not measured after the fact --
    which is precisely why Chen-Hou pin these two functionals.
    """

    WX_REF, THXX_REF = 1.19620314, 1.79819132     # polar_gauge_gate.py

    def __init__(self, N=32, XMAX=25.0, alpha=None, Nb=None, eps_b=1e-3,
                 outer="neumann", constraint="d2"):
        pc = _mod("pc", "polar_corner.py")
        self.C = pc.Corner(N, N if Nb is None else Nb, XMAX,
                           filter_on=False, eps_b=eps_b, outer=outer)
        if alpha is not None:                      # outer self-consistency loop on alpha
            self.C.a0 = float(alpha)
            self.C.mu = 2.0 + self.C.a0
            self.C.E = np.exp(self.C.a0 * self.C.x)[:, None]
            self.C._build_poisson()
        C = self.C
        self.constraint = str(constraint)
        if self.constraint not in ("d2", "d1", "ratio"):
            raise ValueError(
                f"unknown constraint {constraint!r}, expected 'd2', 'd1' or 'ratio'")
        m = np.ones((C.nx, C.nb), dtype=bool)
        m[0, :] = False              # corner row: Ot = Bt = 0 exactly
        m[:, -1] = False             # axis column: pinned
        self.mask = np.concatenate([m.ravel(), m.ravel()])
        self.idx = np.where(self.mask)[0]
        self.n2 = C.nx * C.nb
        # unknown vector = [field ; c_l ; c_w]
        self.x0 = np.concatenate([self.pack(C.Ot0, C.Bt0),
                                  [C.P["cl"], C.P["cw"]]])

    def _axis_weights(self):
        """D = diag(cos b, cos^2 b): absorbs the MEASURED angular vanishing at the
        symmetry axis into the unknowns.

        WHY. The fixed-point drift between resolutions was measured to have 0.99 overlap
        with the 40 smallest singular directions of L (chance 0.128), and those
        directions are VORTICITY concentrated at the axis (Ot-fraction 0.68-0.99,
        beta-peak/(pi/2) = 0.95-0.997). At beta = pi/2 the beta-advection coefficient
        vanishes (Psi = 0 there) AND Om vanishes linearly, so the operator has almost no
        grip on vorticity there and every bit of discretization error is funnelled in.

        Measured orders at the axis (innermost interval, converged N=36 profile):
            Ot ~ eps^1.10 , Bt ~ eps^2.25 , Pt ~ eps^1.03
        i.e. linear and quadratic, matching section 8. Solving for
        Ot_hat = Ot/cos b and Bt_hat = Bt/cos^2 b makes the unknowns O(1) there.

        Applied at the pack/unpack boundary, this is exactly the diagonal preconditioner
        D^-1 J D -- the physics in `Corner` is untouched, only the coordinates the Newton
        system is expressed in change. `Pt ~ eps^1.03` was checked FIRST, because
        dividing the advection by cos b creates a tan(b) term that is finite only if Pt
        vanishes at least linearly."""
        if not getattr(self, "axis_sub", False):
            return None
        c = np.cos(self.C.b)[None, :]
        c = np.maximum(c, 1e-12)
        n2 = self.C.nx * self.C.nb
        wO = np.broadcast_to(c, (self.C.nx, self.C.nb)).ravel()
        wB = np.broadcast_to(c ** 2, (self.C.nx, self.C.nb)).ravel()
        return np.concatenate([wO, wB])[self.idx]

    def pack(self, Ot, Bt):
        v = np.concatenate([Ot.ravel(), Bt.ravel()])[self.idx]
        w = self._axis_weights()
        return v / w if w is not None else v

    def unpack(self, x):
        C = self.C
        w = self._axis_weights()
        xs = x * w if w is not None else x
        full = np.zeros(2 * self.n2)
        full[self.idx] = xs
        Ot = full[:self.n2].reshape(C.nx, C.nb)
        Bt = full[self.n2:].reshape(C.nx, C.nb)
        # restore the pinned rows from the seed (they are boundary data, not unknowns)
        Ot[0, :] = C.Ot0[0, :]; Ot[:, -1] = C.Ot0[:, -1]
        Bt[0, :] = C.Bt0[0, :]; Bt[:, -1] = C.Bt0[:, -1]
        return Ot, Bt

    def F(self, x, open_norm=False):
        """Residual of the FULL system: the two evolution RHS at the given (c_l, c_w),
        plus the two corner conditions. c_l, c_w are unknowns, not gauge outputs.

        WHAT ||F|| DOES NOT SEE.  The `dO[:, -1] = 0.0` line below zeroes the residual on
        the beta = pi/2 - eps_b column, which is INTERIOR to [0, pi/2] -- the beta grid stops
        short of the axis, so this is not a boundary.  At the converged N=36 solution the
        reported ||F|| over the kept rows is 2.2515e-13 with interior max|dOt| = 6.649e-13,
        while the UNZEROED right-hand side on that column has RMS 3.5965e-02 and
        max|dOt| = 1.232e-01 = 2.5% of max|Ot| = 4.875.  Eleven orders larger.  The same
        solution seen by the open system has ||F|| = 5.99e-3.
        So a ||F|| of 1e-13 from this function is NOT evidence of consistency on that line.
        `open_residual` below reports what the open system sees; call it beside any ||F||
        that is going to be quoted.  The pinning itself is load-bearing and is not a bug:
        with Pt = 0 pinned on that whole line, (Pt_xi + mu Pt) * Bt_beta vanishes
        identically, the Bt block there is exactly degenerate (100.0000% of its Jacobian
        mass on itself), Bt = 0 is an exact solution, and the pinning is what selects the
        nontrivial branch.  Measured cost to alpha: 2.1e-5 at N=36, a floor, not the
        blocker."""
        C = self.C
        Ot, Bt = self.unpack(x[:-2])
        cl, cw = float(x[-2]), float(x[-1])
        _, pO, pB = C.parts(Ot, Bt)
        dO = pO[0] + cl * pO[1] + cw * pO[2]
        dB = pB[0] + cl * pB[1] + cw * pB[2]
        if open_norm:
            return dO, dB
        dO[0, :] = 0.0; dB[0, :] = 0.0
        dO[:, -1] = 0.0; dB[:, -1] = 0.0
        fld = np.concatenate([dO.ravel(), dB.ravel()])[self.idx]
        g1 = float((C.Dx @ Ot)[0, 0]) - self.WX_REF
        g2 = self.g2_of(Bt, Ot=Ot, cl=cl)
        return np.concatenate([fld, [g1, g2]]), cl, cw

    def vt(self, Bt):
        """Vt = Bt / g, with Vt[0,:] = 0 EXACTLY.

        B ~ y1^2 near the corner so Bt ~ xi^2 cos^2 b, while g = 1 - e^-xi ~ xi, hence
        Vt ~ xi cos^2 b and Vt[0,:] = 0 is exact rather than a limit that has to be taken."""
        C = self.C
        Vt = np.zeros_like(Bt)
        Vt[1:, :] = Bt[1:, :] / C.g[1:, None]
        return Vt

    def g2_of(self, Bt, Ot=None, cl=None):
        """The SECOND corner constraint, in one of two forms with identical continuum
        content.  Which one is used decides how much of the profile's unresolved radial
        Chebyshev tail is amplified into the constraint, and therefore into alpha.

            'd2'  Bt_xixi(0, eps_b) = THXX * cos^2(eps_b)          -- the original
            'd1'  Vt_xi (0, eps_b) = THXX/2 * cos^2(eps_b),  Vt = Bt/g

        Both are exact: Bt = THXX/2 * xi^2 cos^2 b + O(xi^3) gives Bt_xixi(0) = THXX cos^2 b
        and Vt_xi(0) = THXX/2 cos^2 b.  And the constant itself IS a first derivative --
        th_xx(0) = 2 v_x(0) exactly, because v = theta/y1 to 2.174e-16 over 383,780
        reference grid points -- so pinning it as a SECOND derivative was a free choice, not
        a necessity.

        WHY IT MATTERS.  alpha reads the pinned pair only through q = THXX/WX^2: the measured
        ratio d(alpha)/d(ln THXX) : d(alpha)/d(ln WX) is -2.00007 / -2.00003 / -2.00009 at
        N = 28/36/44 (from the exact Schur complement, no finite differences), so the only
        harmful quantity is dln q = e2 - 2 e1 where e1, e2 are the two rows' relative errors.
        And e2 dominates because the row norms differ by two powers of N:
        |Dx2[0,:]|_1 = 277 / 1132 / 3200 / 7290 / 14427 / 33600 at N = 20/28/36/44/52/64,
        flat against N^4, while |Dx[0,:]|_1 is only ~N^2 (58.3 at N=28, 208.1 at N=52).
        Applied to the seed's radial Chebyshev tail -- which is spectral down to
        |c_last|/|c|max = 1.6e-6 at N=44 but then FLOORS at ~2e-8 by N=96 -- an N^4 amplifier
        makes the 'd2' condition get WORSE with refinement past N ~ 64, while 'd1' keeps
        converging.  Measured e2:
            d2:  -41.32  +5.51  +2.38  -1.28  +0.83  -1.22 %   at N = 28/36/44/52/64/96
            d1:   +0.94  +0.51  -0.011 +0.092 -0.080 +0.025 %
        a 10x-220x improvement whose decisive property is not its size but that |e2| flattens
        at ~1% while |e2'| falls 38x from N=28 to N=96.  The mechanism is coherence loss, not
        row size: sum|row_k data_k| / |sum| grows as N^3.5 for 'd2' (82 -> 18947 over
        N=20..96) and as N^1.0 for 'd1' (22.8 -> 103.4).

        NOTE ON THE TARGETS.  Neither carries the cos^2(eps_b) that the exact corner form
        strictly wants, so that 'd2' reproduces the shipped behaviour BIT FOR BIT and the two
        variants differ ONLY in the row.  Omitting it is safe and measured: the two beta
        offsets are 5.0e-7 (cos - 1) and 1.0e-6 (cos^2 - 1) relative at eps_b = 1e-3, and
        their ratio is exactly 2.000 -- they lie precisely along the alpha-invariant (1,2)
        log-ray -- so the net effect on alpha is 4e-12, eight orders below the effect being
        measured here.  Correcting one row and not the other would inject a 1e-6 asymmetry
        for no gain."""
        C = self.C
        if self.constraint == "d2":
            return float((C.Dx2 @ Bt)[0, 0]) - self.THXX_REF
        if self.constraint == "d1":
            return float((C.Dx @ self.vt(Bt))[0, 0]) - 0.5 * self.THXX_REF
        # 'ratio': impose the EXACT continuum identity c_l = 2 th_xx(0)/w_x(0) directly,
        #     2 Bt_xixi(0,0) - c_l Ot_xi(0,0) = 0.
        # This pins the RATIO of the two corner derivatives rather than their two absolute
        # values (that is what d2 does, and it lets the solved c_l drift 1.9-10.8% off the
        # free target 2*THXX/WX).  The amplitude is still fixed by g1 (Ot_xi(0,0)=WX_REF).
        # Because it contains c_l, this is the FIRST constraint that couples the c-block:
        # the Jacobian's lower-right 2x2 is no longer zero (see exact_Cc in polar_stability).
        return 2.0 * float((C.Dx2 @ Bt)[0, 0]) - cl * float((C.Dx @ Ot)[0, 0])

    def open_residual(self, x):
        """What the residual looks like WITHOUT the pinned rows zeroed, i.e. what the open
        system sees.  Returns a dict; `axis_rms` is the quantity ||F|| is blind to."""
        dO, dB = self.F(x, open_norm=True)
        C = self.C
        Ot, Bt = self.unpack(x[:-2])
        inner = (slice(1, None), slice(0, -1))
        return dict(
            axis_rms=float(np.sqrt(np.mean(np.concatenate(
                [dO[:, -1] ** 2, dB[:, -1] ** 2])))),
            axis_max_dOt=float(np.abs(dO[:, -1]).max()),
            axis_max_dOt_rel=float(np.abs(dO[:, -1]).max() / max(np.abs(Ot).max(), 1e-300)),
            corner_max=float(max(np.abs(dO[0, :]).max(), np.abs(dB[0, :]).max())),
            interior_rms=float(np.sqrt(np.mean(np.concatenate(
                [dO[inner].ravel() ** 2, dB[inner].ravel() ** 2])))),
            open_rms=float(np.sqrt(np.mean(np.concatenate(
                [dO.ravel() ** 2, dB.ravel() ** 2])))))

    def jac(self, x, eps_rel=1e-6):
        n = x.size
        scale = max(np.abs(x).max(), 1e-300)
        eps = eps_rel * scale
        J = np.empty((n, n))
        e = np.zeros(n)
        for j in range(n):
            e[j] = 1.0
            fp, _, _ = self.F(x + eps * e)
            fm, _, _ = self.F(x - eps * e)
            J[:, j] = (fp - fm) / (2 * eps)
            e[j] = 0.0
        return J

    def solve(self, steps=8, tol=1e-10, verbose=True):
        C = self.C
        x = self.x0.copy()
        f, cl, cw = self.F(x)
        r0 = np.linalg.norm(f) / np.sqrt(f.size)
        if verbose:
            print(f"  init: ||F||={r0:.6e}  ||x||={np.linalg.norm(x)/np.sqrt(x.size):.6e}"
                  f"  c_l={cl:.6f} c_w={cw:.6f} alpha={cw/cl:.6f}")
            print(f"  {'it':>3s} {'||F||':>13s} {'ratio':>9s} {'||x||':>12s} "
                  f"{'damp':>6s} {'c_l':>10s} {'alpha':>10s}")
        prev = r0
        for it in range(steps):
            J = self.jac(x)
            try:
                dx = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                dx = -np.linalg.lstsq(J, f, rcond=None)[0]
            # damped linesearch: never accept a step that raises the residual
            lam, best = 1.0, None
            for _ in range(8):
                xt = x + lam * dx
                ft, clt, cwt = self.F(xt)
                rt = np.linalg.norm(ft) / np.sqrt(ft.size)
                if rt < prev:
                    best = (xt, ft, rt, lam, clt, cwt)
                    break
                lam *= 0.5
            if best is None:
                if verbose:
                    print("   linesearch failed -- no damping reduced the residual")
                break
            x, f, r, lam, cl, cw = best
            if verbose:
                print(f"  {it:3d} {r:13.6e} {r/prev:9.4f} "
                      f"{np.linalg.norm(x)/np.sqrt(x.size):12.6e} {lam:6.3f} "
                      f"{cl:10.6f} {cw/cl:10.6f}", flush=True)
            # THE TRAP: a residual falling by exactly the damping factor means the FIELD
            # is marching to zero, not the residual to a root.
            if abs(r / prev - (1 - lam)) < 1e-3 and np.linalg.norm(x) < 0.1 * np.linalg.norm(self.x0):
                print("   WARNING: residual falling by the damping factor and the field is"
                      " collapsing -- this is the converge-to-zero failure, not a solution")
            prev = r
            if r < tol:
                break
        return x, f, prev, r0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()
    S = NewtonSolver(a.N)
    C = S.C
    print(f"NEWTON on the corner frame, {C.nx}x{C.nb}, unknowns {S.idx.size}")
    print(f"  reference c_l={C.P['cl']:.6f} c_w={C.P['cw']:.6f} alpha={C.a0:.6f}\n")
    x, f, r, r0 = S.solve(steps=a.steps)
    print(f"\n  residual {r0:.4e} -> {r:.4e}   (factor {r0/max(r,1e-300):.4g})")
    Ot, Bt = S.unpack(x[:-2])
    I = (slice(2, -2), slice(2, -2))
    print(f"  field norm vs seed: |Ot|max {np.abs(Ot[I]).max():.5f} "
          f"(seed {np.abs(C.Ot0[I]).max():.5f})   "
          f"|Bt|max {np.abs(Bt[I]).max():.5f} (seed {np.abs(C.Bt0[I]).max():.5f})")
    print(f"  deviation from seed: {np.abs(Ot[I]-C.Ot0[I]).max()/np.abs(C.Ot0[I]).max():.4e}")
    np.savez(pathlib.Path.home() / "parzival/runs/polar_newton.npz",
             Ot=Ot, Bt=Bt, x=C.x, b=C.b, alpha=C.a0)
    print("  saved -> ~/parzival/runs/polar_newton.npz")


if __name__ == "__main__":
    main()
