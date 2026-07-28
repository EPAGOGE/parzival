"""
STABILITY SPECTRUM of the converged profile, done properly.

Three fixes over the ad-hoc version in section 21-22:

1. **B and Cg are EXACT, not finite-differenced.**
   `B = d(RHS)/dc` is literally `(LO, LB)` and `(MO, MB)` from `Corner.parts()` -- the RHS
   is AFFINE in `c_l, c_w` by construction. And the two constraints are LINEAR in the
   field, so `Cg` is exactly `Dx[0,:]` (on the Ot block) and `Dx2[0,:]` (on the Bt block).
   This matters at higher N: Chebyshev boundary rows of `Dx2` scale like N^4 (~1e7 at
   N=56), and finite-differencing that row with eps ~ 1e-6 puts roundoff exactly where the
   gauge lives. Only `A = d(RHS)/d(field)` needs finite differences.

2. **Modified Newton.** The Jacobian is reused across steps and rebuilt only when the
   residual stalls. Newton needs ~4 steps and the alpha loop ~3, so the naive version
   built ~12 dense Jacobians per resolution; this builds 2-3.

3. **Grid-scale modes separated by their measured scaling, not by a guess.** They are
   identified by `|Im| ~ N` (measured max|Im| = 24.6 / 35.8 / 47.2 at N = 24 / 32 / 40)
   and by their count growing with N. Physical modes sit still under refinement; spurious
   ones move. The report is therefore a RECURRENCE table across N, not a single leading
   eigenvalue.

WHAT IS ALREADY ESTABLISHED (do not re-derive)
  - converged profile, `||F|| = 2.1e-12`, alpha self-consistent to 7.9e-7,
    `alpha = -0.342108` against Chen-Hou's `-0.342400` (0.085%)
  - the `+1.05` mode of the UNCONSTRAINED operator is a SYMMETRY direction: overlap with
    `span{v_amp, v_trans}` = 0.993 / 0.999 / 1.000 at N = 28 / 36 / 44, against a random
    control at ~0.03. The constraints remove it correctly.
  - `dg/dc = 0` EXACTLY, so the constrained operator is the PROJECTION
    `L = [I - B (Cg B)^-1 Cg] A`, never a Schur complement.
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import numpy.linalg as la

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(HERE / fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class Stability:
    def __init__(self, N, alpha=None, XMAX=25.0, Nb=None, eps_b=1e-3, outer="neumann",
                 constraint="d2"):
        pn = _mod("pn", "polar_newton.py")
        self.S = pn.NewtonSolver(N, alpha=alpha, XMAX=XMAX, Nb=Nb, eps_b=eps_b,
                                 outer=outer, constraint=constraint)
        self.C = self.S.C
        self.n = self.S.idx.size

    # ---- exact blocks --------------------------------------------------------
    def exact_B(self, Ot, Bt):
        """d(field RHS)/d(c_l, c_w) -- exact, since the RHS is affine in them."""
        C = self.C
        _, pO, pB = C.parts(Ot, Bt)
        cols = []
        for iO, iB in ((1, 1), (2, 2)):        # (LO,LB) then (MO,MB)
            dO = pO[iO].copy(); dB = pB[iB].copy()
            dO[0, :] = 0.0; dB[0, :] = 0.0
            dO[:, -1] = 0.0; dB[:, -1] = 0.0
            cols.append(np.concatenate([dO.ravel(), dB.ravel()])[self.S.idx])
        return np.stack(cols, axis=1)

    def exact_Cg(self, cl=None):
        """d(constraints)/d(field) -- exact:

            g1 = (Dx @ Ot)[0,0]                                      (all modes)
            g2 = (Dx2 @ Bt)[0,0]                      constraint='d2'
            g2 = (Dx @ (Bt/g))[0,0]                   constraint='d1'
            g2 = 2(Dx2 @ Bt)[0,0] - c_l (Dx @ Ot)[0,0]  constraint='ratio'

        'd2'/'d1' touch the Bt block only and are c-independent.  'ratio' also touches the
        Ot block (coefficient -c_l) and therefore needs c_l -- and its c-derivative lives in
        exact_Cc, NOT here.  The k = 0 entry of the d1 row is 0 because Vt[0,:] = 0 by
        definition (B ~ y1^2, g ~ xi), which keeps 1/g off the singular node."""
        C = self.C
        n2 = C.nx * C.nb
        cons = getattr(self.S, "constraint", "d2")
        Cg = np.zeros((2, 2 * n2))
        for k in range(C.nx):                  # g1 touches the Ot block only
            Cg[0, k * C.nb + 0] = C.Dx[0, k]
        for k in range(C.nx):                  # g2
            if cons == "d2":
                Cg[1, n2 + k * C.nb + 0] = C.Dx2[0, k]
            elif cons == "d1":
                Cg[1, n2 + k * C.nb + 0] = 0.0 if k == 0 else C.Dx[0, k] / C.g[k]
            else:                              # ratio: Bt block 2 Dx2, Ot block -c_l Dx
                if cl is None:
                    raise ValueError("exact_Cg needs cl for constraint='ratio'")
                Cg[1, n2 + k * C.nb + 0] = 2.0 * C.Dx2[0, k]
                Cg[1, k * C.nb + 0] = -cl * C.Dx[0, k]
        return Cg[:, self.S.idx]

    def exact_Cc(self, Ot):
        """d(constraints)/d(c_l, c_w) -- the Jacobian's lower-right 2x2 block.

        Zero for d2/d1 (their constraints do not contain c).  For 'ratio',
        g2 = 2 Bt_xixi - c_l Ot_xi, so dg2/dc_l = -Ot_xi(0,0), dg2/dc_w = 0.  This is the
        block newton_exact used to hard-code to zero; wiring it makes Newton exact for the
        coupled constraint (a rank-one occupant, so it costs nothing)."""
        Cc = np.zeros((2, 2))
        if getattr(self.S, "constraint", "d2") == "ratio":
            Cc[1, 0] = -float((self.C.Dx @ Ot)[0, 0])
        return Cc

    def A_fd(self, x, eps_rel=1e-6):
        """d(field RHS)/d(field) at fixed c -- the only block needing finite differences."""
        S = self.S
        n = self.n
        xf, cl, cw = x[:-2], float(x[-2]), float(x[-1])
        scale = max(np.abs(xf).max(), 1e-300)
        eps = eps_rel * scale

        def rhs_field(v):
            Ot, Bt = S.unpack(v)
            _, pO, pB = self.C.parts(Ot, Bt)
            dO = pO[0] + cl * pO[1] + cw * pO[2]
            dB = pB[0] + cl * pB[1] + cw * pB[2]
            dO[0, :] = 0.0; dB[0, :] = 0.0
            dO[:, -1] = 0.0; dB[:, -1] = 0.0
            return np.concatenate([dO.ravel(), dB.ravel()])[S.idx]

        A = np.empty((n, n))
        e = np.zeros(n)
        for j in range(n):
            e[j] = 1.0
            A[:, j] = (rhs_field(xf + eps * e) - rhs_field(xf - eps * e)) / (2 * eps)
            e[j] = 0.0
        return A

    def A_exact(self, x):
        """d(field RHS)/d(field) ANALYTICALLY -- no finite differences anywhere.

        The RHS is a QUADRATIC polynomial in the fields, so every block is closed-form.
        Writing dP = Poisson(dOt) (LINEAR in dOt, one back-substitution on the already
        prefactored operator):

          d(advO) = (dP_x + mu dP) Ot_b + (Pt_x + mu Pt) dOt_b
                    - dP_b (Ot_x + a0 Ot) - Pt_b (dOt_x + a0 dOt)
          d(srcO) = G cos b (dBt_x + (1+2a0) dBt) - sin b dBt_b
          d(advB) = (dP_x + mu dP) Bt_b + (Pt_x + mu Pt) dBt_b
                    - dP_b (Bt_x + (1+2a0) Bt) - Pt_b (dBt_x + (1+2a0) dBt)

        Both terms of each bilinear product appear -- dropping the dP ones would be the
        classic "frozen-velocity Jacobian" error.

        This is not only exact, it is CHEAPER: one Poisson solve per column instead of the
        two residual evaluations central differences need. The FD version breaks down at
        N >= 56 because the operator carries Chebyshev boundary rows scaling like N^2
        (Dx) and N^4 (Dx2), plus 1/g ~ N^2 at the first node, so eps = 1e-6 * scale stops
        resolving the Jacobian long before the discretization itself fails.
        """
        C = self.C
        S = self.S
        n = self.n
        n2 = C.nx * C.nb
        Ot, Bt = S.unpack(x[:-2])
        cl, cw = float(x[-2]), float(x[-1])
        a0, mu, G, E = C.a0, C.mu, C.G, C.E
        Ginv = np.zeros_like(G)
        np.divide(1.0, G, out=Ginv, where=(G > 1e-13))
        EG = E * Ginv

        Pt = C.poisson(Ot)
        Pt_x, Pt_b = C.dx(Pt), C.db(Pt)
        Ot_x, Ot_b = C.dx(Ot), C.db(Ot)
        Bt_x, Bt_b = C.dx(Bt), C.db(Bt)
        PmuP = Pt_x + mu * Pt
        OxaO = Ot_x + a0 * Ot
        BxbB = Bt_x + (1.0 + 2.0 * a0) * Bt

        A = np.empty((n, n))
        full = np.zeros(2 * n2)
        for j in range(n):
            full[:] = 0.0
            full[S.idx[j]] = 1.0
            dOt = full[:n2].reshape(C.nx, C.nb)
            dBt = full[n2:].reshape(C.nx, C.nb)
            dP = C.poisson(dOt)                      # LINEAR in dOt
            dP_x, dP_b = C.dx(dP), C.db(dP)
            dOt_x, dOt_b = C.dx(dOt), C.db(dOt)
            dBt_x, dBt_b = C.dx(dBt), C.db(dBt)
            dadvO = ((dP_x + mu * dP) * Ot_b + PmuP * dOt_b
                     - dP_b * OxaO - Pt_b * (dOt_x + a0 * dOt))
            dsrcO = G * C.cosb * (dBt_x + (1.0 + 2.0 * a0) * dBt) - C.sinb * dBt_b
            dadvB = ((dP_x + mu * dP) * Bt_b + PmuP * dBt_b
                     - dP_b * BxbB - Pt_b * (dBt_x + (1.0 + 2.0 * a0) * dBt))
            rO = EG * (-dadvO + dsrcO) + cl * (-G * (dOt_x + a0 * dOt)) + cw * dOt
            rB = EG * (-dadvB) + cl * (-G * (dBt_x + (1.0 + 2.0 * a0) * dBt) + dBt) \
                + cw * (2.0 * dBt)
            rO[0, :] = 0.0; rB[0, :] = 0.0
            rO[:, -1] = 0.0; rB[:, -1] = 0.0
            A[:, j] = np.concatenate([rO.ravel(), rB.ravel()])[S.idx]
        return A

    def spectrum(self, x, exact=True):
        Ot, Bt = self.S.unpack(x[:-2])
        A = self.A_exact(x) if exact else self.A_fd(x)
        B = self.exact_B(Ot, Bt)
        Cg = self.exact_Cg(cl=float(x[-2]))
        CB = Cg @ B
        P = np.eye(self.n) - B @ la.solve(CB, Cg)
        w, V = la.eig(P @ A)
        o = np.argsort(-w.real)
        return w[o], V[:, o], float(la.cond(CB)), A, B, Cg


def newton_exact(St, x0, steps=40, tol=1e-11, verbose=False):
    """Newton with the FULL Jacobian assembled EXACTLY:

        J = [[ A , B ],
             [ Cg, 0 ]]

    every block closed-form (A_exact, exact_B, exact_Cg; the lower-right is 0 because the
    constraints do not depend on c). No finite differences anywhere, so the eps that broke
    FD Newton at N >= 56 is simply gone.

    Returns (x, f, r, nsteps) -- nsteps is the number of ACCEPTED steps, and callers must
    check it. If the linesearch fails on iteration 0 this function returns x0 untouched,
    which is not a solution and used to be indistinguishable from one.

    steps was 10.  The linesearch accepts ANY decrease, so the damped phase can run 16-22
    iterations before the quadratic phase fires; at steps=10 that made converging runs look
    like non-existent solutions.  Measured at N=36 cold: with steps=40 and nothing else
    changed, eps_b = 1e-2 reaches 9.218e-14 in 21 steps, 3e-3 reaches 4.982e-14 in 22, and
    6e-4 reaches 6.973e-14 in 11 -- all three previously "did not converge"."""
    S = St.S
    x = x0.copy()
    f, cl, cw = S.F(x)
    r = np.linalg.norm(f) / np.sqrt(f.size)
    prev = r
    taken = 0
    for it in range(steps):
        A = St.A_exact(x)
        Ot, Bt = S.unpack(x[:-2])
        B = St.exact_B(Ot, Bt)
        Cg = St.exact_Cg(cl=float(x[-2]))
        n = St.n
        J = np.zeros((n + 2, n + 2))
        J[:n, :n] = A
        J[:n, n:] = B
        J[n:, :n] = Cg
        J[n:, n:] = St.exact_Cc(Ot)          # zero for d2/d1; rank-one for 'ratio'
        try:
            dx = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            dx = -np.linalg.lstsq(J, f, rcond=None)[0]
        lam, best = 1.0, None
        for _ in range(10):
            ft, clt, cwt = S.F(x + lam * dx)
            rt = np.linalg.norm(ft) / np.sqrt(ft.size)
            if rt < prev:
                best = (x + lam * dx, ft, rt, lam)
                break
            lam *= 0.5
        if best is None:
            break
        x, f, r, lam = best
        taken += 1
        if verbose:
            print(f"      it{it} ||F||={r:.4e} ratio={r/prev:.4f} damp={lam:.3f} "
                  f"c_l={float(x[-2]):.6f}", flush=True)
        prev = r
        if r < tol:
            break
    return x, f, r, taken


CL_STAR = 2.0 * 1.79819132 / 1.19620314        # = 3.00649824, see below


def converge_exact(N, alpha=None, XMAX=25.0, x0=None, verbose=False,
                   Nb=None, eps_b=1e-3, outer="neumann", constraint="d2",
                   outer_steps=24,
                   theta=0.5, tol=1e-11, strict=True):
    """Newton (exact Jacobian) + outer alpha self-consistency loop.

    Returns (St, x, r, cl, cw, info).  `info` is a dict and MUST be checked; `strict=True`
    raises rather than returning a number that is not a solution.

    FOUR HARNESS DEFECTS FIXED HERE, all of which manufactured false data rather than
    biasing it.  They are recorded in full because every alpha ever quoted from this file
    predates them.

    1. PHANTOM PERFECT RESULT.  newton_exact used to return `x` unchanged when the
       linesearch failed on iteration 0.  This function then computed an = c_w/c_l from the
       SEED, found |an - a| = 0 exactly, broke, and returned normally.  Measured on the
       shipped code at N=28, XMAX=22:  alpha = -0.34240009311696556, i.e. -1e-6 % from
       Chen-Hou and apparently the most accurate result in the project by four orders of
       magnitude -- at ||F|| = 1.767946e-02, with alpha - seed_alpha = 0.000000e+00.  Now
       newton_exact reports its accepted-step count and zero steps is a hard failure.

    2. WARM START DISCARDED.  `x0 = None` executed UNCONDITIONALLY inside the loop, so
       `start` resolved to the seed on every pass and the previous converged iterate was
       thrown away.  Cost 4-5x the Jacobians, and turned PATH failures into apparent
       non-existence: kept warm, eps_b = 1.2e-3 converges to 7.75e-14 and 6e-4 to 3.83e-13
       where the cold loop stalls at 3.1e-3 and 1.6e-3.

    3. UNDAMPED OUTER MAP, NO CONVERGENCE TEST, HARD 8-PASS CAP.  alpha := c_w/c_l is a
       fixed-point iteration whose multiplier is < -1 in places.  At XMAX=15/N=44 it enters
       a clean period-2 cycle (-0.338669, -0.343791, -0.338605, -0.343651, ...) and the
       8th iterate was returned as if converged; the "failure residual 1.73e-2" quoted in
       POLAR_SPEC sections 26/33/34 is that cycle's amplitude, not a discretisation
       property, so those residuals are not comparable across (N, XMAX).  Now damped,
       alpha <- alpha + theta*(c_w/c_l - alpha), and non-convergence is reported.

    4. c_l HAS A FREE EXACT TARGET AND NOBODY WAS USING IT.  th_xx(0) = 2 v_x(0) exactly
       (v = theta/y1 to 2.174e-16 over 383,780 reference points), so
       c_l* = 2*THXX_REF/WX_REF = 3.00649824.  The corner row is excluded from the residual
       and from the unknowns, so this is a GENUINE independent residual -- and every
       converged run violates it by 1.87% to 9.13%.  Reported in `info` so it is impossible
       to quote an alpha without it.

    WHAT ALPHA ACTUALLY SEES.  alpha is the DIFFERENTIAL of the two gauge errors:
    with d_l, d_w the relative errors of c_l, c_w against (c_l*, alpha_ref*c_l*),
    d_alpha = (d_w - d_l)/(1 + d_l) reproduces the measured alpha error to <= 4.5e-7 on
    every row, corr(|differential|, |d_alpha|) = +1.0000 exactly while
    corr(|common mode|, |d_alpha|) = -0.183.  The common mode is an exact discrete symmetry
    (Ot -> s Ot, Bt -> s^2 Bt, (c_l,c_w) -> s(c_l,c_w), residual reproduced to 3.2e-12 over
    s in [0.5,7]) that alpha is blind to, and it is 1.4x to 767x LARGER than the error
    alpha shows.  So any budget quoting a c_l or c_w error, or a percentage of the field, is
    measuring mostly the direction alpha cannot see."""
    a = alpha
    hist = []
    St = x = None
    r = float("inf")
    cl = cw = float("nan")
    conv = False
    for k in range(outer_steps):
        St = Stability(N, alpha=a, XMAX=XMAX, Nb=Nb, eps_b=eps_b, outer=outer,
                       constraint=constraint)
        start = St.S.x0 if x0 is None else x0
        x, f, r, taken = newton_exact(St, start, tol=tol, verbose=verbose)
        if taken == 0:
            msg = (f"newton_exact accepted ZERO steps at N={N} XMAX={XMAX} eps_b={eps_b} "
                   f"(outer pass {k}): returned the start point, ||F||={r:.3e}. "
                   f"Any alpha from this is the seed's, not a solution's.")
            if strict:
                raise RuntimeError(msg)
            return St, x, r, float("nan"), float("nan"), dict(
                converged=False, reason="zero_newton_steps", F=r, outer_passes=k + 1)
        cl, cw = float(x[-2]), float(x[-1])
        an = cw / cl
        x0 = x                                  # KEEP the warm start
        hist.append((an, r, taken))
        if a is not None and abs(an - a) < 1e-9 and r < tol:
            conv = True
            a = an
            break
        a = an if a is None else a + theta * (an - a)
    info = dict(converged=bool(conv), F=float(r), outer_passes=len(hist),
                alpha=cw / cl, alpha_gap=abs(hist[-1][0] - a) if hist else float("nan"),
                cl=cl, cl_star=CL_STAR, d_cl=(cl - CL_STAR) / CL_STAR,
                newton_steps=[h[2] for h in hist], alpha_hist=[h[0] for h in hist])
    if strict and not conv:
        raise RuntimeError(
            f"outer alpha loop did NOT converge at N={N} XMAX={XMAX} eps_b={eps_b}: "
            f"||F||={r:.3e} after {len(hist)} passes, alpha history "
            + " ".join(f"{h[0]:.6f}" for h in hist[-6:]))
    return St, x, r, cl, cw, info


def converge(N, alpha=None, verbose=False):
    """Newton + outer alpha loop, returning the converged state."""
    a = alpha
    for _ in range(5):
        St = Stability(N, alpha=a)
        x, f, r, r0 = St.S.solve(steps=8, verbose=verbose)
        cl, cw = float(x[-2]), float(x[-1])
        an = cw / cl
        if a is not None and abs(an - a) < 1e-7:
            break
        a = an
    return St, x, r, cl, cw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[36, 44, 56])
    ap.add_argument("--imcut", type=float, default=3.0)
    a_ = ap.parse_args()

    res = {}
    for N in a_.Ns:
        St, x, r, cl, cw = converge(N)
        w, V, cCB, A, B, Cg = St.spectrum(x)
        res[N] = (w, cl, cw, r, St.n)
        lo = w[(np.abs(w.imag) < a_.imcut) & (np.abs(w.real) > 1e-7)]
        print(f"N={N:2d} dim={St.n:5d} ||F||={r:.2e} c_l={cl:.6f} alpha={cw/cl:.6f} "
              f"cond(CgB)={cCB:.3g}")
        print(f"   max|Im|={np.abs(w.imag).max():.1f}  unstable(Re>1e-6)={int((w.real>1e-6).sum())}"
              f"  low-|Im| nonzero={lo.size}")
        print("   top low-|Im|: " +
              " ".join(f"{z.real:+.4f}{z.imag:+.4f}i" for z in lo[:6]), flush=True)

    if len(res) > 1:
        Ns = sorted(res)
        fine = res[Ns[-1]][0]
        lof = fine[(np.abs(fine.imag) < a_.imcut) & (np.abs(fine.real) > 1e-7)]
        print(f"\nRECURRENCE across N (a physical mode sits still; a spurious one moves):")
        print(f"  {'N=%d' % Ns[-1]:>20s} | " +
              " | ".join(f"nearest at N={m}" for m in Ns[:-1]))
        for z in lof[:8]:
            cells = []
            for m in Ns[:-1]:
                wm = res[m][0]
                wm = wm[(np.abs(wm.imag) < a_.imcut) & (np.abs(wm.real) > 1e-7)]
                if wm.size == 0:
                    cells.append("      --      ")
                    continue
                j = int(np.argmin(np.abs(wm - z)))
                cells.append(f"{wm[j].real:+.4f}{wm[j].imag:+.4f}i d={abs(wm[j]-z):.3f}")
            print(f"  {z.real:+.4f}{z.imag:+.4f}i | " + " | ".join(cells))
        print("\n  d = distance to the nearest low-|Im| eigenvalue at the coarser N.")
        print("  d small and shrinking with N  -> converging, candidate PHYSICAL mode.")
        print("  d large or erratic            -> unresolved or spurious.")


if __name__ == "__main__":
    main()
