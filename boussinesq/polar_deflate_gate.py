"""GATE THE DEFLATION MACHINERY before believing any atlas it produces.

The dangerous failure is silent: if the pole is not actually planted, deflated Newton walks
straight back to the same root, `is_new` rejects it, and the run reports DRY -- a FALSE
NEGATIVE that would be read as "there is no second root", which is exactly the conclusion
under test.  So every claim the machinery rests on is checked here first.

  1. SIMPLICITY OF THE ROOT.  Farrell's guarantee needs a simple root.  Verify (a) the
     bordered J is nonsingular at the converged root, and (b) the amplitude scaling
     symmetry is genuinely BROKEN by the two corner constraints -- i.e. the constraint
     residual is nonzero for s != 1, so the scaling orbit is not a curve of solutions.
  2. THE JACOBIAN OF THE DEFLATED RESIDUAL IS EXACT.  G' = m J + F grad_m^T is claimed
     analytically; check it against finite differences of G.
  3. THE POLE ACTUALLY REPELS.  Start deflated Newton AT the known root plus a small
     perturbation.  Undeflated Newton returns to it; deflated Newton must NOT.
  4. THE SHIFT MATTERS.  With shift=0 the deflated residual decays at large norm, so a
     norm-based test can report success at absurd amplitude.  Demonstrate the difference
     rather than trusting the citation.
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


pst = _mod("pst", "polar_stability.py")
pdf = _mod("pdf", "polar_deflate.py")


def main(N=28, constraint="d1"):
    print(f"Converging the base root at N={N}, constraint={constraint} ...", flush=True)
    St, x, r, cl, cw, info = pst.converge_exact(
        N, constraint=constraint, strict=False, outer_steps=80)
    assert info["converged"], f"base solve failed: ||F||={r:.2e}"
    S = St.S
    a0 = cw / cl
    print(f"  alpha={a0:+.8f} c_l={cl:.6f} ||F||={r:.2e}\n")

    # --- 1. simplicity -----------------------------------------------------
    print("1. IS THE ROOT SIMPLE?")
    A = St.A_exact(x)
    Ot, Bt = S.unpack(x[:-2])
    B, Cg = St.exact_B(Ot, Bt), St.exact_Cg()
    n = St.n
    J = np.zeros((n + 2, n + 2))
    J[:n, :n], J[:n, n:], J[n:, :n] = A, B, Cg
    sv = la.svdvals(J)
    print(f"   bordered J: sigma_max={sv[0]:.3e} sigma_min={sv[-1]:.3e} "
          f"cond={sv[0]/sv[-1]:.3e}  -> {'NONSINGULAR' if sv[-1] > 0 else 'SINGULAR'}")
    print("   does the amplitude scaling symmetry survive the constraints? "
          "(it must NOT, or the root is a curve)")
    print(f"   {'s':>8} {'g1 (Ot constraint)':>20} {'g2 (Bt constraint)':>20}")
    for s in (0.98, 0.99, 1.0, 1.01, 1.02):
        xs = x.copy()
        xs[:-2] = S.pack(Ot * s, Bt * s * s)
        xs[-2], xs[-1] = cl * s, cw * s
        f_s, _, _ = S.F(xs)
        print(f"   {s:8.2f} {f_s[-2]:20.6e} {f_s[-1]:20.6e}")
    print("   -> nonzero off s=1 means the orbit is NOT a solution curve: root isolated.\n")

    # --- 2. exactness of the deflated Jacobian -----------------------------
    print("2. IS G' = m J + F grad_m^T EXACT?")
    centres = [x.copy()]
    xp = x + 1e-3 * np.random.default_rng(0).standard_normal(x.size) * max(la.norm(x), 1.0) / np.sqrt(x.size)
    m, gm = pdf._m_and_grad(xp, centres)
    f_p, _, _ = S.F(xp)
    A = St.A_exact(xp)
    Otp, Btp = S.unpack(xp[:-2])
    Bp, Cgp = St.exact_B(Otp, Btp), St.exact_Cg()
    Jp = np.zeros((n + 2, n + 2))
    Jp[:n, :n], Jp[:n, n:], Jp[n:, :n] = A, Bp, Cgp
    G_analytic = m * Jp + np.outer(f_p, gm)
    # finite-difference a few random columns of G
    rng = np.random.default_rng(1)
    cols = rng.choice(x.size, size=6, replace=False)
    h = 1e-7 * max(la.norm(xp), 1.0)
    errs = []
    for j in cols:
        e = np.zeros(x.size)
        e[j] = h
        mp, _ = pdf._m_and_grad(xp + e, centres)
        mm, _ = pdf._m_and_grad(xp - e, centres)
        fp, _, _ = S.F(xp + e)
        fm, _, _ = S.F(xp - e)
        num = (mp * fp - mm * fm) / (2 * h)
        errs.append(la.norm(num - G_analytic[:, j]) / max(la.norm(num), 1e-300))
    print(f"   relative error on {len(cols)} random columns: "
          f"max={max(errs):.3e} median={np.median(errs):.3e}")
    print(f"   -> {'EXACT' if max(errs) < 1e-5 else 'MISMATCH -- do not trust the atlas'}\n")

    # --- 3. does the pole repel? ------------------------------------------
    print("3. DOES THE POLE ACTUALLY REPEL?  (the false-negative guard)")
    rng = np.random.default_rng(2)
    pert = rng.standard_normal(x.size)
    pert *= 1e-2 * la.norm(x) / la.norm(pert)
    xstart = x + pert
    xu, fu, ru, tu = pst.newton_exact(St, xstart.copy(), steps=40, verbose=False)
    au = float(xu[-1]) / float(xu[-2])
    print(f"   UNDEFLATED from root+1%: alpha={au:+.8f} ||F||={ru:.2e} "
          f"steps={tu}  d(alpha)={abs(au-a0):.2e}")
    xd, fd, rd, td, okd = pdf.deflated_newton(St, xstart.copy(), centres, verbose=False)
    ad = float(xd[-1]) / float(xd[-2]) if abs(float(xd[-2])) > 1e-30 else np.nan
    dist = la.norm(xd - x) / max(la.norm(x), 1e-300)
    print(f"   DEFLATED   from root+1%: alpha={ad:+.8f} ||F||={rd:.2e} "
          f"steps={td} converged={okd}  relative distance from the deflated root={dist:.2e}")
    ok3 = dist > 1e-3
    print(f"   -> {'POLE WORKS (pushed away)' if ok3 else 'POLE DID NOT REPEL -- any DRY result would be a FALSE NEGATIVE'}\n")

    # --- 4. the shift ------------------------------------------------------
    print("4. WHY shift=1 IS MANDATORY.  ||m F|| at increasing distance from the root:")
    print(f"   {'||u-r||/||r||':>14} {'m (shift=1)':>14} {'m (shift=0)':>14} "
          f"{'||G|| shift=0':>15}")
    for scale in (1e-2, 1e-1, 1.0, 1e1, 1e2):
        xt = x + scale * la.norm(x) * pert / la.norm(pert)
        ft, _, _ = S.F(xt)
        rt = float(la.norm(ft) / np.sqrt(ft.size))
        m1, _ = pdf._m_and_grad(xt, centres, shift=1.0)
        m0, _ = pdf._m_and_grad(xt, centres, shift=0.0)
        print(f"   {scale:14.2e} {m1:14.6f} {m0:14.6e} {m0*rt:15.3e}")
    print("   -> with shift=0 the deflated residual DECAYS with distance, so a norm-based\n"
          "      linesearch can report convergence at absurd amplitude. With shift=1,\n"
          "      m -> 1 far away and the far field is undisturbed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=28)
    ap.add_argument("--constraint", default="d1")
    a = ap.parse_args()
    main(a.N, a.constraint)
