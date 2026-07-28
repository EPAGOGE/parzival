"""DEFLATION: make every root we have already found REPULSIVE, and re-solve.

THE QUESTION THIS EXISTS TO ANSWER.  After the d1 constraint change and the eps_b -> 0
extrapolation, alpha converges cleanly: -0.331331 at N=28 and -0.331565 at N=44, agreeing
to 2.3e-4 (0.07%), each extrapolated linearly AND quadratically to 2e-7.  Chen-Hou report
-0.34240009.  A stable +3.16% offset that no longer looks like scatter.

Two explanations remain, and they are distinguishable:
  (a) a residual discretisation systematic in our scheme, or
  (b) WE ARE ON A DIFFERENT ROOT.
(b) has never been tested, and it is not far-fetched here: Newton has been single-started
from ONE seed (an interpolation of Chen-Hou's stored profile) in a basin known to be
fractal -- six earlier Cartesian configurations found three DIFFERENT wrong roots, and the
project's own notes record "the basin is too narrow to guess into".  Newton basin
boundaries for even a cubic polynomial are Julia sets with the Wada property: every
boundary point touches every basin, so an epsilon of seed uncertainty gives almost no
information about which root you land on.

Deflation decides it.  Plant a pole at the root we found; Newton then structurally CANNOT
return there and is forced to whatever else exists.  If a root sits at -0.3424 next door,
this finds it and the whole 3.16% gap dissolves into "we were in the wrong basin".  If
deflation comes back dry, (b) is dead and the gap is discretisation -- also worth knowing,
and cheaper to learn this way than by another resolution ladder.

THE OPERATOR (Farrell et al., arXiv:1410.5620, in refs/mathlit/deflation-continuation/):

    G(u) = M(u) F(u),     M(u) = prod_i ( ||u - r_i||^{-p} + alpha ) * I

THE SHIFT alpha = 1 IS MANDATORY, NOT COSMETIC.  With alpha = 0 the deflated residual
DECAYS as ||u - r|| -> infinity, so a residual-based linesearch happily reports spurious
convergence at enormous norm -- the paper measures a residual of 1e-13 at x ~ -1.2e8.  The
shift keeps M -> 1 far from the deflated roots, so far-field behaviour is unchanged and
only the neighbourhood of r_i is distorted.

THE JACOBIAN IS EXACT AND CHEAP.  M is a SCALAR function m(u) times the identity, so

    G'(u) = m(u) F'(u) + F(u) grad_m(u)^T

a rank-one update to the scaled Jacobian.  No finite differences, and since J is already
dense at these resolutions there is nothing to be clever about -- just add the outer
product.  With m = prod_i m_i and m_i = ||u-r_i||^{-p} + alpha,
    grad_m = m * sum_i ( -p ||u-r_i||^{-p-2} (u - r_i) / m_i ).

VALIDITY, checked rather than assumed.  Farrell's guarantee requires the deflated root to
be SIMPLE.  Two things make that true here and both are verified in the gate:
  * The two corner constraints BREAK the amplitude scaling symmetry completely.  Under
    Ot -> s Ot, Bt -> s^2 Bt the constraints become s*WX and s^2*THXX, so only s = 1
    satisfies both -- the scaling ORBIT collapses to a point and the root is isolated.
    (This is why deflation is safe here but would NOT be safe on the trivial root
    Om = B = 0, which solves the system for any (c_l, c_w) and is therefore a
    2-parameter family -- a non-simple root outside the theorem.  `Atlas.classify`
    labels it so it never gets deflated as if it were a point.)
  * J is nonsingular at our converged roots (cond ~ 1e7..1e10, sigma_min bounded away
    from zero once the projection's null directions are excluded).

"NON-SIMILAR" IS DOING REAL WORK.  Two numerically distinct vectors can be the same
solution.  Within one (N, XMAX, eps_b, constraint) the constraints pin the amplitude, so
sameness is just proximity -- but the INVARIANT LABEL is alpha = c_w/c_l, because alpha is
exactly the scaling-invariant coordinate (the common mode of (c_l, c_w) is the symmetry
direction alpha cannot see).  So a root is reported as physically NEW when its alpha
differs; two roots with the same alpha and different amplitudes would be the same solution
seen twice, and the constraints make that impossible anyway.
"""
import importlib.util
import json
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


# ---------------------------------------------------------------------------
class Atlas:
    """The set of roots found so far, with invariant labels and a sameness test."""

    def __init__(self, tol_rel=1e-6, tol_alpha=1e-7):
        self.roots = []            # list of dict(x, alpha, cl, cw, F, kind)
        self.tol_rel = tol_rel
        self.tol_alpha = tol_alpha

    def classify(self, St, x, r):
        """PHYSICAL / TRIVIAL / SUSPECT.  TRIVIAL must never be deflated as a point: it is
        a 2-parameter family (Om = B = 0 solves the system for ANY c_l, c_w) and therefore
        a non-simple root that Farrell's theorem does not cover."""
        S = St.S
        Ot, Bt = S.unpack(x[:-2])
        amp = float(max(np.abs(Ot).max(), np.abs(Bt).max()))
        cl = float(x[-2])
        if amp < 1e-8 or abs(cl) < 1e-8:
            return "TRIVIAL"
        cls_ = 2.0 * S.THXX_REF / S.WX_REF
        if r > 1e-9:
            return "SUSPECT"
        if abs((cl - cls_) / cls_) > 0.5:
            return "SUSPECT"
        return "PHYSICAL"

    def is_new(self, x, alpha):
        for R in self.roots:
            if abs(alpha - R["alpha"]) < self.tol_alpha:
                return False
            d = la.norm(x - R["x"]) / max(la.norm(x), 1e-300)
            if d < self.tol_rel:
                return False
        return True

    def add(self, St, x, r):
        cl, cw = float(x[-2]), float(x[-1])
        a = cw / cl
        kind = self.classify(St, x, r)
        rec = dict(x=x.copy(), alpha=a, cl=cl, cw=cw, F=float(r), kind=kind)
        self.roots.append(rec)
        return rec

    def deflatable(self):
        """Only simple roots.  TRIVIAL is a family, not a point."""
        return [R for R in self.roots if R["kind"] != "TRIVIAL"]

    def summary(self):
        return [dict(alpha=R["alpha"], cl=R["cl"], F=R["F"], kind=R["kind"])
                for R in self.roots]


# ---------------------------------------------------------------------------
def _m_and_grad(x, centres, p=2.0, shift=1.0):
    """m(x) = prod_i (||x-r_i||^-p + shift)  and  grad m."""
    m = 1.0
    acc = np.zeros_like(x)
    for r in centres:
        d = x - r
        nd = float(la.norm(d))
        if nd < 1e-14:
            return np.inf, acc
        mi = nd ** (-p) + shift
        m *= mi
        acc += (-p * nd ** (-p - 2.0) / mi) * d
    return m, m * acc


def deflated_newton(St, x0, centres, steps=60, tol=1e-11, p=2.0, shift=1.0,
                    verbose=False):
    """Newton on G(u) = m(u) F(u), with the EXACT rank-one-updated Jacobian.

    The convergence test is on the UNDEFLATED residual ||F||, never on ||G||: m -> 1 far
    from the centres but is large near them, and testing ||G|| is precisely how the
    alpha=0 pathology in the paper produces false convergence."""
    S = St.S
    x = x0.copy()
    f, cl, cw = S.F(x)
    r = float(la.norm(f) / np.sqrt(f.size))
    prev_g = np.inf
    taken = 0
    for it in range(steps):
        m, gm = _m_and_grad(x, centres, p, shift)
        if not np.isfinite(m):
            if verbose:
                print("    landed on a deflated centre exactly -- aborting", flush=True)
            return x, f, r, taken, False
        A = St.A_exact(x)
        Ot, Bt = S.unpack(x[:-2])
        B, Cg = St.exact_B(Ot, Bt), St.exact_Cg()
        n = St.n
        J = np.zeros((n + 2, n + 2))
        J[:n, :n], J[:n, n:], J[n:, :n] = A, B, Cg
        G = m * J + np.outer(f, gm)                 # exact, rank-one update
        g = m * f
        try:
            dx = la.solve(G, -g)
        except la.LinAlgError:
            dx = -la.lstsq(G, g, rcond=None)[0]
        lam, best = 1.0, None
        for _ in range(14):
            xt = x + lam * dx
            mt, _ = _m_and_grad(xt, centres, p, shift)
            if not np.isfinite(mt):
                lam *= 0.5
                continue
            ft, _, _ = S.F(xt)
            gt = float(la.norm(mt * ft) / np.sqrt(ft.size))
            if gt < prev_g:
                best = (xt, ft, float(la.norm(ft) / np.sqrt(ft.size)), gt)
                break
            lam *= 0.5
        if best is None:
            break
        x, f, r, prev_g = best
        taken += 1
        if verbose:
            print(f"    def-it{it:02d} ||F||={r:.4e} ||G||={prev_g:.4e} "
                  f"m={m:.4g} alpha={float(x[-1])/float(x[-2]):+.8f}", flush=True)
        if r < tol:
            break
    return x, f, r, taken, r < tol


# ---------------------------------------------------------------------------
def hunt(N, XMAX=25.0, eps_b=1e-3, constraint="d1", rounds=6, p=2.0, shift=1.0,
         seeds=None, verbose=True):
    """Find the first root normally, then deflate and re-solve until it comes up dry.

    `seeds` optionally supplies extra informed starts (the multi-start half of the idea:
    in a fractal basin, seed diversity buys more than iteration count).  Each seed is
    tried against the CURRENT deflation set, so a seed that would have fallen back into an
    already-found root is pushed out of it instead of wasting the round."""
    pst = _mod("pst", "polar_stability.py")
    pnl = _mod("pnl", "polar_nlens.py")
    atlas = Atlas()

    # round 0: the ordinary solve, no deflation
    St, x, r, cl, cw, info = pst.converge_exact(
        N, XMAX=XMAX, eps_b=eps_b, constraint=constraint, strict=False, outer_steps=80)
    if not (info.get("converged") and np.isfinite(cl)):
        print(f"  base solve did not converge (||F||={r:.2e}); nothing to deflate from")
        return atlas, St
    rec = atlas.add(St, x, r)
    if verbose:
        print(f"  root 0: alpha={rec['alpha']:+.8f} c_l={rec['cl']:.6f} "
              f"||F||={r:.2e} [{rec['kind']}]", flush=True)
        pnl.render(pnl.flags_for(St, x))

    pool = list(seeds or [])
    for k in range(1, rounds + 1):
        centres = [R["x"] for R in atlas.deflatable()]
        if not centres:
            break
        start = pool.pop(0) if pool else St.S.x0.copy()
        if verbose:
            print(f"  round {k}: deflating {len(centres)} root(s), "
                  f"{'informed seed' if pool or seeds else 'seed restart'}", flush=True)
        xn, fn, rn, taken, ok = deflated_newton(
            St, start, centres, p=p, shift=shift, verbose=verbose)
        if not ok or taken == 0:
            if verbose:
                print(f"    DRY: ||F||={rn:.2e} after {taken} steps -- "
                      f"no further root from this start", flush=True)
            continue
        a = float(xn[-1]) / float(xn[-2])
        if not atlas.is_new(xn, a):
            if verbose:
                print(f"    returned an ALREADY-KNOWN root (alpha={a:+.8f}) despite "
                      f"deflation -- check the pole is actually planted", flush=True)
            continue
        rec = atlas.add(St, xn, rn)
        if verbose:
            print(f"  root {len(atlas.roots)-1}: alpha={rec['alpha']:+.8f} "
                  f"c_l={rec['cl']:.6f} ||F||={rn:.2e} [{rec['kind']}]  *** NEW ***",
                  flush=True)
            pnl.render(pnl.flags_for(St, xn))
    return atlas, St


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=28)
    ap.add_argument("--xmax", type=float, default=25.0)
    ap.add_argument("--eps-b", type=float, default=1e-3)
    ap.add_argument("--constraint", default="d1")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--p", type=float, default=2.0)
    ap.add_argument("--shift", type=float, default=1.0)
    a = ap.parse_args()
    atlas, St = hunt(a.N, XMAX=a.xmax, eps_b=a.eps_b, constraint=a.constraint,
                     rounds=a.rounds, p=a.p, shift=a.shift)
    print("\nROOT ATLAS")
    print(f"  {'alpha':>14} {'c_l':>12} {'||F||':>10}  kind")
    for R in atlas.summary():
        print(f"  {R['alpha']:+14.8f} {R['cl']:12.6f} {R['F']:10.2e}  {R['kind']}")
    print(f"\n  reference alpha = -0.34240009   c_l* = "
          f"{2*St.S.THXX_REF/St.S.WX_REF:.8f}")
    out = pathlib.Path(HERE / "runs" / f"atlas_N{a.N}_{a.constraint}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(atlas.summary(), indent=1))
    print(f"  wrote {out}")
