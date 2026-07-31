"""
RESOLVENT NORM and TRANSIENT GROWTH -- the right objects for a non-normal operator.

WHY NOT EIGENVALUES
-------------------
Section 28 measured the eigenvalue condition numbers of the constrained operator `L`:

    MEDIAN kappa ~ 3e15  (machine epsilon moves the typical eigenvalue by ~0.7)
    the unstable complex pairs chased since section 21: kappa ~ 1e14 - 1e15
    departure from normality: 1.414 for L (against 0.10 for the unconstrained A)

So the eigenvalues are not measurable, the N-study could never have converged, and the
"unstable oscillatory mode" was an artifact of a nearly-defective operator.

THE REPLACEMENTS, and they are WELL-CONDITIONED
-----------------------------------------------
  RESOLVENT NORM      R(s) = ||(sI - L)^-1||_2 = 1 / sigma_min(sI - L)
  TRANSIENT GROWTH    G(t) = ||exp(L t)||_2

Both are singular-value quantities, hence Lipschitz in the matrix (a perturbation of size
eps moves them by at most eps). They cannot be hypersensitive the way eigenvalues are.

They also answer the PHYSICAL question directly, which eigenvalues do not for a
non-normal operator. This is the lesson of hydrodynamic stability: pipe flow is linearly
STABLE at every Reynolds number where it demonstrably becomes turbulent -- every
eigenvalue in the left half plane, while `G(t)` reaches 1e3 or more before decaying.
Eigenvalues give the `t -> infinity` fate; `G(t)` gives what actually happens.

READING THE OUTPUT
  G(t) <= 1 for all t                  -> genuinely stable, nothing amplifies
  G(t) peaks >> 1 then decays          -> STABLE spectrum, large TRANSIENT growth; a
                                          finite-amplitude perturbation can still be
                                          amplified enormously before decaying
  G(t) grows without bound             -> genuine instability
  R(s) large far into Re(s) < 0        -> the pseudospectrum protrudes into the right
                                          half plane: eigenvalues say stable, the
                                          operator behaves unstably

THE TEST THAT MATTERS: unlike the eigenvalues, these MUST converge in N. If they do, this
project finally has a measurable object.

sigma_min is obtained by inverse power iteration on an LU factorisation rather than a full
SVD per grid point -- one `n^3/3` factorisation plus a few `n^2` solves, instead of a full
`n^3` SVD.
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(HERE / fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def build_L(N, XMAX=25.0):
    """Converged profile, then the gauge-projected linearisation."""
    pst = _mod("pst", "polar_stability.py")
    St, x, r, cl, cw, _info = pst.converge_exact(N, XMAX=XMAX)
    Ot, Bt = St.S.unpack(x[:-2])
    A = St.A_exact(x)
    B = St.exact_B(Ot, Bt)
    Cg = St.exact_Cg()
    n = St.n
    L = (np.eye(n) - B @ la.solve(Cg @ B, Cg)) @ A
    return L, r, cl, cw, n


def sigma_min(M, iters=12):
    """Smallest singular value by inverse power iteration on an LU factorisation.
    Falls back to a full SVD if the factorisation fails."""
    try:
        lu = sla.lu_factor(M)
    except Exception:
        return float(la.svd(M, compute_uv=False)[-1])
    n = M.shape[0]
    v = np.random.default_rng(0).standard_normal(n)
    v /= la.norm(v)
    s = 0.0
    for _ in range(iters):
        w = sla.lu_solve(lu, v)                 # M^-1 v
        u = sla.lu_solve(lu, w, trans=2)        # M^-H (M^-1 v)
        nrm = la.norm(u)
        if nrm == 0 or not np.isfinite(nrm):
            return float(la.svd(M, compute_uv=False)[-1])
        v = u / nrm
        s = nrm
    return 1.0 / np.sqrt(s) if s > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[28, 36])
    ap.add_argument("--tmax", type=float, default=12.0)
    ap.add_argument("--nt", type=int, default=25)
    a_ = ap.parse_args()

    store = {}
    for N in a_.Ns:
        L, r, cl, cw, n = build_L(N)
        print(f"N={N}  dim={n}  ||F||={r:.2e}  c_l={cl:.6f}  alpha={cw/cl:.6f}", flush=True)

        # ---- TRANSIENT GROWTH ------------------------------------------------
        ts = np.linspace(0.0, a_.tmax, a_.nt)
        G = []
        for t in ts:
            E = sla.expm(L * t)
            G.append(float(la.svd(E, compute_uv=False)[0]))
        G = np.array(G)
        imax = int(np.argmax(G))
        print(f"   TRANSIENT GROWTH  max ||e^(Lt)|| = {G.max():.4e} at t = {ts[imax]:.2f}"
              f"   G(0)={G[0]:.3f}  G(tmax)={G[-1]:.4e}")
        print("     t:  " + " ".join(f"{t:7.2f}" for t in ts[::4]))
        print("     G:  " + " ".join(f"{g:7.3g}" for g in G[::4]), flush=True)

        # ---- RESOLVENT along the real axis -----------------------------------
        svals = np.linspace(-2.0, 2.0, 21)
        R = []
        for s in svals:
            R.append(1.0 / max(sigma_min(s * np.eye(n) - L), 1e-300))
        R = np.array(R)
        print("   RESOLVENT NORM on the real axis  R(s) = 1/sigma_min(sI - L):")
        print("     s:  " + " ".join(f"{s:7.2f}" for s in svals[::2]))
        print("     R:  " + " ".join(f"{v:7.2g}" for v in R[::2]), flush=True)
        store[N] = (ts, G, svals, R)
        print(flush=True)

    if len(store) > 1:
        Ns = sorted(store)
        print("CONVERGENCE IN N -- the test that eigenvalues FAILED:")
        t0, G0, s0, R0 = store[Ns[0]]
        t1, G1, s1, R1 = store[Ns[-1]]
        gm0, gm1 = G0.max(), G1.max()
        print(f"   max transient growth: N={Ns[0]} -> {gm0:.4e}   N={Ns[-1]} -> {gm1:.4e}"
              f"   rel diff {abs(gm1-gm0)/max(gm1,1e-300):.3e}")
        rd = np.abs(R1 - R0) / np.maximum(np.abs(R1), 1e-300)
        print(f"   resolvent norm on the real axis: median rel diff {np.median(rd):.3e}"
              f"   max {rd.max():.3e}")
        print("\n   Small differences => these ARE measurable, unlike the eigenvalues")
        print("   (median eigenvalue kappa ~ 3e15). Large differences => even these need")
        print("   more care, and the operator is harder than non-normality alone explains.")


if __name__ == "__main__":
    main()
