"""Q5 -- invariant zeros of (M, Bc, Cg), by a route that never forms Pi_c M.

polar_zeros.rosenbrock_zeros computes them as the finite generalized eigenvalues
of [[A,B],[Cg,0]] - lambda diag(I,0) by dense QZ.  At production that pencil is
4762 x 4762 (6442 at root B) and QZ is O(30 n^3).  The SPARSE equivalent is the
descriptor pencil spectrum.py already certified (G2b: pencil (E,J) vs the
compressed generator, 3.734e-10):

    J v = lambda E v      ->      (J - sigma E)^-1 E v = theta v,  lambda = sigma + 1/theta

One sparse LU of (J - sigma E), then Arnoldi.  The index-2 infinite eigenvalues sit
at theta = 0 and are exactly what 'largest |theta|' discards -- the structural
reason control theory uses a pencil algorithm here instead of an eigensolver on an
explicitly formed projection.

This is the INDEPENDENT route.  Its disagreement with diag(Schur(Lred)) -- which
went through the blocked compression, the oblique index-2 projection, the
Householder quotient and a dense complex Schur -- is the conditioning measurement
the premortem demands before any eigenvalue may be printed.
"""
import sys, os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
os.chdir("/private/tmp/claude-501/-Users-epagogellc/"
         "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
sys.path.insert(0, os.getcwd())
import q345
log = q345.log

LAB = sys.argv[1] if len(sys.argv) > 1 else "A"
SHIFTS = [complex(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                               else ["-0.3+0.2j", "0.5+0j", "0.05+0.05j"])]
K = 24

real, S, a, z = q345.spectrum.load_production(LAB)
E = real.E.tocsc()
Jc = real.Jc
log(f"[{LAB}] descriptor pencil (E,J)  N={real.N}  rank(E)={real.rank_E}  "
    f"finite count = dim ker(Cg) = {real.n_finite}")

ev_dense = np.load(f"q345_ev_{LAB}.npy") if os.path.exists(f"q345_ev_{LAB}.npy") else None

for sig in SHIFTS:
    t0 = time.time()
    lu = spla.splu((Jc - sig * E).tocsc())
    t_lu = time.time() - t0
    OP = spla.LinearOperator((real.N, real.N), dtype=complex,
                             matvec=lambda v: lu.solve(E @ v))
    t0 = time.time()
    th = spla.eigs(OP, k=K, which="LM", return_eigenvectors=False, tol=1e-12,
                   maxiter=5000)
    lam = sig + 1.0 / th
    lam = lam[np.argsort(-lam.real)]
    log(f"\n[{LAB}] shift sigma = {sig}   [LU {t_lu:.0f}s, Arnoldi {time.time()-t0:.0f}s]")
    log(f"    {K} pencil eigenvalues nearest the shift, rightmost first:")
    for v in lam[:10]:
        line = f"      {v.real:+.10e} {v.imag:+.10e}i"
        if ev_dense is not None:
            d = np.abs(ev_dense - v)
            j = int(np.argmin(d))
            line += (f"    nearest dense-Schur {ev_dense[j].real:+.8e}"
                     f"{ev_dense[j].imag:+.8e}i   |diff| = {d[j]:.3e}")
        log(line)
    if ev_dense is not None:
        worst = max(float(np.abs(ev_dense - v).min()) for v in lam)
        rmax_sparse = float(lam.real.max())
        log(f"    worst |sparse pencil - nearest dense Schur| over all {K} = {worst:.4e}")
        log(f"    rightmost from this shift: {rmax_sparse:+.10e}"
            f"   (dense Schur global max Re = {ev_dense.real.max():+.10e})")
log("DONE")
