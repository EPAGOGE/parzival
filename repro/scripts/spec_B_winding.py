"""ADJUDICATION B -- replace the failed Hutchinson count with an exact winding count.

D2 proposed counting RHP eigenvalues by  N = (1/2pi i) oint tr((zE-J)^-1 E) dz  with a
~30-probe Hutchinson estimator.  Adjudication A measured that estimator returning 2.53
where the exact answer is 1.  It is dead.

REPLACEMENT: the same integral is  d/dz log det(zE - J), so the count is the WINDING
NUMBER of det(zE - J).  det comes free from the LU that the resolvent already needs:
SuperLU gives  A = Pr^T L U Pc^T  with L unit lower triangular, so

    det(A) = sign(perm_r) * sign(perm_c) * prod(diag(U))

and log|det| + i*arg(det) is exact.  Permutation parity is computed exactly (cycle
decomposition), so there is NO sign ambiguity between contour points -- the failure mode
that would otherwise inject spurious jumps of pi.  Unwrap arg continuously; the count is
(arg_end - arg_start)/2pi, which must come out an INTEGER.  Distance from an integer is
the method's own error bar.

Validated here against the QZ ground truth of Adjudication A, with ADAPTIVE refinement
(bisect any arc where |delta arg| > pi/2) so the cost is measured, not guessed.
"""
import importlib.util, pathlib, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla

H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
spec = importlib.util.spec_from_file_location("pc", str(H / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(spec); sys.modules["pc"] = pc; spec.loader.exec_module(pc)


def perm_parity(p):
    p = np.asarray(p).copy(); n = p.size
    seen = np.zeros(n, bool); par = 0
    for i in range(n):
        if seen[i]:
            continue
        j = i; ln = 0
        while not seen[j]:
            seen[j] = True; j = p[j]; ln += 1
        par ^= (ln - 1) & 1
    return -1.0 if par else 1.0


def logdet(Kz):
    """(log|det|, arg det) of a sparse complex matrix, exact up to roundoff."""
    lu = spla.splu(Kz.tocsc(), permc_spec="COLAMD",
                   options=dict(SymmetricMode=False))
    d = lu.U.diagonal()
    s = perm_parity(lu.perm_r) * perm_parity(lu.perm_c)
    return np.log(np.abs(d)).sum(), np.angle(d).sum() + (np.pi if s < 0 else 0.0)


def winding(Kfun, path, tol_arg=np.pi / 2, max_pts=4000, verbose=False):
    """Adaptive winding of det along a closed parametrised path t in [0,1)."""
    ts = list(np.linspace(0.0, 1.0, 33))          # includes t=1 == t=0
    cache = {}
    def A(t):
        if t not in cache:
            cache[t] = logdet(Kfun(path(t)))[1]
        return cache[t]
    for _ in range(60):
        vals = [A(t) for t in ts]
        # unwrap
        un = np.unwrap(np.array(vals))
        d = np.abs(np.diff(un))
        bad = np.where(d > tol_arg)[0]
        if bad.size == 0 or len(ts) > max_pts:
            break
        new = []
        for k in bad:
            new.append(0.5 * (ts[k] + ts[k + 1]))
        ts = sorted(set(ts) | set(new))
    vals = np.array([A(t) for t in ts]); un = np.unwrap(vals)
    w = (un[-1] - un[0]) / (2 * np.pi)
    return w, len(cache), float(np.abs(np.diff(un)).max())


# ------------------------------------------------------------------ small case
def build(S, z):
    Nx, Nb = S.Nx, S.Nb; n2 = Nx * Nb
    J = S.jacobian(z).tocsr()
    liveT = np.setdiff1d(np.arange(n2), np.union1d(S.rT_pin, S.rT_c0))
    fr = np.concatenate([liveT, n2 + liveT])
    return J, n2, liveT, fr


S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(6, 10, 5), Nb=10,
                       eps_b=1e-3, alpha=-0.3447)
z0 = S.pack(S.A0, S.B0, S.P0, S.P["cl"], S.P["cw"])
J, n2, liveT, fr = build(S, z0)
N = J.shape[0]
mask = np.zeros(N); mask[liveT] = 1.0; mask[n2 + liveT] = 1.0
Esp = sp.diags(mask, format="csc")
Jc = J.tocsc()

# ground truth from QZ
Jd = np.asarray(J.todense()); Ed = np.diag(mask)
AA, BB, aa, bb, Q, Zq = sla.ordqz(Jd, Ed, output="complex")
fin = np.abs(bb) > 1e-12 * max(abs(aa).max(), 1.0)
lam = aa[fin] / bb[fin]
print(f"GROUND TRUTH (QZ): {fin.sum()} finite eigenvalues, "
      f"{int((lam.real>0).sum())} with Re>0, max Re = {lam.real.max():+.5f}")

Kf = lambda z: (z * Esp - Jc)

tests = [
    ("circle |z|=1e6", lambda t: 1e6 * np.exp(2j * np.pi * t)),
    ("circle |z|=1",   lambda t: 1.0 * np.exp(2j * np.pi * t)),
    ("circle |z-5|=4.99", lambda t: 5.0 + 4.99 * np.exp(2j * np.pi * t)),
    ("circle |z|=0.1", lambda t: 0.1 * np.exp(2j * np.pi * t)),
]
print()
for name, path in tests:
    t0 = time.time()
    w, npts, dmax = winding(Kf, path)
    # direct count
    zs = np.array([path(t) for t in np.linspace(0, 1, 4000, endpoint=False)])
    ctr = zs.mean(); R = np.abs(zs - ctr).mean()
    direct = int((np.abs(lam - ctr) < R).sum())
    print(f"  {name:22s} winding = {w:+10.6f}  -> {round(w):+3d}   direct = {direct:+3d}"
          f"   [{npts} LU, {time.time()-t0:.1f}s, max|dArg| = {dmax:.3f}]"
          f"   {'OK' if round(w)==direct else '*** MISMATCH ***'}")

# ------------------------------------- the RHP contour: D-shape, the production shape
print("\nRHP CONTOUR (the production shape): imaginary segment + right half circle")
nrmL = 20.79 * 1.05   # small-case ||L|| bound; use spectral radius margin
def dpath(t, R=nrmL):
    """t in [0,1): up the imaginary axis from -iR to +iR, then the right半 circle back."""
    if t < 0.5:
        return 1j * R * (-1.0 + 4.0 * t)               # -iR -> +iR
    th = np.pi / 2 - 2 * np.pi * (t - 0.5)             # +iR -> -iR the right way
    return R * np.exp(1j * th)
t0 = time.time()
w, npts, dmax = winding(Kf, dpath)
direct = int(((lam.real > 0) & (np.abs(lam) < nrmL)).sum())
print(f"  winding = {w:+.6f} -> {round(w):+d}   direct RHP count = {direct}"
      f"   [{npts} LU, {time.time()-t0:.1f}s, max|dArg| = {dmax:.3f}]"
      f"   {'OK' if round(w)==direct else '*** MISMATCH ***'}")
