"""q345.py -- Q3 pseudospectra / Q4 transient growth / Q5 invariant zeros / Q6 repeat.

The instrument spectrum.py built is a SPARSE resolvent: one LU per z, ~20 s at
production size.  That buys ~17 points -- enough for two axis scans, not for a
pseudospectral picture and nowhere near enough to locate the RHP minimum.

This file adds the dense route on the QUOTIENT, which is the same operator
spectrum.py's gates certified, in a form where a grid point costs O(n^2):

  1. COMPRESSION, blocked.  M = R_free J Z with Z the prolongation.  spectrum.py
     builds it one unit vector at a time (n_f sparse solves).  Here the P-slaving
     is one multi-RHS SuperLU solve and the rest is two sparse-times-dense
     products.  Same matrix, ~200x faster, verified column-wise against
     Realization.apply_M.

  2. QUOTIENT BY HOUSEHOLDER, not by an explicit basis.  ker(Cg) has codimension
     2, so the orthonormal basis Z0 is I minus a rank-2 reflector product.  Never
     form the n_f x n_f Q: apply the two Householder reflectors to Ld on both
     sides (four O(n^2) rank-1 updates) and read Lred off the trailing block.
     FREE CHECK: rows 0:2 of the transformed matrix must vanish, because
     range(Pi_c M) = ker(Cg) is exactly orthogonal to the first two columns of Q.

  3. COMPLEX SCHUR ONCE.  sigma_min(z I - Lred) = sigma_min(z I - T) exactly for
     T the Schur form.  Per grid point: overwrite the diagonal of a single stored
     T (O(n)) and run inverse power iteration with triangular solves (O(n^2)),
     warm-started from the previous point.  ~30 ms/point at n = 4758 instead of
     ~20 s.

WHY THE CROSSING VERDICT IS THEN A ONE-DIMENSIONAL QUESTION.  The resolvent norm
admits no local maximum in the resolvent set (Davies-Shargorodsky, Hilbert
space), and sigma_min(zI-L) >= |z| - ||L|| forces the RHP minimum into
|z| <= ||L||.  So

    min_{Re z >= 0} sigma_min(z I - L)

is attained EITHER on the imaginary axis OR at a point of the spectrum with
Re > 0.  The verdict therefore needs exactly two measurements: a dense imaginary
axis scan, and an answer to "is there spectrum in the open RHP".  The 2-D map is
run anyway as the empirical control on that argument, not as its substitute.
"""
import importlib.util
import pathlib
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SCRATCH = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/"
                       "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


spectrum = _mod("spectrum", SCRATCH / "spectrum.py")
pz = _mod("pz", pathlib.Path("/Users/epagogellc/parzival/boussinesq/polar_zeros.py"))


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 1. blocked compression
# ---------------------------------------------------------------------------
def dense_M_blocked(real):
    """M = R_free J Z, built with ONE multi-RHS solve instead of n_f single ones."""
    n2, n_f, N = real.n2, real.n_f, real.N
    t0 = time.time()
    # U = Z1: free coordinates + the C0 duplicate rows that copy their partner
    rows = np.concatenate([real.free, real.partner_l, n2 + real.partner_l])
    cols = np.concatenate([np.arange(n_f), real._pos_r, real._nl + real._pos_r])
    U = sp.csc_matrix((np.ones(rows.size), (rows, cols)), shape=(N, n_f))
    UA, UB = U[:n2, :], U[n2:2 * n2, :]
    rhs = np.asarray((real.JPA @ UA + real.JPB @ UB).todense())
    if real._lu_P is None:
        real._lu_P = spla.splu(real.JPP)
    XP = -real._lu_P.solve(rhs)                       # n2 x n_f, dense
    t_pro = time.time() - t0

    Jf = real.J[real.free, :].tocsr()
    M = np.asarray((Jf[:, :2 * n2] @ U[:2 * n2, :]).todense())
    M += Jf[:, 2 * n2:3 * n2] @ XP
    return M, t_pro, time.time() - t0


def verify_M(real, M, ntrial=4, seed=5):
    """The blocked M against Realization.apply_M, which the gates certified."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(ntrial):
        v = rng.standard_normal(real.n_f)
        worst = max(worst, np.linalg.norm(M @ v - real.apply_M(v))
                    / np.linalg.norm(real.apply_M(v)))
    return float(worst)


def householder2(C):
    """Two Householder reflectors H0, H1 with H1 H0 C upper triangular, C: n x 2.

    Returns V (n x 2, unit columns, zero above the pivot).  H_j = I - 2 v_j v_j^T.
    Q = H0 H1;  Q[:, :2] spans range(C);  Q[:, 2:] spans ker(C^T)."""
    C = np.array(C, dtype=float, copy=True)
    n = C.shape[0]
    V = np.zeros((n, 2))
    for j in range(2):
        x = C[j:, j].copy()
        nx = np.linalg.norm(x)
        if nx == 0.0:
            continue
        alpha = -np.sign(x[0]) * nx if x[0] != 0.0 else -nx
        v = x.copy()
        v[0] -= alpha
        nv = np.linalg.norm(v)
        if nv == 0.0:
            continue
        v /= nv
        V[j:, j] = v
        C[j:, j:] -= 2.0 * np.outer(v, v @ C[j:, j:])
    return V


def congruence(X, V):
    """X <- Q^T X Q in place, Q = H0 H1, applied as H1 (H0 X H0) H1.  O(n^2)."""
    for j in (0, 1):
        v = V[:, j]
        X -= 2.0 * np.outer(v, v @ X)
        X -= 2.0 * np.outer(X @ v, v)
    return X


def build_quotient(label, check=True):
    """Lred = the compressed generator on ker(Cg), n_finite x n_finite."""
    t0 = time.time()
    real, S, a, z = spectrum.load_production(label)
    cfg = spectrum.ROOTS[label][1]
    res_rms = float(np.linalg.norm(S.residual(z)) / np.sqrt(real.N))
    log(f"[{label}] root  degs={tuple(cfg['degs'])} Nb={cfg['Nb']} "
        f"eps_b={cfg['eps_b']:g}   alpha={a:+.8f}  h_id={S.h_id(z):+.4e}  "
        f"||F||_rms={res_rms:.3e}  converged=True")
    log(f"[{label}] N={real.N}  n_f={real.n_f}  dim ker(Cg)={real.n_finite}  "
        f"rank(E)={real.rank_E}   [load {time.time()-t0:.1f}s]")

    M, t_pro, t_M = dense_M_blocked(real)
    log(f"[{label}] blocked M: {M.shape}  [{t_M:.1f}s, prolongation {t_pro:.1f}s]")
    if check:
        w = verify_M(real, M)
        log(f"[{label}] CHECK blocked M vs Realization.apply_M: {w:.3e}")
        assert w < 1e-10, "blocked compression disagrees with the certified route"

    # oblique index-2 projection: Ld = Pi_c M,  Pi_c = I - Bc (Cg Bc)^-1 Cg
    Ld = M - real.Bc @ np.linalg.solve(real.CgBc, real.Cg @ M)
    del M
    V = householder2(real.Cg.T)
    congruence(Ld, V)
    lead = float(np.linalg.norm(Ld[:2, :]))
    Lred = np.ascontiguousarray(Ld[2:, 2:])
    del Ld
    log(f"[{label}] quotient by Householder: ||W[:2,:]|| = {lead:.3e}   "
        f"(exact 0: range(Pi_c M) = ker(Cg))")
    return dict(label=label, real=real, S=S, alpha=a, z=z, cfg=cfg,
                res_rms=res_rms, Lred=Lred, lead=lead)


# ---------------------------------------------------------------------------
# 2. sigma_min on a Schur form
# ---------------------------------------------------------------------------
class SchurSigma:
    """sigma_min(z I - L) for many z, from ONE complex Schur form.

    T is stored once.  Each z only rewrites the diagonal (O(n)); the solves are
    LAPACK ztrtrs (O(n^2)).  Inverse power iteration on (A^-1)^H A^-1, warm
    started from the previous grid point -- on a raster the start vector is
    already nearly the right singular vector and 3-8 iterations suffice.
    """

    def __init__(self, Lred, tol=1e-9, maxit=300, T=None):
        t0 = time.time()
        if T is None:
            self.T, _Zs = sla.schur(Lred, output="complex")
            del _Zs
        else:
            self.T = T
        self.d = np.ascontiguousarray(np.diag(self.T).copy())
        self.n = self.T.shape[0]
        self.tol, self.maxit = tol, maxit
        self.t_schur = time.time() - t0
        self.ev = self.d.copy()               # Schur diagonal = the eigenvalues
        self._v = None
        self.calls = 0
        self.iters = 0

    def _set(self, z):
        # A = z I - T.  Store -(T - z I): keep T, negate the solves' sign by
        # solving with (T - zI) and negating the result -- but sigma_min is
        # invariant under the global sign, so just use (T - z I).
        idx = np.arange(self.n)
        self.T[idx, idx] = self.d - z

    def _restore(self):
        idx = np.arange(self.n)
        self.T[idx, idx] = self.d

    def sigma_min(self, z, warm=True):
        self._set(z)
        try:
            if warm and self._v is not None:
                v = self._v.copy()
            else:
                v = np.random.default_rng(0).standard_normal(self.n) \
                    + 1j * np.random.default_rng(1).standard_normal(self.n)
            v /= np.linalg.norm(v)
            lam_old = 0.0
            it = 0
            for it in range(1, self.maxit + 1):
                y = sla.solve_triangular(self.T, v, lower=False,
                                         check_finite=False)
                w = sla.solve_triangular(self.T, y, lower=False, trans="C",
                                         check_finite=False)
                lam = np.linalg.norm(w)
                v = w / lam
                if it > 1 and abs(lam - lam_old) <= self.tol * lam:
                    break
                lam_old = lam
            self._v = v
            self.calls += 1
            self.iters += it
            return float(1.0 / np.sqrt(lam)), it
        finally:
            self._restore()

    def norm_R(self, z):
        s, it = self.sigma_min(z)
        return 1.0 / s, it


# ---------------------------------------------------------------------------
# 3. scans
# ---------------------------------------------------------------------------
def scan(ss, zs, tag="", every=25):
    out = np.empty(len(zs))
    t0 = time.time()
    for i, z in enumerate(zs):
        out[i], _ = ss.sigma_min(z)
        if every and (i + 1) % every == 0:
            log(f"    [{tag}] {i+1}/{len(zs)}  z={zs[i]:+.4g}  "
                f"sigma_min={out[i]:.6e}  [{time.time()-t0:.0f}s]")
    return out


def grid_map(ss, re, im, tag=""):
    """Boustrophedon raster so the warm start is always a neighbour."""
    S = np.empty((im.size, re.size))
    t0 = time.time()
    for j, y in enumerate(im):
        cols = range(re.size) if j % 2 == 0 else range(re.size - 1, -1, -1)
        for i in cols:
            S[j, i], _ = ss.sigma_min(complex(re[i], y))
        if (j + 1) % 5 == 0 or j == im.size - 1:
            log(f"    [{tag}] row {j+1}/{im.size}  y={y:+.4g}  "
                f"min={S[j].min():.4e}  [{time.time()-t0:.0f}s]")
    return S
