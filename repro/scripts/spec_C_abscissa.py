"""ADJUDICATION C -- the numerical abscissa, measured directly, in FOUR norms.

Why this is the load-bearing measurement and not a Q4 footnote:  every eigenvalue of L
satisfies  Re lam <= omega(L) = lam_max((L + L^H)/2).  So omega < 0 is a ONE-SHOT,
contour-free certificate that the right half plane is empty, and omega > 0 bounds the
region a contour count would have to sweep.  D1 and D2 both reported omega only through
the weak inequality  omega >= Re z - 1/||R(z)||  from resolvent probes (D1: >= 90.196 at
z=100, "still climbing"; D2: Kreiss K >= 89.5).  Neither computed it.  Lanczos on the
Hermitian part computes it exactly, and each apply costs one cached Poisson solve.

omega is NORM-DEPENDENT (open tension #16).  So measure it in a ladder:
  W0  raw collocation l2                      -- what every number in both derivations used
  W1  quadrature-weighted L2(dxi dbeta)       -- an honest FUNCTION-SPACE norm
  W2  physical L2(dy) of (Om, Th) on the wedge -- undoes the substitution AND the division
  W3(gamma)  W1 * e^{2 gamma xi}              -- a one-parameter tilt, swept for a zero crossing

The forward operator is assembled from the solver's own blocks; the adjoint is the exact
transpose of that assembly (no finite differences, no re-derivation).
"""
import importlib.util, pathlib, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/"
                   "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
spec = importlib.util.spec_from_file_location("pc", str(H / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(spec); sys.modules["pc"] = pc; spec.loader.exec_module(pc)

d = np.load(SCR / "hunt_fields/rung_00_a-0.344712.npz")
a, zroot = float(d["a"]), d["z"]
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12), Nb=36,
                       eps_b=1e-4, alpha=a)
S.adopt_seed(zroot)
Nx, Nb = S.Nx, S.Nb; n2 = Nx * Nb
J = S.jacobian(zroot).tocsr(); N = J.shape[0]
res = S.residual(zroot)
print(f"ROOT  alpha={a:.8f}  ||F||_rms={np.linalg.norm(res)/np.sqrt(res.size):.3e}"
      f"  h_id={S.h_id(zroot):+.3e}   grid ({Nx}x{Nb}) N={N}")

liveT = np.setdiff1d(np.arange(n2), np.union1d(S.rT_pin, S.rT_c0))
fr = np.concatenate([liveT, n2 + liveT]); nf = fr.size
part_l = np.array(sorted(S.rT_c0), dtype=int)
part_r = np.array([S.partner[int(r)] for r in part_l], dtype=int)
spec_p = np.concatenate([S.rP_bedge, S.rP_outer, S.rP_c0, S.rP_c1, S.rP_cornerI])
coefP = np.broadcast_to(-(S.XI * S.G1 ** 2), (Nx, Nb)).ravel().copy(); coefP[spec_p] = 0.0
lp = S._lp_factor()
Lp = J[2 * n2:3 * n2, 2 * n2:3 * n2].tocsc()
lpH = spla.splu(Lp.conj().T.tocsc())
Jc = J.tocsc(); JH = J.conj().T.tocsr()
Bc = np.asarray(J[:, [N - 2, N - 1]].todense())[fr, :]
Cg = np.asarray(J[[N - 2, N - 1], :].todense())[:, fr]
CB = Cg @ Bc
print(f"  n_f={nf}   cond(Cg Bc)={np.linalg.cond(CB):.4f}   det={np.linalg.det(CB):+.5f}")


def Zap(x):
    u = np.zeros(N, dtype=x.dtype); u[fr] = x
    u[part_l] = u[part_r]; u[n2 + part_l] = u[n2 + part_r]
    u[2 * n2:3 * n2] = lp.solve(coefP * u[:n2])
    return u


def ZapH(v):
    w = v[:2 * n2].copy()
    w[:n2] += coefP.conj() * lpH.solve(v[2 * n2:3 * n2])
    t = w.copy()
    t[part_r] += w[part_l]; t[n2 + part_r] += w[n2 + part_l]
    return t[fr]


def Mv(x):    return (Jc @ Zap(x))[fr]
def MHv(w):
    v = np.zeros(N, dtype=w.dtype); v[fr] = w
    return ZapH(JH @ v)
def Lv(x):
    y = Mv(x); return y - Bc @ np.linalg.solve(CB, Cg @ y)
def LHv(y):
    return MHv(y - Cg.conj().T @ np.linalg.solve(CB.conj().T, Bc.conj().T @ y))


# sanity: <Lx, y> == <x, L^H y>
rng = np.random.default_rng(3)
x = rng.standard_normal(nf); y = rng.standard_normal(nf)
lhs = float(np.dot(Lv(x), y)); rhs = float(np.dot(x, LHv(y)))
print(f"  adjoint check: <Lx,y>={lhs:.8e}  <x,L^H y>={rhs:.8e}  "
      f"rel={abs(lhs-rhs)/abs(lhs):.2e}")

# ---------------------------------------------------------------- norm ladder
xi = S.x
dxi = np.zeros(Nx)
mid = 0.5 * (xi[1:] + xi[:-1])
edges_x = np.concatenate([[xi[0]], mid, [xi[-1]]])
dxi = np.diff(edges_x)
dxi = np.maximum(dxi, 1e-14)
bb = S.b
midb = 0.5 * (bb[1:] + bb[:-1])
edges_b = np.concatenate([[bb[0]], midb, [bb[-1]]])
dbe = np.diff(edges_b)
quad = np.outer(dxi, dbe).ravel()                     # dxi dbeta on the grid
r = np.exp(xi) - 1.0
geoA = (np.exp(2 * a * xi) * xi ** 2 * r * np.exp(xi))            # |Om|^2 from |A|^2
geoB = (np.exp(2 * (1 + 2 * a) * xi) * xi ** 4 * r * np.exp(xi))  # |Th|^2 from |B|^2
w_A2 = np.outer(geoA, np.ones(Nb)).ravel(); w_B2 = np.outer(geoB, np.ones(Nb)).ravel()


def make_w(kind, gamma=0.0):
    if kind == "W0":
        return np.ones(nf)
    if kind == "W1":
        return np.concatenate([quad[liveT], quad[liveT]])
    if kind == "W2":
        return np.concatenate([(quad * w_A2)[liveT], (quad * w_B2)[liveT]])
    if kind == "W3":
        t = np.exp(2 * gamma * np.repeat(xi, Nb))
        return np.concatenate([(quad * t)[liveT], (quad * t)[liveT]])
    raise ValueError


def abscissa(w, k=1):
    """lam_max of the Hermitian part of D L D^-1 restricted to D ker(Cg)."""
    dv = np.sqrt(w / w.max())
    inv = 1.0 / dv
    Cw = Cg * inv[None, :]                       # Cg D^-1
    Qw, _ = np.linalg.qr(Cw.T.conj())
    def Pk(v): return v - Qw @ (Qw.conj().T @ v)
    def Hv(v):
        v = Pk(v)
        a1 = dv * Lv(inv * v)
        a2 = inv * LHv(dv * v)
        return Pk(0.5 * (a1 + a2))
    op = spla.LinearOperator((nf, nf), matvec=Hv, rmatvec=Hv, dtype=float)
    hi = spla.eigsh(op, k=k, which="LA", return_eigenvectors=False, tol=1e-8,
                    maxiter=20000)
    lo = spla.eigsh(op, k=1, which="SA", return_eigenvectors=False, tol=1e-8,
                    maxiter=20000)
    return float(np.max(hi)), float(np.min(lo))


print("\n--- numerical abscissa omega(L) = lam_max Herm(L), by norm ---")
print(f"{'norm':>28s} {'omega(L)':>14s} {'lam_min Herm':>16s}   {'weight span':>12s}")
for kind, gam in [("W0", 0), ("W1", 0), ("W2", 0)]:
    w = make_w(kind, gam)
    t0 = time.time(); hi, lo = abscissa(w)
    print(f"{kind:>28s} {hi:>+14.5e} {lo:>+16.5e}   "
          f"{w.max()/w.min():>12.2e}   [{time.time()-t0:.1f}s]")

print("\n--- W3(gamma): sweep the exponential tilt for a zero crossing ---")
for gam in (-1.5, -1.0, -0.6, -0.3446, 0.0, 0.3446, 1.0):
    w = make_w("W3", gam)
    t0 = time.time(); hi, lo = abscissa(w)
    print(f"      gamma={gam:+7.4f}   omega={hi:+.5e}   lam_min={lo:+.5e}"
          f"   [{time.time()-t0:.1f}s]")
