"""Q3 PLAN + the numbers that decide whether it is affordable and trustworthy.

(a) where the DILATION's discrete breakage lives (may it be deflated? no -- but is it
    truncation or boundary?)
(b) the P-block conditioning -- the two smallest left singular vectors of J live 99.9%
    on the P rows, so J_PP is the noise floor of everything downstream
(c) cost model: one sparse LU of (z E - J) per grid point
(d) GATE G2 prototype: resolvent two ways (explicit dense reduced generator vs sparse
    bordered solve) on a small configuration
(e) resolvent norm at a few z: does the transport prediction hold?  With
    RO' -> -c_l A_xi as xi -> infinity (verified: the zeroth-order terms cancel at
    alpha self-consistency), the far field is pure OUTWARD advection at speed c_l on a
    finite interval with an inflow pin at xi=0 -- a NILPOTENT semigroup, empty
    continuum spectrum, and ||R(z)|| ~ exp(-Re(z) * XMAX / c_l) as Re z -> -infinity.
"""
import importlib.util, pathlib, sys, time
import numpy as np
import scipy.sparse as spr
import scipy.sparse.linalg as spla
import scipy.linalg as la

H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/"
                   "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
spec = importlib.util.spec_from_file_location("pc", str(H / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(spec); sys.modules["pc"] = pc; spec.loader.exec_module(pc)


def build(degs, Nb, eps_b, alpha, zst=None):
    S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=degs, Nb=Nb,
                           eps_b=eps_b, alpha=alpha)
    if zst is not None:
        S.adopt_seed(zst)
    return S


def index_sets(S):
    n2 = S.Nx * S.Nb
    N = 3 * n2 + 2
    pin = np.concatenate([S.rT_pin, S.rT_pin + n2])
    c0 = np.concatenate([S.rT_c0, S.rT_c0 + n2])
    prow = np.arange(2 * n2, 3 * n2)
    grow = np.array([N - 2, N - 1])
    alg = np.concatenate([pin, c0, prow, grow])
    dyn = np.setdiff1d(np.arange(N), alg)
    return n2, N, pin, c0, prow, grow, alg, dyn


# =============================================================== full-size root ==
d = np.load(SCR / "hunt_fields/rung_00_a-0.344712.npz")
a, z = float(d["a"]), d["z"]
S = build((16, 40, 12), 36, 1e-4, a, z)
n2, N, pin, c0, prow, grow, alg, dyn = index_sets(S)
J = S.jacobian(z).tocsr()
E = spr.diags(np.isin(np.arange(N), dyn).astype(float), format="csr")
cl, cw = float(z[-2]), float(z[-1])
print(f"root alpha={a:+.8f} c_l={cl:.6f} h_id={S.h_id(z):+.3e}  N={N} dyn={len(dyn)}",
      flush=True)

# ---- (a) localize the dilation's discrete breakage ---------------------------
A, B, P, _, _ = S.unpack(z)
x, G1, mu, a0 = S.XI, S.G1, S.mu, S.a0
dA = G1 * (A + x * ((S.Dx @ A) + a0 * A))
dB = G1 * (2.0 * B + x * ((S.Dx @ B) + (1.0 + 2.0 * a0) * B)) - B
dP = G1 * (2.0 * P + x * ((S.Dx @ P) + mu * P)) - 2.0 * P
v_d = np.concatenate([dA.ravel(), dB.ravel(), dP.ravel(), [0.0, 0.0]])
v_d /= np.linalg.norm(v_d)
w = J @ v_d
cov = np.zeros(N, bool); cov[dyn] = True
print("\n[a] DILATION breakage, localized on the covariant rows", flush=True)
tot_cov = np.linalg.norm(w[cov])
for lab, sel in (("A block", cov & (np.arange(N) < n2)),
                 ("B block", cov & (np.arange(N) >= n2) & (np.arange(N) < 2 * n2))):
    print(f"    {lab}: {np.linalg.norm(w[sel]):.3e}", flush=True)
ii = np.arange(N) % n2 // S.Nb            # radial node index of each field row
for lo, hi, lab in ((0, 5, "xi<x[5]  (corner panel head)"),
                    (5, S.Nx - 5, "interior"),
                    (S.Nx - 5, S.Nx, "outer 5 nodes")):
    sel = cov & (np.arange(N) < 2 * n2) & (ii >= lo) & (ii < hi)
    print(f"    radial band {lab:<28s}: {np.linalg.norm(w[sel]):.3e} "
          f"({np.linalg.norm(w[sel])/tot_cov:6.2%})", flush=True)

# ---- (b) the P block: the real noise floor -----------------------------------
Jpp = J[prow][:, np.arange(2 * n2, 3 * n2)].tocsc()
lup = spla.splu(Jpp); lupT = spla.splu(Jpp.T.tocsc())
xv = np.random.default_rng(1).standard_normal(n2); xv /= np.linalg.norm(xv)
for _ in range(80):
    xv = lup.solve(lupT.solve(xv)); xv /= np.linalg.norm(xv)
smin_p = np.linalg.norm(Jpp @ xv)
nrm_p = spla.norm(Jpp) if hasattr(spla, "norm") else np.sqrt((Jpp.multiply(Jpp)).sum())
print(f"\n[b] P BLOCK  sigma_min(J_PP) = {smin_p:.4e}   ||J_PP||_F = {nrm_p:.4e}   "
      f"ratio = {smin_p/nrm_p:.3e}", flush=True)
jb = np.argmax(np.abs(xv.reshape(S.Nx, S.Nb)).max(axis=0))
ib = np.argmax(np.abs(xv.reshape(S.Nx, S.Nb)).max(axis=1))
print(f"    softest P mode peaks at radial node {ib} (xi={S.x[ib]:.4f}), "
      f"beta node {jb} (beta={S.b[jb]:.4f});  corner-circle share "
      f"{np.linalg.norm(xv.reshape(S.Nx,S.Nb)[0])/np.linalg.norm(xv):.4f}", flush=True)

# ---- (c) cost model ----------------------------------------------------------
print("\n[c] COST of one grid point of the resolvent scan", flush=True)
for zv in (0.0, 0.7 + 0.9j):
    K = (zv * E - J).tocsc()
    t0 = time.time(); luK = spla.splu(K); t1 = time.time()
    rhs = np.random.default_rng(0).standard_normal(N).astype(K.dtype)
    t2 = time.time()
    for _ in range(10):
        luK.solve(rhs)
    t3 = time.time()
    print(f"    z={zv!s:>10s}: splu {t1-t0:6.2f}s   nnz(LU)={luK.nnz:.3e}   "
          f"solve {(t3-t2)/10*1e3:6.1f} ms   -> ~{t1-t0 + 60*(t3-t2)/10:5.2f}s "
          f"per z with 30 power-iteration pairs", flush=True)

# ---- (e) resolvent scan on a coarse line ------------------------------------
Cg = J[grow, :].toarray()
QGd = np.linalg.qr(Cg.T)[0][dyn, :]
QGd = np.linalg.qr(QGd)[0]


def rnorm(zv, iters=40, seed=5):
    K = (zv * E - J).tocsc()
    luK = spla.splu(K)
    cx = np.iscomplexobj(K)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(len(dyn)) + (1j * rng.standard_normal(len(dyn)) if cx else 0)
    v /= np.linalg.norm(v)
    s = 0.0

    def ap(f, tr):
        f = f - QGd @ (QGd.conj().T @ f)
        rhs = np.zeros(N, dtype=complex if cx else float); rhs[dyn] = f
        wv = luK.solve(rhs, trans=tr)[dyn]
        return wv - QGd @ (QGd.conj().T @ wv)
    for _ in range(iters):
        u = ap(ap(v, "N"), "H" if cx else "T")
        nn = np.linalg.norm(u); v = u / nn; s = nn
    return np.sqrt(s)


print("\n[e] RESOLVENT NORM on the real axis (quotient, 2-norm in collocation coords)",
      flush=True)
print(f"    transport prediction: ||R(z)|| ~ exp(-Re z * XMAX/c_l), "
      f"XMAX/c_l = {25.0/cl:.3f}", flush=True)
for zv in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
    t0 = time.time(); rv = rnorm(zv); t1 = time.time()
    print(f"      z={zv:+5.2f}   ||R(z)|| = {rv:.6e}   ({t1-t0:.1f}s)", flush=True)

# ============================================== (d) GATE G2 on a small config ==
print("\n[d] GATE G2 -- resolvent two ways on a SMALL configuration", flush=True)
Ss = build((6, 10, 5), 10, 1e-3, a)
zs = Ss.pack(Ss.A0, Ss.B0, Ss.P0, Ss.P["cl"], Ss.P["cw"])
n2s, Ns, pins, c0s, prows, grows, algs, dyns = index_sets(Ss)
Js = Ss.jacobian(zs).tocsr()
Es = spr.diags(np.isin(np.arange(Ns), dyns).astype(float), format="csr")
print(f"    small config: Nx={Ss.Nx} Nb={Ss.Nb} N={Ns} dyn={len(dyns)}", flush=True)

# explicit reduced generator
free = np.setdiff1d(np.arange(n2s), np.concatenate([Ss.rT_pin, Ss.rT_c0]))
nf = free.size
Rm = np.zeros((n2s, nf))
Rm[free, np.arange(nf)] = 1.0
pos = {int(f): i for i, f in enumerate(free)}
for r in Ss.rT_c0:
    Rm[int(r), pos[int(Ss.partner[int(r)])]] = 1.0
R2 = la.block_diag(Rm, Rm)
Jd = Js.toarray()
fieldc = np.arange(2 * n2s)
Pc = np.arange(2 * n2s, 3 * n2s)
cc = np.array([Ns - 2, Ns - 1])
Smap = -la.solve(Jd[np.ix_(prows, Pc)], Jd[np.ix_(prows, fieldc)])
M = (Jd[np.ix_(dyns, fieldc)] + Jd[np.ix_(dyns, Pc)] @ Smap) @ R2
Nc = Jd[np.ix_(dyns, cc)]
Gm = Jd[np.ix_(grows, fieldc)] @ R2
GN = Gm @ Nc
L = (np.eye(2 * nf) - Nc @ la.solve(GN, Gm)) @ M
U_, s_, Vt_ = la.svd(Gm)
Z = Vt_[2:].T.conj()
LZ = Z.T @ L @ Z
print(f"    reduced dims: free {nf} per field, state {2*nf}, ker(G) {Z.shape[1]}; "
      f"cond(GN)={np.linalg.cond(GN):.3e}", flush=True)
for zv in (0.0, 0.3, -0.8 + 0.4j):
    dense = 1.0 / la.svdvals(zv * np.eye(Z.shape[1]) - LZ)[-1]
    # sparse bordered route
    K = (zv * Es - Js).tocsc()
    luK = spla.splu(K)
    cx = np.iscomplexobj(K)
    QG = np.linalg.qr(np.linalg.qr(Jd[np.ix_(grows, np.arange(Ns))].T)[0][dyns, :])[0]

    def ap(f, tr):
        f = f - QG @ (QG.conj().T @ f)
        rhs = np.zeros(Ns, dtype=complex if cx else float); rhs[dyns] = f
        wv = luK.solve(rhs, trans=tr)[dyns]
        return wv - QG @ (QG.conj().T @ wv)
    Rmat = np.column_stack([ap(np.eye(len(dyns), dtype=complex if cx else float)[:, j],
                               "N") for j in range(len(dyns))])
    sparse_n = la.svdvals(Rmat)[0]
    print(f"      z={zv!s:>12s}  dense reduced {dense:.10e}   sparse bordered "
          f"{sparse_n:.10e}   rel diff {abs(dense-sparse_n)/dense:.3e}", flush=True)
print("done", flush=True)
