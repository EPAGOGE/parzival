"""G2 (two routes) + Q3 cost probe.

(1) ||J||_2 so cond(J) and cond(L) are comparable numbers.
(2) SMALL CONFIG, pure linear algebra: build the reduced generator L two ways --
    (i) the sparse-solve realization L^-1 f = [J^-1 (f on live rows)]_free
    (ii) an EXPLICIT DENSE reduction: eliminate dP by a dense Poisson solve, fold the
         duplicates, form M, then L = [I - Bc(CgBc)^-1 Cg] M and invert it densely.
    They must agree.  This validates embed/embedT/Linv on an object small enough to
    see whole.
(3) One complex resolvent point at the production size, timed: the cost model for Q3.
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


def build(S, z):
    Nx, Nb = S.Nx, S.Nb; n2 = Nx * Nb
    J = S.jacobian(z).tocsr()
    allr = np.arange(n2)
    liveT = np.setdiff1d(allr, np.union1d(S.rT_pin, S.rT_c0))
    free_rows = np.concatenate([liveT, n2 + liveT])
    part_l = np.array(sorted(S.rT_c0), dtype=int)
    part_r = np.array([S.partner[int(r)] for r in part_l], dtype=int)
    spec_p = np.concatenate([S.rP_bedge, S.rP_outer, S.rP_c0, S.rP_c1, S.rP_cornerI])
    coefP = np.broadcast_to(-(S.XI * S.G1 ** 2), (Nx, Nb)).ravel().copy()
    coefP[spec_p] = 0.0
    return dict(J=J, n2=n2, liveT=liveT, free_rows=free_rows, part_l=part_l,
                part_r=part_r, coefP=coefP, Nx=Nx, Nb=Nb)


def sparse_route(S, K):
    J, n2, fr = K["J"], K["n2"], K["free_rows"]
    lu = spla.splu(J.tocsc())
    def Linv(f):
        rhs = np.zeros(J.shape[0], dtype=f.dtype); rhs[fr] = f
        return lu.solve(rhs)[fr]
    return Linv


def dense_route(S, K):
    """Explicit reduction, everything dense.  Independent of embed/Linv."""
    J, n2, fr = K["J"], K["n2"], K["free_rows"]
    Jd = np.asarray(J.todense())
    nf = fr.size
    # Poisson slaving: dP = Lp^-1 diag(coefP) dA   (dense)
    Lp = Jd[2 * n2:3 * n2, 2 * n2:3 * n2]
    Pmap = np.linalg.solve(Lp, np.diag(K["coefP"]))          # n2 x n2
    # embedding matrix Z: free coords -> full unknown vector
    Z = np.zeros((J.shape[1], nf))
    Z[fr, np.arange(nf)] = 1.0
    Z[K["part_l"], :] = Z[K["part_r"], :]
    Z[n2 + K["part_l"], :] = Z[n2 + K["part_r"], :]
    Z[2 * n2:3 * n2, :] = Pmap @ Z[:n2, :]
    M = Jd[np.ix_(fr, np.arange(J.shape[1]))] @ Z
    Bc = Jd[np.ix_(fr, [J.shape[1] - 2, J.shape[1] - 1])]
    Cg = Jd[np.ix_([J.shape[0] - 2, J.shape[0] - 1], fr)]
    L = M - Bc @ np.linalg.solve(Cg @ Bc, Cg @ M)
    return L, Bc, Cg


# ---------------------------------------------------------------------------
print("=== (2) SMALL CONFIG, two routes ===", flush=True)
Ss = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(6, 10, 5), Nb=10,
                        eps_b=1e-3, alpha=-0.3447)
zs = Ss.pack(Ss.A0, Ss.B0, Ss.P0, Ss.P["cl"], Ss.P["cw"])
Ks = build(Ss, zs)
print(f"    dims: full={Ks['J'].shape[0]}  n_f={Ks['free_rows'].size}", flush=True)
Linv_s = sparse_route(Ss, Ks)
Ld, Bcd, Cgd = dense_route(Ss, Ks)
nf = Ks["free_rows"].size
Qc, _ = np.linalg.qr(Cgd.T)
Pk = lambda x: x - Qc @ (Qc.T @ x)
rng = np.random.default_rng(7)
errs, mags = [], []
for t in range(6):
    f = Pk(rng.standard_normal(nf))
    x_sparse = Linv_s(f)
    x_dense = np.linalg.solve(Ld, f)
    e = np.linalg.norm(x_sparse - x_dense) / np.linalg.norm(x_dense)
    errs.append(e); mags.append(np.linalg.norm(x_dense))
print(f"    L^-1 sparse-vs-dense rel diff: max={max(errs):.3e} med={np.median(errs):.3e}"
      f"   (gate G2 needs 1e-8)", flush=True)
sv = np.linalg.svd(Ld, compute_uv=False)
print(f"    dense L: ||L||={sv[0]:.4e}  sigma_min(full)={sv[-1]:.4e}  "
      f"sigma_min on ker(Cg) via dense: ", end="", flush=True)
Z0 = np.linalg.svd(Cgd)[2][2:].T.conj()
sv2 = np.linalg.svd(Z0.T @ Ld @ Z0, compute_uv=False)
print(f"{sv2[-1]:.4e}   (||.||={sv2[0]:.4e})", flush=True)
# and via the sparse route, same object
Amat = np.column_stack([Z0.T @ Linv_s(Z0[:, j]) for j in range(Z0.shape[1])])
sv3 = np.linalg.svd(Amat, compute_uv=False)
print(f"    1/||L^-1|| on ker(Cg), sparse route = {1.0/sv3[0]:.4e}   "
      f"rel diff vs dense = {abs(1.0/sv3[0]-sv2[-1])/sv2[-1]:.3e}", flush=True)

# ---------------------------------------------------------------------------
print("\n=== (1)+(3) PRODUCTION SIZE ===", flush=True)
d = np.load(SCR / "hunt_fields/rung_00_a-0.344712.npz")
a, z = float(d["a"]), d["z"]
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12), Nb=36,
                       eps_b=1e-4, alpha=a)
S.adopt_seed(z)
K = build(S, z)
J = K["J"]; n2 = K["n2"]; fr = K["free_rows"]; nf = fr.size
Jd_op = spla.LinearOperator(J.shape, matvec=lambda v: J @ v,
                            rmatvec=lambda v: J.T @ v)
x = np.random.default_rng(0).standard_normal(J.shape[1])
for _ in range(200):
    y = J @ x; x2 = J.T @ y; nn = np.linalg.norm(x2); x = x2 / nn
nJ2 = np.sqrt(nn)
print(f"    ||J||_2 = {nJ2:.4e}   sigma_min(J) = 2.5356e-06  ->  cond(J) = "
      f"{nJ2/2.5356e-06:.3e}", flush=True)
q = np.load(SCR / "quotient_state.npz")
print(f"    ||L||_2 = {float(q['nrmL']):.4e}  sigma_min(L|kerCg) = "
      f"{float(q['s_ker']):.4e}  ->  cond(L) = {float(q['nrmL'])/float(q['s_ker']):.3e}",
      flush=True)

Cg = np.asarray(J[-2:, :].todense())[:, fr]
Qc, _ = np.linalg.qr(Cg.T)
Pk = lambda v: v - Qc @ (Qc.T @ v)
mask = np.zeros(J.shape[0]); mask[K["liveT"]] = 1.0; mask[n2 + K["liveT"]] = 1.0
E = sp.diags(mask, format="csc")

def resolvent_norm(zz, iters=60, seed=0):
    Mz = (zz * E - J.tocsc()).tocsc()
    t0 = time.time(); lu = spla.splu(Mz); tlu = time.time() - t0
    luH = spla.splu(Mz.conj().T.tocsc())
    def R(f):
        rhs = np.zeros(J.shape[0], dtype=complex); rhs[fr] = f
        return lu.solve(rhs)[fr]
    def RH(f):
        rhs = np.zeros(J.shape[0], dtype=complex); rhs[fr] = f
        return luH.solve(rhs)[fr]
    v = Pk(np.random.default_rng(seed).standard_normal(nf).astype(complex))
    v /= np.linalg.norm(v); s = 0.0
    t1 = time.time()
    for _ in range(iters):
        y = Pk(R(v)); w = Pk(RH(y)); nn = np.linalg.norm(w)
        v = w / nn; s = np.sqrt(nn)
    return s, tlu, time.time() - t1

for zz in (0.0 + 0j, 1.0 + 0j, -1.0 + 0j, 0.0 + 1.0j, 0.5 + 0.5j):
    s, tlu, tit = resolvent_norm(zz)
    print(f"    z={zz.real:+.2f}{zz.imag:+.2f}i   ||R(z)||={s:.4e}   "
          f"1/||R||={1/s:.4e}   [LU {tlu:.2f}s + 60 it {tit:.2f}s]", flush=True)
