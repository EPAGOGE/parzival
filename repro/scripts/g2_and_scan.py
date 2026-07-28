"""G2 done right + the z ~ -1 feature + the noise floor + LU cost model.

WHY THE FIRST G2 ATTEMPT LOOKED LIKE A FAILURE.  L = Pi_c M is EXACTLY rank-deficient
by 2 on the full free space (Pi_c is an oblique projector of rank n_f - 2, so
ker L = M^-1(range Bc) is 2-dimensional).  Measured: sigma_min(dense L) = 1.26e-16
against ||L|| = 5.15e2.  So `solve(L, f)` on the FULL space is a singular solve and the
two routes may differ by any null vector -- which is what the 9.3e-1 "disagreement" was.
The operator that exists is L restricted to ker(Cg), and THAT is what G2 must compare.
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
    liveT = np.setdiff1d(np.arange(n2), np.union1d(S.rT_pin, S.rT_c0))
    fr = np.concatenate([liveT, n2 + liveT])
    pl = np.array(sorted(S.rT_c0), dtype=int)
    pr = np.array([S.partner[int(r)] for r in pl], dtype=int)
    sp_p = np.concatenate([S.rP_bedge, S.rP_outer, S.rP_c0, S.rP_c1, S.rP_cornerI])
    cP = np.broadcast_to(-(S.XI * S.G1 ** 2), (Nx, Nb)).ravel().copy(); cP[sp_p] = 0.0
    mask = np.zeros(J.shape[0]); mask[liveT] = 1.0; mask[n2 + liveT] = 1.0
    return dict(J=J, n2=n2, liveT=liveT, fr=fr, pl=pl, pr=pr, cP=cP, mask=mask)


# ===================== G2, small config, compressed operator ===============
print("=== G2: two routes, compressed onto ker(Cg) ===", flush=True)
Ss = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(6, 10, 5), Nb=10,
                        eps_b=1e-3, alpha=-0.3447)
zs = Ss.pack(Ss.A0, Ss.B0, Ss.P0, Ss.P["cl"], Ss.P["cw"])
K = build(Ss, zs); J = K["J"]; n2 = K["n2"]; fr = K["fr"]; nf = fr.size
Jd = np.asarray(J.todense())
Lp = Jd[2 * n2:3 * n2, 2 * n2:3 * n2]
Pmap = np.linalg.solve(Lp, np.diag(K["cP"]))
Z = np.zeros((J.shape[1], nf)); Z[fr, np.arange(nf)] = 1.0
Z[K["pl"], :] = Z[K["pr"], :]; Z[n2 + K["pl"], :] = Z[n2 + K["pr"], :]
Z[2 * n2:3 * n2, :] = Pmap @ Z[:n2, :]
M = Jd[fr, :] @ Z
Bc = Jd[np.ix_(fr, [J.shape[1] - 2, J.shape[1] - 1])]
Cg = Jd[np.ix_([J.shape[0] - 2, J.shape[0] - 1], fr)]
Ld = M - Bc @ np.linalg.solve(Cg @ Bc, Cg @ M)
Z0 = np.linalg.svd(Cg, full_matrices=True)[2][2:].T.conj()          # ker(Cg)
lu = spla.splu(J.tocsc())
def Linv(f):
    rhs = np.zeros(J.shape[0], dtype=f.dtype); rhs[fr] = f
    return lu.solve(rhs)[fr]
Rsp = np.column_stack([Z0.T @ Linv(Z0[:, j]) for j in range(Z0.shape[1])])
Ldn = Z0.T @ Ld @ Z0
Rdn = np.linalg.inv(Ldn)
rel = np.linalg.norm(Rsp - Rdn) / np.linalg.norm(Rdn)
print(f"    dims full={J.shape[0]} n_f={nf} quotient={Z0.shape[1]}", flush=True)
print(f"    sigma_min(L on the FULL free space) = "
      f"{np.linalg.svd(Ld, compute_uv=False)[-1]:.3e}  vs ||L||="
      f"{np.linalg.svd(Ld, compute_uv=False)[0]:.3e}   (exact rank deficiency 2)",
      flush=True)
print(f"    [G2] ||Z0^T L^-1 Z0  -  (Z0^T L Z0)^-1||_F / ||.||_F = {rel:.3e}"
      f"    (gate: 1e-8)   -> {'PASS' if rel < 1e-8 else 'FAIL'}", flush=True)
sv = np.linalg.svd(Ldn, compute_uv=False)
print(f"    quotient operator: ||L||={sv[0]:.4e} sigma_min={sv[-1]:.4e} "
      f"cond={sv[0]/sv[-1]:.3e}", flush=True)

# ===================== production: LU ordering + real-axis scan ============
print("\n=== production size: LU cost, noise floor, real-axis scan ===", flush=True)
d = np.load(SCR / "hunt_fields/rung_00_a-0.344712.npz")
a, zz = float(d["a"]), d["z"]
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12), Nb=36,
                       eps_b=1e-4, alpha=a)
S.adopt_seed(zz)
K = build(S, zz); J = K["J"]; n2 = K["n2"]; fr = K["fr"]; nf = fr.size
E = sp.diags(K["mask"], format="csc")
Cg = np.asarray(J[-2:, :].todense())[:, fr]
Qc, _ = np.linalg.qr(Cg.T); Pk = lambda v: v - Qc @ (Qc.T @ v)

Jcsc = J.tocsc()
for spec_ in ("COLAMD", "MMD_AT_PLUS_A", "MMD_ATA"):
    t0 = time.time()
    try:
        f_ = spla.splu(Jcsc, permc_spec=spec_)
        print(f"    LU permc_spec={spec_:<14s} {time.time()-t0:6.2f}s  nnz(L)+nnz(U)="
              f"{f_.L.nnz + f_.U.nnz:,}", flush=True)
    except Exception as ex:
        print(f"    LU permc_spec={spec_:<14s} FAILED {ex}", flush=True)

def rnorm(zv, iters=60, seed=0, permc="MMD_AT_PLUS_A"):
    Mz = (zv * E - Jcsc).tocsc()
    lu = spla.splu(Mz, permc_spec=permc); luH = spla.splu(Mz.conj().T.tocsc(), permc_spec=permc)
    def R(f):
        r = np.zeros(J.shape[0], dtype=complex); r[fr] = f; return lu.solve(r)[fr]
    def RH(f):
        r = np.zeros(J.shape[0], dtype=complex); r[fr] = f; return luH.solve(r)[fr]
    v = Pk(np.random.default_rng(seed).standard_normal(nf).astype(complex))
    v /= np.linalg.norm(v); s = 0.0
    for _ in range(iters):
        y = Pk(R(v)); w = Pk(RH(y)); nn = np.linalg.norm(w); v = w / nn; s = np.sqrt(nn)
    # noise floor: one step of iterative refinement on the extremal solve
    b = np.zeros(J.shape[0], dtype=complex); b[fr] = v
    x = lu.solve(b); res = b - Mz @ x
    dx = lu.solve(res)
    floor = np.linalg.norm(dx) / max(np.linalg.norm(x), 1e-300)
    return s, floor

print("    real-axis scan (1/||R|| == sigma_min of the pencil on the quotient):",
      flush=True)
for zv in (-1.30, -1.15, -1.0360, -1.00, -0.95, -0.85, -0.60, 0.0, 0.5, 1.0, 2.0):
    t0 = time.time(); s, fl = rnorm(complex(zv))
    print(f"      z={zv:+8.4f}   ||R||={s:.4e}   1/||R||={1/s:.4e}   "
          f"refine rel={fl:.2e}   [{time.time()-t0:.1f}s]", flush=True)
print(f"    c_w = {float(zz[-1]):.6f}   c_l = {float(zz[-2]):.6f}   "
      f"alpha = {a:+.8f}   1+2a0 = {1+2*a:.6f}", flush=True)
