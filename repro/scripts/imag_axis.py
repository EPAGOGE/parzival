"""The decisive Q3 number: min over the imaginary axis of 1/||R(iy)|| = the eps at
which the pseudospectrum first REACHES Re z = 0.  Plus the Kreiss lower bound and a
corrected cost model (one LU per z, using trans='H' instead of a second factorization).
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
a, z0 = float(d["a"]), d["z"]
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12), Nb=36,
                       eps_b=1e-4, alpha=a)
S.adopt_seed(z0)
Nx, Nb = S.Nx, S.Nb; n2 = Nx * Nb
J = S.jacobian(z0).tocsc()
liveT = np.setdiff1d(np.arange(n2), np.union1d(S.rT_pin, S.rT_c0))
fr = np.concatenate([liveT, n2 + liveT]); nf = fr.size
mask = np.zeros(J.shape[0]); mask[liveT] = 1.0; mask[n2 + liveT] = 1.0
E = sp.diags(mask, format="csc")
Cg = np.asarray(J[-2:, :].todense())[:, fr]
Qc, _ = np.linalg.qr(Cg.T); Pk = lambda v: v - Qc @ (Qc.T @ v)

def rnorm(zv, iters=40, seed=0, tol=1e-6):
    t0 = time.time()
    Mz = (zv * E - J).tocsc()
    lu = spla.splu(Mz, permc_spec="COLAMD")
    tlu = time.time() - t0
    def R(f):
        r = np.zeros(J.shape[0], dtype=complex); r[fr] = f; return lu.solve(r)[fr]
    def RH(f):
        r = np.zeros(J.shape[0], dtype=complex); r[fr] = f
        return lu.solve(r, trans="H")[fr]
    v = Pk(np.random.default_rng(seed).standard_normal(nf).astype(complex))
    v /= np.linalg.norm(v); s = prev = 0.0; nit = 0
    for k in range(iters):
        y = Pk(R(v)); w = Pk(RH(y)); nn = np.linalg.norm(w); v = w / nn
        s = np.sqrt(nn); nit = k + 1
        if k > 3 and abs(s - prev) < tol * s:
            break
        prev = s
    b = np.zeros(J.shape[0], dtype=complex); b[fr] = v
    x = lu.solve(b); dx = lu.solve(b - Mz @ x)
    return s, np.linalg.norm(dx) / np.linalg.norm(x), time.time() - t0, tlu, nit

print("IMAGINARY AXIS  (eps at which the pseudospectrum reaches Re z = 0)", flush=True)
best = (np.inf, None)
for y in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
    s, fl, tt, tlu, nit = rnorm(1j * y)
    inv = 1.0 / s
    if inv < best[0]: best = (inv, y)
    print(f"   z={0.0:+.2f}{y:+.2f}i   ||R||={s:.4e}  1/||R||={inv:.4e}  "
          f"refine={fl:.1e}  [{tt:.1f}s, LU {tlu:.1f}s, {nit} it]", flush=True)
print(f"   MIN over the sampled imaginary axis: 1/||R|| = {best[0]:.4e} at y={best[1]}",
      flush=True)

print("\nRIGHT HALF PLANE -- Kreiss constant  K = sup_{Re z>0} Re(z)||R(z)||", flush=True)
K = 0.0
for x in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    s, fl, tt, tlu, nit = rnorm(complex(x))
    K = max(K, x * s)
    print(f"   z={x:+.2f}   ||R||={s:.4e}   Re(z)||R||={x*s:.4e}   [{tt:.1f}s]", flush=True)
print(f"   K >= {K:.4e}   =>   sup_t ||e^(tL)|| >= {K:.4e}   (Kreiss lower bound)",
      flush=True)
