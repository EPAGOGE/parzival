"""Operator scale, noise floor, numerical-abscissa lower bound, imaginary axis.

||R(z)|| <= 1/(Re z - omega(L)) whenever Re z > omega(L), so a single measured
resolvent norm at large real z is a RIGOROUS lower bound on the numerical abscissa:
    omega(L) >= Re z - 1/||R(z)||.
"""
import importlib.util, pathlib, sys, time
import numpy as np
import scipy.sparse as spr
import scipy.sparse.linalg as spla

H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/"
                   "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
spec = importlib.util.spec_from_file_location("pc", str(H / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(spec); sys.modules["pc"] = pc; spec.loader.exec_module(pc)

d = np.load(SCR / "hunt_fields/rung_00_a-0.344712.npz")
a, z = float(d["a"]), d["z"]
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12), Nb=36,
                       eps_b=1e-4, alpha=a)
S.adopt_seed(z)
n2 = S.Nx * S.Nb; N = 3 * n2 + 2
pin = np.concatenate([S.rT_pin, S.rT_pin + n2])
c0 = np.concatenate([S.rT_c0, S.rT_c0 + n2])
prow = np.arange(2 * n2, 3 * n2); grow = np.array([N - 2, N - 1])
alg = np.concatenate([pin, c0, prow, grow])
dyn = np.setdiff1d(np.arange(N), alg)
J = S.jacobian(z).tocsr()
E = spr.diags(np.isin(np.arange(N), dyn).astype(float), format="csr")
Cg = J[grow, :].toarray()
QGd = np.linalg.qr(np.linalg.qr(Cg.T)[0][dyn, :])[0]

# ---- operator scale: ||L||_F by Hutchinson, using the forward map -------------
free = np.setdiff1d(np.arange(n2), np.concatenate([S.rT_pin, S.rT_c0]))
nf = free.size
rows = list(free); cols = list(range(nf)); vals = [1.0] * nf
pos = {int(f): i for i, f in enumerate(free)}
for r in S.rT_c0:
    rows.append(int(r)); cols.append(pos[int(S.partner[int(r)])]); vals.append(1.0)
Rm = spr.csr_matrix((vals, (rows, cols)), shape=(n2, nf))
R2 = spr.block_diag([Rm, Rm], format="csr")
Jpp = J[prow][:, np.arange(2 * n2, 3 * n2)].tocsc()
lup = spla.splu(Jpp)
Jpf = J[prow][:, np.arange(2 * n2)].tocsr()
Jdf = J[dyn][:, np.arange(2 * n2)].tocsr()
JdP = J[dyn][:, np.arange(2 * n2, 3 * n2)].tocsr()
Nc = J[dyn][:, [N - 2, N - 1]].toarray()
Gf = Cg[:, :2 * n2] @ R2
GN = Gf @ Nc
assert dyn.size == 2 * nf, (dyn.size, 2 * nf)


def Lapply(q):
    q = q - QGd @ (QGd.T @ q)
    f = R2 @ q
    dP = lup.solve(-(Jpf @ f))
    raw = Jdf @ f + JdP @ dP
    c = -np.linalg.solve(GN, Gf @ raw)
    return raw + Nc @ c


rng = np.random.default_rng(0)
acc = []
for _ in range(24):
    g = rng.standard_normal(2 * nf)
    acc.append(np.linalg.norm(Lapply(g)) ** 2 / np.linalg.norm(g) ** 2)
LF = np.sqrt(np.mean(acc) * 2 * nf)
print(f"||L||_F ~ {LF:.4e}  (Hutchinson, 24 probes; state dim {2*nf})", flush=True)
print(f"    => ||L||_2 in [{LF/np.sqrt(2*nf):.3e}, {LF:.3e}]", flush=True)


def rnorm(zv, iters=40, seed=5):
    K = (zv * E - J).tocsc()
    luK = spla.splu(K)
    cx = np.iscomplexobj(K)
    r2 = np.random.default_rng(seed)
    v = r2.standard_normal(len(dyn)) + (1j * r2.standard_normal(len(dyn)) if cx else 0)
    v /= np.linalg.norm(v); s = 0.0

    def ap(f, tr):
        f = f - QGd @ (QGd.conj().T @ f)
        rhs = np.zeros(N, dtype=complex if cx else float); rhs[dyn] = f
        wv = luK.solve(rhs, trans=tr)[dyn]
        return wv - QGd @ (QGd.conj().T @ wv)
    for _ in range(iters):
        u = ap(ap(v, "N"), "H" if cx else "T")
        nn = np.linalg.norm(u); v = u / nn; s = nn
    return np.sqrt(s)


print("\nRIGHT-HALF-PLANE probe + numerical abscissa lower bound", flush=True)
best = -np.inf
for zv in (2.0, 5.0, 20.0, 100.0):
    t0 = time.time(); rv = rnorm(zv); t1 = time.time()
    lb = zv - 1.0 / rv
    best = max(best, lb)
    print(f"    z={zv:7.1f}  ||R(z)||={rv:.5e}   omega(L) >= {lb:10.4f}   "
          f"({t1-t0:.1f}s)", flush=True)
print(f"    numerical abscissa LOWER BOUND (collocation 2-norm): "
      f"omega(L) >= {best:.4f}", flush=True)

print("\nIMAGINARY-AXIS / off-axis preview (conjugate-symmetric, Im>=0 only)",
      flush=True)
for zv in (1.0j, 2.0j, 0.5 + 1.0j, 1.0 + 1.0j):
    t0 = time.time(); rv = rnorm(zv); t1 = time.time()
    print(f"    z={zv!s:>12s}  ||R(z)||={rv:.5e}   eps-level 1/||R||={1/rv:.3e}   "
          f"({t1-t0:.1f}s)", flush=True)
print("done", flush=True)
