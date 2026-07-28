"""Two things to settle:
  (1) M vs QZ differ by up to 146 -- is that a real disagreement or a sorting artifact?
  (2) QZ has NOTHING near +1, but the ambient P@A route reported +1.048/+1.053 as the
      symmetry mode. One of the two is wrong. Decide with RESIDUALS, not with matching.
"""
import sys, pathlib, numpy as np, scipy.linalg as la
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
import importlib.util


def mod(name, fn):
    sp = importlib.util.spec_from_file_location(name, str(H / fn))
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m


pst = mod("pst", "polar_stability.py")
pz = mod("pz", "polar_zeros.py")

N = 28
St, x, r, cl, cw = pst.converge_exact(N)
w_old, V, cCB, A, B, Cg = St.spectrum(x)
M, M_orth, Z = pz.compress(A, B, Cg)
w_M = la.eigvals(M)
w_qz, n_inf, al, be = pz.rosenbrock_zeros(A, B, Cg)
n, m = A.shape[0], Cg.shape[0]
print(f"N={N} n={n}  ||A||={la.norm(A):.4g}  ||B||={la.norm(B):.4g}  ||Cg||={la.norm(Cg):.4g}")

# --- proper set comparison: greedy nearest, report the WORST unmatched -------
def setdiff(a, b, lab):
    a = np.asarray(a); b = np.asarray(b)
    d = np.abs(a[:, None] - b[None, :])
    ia = d.min(axis=1)
    print(f"  {lab}: worst nearest-neighbour distance = {ia.max():.4e} "
          f"(median {np.median(ia):.3e}); {int((ia > 1e-6).sum())}/{a.size} unmatched at 1e-6")
    bad = np.argsort(-ia)[:5]
    for i in bad:
        print(f"      {a[i].real:+.6f}{a[i].imag:+.6f}i   dist {ia[i]:.3e}")


setdiff(w_M, w_qz, "M -> QZ")
setdiff(w_qz, w_M, "QZ -> M")
nz = w_old[np.abs(w_old) > 1e-9]
setdiff(nz, w_M, "P@A(nonzero) -> M")

# --- residuals: which set actually satisfies the DAE eigenproblem? ----------
# lambda is an invariant zero iff  [[A - lam I, B],[Cg, 0]] is singular.
def pencil_smin(lam):
    K = np.zeros((n + m, n + m), dtype=complex)
    K[:n, :n] = A - lam * np.eye(n)
    K[:n, n:] = B
    K[n:, :n] = Cg
    return la.svdvals(K)[-1]


scale = la.norm(A)
print(f"\n  sigma_min of [[A-lam,B],[Cg,0]]  (relative to ||A|| = {scale:.4g})")
print("  a zero of the pencil => sigma_min ~ 0.  a non-zero => sigma_min = O(scale).")
cands = {
    "P@A  +1.05-ish": pz.nearest(w_old, 1.0),
    "QZ   top real ": w_qz[0],
    "M    top real ": w_M[np.argsort(-w_M.real)][0],
    "QZ   nearest 1": pz.nearest(w_qz, 1.0),
    "control  +1.00": 1.0 + 0j,
    "control  +3.70": 3.7 + 0j,
}
for k, v in cands.items():
    if np.isnan(np.real(v)):
        continue
    print(f"    {k}  lam = {np.real(v):+.6f}{np.imag(v):+.6f}i   "
          f"sigma_min = {pencil_smin(v):.4e}   rel {pencil_smin(v)/scale:.3e}")

# --- what does the ambient route actually have near +1? ---------------------
real_old = w_old[np.abs(w_old.imag) < 1e-6]
real_old = real_old[np.argsort(-real_old.real)]
print(f"\n  ambient P@A real eigenvalues in (-2, 8): "
      + " ".join(f"{z.real:+.5f}" for z in real_old if -2 < z.real < 8))
real_qz = w_qz[np.abs(w_qz.imag) < 1e-6]
real_qz = real_qz[np.argsort(-real_qz.real)]
print(f"  QZ finite real eigenvalues in (-2, 8):    "
      + " ".join(f"{z.real:+.5f}" for z in real_qz if -2 < z.real < 8))
real_M = w_M[np.abs(w_M.imag) < 1e-6]
real_M = real_M[np.argsort(-real_M.real)]
print(f"  compressed M real eigenvalues in (-2, 8): "
      + " ".join(f"{z.real:+.5f}" for z in real_M if -2 < z.real < 8))
