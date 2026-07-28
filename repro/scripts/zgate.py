"""Gate polar_zeros against the existing (contaminated) route at one N.

Three things must hold before any of it is used:
  1. spec(compressed M) == spec(QZ finite zeros)                 -- the identification
  2. spec(P@A) == spec(M) union {0, 0}                           -- against the old route
  3. the symbol S(k) reproduces the hand-derived far-field limit  -- p_inf = -i c_l k
"""
import sys, pathlib, numpy as np, scipy.linalg as la
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
sys.path.insert(0, str(H))
import importlib.util


def mod(name, fn):
    sp = importlib.util.spec_from_file_location(name, str(H / fn))
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m


pst = mod("pst", "polar_stability.py")
pz = mod("pz", "polar_zeros.py")

N = 28
St, x, r, cl, cw = pst.converge_exact(N, verbose=False)
print(f"N={N} dim={St.n} ||F||={r:.2e} c_l={cl:.6f} c_w={cw:.6f} alpha={cw/cl:.8f}")
print(f"  self-consistency residual  c_w - c_l*a0 = {cw - cl*St.C.a0:.3e}")

w_old, V, cCB, A, B, Cg = St.spectrum(x)
M, M_orth, Z = pz.compress(A, B, Cg)
w_M = la.eigvals(M); w_M = w_M[np.argsort(-w_M.real)]
w_qz, n_inf, al, be = pz.rosenbrock_zeros(A, B, Cg)

print(f"\n  ambient  P@A : {w_old.size} eigenvalues")
print(f"  compressed M : {w_M.size}  (n-2 = {St.n-2})")
print(f"  QZ pencil    : {w_qz.size} finite + {n_inf} infinite  (m=2 expected infinite)")


def match(a, b, lab):
    a = np.sort_complex(a); b = np.sort_complex(b)
    if a.size != b.size:
        print(f"  {lab}: SIZE MISMATCH {a.size} vs {b.size}"); return
    d = np.abs(a - b).max()
    print(f"  {lab}: max|diff| = {d:.3e}  {'OK' if d < 1e-6*max(1,np.abs(a).max()) else 'FAIL'}")


match(w_M, w_qz, "M   vs QZ                ")
nz = w_old[np.abs(w_old) > 1e-9]
match(np.sort_complex(nz)[-w_M.size:] if nz.size >= w_M.size else nz,
      w_M[:nz.size] if nz.size <= w_M.size else w_M, "P@A(nonzero) vs M (sizes) ")
print(f"  P@A near-zero count (|w|<1e-9): {(np.abs(w_old)<=1e-9).sum()}  (expect 2)")

pn_, sin_min = pz.proj_norm(B, Cg, Z)
print(f"\n  cond(Cg B)      = {cCB:.4g}      <- what we have been reporting")
print(f"  ||P|| = 1/sin   = {pn_:.4g}   (theta_min = {np.degrees(np.arcsin(sin_min)):.4f} deg)")
print(f"  departure from normality:  A {pz.departure_from_normality(A):.4g}"
      f"   Z^TAZ {pz.departure_from_normality(M_orth):.4g}"
      f"   M {pz.departure_from_normality(M):.4g}")

print("\n  top finite zeros (QZ):")
for z in w_qz[:8]:
    print(f"    {z.real:+.6f} {z.imag:+.6f}i")
lam = pz.nearest(w_qz, 1.0)
print(f"  symmetry mode  lambda_sym = {lam.real:+.6f}{lam.imag:+.6f}i   |lam-1| = {abs(lam-1):.4e}")

# --- symbol check ---------------------------------------------------------
print("\n  FAR-FIELD SYMBOL.  hand-derived limit: p_inf(k) = -i c_l k on both components")
for k in (0.0, 1.0, 5.0):
    Sk = pz.far_field_symbol(St, x, k)
    ev = la.eigvals(Sk)
    print(f"    k={k:5.1f}  max|Re| = {np.abs(ev.real).max():.4e}   "
          f"Im range [{ev.imag.min():+.4f},{ev.imag.max():+.4f}]   "
          f"predicted Im = {-cl*k:+.4f}")

th, h, ext = pz.essential_numerical_range(St, x)
print(f"\n  W_e support:  max Re = {h[0]:+.4e}   min Re = {-h[len(th)//2]:+.4e}")
print(f"                max Im = {h[len(th)//4]:+.4e}")
for z in list(w_qz[:6]) + [lam]:
    print(f"    {z.real:+.5f}{z.imag:+.5f}i  inside W_e: {pz.support_contains(th, h, z)}")
