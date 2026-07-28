"""ADJUDICATION E -- the obliqueness metric BOTH derivations reported wrongly.

D1 and D2 both quote cond(Cg Bc) = 6.2603 as the structural conditioning of the index-2
reduction.  polar_zeros.py's docstring says in as many words that this is NOT the
governing quantity: for the oblique projector P = I - B(CgB)^-1 Cg,

    ||P|| = ||I - P|| = 1 / sin(theta_min(ker Cg, range B))

(Kato; Xu-Zikatanov 1307.4393 Cor 4.2) -- and cond(Cg B) is not it.  Measure the right
one, with polar_zeros' own stable routine (sines directly, never sqrt(1-cos^2)).
Also confirm the far-field symbol, hence W_e, for the CORNER-REGULARIZED rows.
"""
import importlib.util, pathlib, sys
import numpy as np, scipy.sparse as sp, scipy.linalg as la

H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/"
                   "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
def mod(n, f):
    s = importlib.util.spec_from_file_location(n, str(H / f))
    m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
pc = mod("pc", "polar_cornerreg.py"); pz = mod("pz", "polar_zeros.py")

d = np.load(SCR / "hunt_fields/rung_00_a-0.344712.npz")
a, zroot = float(d["a"]), d["z"]
S = pc.CornerRegSolver(edges=(0.0,2.0,15.0,25.0), degs=(16,40,12), Nb=36, eps_b=1e-4, alpha=a)
S.adopt_seed(zroot)
Nx, Nb = S.Nx, S.Nb; n2 = Nx*Nb
J = S.jacobian(zroot).tocsr(); N = J.shape[0]
liveT = np.setdiff1d(np.arange(n2), np.union1d(S.rT_pin, S.rT_c0))
fr = np.concatenate([liveT, n2+liveT])
Bc = np.asarray(J[:, [N-2, N-1]].todense())[fr, :]
Cg = np.asarray(J[[N-2, N-1], :].todense())[:, fr]
CB = Cg @ Bc
pn, smin = pz.proj_norm(Bc, Cg)
print(f"  cond(Cg Bc)          = {np.linalg.cond(CB):.4f}      <- what D1 and D2 both quote")
print(f"  sin(theta_min)       = {smin:.6e}")
print(f"  ||P|| = ||I-P||      = {pn:.4f}          <- what governs the obliqueness")
print(f"  ratio ||P||/cond     = {pn/np.linalg.cond(CB):.4f}")
print(f"  ||Bc||={np.linalg.norm(Bc,2):.4e}  ||Cg||={np.linalg.norm(Cg,2):.4e}")
print(f"  det(Cg Bc) = {np.linalg.det(CB):+.6f}   Cg Bc =\n{CB}")

# far-field symbol of the CORNERREG rows, read off the assembled Jacobian
print("\n  far-field check (outermost radial nodes): coefficient of the xi-derivative")
cl, cw = float(zroot[-2]), float(zroot[-1])
xi = S.x; G1 = np.asarray(S.G1).ravel(); g = xi*G1
E1 = np.exp(a*xi)/np.maximum(G1, 1e-300)
print(f"    xi_max = {xi[-1]:.3f}   G1 = {G1[-1]:.4e}   g = {g[-1]:.10f}   "
      f"E1 = e^(a0 xi)/G1 = {E1[-1]:.4e}")
print(f"    c_l = {cl:.8f}   c_w = {cw:.8f}   a0 = {a:.8f}   c_w/c_l = {cw/cl:.8f}   "
      f"a0 - c_w/c_l = {a - cw/cl:.2e}")
print(f"    RO' zeroth-order coefficient (c_w - a0 c_l) = {cw - a*cl:+.3e}")
print(f"    RB' zeroth-order coefficient 2(c_w - a0 c_l) = {2*(cw - a*cl):+.3e}")
print(f"    => symbol p(k) = -i c_l k on BOTH components;  W_e = i R;  max Re W_e = 0")
