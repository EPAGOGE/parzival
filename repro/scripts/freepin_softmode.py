"""Soft-mode diagnostic for the freed system at the ground seed.

Measures, at the ground grid (16,40,12)/Nb36/eps_b=1e-4:
  1. the exact (undamped) Newton step from the seed: where does the linear
     model put the freed root?  (dTH, dcl, dcw, field norms)
  2. sigma_min of J_free (inverse power iteration on (J^T J)^-1 via splu)
     and the composition of its right singular vector: TH/cl/cw components,
     per-block field norms, alignment with the gI-line and the T_s grading.
  3. same sigma_min estimate for the BASE pinned Jacobian at the same state
     (the corner-junction cluster ~5e-9 is known and shared; the question is
     whether freeing adds a NEW soft direction carrying TH/cl).
"""
import pathlib
import sys

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from freepin import FreePinSolver  # noqa: E402

SCR = pathlib.Path(__file__).parent
SEED = SCR / "hunt_fields/rung_00_a-0.344712.npz"
S, z0 = FreePinSolver.from_seed(SEED, degs=(16, 40, 12), Nb=36, eps_b=1e-4)
n2, n3 = S._n2, S._n3
WX = S.WX_REF

F = S.residual(z0)
J = S.jacobian(z0).tocsc()
lu = spla.splu(J)

# 1. exact Newton step
dz = lu.solve(-F)
dTH, dcl, dcw = dz[n3], dz[-2], dz[-1]
cl0, cw0 = float(z0[-2]), float(z0[-1])
da = (cw0 + dcw) / (cl0 + dcl) - cw0 / cl0
print("=== exact Newton step from ground seed ===")
print(f"dTH = {dTH:+.6e}   dcl = {dcl:+.6e}   dcw = {dcw:+.6e}")
print(f"implied dalpha = {da:+.6e}")
print(f"|dA| = {np.linalg.norm(dz[:n2]):.3e}  |dB| = {np.linalg.norm(dz[n2:2*n2]):.3e}"
      f"  |dP| = {np.linalg.norm(dz[2*n2:n3]):.3e}")
zn = z0 + dz
Fn = S.residual(zn)
print(f"||F|| after one full step: {np.linalg.norm(Fn)/np.sqrt(Fn.size):.3e} rms "
      f"(seed was {np.linalg.norm(F)/np.sqrt(F.size):.3e})")


def sigma_min_vec(lu_, n, iters=30, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x /= np.linalg.norm(x)
    lam = None
    for _ in range(iters):
        y = lu_.solve(lu_.solve(x), trans="T") if False else None
        # right singular vector: power iteration on (J^T J)^{-1} = J^{-1} J^{-T}
        y = lu_.solve(lu_.solve(x, trans="T"))
        lam = np.linalg.norm(y)
        x = y / lam
    return 1.0 / np.sqrt(lam), x


sig, v = sigma_min_vec(lu, J.shape[0])
print("\n=== J_free smallest singular value (inverse power, 30 it) ===")
print(f"sigma_min(J_free) ~ {sig:.3e}")
vTH, vcl, vcw = v[n3], v[-2], v[-1]
print(f"v: TH = {vTH:+.4f}  cl = {vcl:+.4f}  cw = {vcw:+.4f}")
print(f"   |A| = {np.linalg.norm(v[:n2]):.4f}  |B| = {np.linalg.norm(v[n2:2*n2]):.4f}"
      f"  |P| = {np.linalg.norm(v[2*n2:n3]):.4f}")
# gI-line alignment: along the identity, dcl = (2/WX) dTH
if abs(vTH) > 1e-8:
    print(f"   dcl/dTH along v = {vcl/vTH:+.4f}   (gI line: {2/WX:+.4f};"
          f" T_s grading would need dcl/cl=dTH/(2 TH))")
    print(f"   dcw/dTH along v = {vcw/vTH:+.4f}   (alpha-preserving would be"
          f" {cw0/cl0*2/WX:+.4f})")

# 3. base pinned Jacobian at the stripped state
Jb = S.jacobian_base_probe = None
import importlib  # noqa: E402
Fb_state = np.concatenate([z0[:n3], z0[-2:]])
S._refresh(float(z0[n3]))
Jb = super(FreePinSolver, S).jacobian(Fb_state).tocsc()
lub = spla.splu(Jb)
sigb, vb = sigma_min_vec(lub, Jb.shape[0])
print("\n=== J_base (pinned) smallest singular value at same state ===")
print(f"sigma_min(J_base) ~ {sigb:.3e}")
print(f"vb: cl = {vb[-2]:+.4f}  cw = {vb[-1]:+.4f}  "
      f"|A| = {np.linalg.norm(vb[:n2]):.4f}  |B| = {np.linalg.norm(vb[n2:2*n2]):.4f}"
      f"  |P| = {np.linalg.norm(vb[2*n2:n3]):.4f}")
print(f"\nratio sigma_min(base)/sigma_min(free) = {sigb/sig:.2f}")
