"""Tension #10 discriminator: soft-mode measurement at deg24/eps_b=1e-5.

Does the identity-line near-null (ground: sigma_min(J_free)=2.98e-8 vs
base 2.54e-6, 85x) stiffen or persist at the branch grid?
State: the pinned branch root (branch1_deg24_56.npz), TH=THXX_REF.
"""
import pathlib
import sys

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from freepin import FreePinSolver  # noqa: E402

SCR = pathlib.Path(__file__).parent
SEED = SCR / "hunt_fields/branch1_deg24_56.npz"
S, z0 = FreePinSolver.from_seed(SEED, degs=(24, 56, 12), Nb=36, eps_b=1e-5)
n2, n3 = S._n2, S._n3
WX = S.WX_REF

J = S.jacobian(z0).tocsc()
lu = spla.splu(J)
Fb_state = np.concatenate([z0[:n3], z0[-2:]])
S._refresh(float(z0[n3]))
Jb = super(FreePinSolver, S).jacobian(Fb_state).tocsc()
lub = spla.splu(Jb)


def sigma_min_vec(lu_, n, iters=30, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x /= np.linalg.norm(x)
    lam = None
    for _ in range(iters):
        y = lu_.solve(lu_.solve(x, trans="T"))
        lam = np.linalg.norm(y)
        x = y / lam
    return 1.0 / np.sqrt(lam), x


sig, v = sigma_min_vec(lu, J.shape[0])
sigb, vb = sigma_min_vec(lub, Jb.shape[0])
cl0, cw0 = float(z0[-2]), float(z0[-1])
print(f"deg24/eps1e-5 at pinned branch root (cl={cl0:.6f}):")
print(f"sigma_min(J_free) ~ {sig:.3e}    sigma_min(J_base) ~ {sigb:.3e}"
      f"    ratio base/free = {sigb/sig:.2f}")
vTH, vcl, vcw = v[n3], v[-2], v[-1]
print(f"v_free: TH={vTH:+.4f} cl={vcl:+.4f} cw={vcw:+.4f}  "
      f"|A|={np.linalg.norm(v[:n2]):.4f} |B|={np.linalg.norm(v[n2:2*n2]):.4f}"
      f" |P|={np.linalg.norm(v[2*n2:n3]):.4f}")
if abs(vTH) > 1e-8:
    print(f"  dcl/dTH = {vcl/vTH:+.4f} (gI line {2/WX:+.4f})   "
          f"dcw/dTH = {vcw/vTH:+.4f} (alpha-preserving {cw0/cl0*2/WX:+.4f})")
print(f"ground comparison: free 2.982e-8, base 2.536e-6, ratio 85.02")
