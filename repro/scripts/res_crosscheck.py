"""Resolution cross-check of the corner measurements using the res-study files."""
import sys
import importlib.util
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad"
HF = SCRATCH + "/hunt_fields"
spec = importlib.util.spec_from_file_location(
    "pc", "/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py")
pc = importlib.util.module_from_spec(spec)
sys.modules["pc"] = pc
spec.loader.exec_module(pc)

for fn, degs in (("branch1_deg16_40.npz", (16, 40, 12)),
                 ("branch1_deg24_56.npz", (24, 56, 12))):
    d = np.load(f"{HF}/{fn}")
    z = d["z"]
    S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=degs, Nb=36,
                           eps_b=1e-5 if "deg" in fn else 1e-4, alpha=float(d["a"]))
    n = S.Nx * S.Nb
    assert z.size == 3 * n + 2, (fn, z.size, 3 * n + 2)
    A = z[:n].reshape(S.Nx, S.Nb)
    P = z[2 * n:3 * n].reshape(S.Nx, S.Nb)
    cl, cw = float(z[3 * n]), float(z[3 * n + 1])
    c = (cl - 2 * cw) / 4
    a = float(d["a"])
    print(f"{fn}: degs={degs} Nx={S.Nx} a_frozen={a:+.9f}")
    print(f"  cl={cl:+.9f} cw={cw:+.9f} cw/cl={cw / cl:+.9f} c={c:+.9f}")
    # corner fit
    ep = S.eps_b
    kk = np.pi / (np.pi / 2 - 2 * ep)
    s = np.sin(kk * (S.b - ep))
    cfit = float(np.dot(P[0, :], s) / np.dot(s, s))
    res = float(np.linalg.norm(P[0, :] - cfit * s) / np.linalg.norm(P[0, :]))
    print(f"  P0 fit sin(k(b-eps)): c_fit={cfit:+.9f} rel resid={res:.3e}")
    # corner radial derivative of A
    o0, o1 = int(S.offs[0]), int(S.offs[1])
    D0 = S.Dx[o0:o1, o0:o1].toarray()
    dA0 = (D0 @ A[o0:o1, :])[0, :]
    cb = np.cos(S.b)
    coef = float(np.dot(dA0, cb) / np.dot(cb, cb))
    print(f"  dA/dxi(0,:): rms={np.sqrt(np.mean(dA0 ** 2)):.6e} "
          f"cosb-coef={coef:+.6e}")
    print(f"  first off-corner nodes xi={S.x[1]:.6f},{S.x[2]:.6f}: "
          f"|A(1)-A(0)|rms={np.sqrt(np.mean((A[1] - A[0]) ** 2)):.4e} "
          f"|A(2)-A(0)|rms={np.sqrt(np.mean((A[2] - A[0]) ** 2)):.4e}")
