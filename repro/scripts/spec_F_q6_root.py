"""Q6 -- converge the GROUND root at a second resolution, with the standing discipline:
converged FLAG, accepted-step count, and h_id printed beside alpha.  Never a bare residual."""
import importlib.util, pathlib, sys, time
import numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/"
                   "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
s = importlib.util.spec_from_file_location("pc", str(H / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(s); sys.modules["pc"] = pc; s.loader.exec_module(pc)

for degs, Nb, eps_b in [((24, 56, 12), 36, 1e-4), ((16, 40, 12), 36, 5e-5)]:
    t0 = time.time()
    S, z, r, info = pc.converge(edges=(0.0, 2.0, 15.0, 25.0), degs=degs, Nb=Nb,
                                eps_b=eps_b, tol=1e-11, outer=60)
    res = S.residual(z)
    tag = f"{degs}/Nb{Nb}/eps_b={eps_b:g}"
    print(f"{tag}: converged={info.get('converged')} passes={info.get('passes')} "
          f"reason={info.get('reason','-')}", flush=True)
    print(f"    alpha={info.get('alpha', float('nan')):.8f}  c_l={info.get('cl', float('nan')):.8f}  "
          f"h_id={S.h_id(z):+.4e}  ||F||_rms={np.linalg.norm(res)/np.sqrt(res.size):.3e}  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    if info.get("converged"):
        np.savez(SCR / f"q6_root_{degs[0]}_{degs[1]}_{eps_b:g}.npz", z=z, a=info["alpha"])
