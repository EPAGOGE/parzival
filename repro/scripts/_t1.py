import importlib.util, pathlib, sys, time, numpy as np
sys.path.insert(0,'.')
import q345
t0=time.time()
d = q345.build_quotient("A")
L = d["Lred"]
q345.log(f"Lred {L.shape}  [{time.time()-t0:.1f}s total]")
t=time.time(); s = np.linalg.svd(L, compute_uv=False); q345.log(f"svd {time.time()-t:.1f}s")
q345.log(f"||L||_2 = {s[0]:.6e}   (promoted invariant: 1.112468e+03)")
q345.log(f"sigma_min(L|kerCg) = {s[-1]:.6e}   (spectrum.py G1: 3.970084e-03)")
q345.log(f"cond = {s[0]/s[-1]:.4e}")
np.save("q345_Lred_A.npy", L)
