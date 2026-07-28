"""Dust-free corner-panel sweep: deg 8..14 on [0,2], everything else frozen
((.,56,12), Nb=36, eps_b=1e-5). x1 = 1-cos(pi/deg) stays above CORNER_DUST=0.025
through deg 14, so ZERO de-collocation rows -- the rule is geometrically inert.
Question: do alpha and d_cl walk monotonically toward (-0.34240009, 0) as the
corner panel refines toward the degeneracy wall?"""
import importlib.util, pathlib, sys, time
import numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
def mod(n,f):
    sp_=importlib.util.spec_from_file_location(n,str(H/f)); m=importlib.util.module_from_spec(sp_)
    sys.modules[n]=m; sp_.loader.exec_module(m); return m
pp = mod("pp","polar_panels.py")
REF=-0.34240009
print(f"{'deg0':>5} {'x1':>7} {'alpha':>13} {'vs ref':>8} {'d_cl':>8} {'||F||':>9} {'secs':>5}", flush=True)
for d0 in (8,10,12,13,14):
    x1=1-np.cos(np.pi/d0)
    t0=time.time()
    try:
        S,z,r,info = pp.converge(edges=(0.0,2.0,15.0,25.0),degs=(d0,56,12),Nb=36,eps_b=1e-5)
    except Exception as ex:
        print(f"{d0:>5} {x1:7.4f}  RAISED {type(ex).__name__}", flush=True); continue
    if not info.get("converged"):
        print(f"{d0:>5} {x1:7.4f}  NOT CONVERGED ||F||={r:.2e} {info.get('reason','')}", flush=True); continue
    a=info["alpha"]; CLS=2.0*S.THXX_REF/S.WX_REF
    print(f"{d0:>5} {x1:7.4f} {a:+13.8f} {100*(a-REF)/abs(REF):+7.3f}% "
          f"{100*(info['cl']/CLS-1):+7.2f}% {r:9.2e} {time.time()-t0:5.0f}", flush=True)
print(f"ref = {REF}", flush=True)
