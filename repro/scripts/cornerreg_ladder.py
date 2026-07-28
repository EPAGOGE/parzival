"""THE ENDGAME LADDER: corner-regularized solver, corner degree 12 -> 28, straight
through the old x1<0.025 wall (deg >= 15 was impossible; deg 16 was the sick geometry).
Config (d0,56,12)/Nb=36/eps_b=1e-5.  PRE-REGISTERED GATE: d_cl -> 0 AND alpha ->
-0.34240009 within the eps_b/Nb bars (~2e-4)."""
import importlib.util, pathlib, sys, time
import numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
def mod(n,f):
    sp_=importlib.util.spec_from_file_location(n,str(H/f)); m=importlib.util.module_from_spec(sp_)
    sys.modules[n]=m; sp_.loader.exec_module(m); return m
pc = mod("pc","polar_cornerreg.py")
REF=-0.34240009
print(f"{'deg0':>5} {'alpha':>13} {'vs ref':>8} {'d_cl':>8} {'||F||':>9} {'passes':>6} {'secs':>5}", flush=True)
for d0 in (12,14,16,20,24,28):
    t0=time.time()
    try:
        S,z,r,info = pc.converge(edges=(0.0,2.0,15.0,25.0),degs=(d0,56,12),Nb=36,eps_b=1e-5)
    except Exception as ex:
        print(f"{d0:>5}  RAISED {type(ex).__name__}: {ex}", flush=True); continue
    if not info.get("converged"):
        print(f"{d0:>5}  NOT CONVERGED ||F||={r:.2e} {info.get('reason','')}", flush=True); continue
    a=info["alpha"]; CLS=2.0*S.THXX_REF/S.WX_REF
    print(f"{d0:>5} {a:+13.8f} {100*(a-REF)/abs(REF):+7.3f}% "
          f"{100*(info['cl']/CLS-1):+7.2f}% {r:9.2e} {info['passes']:>6} {time.time()-t0:5.0f}", flush=True)
print(f"ref = {REF}  (gate: d_cl -> 0 and alpha within ~2e-4 of ref)", flush=True)
