"""T1 from the jump engine: the Nb ladder -- the only never-varied panel axis.
Config (0,2,15,25)/(16,56,12), eps_b=1e-5 (healed domain). Nb = 48, 60 vs known 36."""
import importlib.util, pathlib, sys, time
import numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
def mod(n,f):
    sp_=importlib.util.spec_from_file_location(n,str(H/f)); m=importlib.util.module_from_spec(sp_)
    sys.modules[n]=m; sp_.loader.exec_module(m); return m
pp = mod("pp","polar_panels.py")
REF=-0.34240009
print(f"{'Nb':>4} {'alpha':>13} {'vs ref':>8} {'d_cl':>8} {'||F||':>9} {'passes':>6} {'secs':>5}", flush=True)
print(f"{36:>4} {-0.33883505:+13.8f} {100*(-0.33883505-REF)/abs(REF):+7.3f}% {-4.61:+7.2f}%   (known)", flush=True)
for NB in (48, 60):
    t0=time.time()
    try:
        S,z,r,info = pp.converge(edges=(0.0,2.0,15.0,25.0),degs=(16,56,12),Nb=NB,eps_b=1e-5)
    except Exception as ex:
        print(f"{NB:>4}  RAISED {type(ex).__name__}: {ex}", flush=True); continue
    if not info.get("converged"):
        print(f"{NB:>4}  NOT CONVERGED ||F||={r:.2e} {info.get('reason','')}", flush=True); continue
    a=info["alpha"]; CLS=2.0*S.THXX_REF/S.WX_REF
    print(f"{NB:>4} {a:+13.8f} {100*(a-REF)/abs(REF):+7.3f}% {100*(info['cl']/CLS-1):+7.2f}% "
          f"{r:9.2e} {info['passes']:>6} {time.time()-t0:5.0f}", flush=True)
print(f"ref = {REF}", flush=True)
