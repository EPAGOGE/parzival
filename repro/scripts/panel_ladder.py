"""PANEL REFINEMENT LADDER -- the run the whole build exists for.

Three panelizations already agree on alpha = -0.341099 to <7e-7, INCLUDING K=3 with the
outer edge at 25 -- i.e. the XMAX=15 vs 25 sensitivity that measured 5.27% at N=36 on the
single grid collapsed to 6e-8 once the mid-band was properly resolved.  Now: walk the
mid-panel degree up (panel-0 capped at <=20 per the corner-dust rule) and watch whether
alpha converges -- and where it lands relative to Chen-Hou's -0.34240009.
Gates per rung: converged flag, ||F||, c_l vs c_l*, dust-set size, secs.
"""
import importlib.util, json, pathlib, sys, time
import numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
def mod(n, f):
    sp_=importlib.util.spec_from_file_location(n,str(H/f)); m=importlib.util.module_from_spec(sp_)
    sys.modules[n]=m; sp_.loader.exec_module(m); return m
pp = mod("pp","polar_panels.py")
REF=-0.34240009
OUT=pathlib.Path("/Users/epagogellc/parzival/boussinesq/runs/panel_ladder.json")
CONFIGS=[((0.0,2.0,15.0,25.0),(16,40,12)),
         ((0.0,2.0,15.0,25.0),(16,56,12)),
         ((0.0,2.0,15.0,25.0),(16,72,14)),
         ((0.0,2.0,15.0,25.0),(16,88,16)),
         ((0.0,2.0,15.0,25.0),(18,104,18))]
res={}
print(f"{'degs':>14} {'n':>6} {'alpha':>13} {'vs ref':>8} {'d_cl':>8} {'||F||':>9} "
      f"{'passes':>6} {'dust':>4} {'secs':>5}", flush=True)
for edges,degs in CONFIGS:
    t0=time.time()
    try:
        S,z,r,info = pp.converge(edges=edges,degs=degs,Nb=36)
    except Exception as ex:
        print(f"{str(degs):>14}  RAISED {type(ex).__name__}: {ex}", flush=True); continue
    if not info.get("converged"):
        print(f"{str(degs):>14}  NOT CONVERGED ||F||={r:.2e} {info}", flush=True); continue
    a=info["alpha"]; cl=info["cl"]; CLS=2.0*S.THXX_REF/S.WX_REF
    n=3*S.Nx*S.Nb+2
    res[str(degs)]=dict(alpha=a,cl=cl,F=float(r),n=n)
    OUT.write_text(json.dumps(res,indent=1))
    print(f"{str(degs):>14} {n:>6} {a:+13.8f} {100*(a-REF)/abs(REF):+7.3f}% "
          f"{100*(cl/CLS-1):+7.2f}% {r:9.2e} {info['passes']:>6} "
          f"{len(S.rT_interp):>4} {time.time()-t0:5.0f}", flush=True)
print(f"\nreference alpha = {REF}", flush=True)
