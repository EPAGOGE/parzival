"""THE FINAL LADDER: corner-regularized solver (both corner fixes in), deep corner
degree, eps_b -> 0.  The adjudicator proved the eps_b wedge carries a xi^(k-2) singular
corner layer (k = pi/(pi/2-2eps_b)) that biases alpha and blocks deep-corner configs at
eps_b=1e-3 -- and that the layer VANISHES as eps_b -> 0.  So: deg0=24, eps ladder
1e-4 -> 1e-5 with warm alpha continuation, extrapolate; then deg0=28 at 1e-5 for the
corner-degree convergence check.  Gates: d_cl (already at 2.6e-5, 100x the campaign's
best) and alpha vs -0.34240009."""
import importlib.util, pathlib, sys, time
import numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
def mod(n,f):
    sp_=importlib.util.spec_from_file_location(n,str(H/f)); m=importlib.util.module_from_spec(sp_)
    sys.modules[n]=m; sp_.loader.exec_module(m); return m
pc = mod("pc","polar_cornerreg.py")
REF=-0.34240009
print(f"{'deg0':>5} {'eps_b':>8} {'alpha':>13} {'vs ref':>8} {'d_cl':>11} {'||F||':>9} {'passes':>6} {'secs':>5}", flush=True)
rows=[]
a_warm=None
for d0,eps in ((24,1e-4),(24,5e-5),(24,2.5e-5),(24,1e-5),(28,1e-5)):
    t0=time.time()
    try:
        S,z,r,info = pc.converge(edges=(0.0,2.0,15.0,25.0),degs=(d0,56,12),Nb=36,eps_b=eps)
    except Exception as ex:
        print(f"{d0:>5} {eps:8.0e}  RAISED {type(ex).__name__}: {ex}", flush=True); continue
    if not info.get("converged"):
        print(f"{d0:>5} {eps:8.0e}  NOT CONVERGED ||F||={r:.2e} {info.get('reason','')}", flush=True); continue
    a=info["alpha"]; CLS=2.0*S.THXX_REF/S.WX_REF; dcl=info["cl"]/CLS-1
    if d0==24: rows.append((eps,a))
    print(f"{d0:>5} {eps:8.0e} {a:+13.8f} {100*(a-REF)/abs(REF):+7.3f}% {dcl:+11.2e} "
          f"{r:9.2e} {info['passes']:>6} {time.time()-t0:5.0f}", flush=True)
if len(rows)>=3:
    es=np.array([p[0] for p in rows]); al=np.array([p[1] for p in rows])
    lin=float(np.polyfit(es,al,1)[-1]); quad=float(np.polyfit(es,al,2)[-1])
    print(f"\n deg0=24 eps_b->0: linear={lin:+.8f} ({100*(lin-REF)/abs(REF):+.3f}%)  "
          f"quadratic={quad:+.8f} ({100*(quad-REF)/abs(REF):+.3f}%)  bar={abs(lin-quad):.1e}", flush=True)
print(f" reference alpha = {REF}", flush=True)
