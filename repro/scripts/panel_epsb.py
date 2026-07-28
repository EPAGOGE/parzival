"""eps_b -> 0 continuation on the PANEL solver, config (0,2,15,25)/(16,56,12)/Nb=36.

Mid-band resolution is converged to 8 digits (alpha = -0.34109904 across degs 40..88), so
the remaining +0.380% vs Chen-Hou has one measured suspect: the beta-domain offset, worth
0.4-1% at these scales on the single grid.  Warm continuation down the ladder, frozen-seed
inner loop (PanelSolver built once per rung, set_alpha only), alpha and state carried
between rungs.  Report alpha, d_cl, and the linear/quadratic eps_b->0 extrapolations.
"""
import importlib.util, json, pathlib, sys, time
import numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
def mod(n, f):
    sp_=importlib.util.spec_from_file_location(n,str(H/f)); m=importlib.util.module_from_spec(sp_)
    sys.modules[n]=m; sp_.loader.exec_module(m); return m
pp = mod("pp","polar_panels.py")
REF=-0.34240009
EDGES=(0.0,2.0,15.0,25.0); DEGS=(16,56,12); NB=36
LADDER=[1e-3,6e-4,3e-4,1e-4,3e-5,1e-5]
OUT=pathlib.Path(H/"runs"/"panel_epsb.json")

def rung(eps_b, a0=None, z0=None):
    S = pp.PanelSolver(edges=EDGES,degs=DEGS,Nb=NB,eps_b=eps_b,alpha=a0)
    a=a0; hist=[]
    for k in range(80):
        if a is not None: S.set_alpha(a)
        z,f,r,taken = S.newton(z0=z0)
        if taken==0: return None,None,dict(fail="zero_steps",F=float(r),passes=k)
        cl,cw=float(z[-2]),float(z[-1]); an=cw/cl; z0=z; hist.append(an)
        if a is not None and abs(an-a)<1e-9 and r<1e-11:
            return z,S,dict(alpha=an,cl=cl,F=float(r),passes=k+1)
        a = an if a is None else a+0.5*(an-a)
    return None,None,dict(fail="outer_cap",hist=hist[-4:])

res={}; z0=None; a0=None
print(f"{'eps_b':>8} {'alpha':>13} {'vs ref':>8} {'d_cl':>8} {'||F||':>9} {'passes':>6} {'secs':>5}",
      flush=True)
for e in LADDER:
    t0=time.time()
    z,S,info = rung(e, a0=a0, z0=None if z0 is None else z0.copy())
    if z is None:
        print(f"{e:8.0e}  FAILED {info}", flush=True); break
    a=info["alpha"]; CLS=2.0*S.THXX_REF/S.WX_REF
    res[f"{e:.0e}"]=dict(alpha=a,cl=info["cl"],F=info["F"])
    OUT.write_text(json.dumps(res,indent=1))
    print(f"{e:8.0e} {a:+13.8f} {100*(a-REF)/abs(REF):+7.3f}% "
          f"{100*(info['cl']/CLS-1):+7.2f}% {info['F']:9.2e} {info['passes']:>6} "
          f"{time.time()-t0:5.0f}", flush=True)
    z0, a0 = z, a

es=np.array([float(k) for k in res]); al=np.array([v["alpha"] for v in res.values()])
if es.size>=4:
    lin=float(np.polyfit(es[-4:],al[-4:],1)[-1])
    quad=float(np.polyfit(es[-min(5,es.size):],al[-min(5,es.size):],2)[-1])
    print(f"\n eps_b->0: linear={lin:+.8f} ({100*(lin-REF)/abs(REF):+.3f}%)  "
          f"quadratic={quad:+.8f} ({100*(quad-REF)/abs(REF):+.3f}%)  "
          f"[bar {abs(lin-quad):.1e}]", flush=True)
print(f" reference alpha = {REF}", flush=True)
