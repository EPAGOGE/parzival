"""T2: re-price the frozen axis column on the panel solver at eps_b=1e-5.

The pinned axis data is interpolated through a corruption band that at eps=1e-5 covers
the whole domain, while the true values shrink like eps (Ot) and eps^2 (Bt).  Perturb the
pinned data and measure d(alpha): x1 (as-is baseline, known -0.33883505), x0 (zeroed --
the parity limit, near-truth at this eps), x2 (doubled -- linearity check).
"""
import importlib.util, pathlib, sys, time
import numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
def mod(n,f):
    sp_=importlib.util.spec_from_file_location(n,str(H/f)); m=importlib.util.module_from_spec(sp_)
    sys.modules[n]=m; sp_.loader.exec_module(m); return m
pp = mod("pp","polar_panels.py")
REF=-0.34240009; BASE=-0.33883505
EDGES=(0.0,2.0,15.0,25.0); DEGS=(16,56,12); NB=36; EPS=1e-5

def run(scale):
    S = pp.PanelSolver(edges=EDGES,degs=DEGS,Nb=NB,eps_b=EPS)
    if scale != 1.0:
        S.Ot0[:,-1]*=scale; S.Bt0[:,-1]*=scale     # pinned axis data, post-construction
    a=None; z0=None
    for k in range(80):
        if a is not None: S.set_alpha(a)
        z,f,r,taken = S.newton(z0=z0)
        if taken==0: return None,dict(fail="zero_steps",F=float(r),passes=k)
        cl,cw=float(z[-2]),float(z[-1]); an=cw/cl; z0=z
        if a is not None and abs(an-a)<1e-9 and r<1e-11:
            return an,dict(cl=cl,F=float(r),passes=k+1)
        a = an if a is None else a+0.5*(an-a)
    return None,dict(fail="outer_cap")

S0 = pp.PanelSolver(edges=EDGES,degs=DEGS,Nb=NB,eps_b=EPS)
mo, mb = float(np.abs(S0.Ot0[:,-1]).max()), float(np.abs(S0.Bt0[:,-1]).max())
fo, fb = float(np.abs(S0.Ot0).max()), float(np.abs(S0.Bt0).max())
print(f"pinned axis data: max|Ot|={mo:.3e} ({100*mo/fo:.3f}% of field) "
      f"max|Bt|={mb:.3e} ({100*mb/fb:.4f}% of field)  [true ~ eps, eps^2 = 1e-5, 1e-10]",
      flush=True)
del S0
print(f"{'variant':>8} {'alpha':>13} {'d(alpha) vs base':>17} {'vs ref':>8} {'passes':>6} {'secs':>5}",
      flush=True)
print(f"{'x1':>8} {BASE:+13.8f} {'(known)':>17} {100*(BASE-REF)/abs(REF):+7.3f}%", flush=True)
for tag,sc in (("x0",0.0),("x2",2.0)):
    t0=time.time()
    a,info = run(sc)
    if a is None:
        print(f"{tag:>8}  FAILED {info}", flush=True); continue
    print(f"{tag:>8} {a:+13.8f} {a-BASE:+17.2e} {100*(a-REF)/abs(REF):+7.3f}% "
          f"{info['passes']:>6} {time.time()-t0:5.0f}", flush=True)
print("done", flush=True)
