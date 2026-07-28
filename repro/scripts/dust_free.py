"""Dust-rule bias test: panel-0 deg 12 puts x1=0.0341 > CORNER_DUST, so NO de-collocation
rows exist anywhere -- same solver, rule inert by geometry. Compare against the dust-rule
result at deg 16 (alpha=-0.33883505, Nb=36, eps=1e-5). Agreement -> rule exonerated;
~1% difference -> the shared rule owned the gap all along."""
import importlib.util, pathlib, sys, time
import numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
def mod(n,f):
    sp_=importlib.util.spec_from_file_location(n,str(H/f)); m=importlib.util.module_from_spec(sp_)
    sys.modules[n]=m; sp_.loader.exec_module(m); return m
pp = mod("pp","polar_panels.py")
REF=-0.34240009; BASE=-0.33883505
for degs in ((12,56,12),(14,56,12)):
    t0=time.time()
    S,z,r,info = pp.converge(edges=(0.0,2.0,15.0,25.0),degs=degs,Nb=36,eps_b=1e-5)
    ndust = len(S.rT_interp)
    if not info.get("converged"):
        print(f"degs={degs}: NOT CONVERGED ||F||={r:.2e} dust={ndust} {info.get('reason','')}", flush=True)
        continue
    a=info["alpha"]; CLS=2.0*S.THXX_REF/S.WX_REF
    print(f"degs={degs}: alpha={a:+.8f} d(alpha) vs dust-rule={a-BASE:+.2e} "
          f"vs ref={100*(a-REF)/abs(REF):+.3f}% d_cl={100*(info['cl']/CLS-1):+.2f}% "
          f"dust={ndust} ||F||={r:.2e} secs={time.time()-t0:.0f}", flush=True)
print("done", flush=True)
