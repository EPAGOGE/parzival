"""Step 1: tighten the quote. Nb=48 check rung + eps=5e-6 depth rung + deg 28 at 5e-6,
then 3-point Richardson in eps at deg24/Nb36 using {3e-5, 1e-5, 5e-6}."""
import importlib.util, pathlib, sys, time
import numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
def mod(n,f):
    sp_=importlib.util.spec_from_file_location(n,str(H/f)); m=importlib.util.module_from_spec(sp_)
    sys.modules[n]=m; sp_.loader.exec_module(m); return m
pc = mod("pc","polar_cornerreg.py")
REF=-0.34240009
KNOWN={(24,36,1e-4):-0.34541032,(24,36,5e-5):-0.34386167,(24,36,3e-5):-0.34312079,
       (24,36,1e-5):-0.34268591,(28,36,1e-5):-0.34270532}
print(f"{'deg0':>5} {'Nb':>4} {'eps_b':>8} {'alpha':>13} {'vs ref':>8} {'d_cl':>11} {'||F||':>9} {'secs':>5}", flush=True)
res=dict(KNOWN)
for d0,nb,eps in ((24,48,1e-5),(24,36,5e-6),(28,36,5e-6)):
    t0=time.time()
    try:
        S,z,r,info = pc.converge(edges=(0.0,2.0,15.0,25.0),degs=(d0,56,12),Nb=nb,eps_b=eps)
    except Exception as ex:
        print(f"{d0:>5} {nb:>4} {eps:8.0e}  RAISED {type(ex).__name__}: {ex}", flush=True); continue
    if not info.get("converged"):
        print(f"{d0:>5} {nb:>4} {eps:8.0e}  NOT CONVERGED ||F||={r:.2e} {info.get('reason','')}", flush=True); continue
    a=info["alpha"]; CLS=2.0*S.THXX_REF/S.WX_REF
    res[(d0,nb,eps)]=a
    print(f"{d0:>5} {nb:>4} {eps:8.0e} {a:+13.8f} {100*(a-REF)/abs(REF):+7.3f}% "
          f"{info['cl']/CLS-1:+11.2e} {r:9.2e} {time.time()-t0:5.0f}", flush=True)
# Richardson at deg24/Nb36 on the smallest three eps
pts=[(e,res[(24,36,e)]) for e in (3e-5,1e-5,5e-6) if (24,36,e) in res]
if len(pts)==3:
    (e1,a1),(e2,a2),(e3,a3)=pts
    # fit a + b*eps + c*eps^2 through 3 points -> a at eps=0
    M=np.array([[1,e,e*e] for e,_ in pts]); v=np.array([a for _,a in pts])
    coef=np.linalg.solve(M,v)
    lin=np.polyfit([p[0] for p in pts],[p[1] for p in pts],1)[-1]
    print(f"\n Richardson (3-pt quad) eps->0: {coef[0]:+.8f} ({100*(coef[0]-REF)/abs(REF):+.4f}%)", flush=True)
    print(f" linear (3-pt):               {lin:+.8f} ({100*(lin-REF)/abs(REF):+.4f}%)   bar={abs(coef[0]-lin):.1e}", flush=True)
if (24,48,1e-5) in res and (24,36,1e-5) in res:
    print(f" Nb 36->48 shift at eps=1e-5: {res[(24,48,1e-5)]-res[(24,36,1e-5)]:+.2e}", flush=True)
print(f" reference alpha = {REF}", flush=True)
