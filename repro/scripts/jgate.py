import importlib.util, sys, time, numpy as np, numpy.linalg as la
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pst=M("pst","polar_stability.py")
print("GATE: analytic Jacobian vs finite differences, at LOW N where FD is still good.\n")
for N in (20,28):
    St,x,r,cl,cw=pst.converge(N)
    t0=time.time(); Ae=St.A_exact(x); te=time.time()-t0
    t0=time.time(); Af=St.A_fd(x);    tf=time.time()-t0
    rel=la.norm(Ae-Af)/max(la.norm(Af),1e-300)
    mx=np.abs(Ae-Af).max()/max(np.abs(Af).max(),1e-300)
    print("  N=%2d  dim %4d  rel||A_exact-A_fd|| = %.3e   max rel = %.3e   %s"
          %(N,St.n,rel,mx,"PASS" if rel<1e-5 else "FAIL"))
    print("        timing: exact %.1fs   FD %.1fs   speedup %.2fx"%(te,tf,tf/max(te,1e-9)),flush=True)
    we=la.eigvals(Ae); wf=la.eigvals(Af)
    we=we[np.argsort(-we.real)]; wf=wf[np.argsort(-wf.real)]
    print("        lead eig: exact %+.6f%+.6fi   FD %+.6f%+.6fi"
          %(we[0].real,we[0].imag,wf[0].real,wf[0].imag))
