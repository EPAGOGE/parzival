import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pn=M("pn","polar_newton.py")
print("CONVERGED NEWTON PROFILE vs RESOLUTION, and the outer alpha self-consistency loop.")
print("  reference: c_l=3.006498  c_w=-1.029425  alpha=-0.342400\n")
print("  %4s %13s %11s %11s %11s   %s"%("N","||F|| final","c_l","c_w","alpha","dev from seed"))
for N in (24,28,36,44):
    S=pn.NewtonSolver(N)
    x,f,r,r0=S.solve(steps=8,verbose=False)
    cl,cw=float(x[-2]),float(x[-1])
    Ot,Bt=S.unpack(x[:-2]); C=S.C; I=(slice(2,-2),slice(2,-2))
    dev=np.abs(Ot[I]-C.Ot0[I]).max()/np.abs(C.Ot0[I]).max()
    print("  %4d %13.4e %11.6f %11.6f %11.6f   %.4e"%(N,r,cl,cw,cw/cl,dev),flush=True)
print("\n  OUTER ALPHA LOOP at N=36 (alpha0 <- c_w/c_l until self-consistent):")
a=None
for it in range(6):
    S=pn.NewtonSolver(36,alpha=a)
    x,f,r,r0=S.solve(steps=8,verbose=False)
    cl,cw=float(x[-2]),float(x[-1]); an=cw/cl
    print("    it %d: alpha0=%s -> c_l=%.6f c_w=%.6f alpha_out=%.6f  ||F||=%.3e  gap=%.2e"
          %(it, ("seed" if a is None else "%.6f"%a), cl,cw,an,r,
            abs(an-(a if a is not None else S.C.a0))),flush=True)
    if a is not None and abs(an-a)<1e-6: print("    CONVERGED"); break
    a=an
