import importlib.util, sys, time, numpy as np, numpy.linalg as la
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pst=M("pst","polar_stability.py")
print("BRACKET: where exactly does it break?  N=44 converges (6.9e-13), N=56 does not (1.7e-2).")
print("Also reporting cond(J) of the exact Newton Jacobian at the FIRST step, which is the")
print("thing that would actually stop a Newton step from descending.\n")
print("  %4s %6s %12s %12s %12s %10s"%("N","dim","||F|| init","||F|| final","cond(J0)","verdict"))
for N in (44,48,52,56):
    t0=time.time()
    St=pst.Stability(N)
    x0=St.S.x0
    f0,_,_=St.S.F(x0)
    A=St.A_exact(x0); Ot,Bt=St.S.unpack(x0[:-2])
    B=St.exact_B(Ot,Bt); Cg=St.exact_Cg(); n=St.n
    J=np.zeros((n+2,n+2)); J[:n,:n]=A; J[:n,n:]=B; J[n:,:n]=Cg
    c=la.cond(J)
    St2,x,r,cl,cw=pst.converge_exact(N)
    print("  %4d %6d %12.3e %12.3e %12.3e %10s   (%.1fm)"
          %(N,n,la.norm(f0)/np.sqrt(f0.size),r,c,
            "OK" if r<1e-9 else "FAIL",(time.time()-t0)/60),flush=True)
