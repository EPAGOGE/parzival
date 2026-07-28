import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pn=M("pn","polar_newton.py")
print("STABILITY SPECTRUM AT A GENUINE FIXED POINT")
print("Every earlier spectrum was taken about a seed with a 1.5% residual and is void.")
print("Here: Newton-converged profile (||F|| ~ 1e-11), alpha self-consistent to 8e-7.\n")
print("The gauged operator is the SCHUR COMPLEMENT of the Newton Jacobian:")
print("  J = [[A, B],[C, D]]  ->  A - B D^-1 C   (field dynamics with c_l,c_w slaved")
print("  to hold the two corner constraints, which IS Chen-Hou's closure).\n")
for N in (28,36,44):
    S=pn.NewtonSolver(N)
    x,f,r,r0=S.solve(steps=8,verbose=False)
    J=S.jac(x)
    n=J.shape[0]-2
    A=J[:n,:n]; B=J[:n,n:]; Cc=J[n:,:n]; D=J[n:,n:]
    try:    Sc = A - B @ np.linalg.solve(D, Cc)
    except np.linalg.LinAlgError: Sc = A
    v=np.linalg.eigvals(Sc); v=v[np.argsort(-v.real)]
    low=v[np.abs(v.imag)<5.0]
    nun=int((v.real>1e-6).sum()); nlow=int((low.real>1e-6).sum())
    cl,cw=float(x[-2]),float(x[-1])
    print("  N=%2d  ||F||=%.2e  c_l=%.5f alpha=%.6f  |  unstable: %d total, %d with |Im|<5"
          %(N,r,cl,cw/cl,nun,nlow))
    print("        leading overall : %+10.5f %+10.5fi"%(v[0].real,v[0].imag))
    for z in low[:3]:
        print("        low-|Im|        : %+10.5f %+10.5fi   %s"
              %(z.real,z.imag,"UNSTABLE" if z.real>1e-6 else "stable"))
    print(flush=True)
