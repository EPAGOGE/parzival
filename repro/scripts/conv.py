import importlib.util, sys, time, numpy as np, numpy.linalg as la
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pn=M("pn","polar_newton.py")

def solve_at(N):
    a=None
    for _ in range(4):
        S=pn.NewtonSolver(N,alpha=a)
        x,f,r,_=S.solve(steps=8,verbose=False)
        cl,cw=float(x[-2]),float(x[-1]); an=cw/cl
        if a is not None and abs(an-a)<1e-6: break
        a=an
    J=S.jac(x); n=J.shape[0]-2
    A=J[:n,:n]; B=J[:n,n:]; Cg=J[n:,:n]
    P=np.eye(n)-B@la.solve(Cg@B,Cg)
    w=la.eigvals(P@A)
    return S,x,r,w,n,cl,cw

RES={}
for N in (36,44,56):
    t0=time.time()
    S,x,r,w,n,cl,cw=solve_at(N)
    RES[N]=(w,cl,cw,r,n)
    lo=w[(np.abs(w.imag)<3.0)&(np.abs(w.real)>1e-7)]
    lo=lo[np.argsort(-lo.real)]
    print("N=%2d dim=%4d ||F||=%.2e c_l=%.6f alpha=%.6f  (%.1f min)"
          %(N,n,r,cl,cw/cl,(time.time()-t0)/60),flush=True)
    print("   top low-|Im| (|Im|<3, nonzero):",
          " ".join("%+.4f%+.4fi"%(z.real,z.imag) for z in lo[:6]),flush=True)
    print("   grid-scale check: max|Im| overall = %.1f, count Re>1e-6 = %d"
          %(np.abs(w.imag).max(),int((w.real>1e-6).sum())),flush=True)

print("\nWHICH EIGENVALUES RECUR ACROSS N?  (a physical mode sits still; a spurious one moves)")
Ns=sorted(RES)
base=RES[Ns[-1]][0]
lob=base[(np.abs(base.imag)<3.0)&(np.abs(base.real)>1e-7)]
lob=lob[np.argsort(-lob.real)][:10]
print("  %-22s %s"%("N=%d candidate"%Ns[-1]," ".join("closest at N=%d"%m for m in Ns[:-1])))
for z in lob:
    row=[]
    for m in Ns[:-1]:
        wm=RES[m][0]; wm=wm[(np.abs(wm.imag)<3.0)&(np.abs(wm.real)>1e-7)]
        j=int(np.argmin(np.abs(wm-z))); d=abs(wm[j]-z)
        row.append("%+.4f%+.4fi (d=%.3f)"%(wm[j].real,wm[j].imag,d))
    print("  %+.4f%+.4fi        %s"%(z.real,z.imag,"   ".join(row)))
