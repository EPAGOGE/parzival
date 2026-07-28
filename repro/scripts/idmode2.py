import importlib.util, sys, numpy as np, numpy.linalg as la
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pn=M("pn","polar_newton.py")

def run(N, alpha=None):
    S=pn.NewtonSolver(N,alpha=alpha)
    x,f,r,_=S.solve(steps=8,verbose=False)
    J=S.jac(x); n=J.shape[0]-2
    A=J[:n,:n]; B=J[:n,n:]; Cg=J[n:,:n]; D=J[n:,n:]
    # d/dt g = Cg (A x + B c) = 0  ->  c = -(Cg B)^-1 Cg A x
    CB=Cg@B
    P=np.eye(n)-B@la.solve(CB,Cg)           # projection enforcing the constraints
    L=P@A
    w,V=la.eig(L); o=np.argsort(-w.real); w,V=w[o],V[:,o]
    return S,x,r,A,L,w,V,n,float(la.norm(D)),float(la.cond(CB))

print("CORRECTED CONSTRAINED OPERATOR.  ||D|| should be ~0 (the constraints do not")
print("depend on c), which is why the earlier Schur complement silently fell back to the")
print("UNCONSTRAINED operator A -- so the previous +1.05 was computed WITHOUT the gauge.\n")
print("  %3s %10s %9s %10s | %-24s | %-24s"
      %("N","||D||","cond(CgB)","alpha","UNCONSTRAINED A: lead","CONSTRAINED L: lead"))
for N in (28,36,44):
    S,x,r,A,L,w,V,n,nD,cCB=run(N)
    wa=la.eigvals(A); wa=wa[np.argsort(-wa.real)]
    la_lo=wa[np.abs(wa.imag)<5.0]; l_lo=w[np.abs(w.imag)<5.0]
    cl,cw=float(x[-2]),float(x[-1])
    print("  %3d %10.2e %9.2e %10.6f | %+9.5f %+9.5fi     | %+9.5f %+9.5fi"
          %(N,nD,cCB,cw/cl,la_lo[0].real,la_lo[0].imag,l_lo[0].real,l_lo[0].imag),flush=True)
    print("      %-24s   next: %s"%("", " ".join("%+.4f"%z.real for z in l_lo[1:4])))

print("\nIs the leading CONSTRAINED mode in span{v_amp, v_trans}?")
for N in (28,36,44):
    S,x,r,A,L,w,V,n,_,_=run(N)
    C=S.C; Ot,Bt=S.unpack(x[:-2]); G=C.G
    vA=np.concatenate([Ot.ravel(),(2*Bt).ravel()])[S.idx]
    vT=np.concatenate([(G*(C.dx(Ot)+C.a0*Ot)).ravel(),
                       (G*(C.dx(Bt)+(1+2*C.a0)*Bt)-Bt).ravel()])[S.idx]
    Q,_=la.qr(np.stack([vA,vT],axis=1))
    lo=np.where(np.abs(w.imag)<5.0)[0]
    v=V[:,lo[0]]; v=np.abs(v); v/=la.norm(v)
    print("  N=%2d  lead constrained %+9.5f   in span = %5.1f%%"
          %(N,w[lo[0]].real,100*la.norm(Q.T@v)),flush=True)
