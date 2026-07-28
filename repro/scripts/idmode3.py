import importlib.util, sys, numpy as np, numpy.linalg as la
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pn=M("pn","polar_newton.py")
rng=np.random.default_rng(0)
print("IS THE UNCONSTRAINED +1.05 MODE A SYMMETRY DIRECTION?")
print("(complex-safe projection this time; previous run took np.abs() of the eigenvector,")
print(" which destroys direction and reported a meaningless 100%.)\n")
print("  %3s %10s | %-11s %8s %8s %8s | %8s"
      %("N","alpha","lead(A)","in vA","in vT","in span","random"))
for N in (28,36,44):
    S=pn.NewtonSolver(N)
    x,f,r,_=S.solve(steps=8,verbose=False)
    J=S.jac(x); n=J.shape[0]-2
    A=J[:n,:n]
    w,V=la.eig(A); o=np.argsort(-w.real); w,V=w[o],V[:,o]
    C=S.C; Ot,Bt=S.unpack(x[:-2]); G=C.G
    vA=np.concatenate([Ot.ravel(),(2*Bt).ravel()])[S.idx]
    vT=np.concatenate([(G*(C.dx(Ot)+C.a0*Ot)).ravel(),
                       (G*(C.dx(Bt)+(1+2*C.a0)*Bt)-Bt).ravel()])[S.idx]
    uA=vA/la.norm(vA); uT=vT/la.norm(vT)
    Q,_=la.qr(np.stack([vA,vT],axis=1))
    v=V[:,0]; v=v/la.norm(v)
    ov=lambda u: float(abs(np.vdot(u,v)))
    rv=rng.standard_normal(n); rv/=la.norm(rv)
    cl,cw=float(x[-2]),float(x[-1])
    print("  %3d %10.6f | %+10.5f %8.3f %8.3f %8.3f | %8.3f"
          %(N,cw/cl,w[0].real,ov(uA),ov(uT),float(la.norm(Q.conj().T@v)),
            float(la.norm(Q.conj().T@rv))),flush=True)
print("\n  (angle between vA and vT is small, so 'in span' can exceed either overlap.)")
print("  random-vector control shows what chance alignment looks like.")
