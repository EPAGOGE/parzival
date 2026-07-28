import importlib.util, sys, numpy as np, numpy.linalg as la
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pn=M("pn","polar_newton.py")

def run(N, alpha=None):
    S=pn.NewtonSolver(N,alpha=alpha)
    x,f,r,r0=S.solve(steps=8,verbose=False)
    J=S.jac(x); n=J.shape[0]-2
    A=J[:n,:n]; B=J[:n,n:]; Cc=J[n:,:n]; D=J[n:,n:]
    Sc=A-B@la.solve(D,Cc)
    w,V=la.eig(Sc); o=np.argsort(-w.real); w,V=w[o],V[:,o]
    return S,x,r,Sc,w,V,n

print("WHAT IS THE +1.05 MODE?  Projection of its eigenvector onto the symmetry span")
print("span{v_amp, v_trans}.  Time-shift ~ Omega + gamma Y.grad Omega lies in that span.\n")
print("  %3s %11s %10s %10s | %10s %9s | %s"
      %("N","||F||","c_l","alpha","lead eig","in span","2nd / 3rd low-|Im|"))
for N in (28,36,44):
    S,x,r,Sc,w,V,n=run(N)
    C=S.C; cl,cw=float(x[-2]),float(x[-1])
    Ot,Bt=S.unpack(x[:-2])
    # symmetry tangents in the SAME packed coordinates the operator acts on
    G=C.G
    vA=np.concatenate([Ot.ravel(),(2*Bt).ravel()])[S.idx]
    vT=np.concatenate([(G*(C.dx(Ot)+C.a0*Ot)).ravel(),
                       (G*(C.dx(Bt)+(1+2*C.a0)*Bt)-Bt).ravel()])[S.idx]
    Q,_=la.qr(np.stack([vA,vT],axis=1))          # orthonormal basis of the span
    v=V[:,0].real if abs(V[:,0].imag).max()<1e-9 else np.abs(V[:,0])
    v=v/la.norm(v)
    frac=la.norm(Q.T@v)                           # fraction of the eigenvector in the span
    low=w[np.abs(w.imag)<5.0]
    print("  %3d %11.2e %10.5f %10.6f | %+10.5f %8.1f%% | %s"
          %(N,r,cl,cw/cl,w[0].real,100*frac,
            " ".join("%+.4f"%z.real for z in low[1:4])),flush=True)

print("\nRESOLUTION: does anything converge?  (with the outer alpha loop run to closure)")
print("  %3s %11s %11s %11s | %11s %11s"%("N","c_l","c_w","alpha","lead eig","2nd eig"))
for N in (28,36,44,52):
    a=None
    for _ in range(3):
        S,x,r,Sc,w,V,n=run(N,alpha=a)
        cl,cw=float(x[-2]),float(x[-1]); an=cw/cl
        if a is not None and abs(an-a)<1e-6: break
        a=an
    low=w[np.abs(w.imag)<5.0]
    print("  %3d %11.6f %11.6f %11.6f | %+11.5f %+11.5f"
          %(N,cl,cw,cw/cl,w[0].real,low[1].real if low.size>1 else np.nan),flush=True)
