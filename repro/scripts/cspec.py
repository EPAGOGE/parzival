import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pc=M("pc","polar_corner.py")

def spec(N):
    C=pc.Corner(N,N,25.0)
    nx,nb=C.nx,C.nb; n2=nx*nb
    m=np.ones((nx,nb),bool); m[0,:]=False; m[:,-1]=False
    idx=np.where(np.concatenate([m.ravel(),m.ravel()]))[0]
    n=idx.size
    O0,B0=C.Ot0.copy(),C.Bt0.copy()
    scale=max(np.abs(O0).max(),np.abs(B0).max()); eps=1e-6*scale
    J=np.empty((n,n)); e=np.zeros(n); full=np.zeros(2*n2)
    for j in range(n):
        e[j]=1.0; full[:]=0.0; full[idx]=e
        dO=full[:n2].reshape(nx,nb); dB=full[n2:].reshape(nx,nb)
        rp=C.rhs(O0+eps*dO,B0+eps*dB); rm=C.rhs(O0-eps*dO,B0-eps*dB)
        col=np.concatenate([((rp[0]-rm[0])/(2*eps)).ravel(),((rp[1]-rm[1])/(2*eps)).ravel()])
        J[:,j]=col[idx]; e[j]=0.0
    v=np.linalg.eig(J)[0]; v=v[np.argsort(-v.real)]
    return C,v,n

print("SPECTRUM IN THE CORNER-INCLUSIVE FRAME (r=0 INCLUDED)")
print("log-polar (corner EXCLUDED) had a converged unstable pair +0.20 +- 0.64i,")
print("period ~9.8, plus grid-scale modes with |Im| ~ N.\n")
for N in (24,28,32):
    C,v,n=spec(N)
    low=v[(np.abs(v.imag)<5.0)]
    nun=int((v.real>1e-6).sum())
    nlow=int((low.real>1e-6).sum())
    print("  N=%2d (dim %4d): unstable total %3d | unstable with |Im|<5 : %3d" % (N,n,nun,nlow))
    print("      leading overall  : %+9.5f %+9.5fi" % (v[0].real,v[0].imag))
    if low.size:
        lo=low[np.argsort(-low.real)][:4]
        for z in lo:
            print("      low-|Im| leading : %+9.5f %+9.5fi   period %7.2f  %s"
                  %(z.real,z.imag,2*np.pi/max(abs(z.imag),1e-9),
                    "UNSTABLE" if z.real>1e-6 else "stable"))
    print(flush=True)
