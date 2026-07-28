import importlib.util, sys, numpy as np, numpy.linalg as la
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pst=M("pst","polar_stability.py")
print("SPECTRUM RESTRICTED TO CHEN-HOU'S ADMISSIBLE SPACE")
print("  eq:normal_vanish: omega = O(|x|^2), theta_x/theta_y = O(|x|^2), i.e. for the")
print("  PERTURBATION (one order faster than the profile's own vanishing):")
print("      dOt_xi(0, b)   = 0   for ALL b      (dOm ~ r^2)")
print("      dBt_xixi(0, b) = 0   for ALL b      (dB  ~ r^3)")
print("  That is 2*nb linear conditions -- far stronger than the two POINT conditions")
print("  the gauge imposes, and it is the space their stability result lives on.\n")
for N in (28,36,44):
    St,x,r,cl,cw = pst.converge(N)
    w,V,cCB,A,B,Cg = St.spectrum(x)
    C=St.C; n2=C.nx*C.nb; idx=St.S.idx; n=St.n
    # build the admissibility constraint matrix R (2*nb x n)
    rows=[]
    for j in range(C.nb):
        rO=np.zeros(2*n2); rB=np.zeros(2*n2)
        for k in range(C.nx):
            rO[k*C.nb+j]      = C.Dx[0,k]
            rB[n2+k*C.nb+j]   = C.Dx2[0,k]
        rows.append(rO[idx]); rows.append(rB[idx])
    R=np.stack(rows,axis=0)
    # orthonormal basis of ker(R)
    U,s,Vt=la.svd(R,full_matrices=True)
    tol=max(R.shape)*np.finfo(float).eps*(s[0] if s.size else 1.0)
    rank=int((s>tol).sum())
    Z=Vt[rank:].T                                  # n x (n-rank)
    # gauge-projected operator, then Galerkin-restrict to the admissible space
    CB=Cg@B; P=np.eye(n)-B@la.solve(CB,Cg); L=P@A
    Lr=Z.T@L@Z
    inv=la.norm((np.eye(n)-Z@Z.T)@L@Z)/max(la.norm(L@Z),1e-300)
    wr=la.eigvals(Lr); wr=wr[np.argsort(-wr.real)]
    lo=wr[(np.abs(wr.imag)<3.0)&(np.abs(wr.real)>1e-7)]
    nun=int((wr.real>1e-6).sum())
    print("  N=%2d  dim %d -> %d (removed %d)  space-invariance residual %.3f"
          %(N,n,Z.shape[1],rank,inv))
    print("        UNRESTRICTED lead low-|Im|: %s"
          %" ".join("%+.4f%+.4fi"%(z.real,z.imag)
                    for z in w[(np.abs(w.imag)<3.0)&(np.abs(w.real)>1e-7)][:2]))
    print("        RESTRICTED   lead low-|Im|: %s"
          %" ".join("%+.4f%+.4fi"%(z.real,z.imag) for z in lo[:4]))
    print("        restricted unstable count (Re>1e-6): %d of %d\n"%(nun,wr.size),flush=True)
