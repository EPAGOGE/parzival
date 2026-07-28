import importlib.util, sys, numpy as np, numpy.linalg as la
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pst=M("pst","polar_stability.py")
print("EIGENVALUE CONDITION NUMBERS -- the cheap pseudospectral magnifying glass.")
print("  kappa(lam) = 1/|y^H x|  (left/right eigenvectors, unit norm).")
print("  A perturbation of size eps moves lam by up to kappa*eps.")
print("  Different N IS a perturbation of the operator -- so if kappa is huge, the")
print("  N-scatter in the leading eigenvalues is PSEUDOSPECTRAL NOISE, not physics.\n")
for N in (28,36,44):
    St,x,r,cl,cw = pst.converge_exact(N)
    Ot,Bt=St.S.unpack(x[:-2])
    A=St.A_exact(x); B=St.exact_B(Ot,Bt); Cg=St.exact_Cg(); n=St.n
    L=(np.eye(n)-B@la.solve(Cg@B,Cg))@A
    w,V=la.eig(L)                 # right eigenvectors
    wl,W=la.eig(L.T)              # left  eigenvectors (of L^T)
    o=np.argsort(-w.real); w,V=w[o],V[:,o]
    # match left to right by eigenvalue
    kap=[]
    for i in range(w.size):
        j=int(np.argmin(np.abs(wl-w[i])))
        xr=V[:,i]/la.norm(V[:,i]); yl=W[:,j]/la.norm(W[:,j])
        d=abs(np.vdot(yl,xr))
        kap.append(1.0/max(d,1e-300))
    kap=np.array(kap)
    lo=[i for i in range(w.size) if abs(w[i].imag)<3.0 and abs(w[i].real)>1e-7][:4]
    print("  N=%2d  ||F||=%.1e   normality dep = %.4f"
          %(N,r,la.norm(L@L.T-L.T@L)/max(la.norm(L)**2,1e-300)))
    for i in lo:
        eps_move = kap[i]*1e-10
        print("     lam = %+9.5f%+9.5fi   kappa = %10.3e   a 1e-10 perturbation moves it by %.2e"
              %(w[i].real,w[i].imag,kap[i],eps_move))
    print("     median kappa over all modes = %.3e   max = %.3e\n"
          %(np.median(kap),kap.max()),flush=True)
