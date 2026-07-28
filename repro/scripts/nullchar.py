import importlib.util, sys, numpy as np, numpy.linalg as la
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pst=M("pst","polar_stability.py")
St,x,r,cl,cw=pst.converge_exact(36)
C=St.C; Ot,Bt=St.S.unpack(x[:-2])
A=St.A_exact(x); B=St.exact_B(Ot,Bt); Cg=St.exact_Cg(); n=St.n
L=(np.eye(n)-B@la.solve(Cg@B,Cg))@A
U,sv,Vt=la.svd(L)
n2=C.nx*C.nb; idx=St.S.idx
print("WHAT ARE THE NEAR-NULL DIRECTIONS THE DRIFT LIVES IN?  N=36, ||F||=%.1e\n"%r)
print("  singular values 1, 10, 100, and the smallest 8 (last two are the projection's):")
print("   %s ... %s"%(np.array2string(sv[[0,9,99]],precision=4),
                      np.array2string(sv[-8:],precision=3)))
def field(v):
    full=np.zeros(2*n2); full[idx]=v
    return full[:n2].reshape(C.nx,C.nb), full[n2:].reshape(C.nx,C.nb)
print("\n  WHERE each near-null right-singular vector lives (energy fraction by xi band):")
bands=[(0,0.5),(0.5,2),(2,5),(5,10),(10,15),(15,20),(20,25)]
print("  %5s %10s | %s"%("k","sigma"," ".join("%9s"%("xi %g-%g"%b) for b in bands)))
for k in range(3, 3+12):                      # skip the 2 null + 1 buffer
    v=Vt[-(k+1)]
    dO,dB=field(v)
    E=dO**2+dB**2; tot=E.sum()
    row=[]
    for lo,hi in bands:
        m=(C.x>=lo)&(C.x<hi)
        row.append(E[m].sum()/max(tot,1e-300))
    # also: which field dominates, and beta localisation
    fo=(dO**2).sum()/max(tot,1e-300)
    jb=int(np.argmax((dO**2+dB**2).sum(axis=0)))
    print("  %5d %10.3e | %s   Ot-frac %.2f  beta_peak/(pi/2) %.3f"
          %(k,sv[-(k+1)]," ".join("%9.3f"%f for f in row),fo,C.b[jb]/(np.pi/2)))
print("\n  reminder: the profile drift between resolutions had 0.99 overlap with the span")
print("  of the 40 smallest of these. Wherever they concentrate is where the")
print("  discretization is failing to pin the fixed point.")
