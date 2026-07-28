import importlib.util, sys, numpy as np, numpy.linalg as la
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pn=M("pn","polar_newton.py"); pst=M("pst","polar_stability.py")

print("BETA RESOLUTION SWEEP -- Nx FIXED, Nb varied.")
print("The near-axis layer is ~0.03 rad wide; at Nb=36 the Chebyshev spacing near")
print("beta=pi/2 is ~0.006, i.e. only ~5 nodes across it. Nx and Nb have been tied")
print("together this whole time. If the axis layer is the blocker, alpha should move")
print("with Nb and settle, at FIXED Nx.\n")
def solve(Nx,Nb):
    a=None
    for _ in range(6):
        St=pst.Stability.__new__(pst.Stability)
        St.S=pn.NewtonSolver.__new__(pn.NewtonSolver)
        # build a Corner with independent Nx, Nb
        pc=M("pc","polar_corner.py")
        C=pc.Corner(Nx,Nb,25.0,filter_on=False)
        if a is not None:
            C.a0=float(a); C.mu=2.0+C.a0
            C.E=np.exp(C.a0*C.x)[:,None]; C._build_poisson()
        S=St.S; S.C=C
        m=np.ones((C.nx,C.nb),bool); m[0,:]=False; m[:,-1]=False
        S.mask=np.concatenate([m.ravel(),m.ravel()]); S.idx=np.where(S.mask)[0]
        S.n2=C.nx*C.nb; S.axis_sub=False
        S.x0=np.concatenate([S.pack(C.Ot0,C.Bt0),[C.P["cl"],C.P["cw"]]])
        St.C=C; St.n=S.idx.size
        x,f,r=pst.newton_exact(St,S.x0,steps=10)
        cl,cw=float(x[-2]),float(x[-1]); an=cw/cl
        if a is not None and abs(an-a)<1e-9: break
        a=an
    return St,x,r,cl,cw
print("  %4s %4s %6s %12s %14s %12s"%("Nx","Nb","dim","||F||","alpha","vs ref"))
for Nx,Nb in ((36,36),(36,52),(36,72),(36,96),(44,72),(44,96)):
    try:
        St,x,r,cl,cw=solve(Nx,Nb)
        a=cw/cl
        print("  %4d %4d %6d %12.3e %14.8f %11.3f%%"
              %(Nx,Nb,St.n,r,a,100*(a-(-0.34240009))/0.34240009),flush=True)
    except Exception as e:
        print("  %4d %4d  FAILED %s"%(Nx,Nb,str(e)[:50]),flush=True)
print("\n  reference alpha = -0.34240009")
