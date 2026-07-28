import importlib.util, sys, numpy as np, numpy.linalg as la
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pst=M("pst","polar_stability.py")
print("AXIS SUBSTITUTION  Ot = cos(b) Ot_hat , Bt = cos^2(b) Bt_hat")
print("applied as a change of UNKNOWNS (= diagonal preconditioner D^-1 J D).")
print("Physics unchanged, so the converged PROFILE must be identical; what should")
print("change is the CONDITIONING and, if the diagnosis is right, the N-convergence.\n")
print("  %4s %6s %12s %12s %12s %14s"%("N","axis","||F||","alpha","cond(J0)","sigma_min(L)"))
for N in (36,44,52):
    for sub in (False,True):
        St=pst.Stability(N); St.S.axis_sub=sub
        St.S.x0=np.concatenate([St.S.pack(St.C.Ot0,St.C.Bt0),[St.C.P["cl"],St.C.P["cw"]]])
        x,f,r=pst.newton_exact(St,St.S.x0,steps=10)
        cl,cw=float(x[-2]),float(x[-1])
        Ot,Bt=St.S.unpack(x[:-2])
        A=St.A_exact(x); B=St.exact_B(Ot,Bt); Cg=St.exact_Cg(); n=St.n
        J=np.zeros((n+2,n+2)); J[:n,:n]=A; J[:n,n:]=B; J[n:,:n]=Cg
        L=(np.eye(n)-B@la.solve(Cg@B,Cg))@A
        sv=la.svd(L,compute_uv=False)
        print("  %4d %6s %12.3e %12.8f %12.3e %14.4e"
              %(N,"ON" if sub else "off",r,cw/cl,la.cond(J),sv[-3]),flush=True)
print("\n  reference alpha = -0.34240009")
