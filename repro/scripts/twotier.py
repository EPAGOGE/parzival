import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pc=M("pc","polar_corner.py")
print("TWO-TIER SUBSTEPPING (Chen-Hou run_pertb.m:56-59): 1 outer step WITH the Poisson")
print("solve, then 30 inner steps with the VELOCITY FROZEN.  48x48, filter on.\n")
K,K2,NIN = 5e-4, 3.75e-4, 30
for ninner,lab in ((0,"one-tier (what we have been doing)"),(NIN,"TWO-TIER, 30 inner")):
    C=pc.Corner(48,48,25.0,filter_on=True)
    I=(slice(2,-2),slice(2,-2))
    r0=np.abs(C.rhs(C.Ot0,C.Bt0)[0][I]).max()
    print("  --- %s ---  res0=%.4e"%(lab,r0))
    print("  %6s %9s %11s %11s %10s %10s"%("outer","tau_eff","max|dOt|","res/res0","c_l","alpha"))
    tau=0.0; best=r0; bad=False
    for k in range(601):
        if k%50==0:
            dO,_,_,cl,cw,_=C.rhs(C.Ot,C.Bt)
            rr=np.abs(dO[I]).max(); best=min(best,rr)
            print("  %6d %9.3f %11.4e %11.4e %10.6f %10.6f"%(k,tau,rr,rr/r0,cl,cw/cl),flush=True)
            if not np.isfinite(rr) or rr>1e4: print("   DIVERGED"); bad=True
        if bad: break
        if ninner: C.step_two_tier(K,K2,ninner); tau+=K+ninner*K2
        else:      C.step(K); tau+=K
    print("  best res %.4e  (ratio %.3g)\n"%(best,best/r0))
