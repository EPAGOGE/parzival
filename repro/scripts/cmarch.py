import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pc=M("pc","polar_corner.py")
print("DOES THE CORNER FRAME'S MARCH CONVERGE?")
print("A spectrum is only a STABILITY spectrum at a FIXED POINT. The seed carries a 1.5%")
print("residual, so the eigenvalues so far are contaminated. Chen-Hou march to 2e-10")
print("FIRST, then linearise.\n")
C=pc.Corner(48,48,25.0,filter_on=True)
I=(slice(2,-2),slice(2,-2)); O0=C.Ot0.copy(); sc=np.abs(O0[I]).max()
r0=np.abs(C.rhs(C.Ot0,C.Bt0)[0][I]).max()
print("  %7s %8s %11s %11s %10s %10s"%("step","tau","max|dOt|","res/res0","c_l","alpha"))
best=r0
for k in range(20001):
    if k%1000==0:
        dO,_,_,cl,cw,_=C.rhs(C.Ot,C.Bt)
        rr=np.abs(dO[I]).max(); best=min(best,rr)
        print("  %7d %8.2f %11.4e %11.4e %10.6f %10.6f"%(k,k*5e-4,rr,rr/r0,cl,cw/cl),flush=True)
        if not np.isfinite(rr) or rr>1e4:
            print("  DIVERGED"); break
    C.step(5e-4)
print("\n  best residual reached: %.4e  (started %.4e, ratio %.3g)"%(best,r0,best/r0))
