import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pst=M("pst","polar_stability.py")

def loop(N, XMAX=25.0, iters=8, verbose=True):
    a=None; hist=[]
    for k in range(iters):
        St=pst.Stability(N,alpha=a,XMAX=XMAX)
        x,f,r,_=St.S.solve(steps=8,verbose=False)
        cl,cw=float(x[-2]),float(x[-1]); an=cw/cl
        a0 = St.C.a0
        gap=abs(an-a0)
        hist.append((a0,an,gap,r,cl,cw))
        if verbose:
            print("     it%d  alpha0=%+.8f -> alpha_out=%+.8f  gap=%.2e  ||F||=%.2e  c_l=%.6f"
                  %(k,a0,an,gap,r,cl),flush=True)
        if gap<1e-8: break
        a=an
    return hist

print("DID THE OUTER ALPHA LOOP ACTUALLY CONVERGE AT EVERY N?")
print("  (converge() allowed only 5 iterations at tol 1e-7; never verified per N)\n")
for N in (28,36,44):
    print("  N=%d:"%N)
    h=loop(N)
    a0,an,gap,r,cl,cw=h[-1]
    ok = gap<1e-8
    print("     -> %s after %d iters, final gap %.2e, alpha=%+.8f, c_l=%.6f\n"
          %("CONVERGED" if ok else "NOT CONVERGED",len(h),gap,an,cl),flush=True)

print("DOMAIN SIZE: does XMAX move the answer? (N=36)")
print("  %6s %12s %12s %10s"%("XMAX","c_l","alpha","||F||"))
for X in (20.0,25.0,30.0,35.0):
    h=loop(36,XMAX=X,verbose=False)
    a0,an,gap,r,cl,cw=h[-1]
    print("  %6.1f %12.6f %12.8f %10.2e   (%d iters, gap %.1e)"%(X,cl,an,r,len(h),gap),flush=True)
print("\n  reference: c_l = 3.006498   alpha = -0.34240009")
