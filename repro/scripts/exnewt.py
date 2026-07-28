import importlib.util, sys, time, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pst=M("pst","polar_stability.py")
print("EXACT-JACOBIAN NEWTON.  FD Newton gave ||F||=1.8e-2 at N=56 (failure).\n")
print("  %4s %6s %12s %12s %14s %9s"%("N","dim","||F||","c_l","alpha","wall"))
for N in (28,36,44,56):
    t0=time.time()
    St,x,r,cl,cw=pst.converge_exact(N)
    print("  %4d %6d %12.3e %12.6f %14.8f %8.1fm  %s"
          %(N,St.n,r,cl,cw/cl,(time.time()-t0)/60,
            "converged" if r<1e-9 else "NOT CONVERGED"),flush=True)
print("\n  reference: c_l = 3.006498   alpha = -0.34240009")
