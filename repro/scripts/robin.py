import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
sp=importlib.util.spec_from_file_location("pm","/Users/epagogellc/parzival/boussinesq/polar_march.py")
pm=importlib.util.module_from_spec(sp); sys.modules["pm"]=pm; sp.loader.exec_module(pm)
print("INNER BC HEAD-TO-HEAD (64x64, filter on, dt 1e-3). Is the frozen inner edge the")
print("persistent forcing driving the c_l drift?\n")
for mode in ("dirichlet","robin"):
    M=pm.March(64,64,-2.0,25.0,filter_on=True,inner=mode)
    I=(slice(2,-2),slice(2,-2)); O0=M.Ot0.copy(); sc=np.abs(O0[I]).max()
    print("  --- inner = %s ---" % mode)
    print("  %8s %7s %11s %11s %10s" % ("step","tau","max|dOt|","drift","c_l"))
    for k in range(12001):
        if k%2000==0:
            dO,_,_,cl,_,_=M.rhs(M.Ot,M.Bt)
            print("  %8d %7.1f %11.4e %11.4e %10.6f"
                  % (k,k*1e-3,np.abs(dO[I]).max(),
                     np.abs(M.Ot[I]-O0[I]).max()/sc,cl),flush=True)
        M.step(1e-3)
    print()
