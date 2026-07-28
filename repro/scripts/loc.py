import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
sp=importlib.util.spec_from_file_location("pm","/Users/epagogellc/parzival/boussinesq/polar_march.py")
pm=importlib.util.module_from_spec(sp); sys.modules["pm"]=pm; sp.loader.exec_module(pm)
M=pm.March(64,64,-2.0,25.0,filter_on=True)
I=(slice(2,-2),slice(2,-2))
print("WHERE does the residual live? If the frozen inflow VALUE is inconsistent with")
print("this domain's own fixed point, the residual must concentrate near s = S0.\n")
def prof(F):
    a=np.abs(F); return a.max(axis=1)          # max over beta, as a function of s
for k in range(6001):
    if k in (0,2000,6000):
        dO,_,_,cl,_,_=M.rhs(M.Ot,M.Bt)
        p=prof(dO); sm=M.s
        tot=np.abs(dO[I]).max()
        j=int(np.argmax(p[2:-2]))+2
        print("tau=%4.1f  max|dOt|=%.4e at s=%+.2f  (domain [-2,25])" % (k*1e-3,tot,sm[j]))
        # binned profile
        edges=[-2,0,2,5,10,15,20,25]
        for lo,hi in zip(edges[:-1],edges[1:]):
            m=(sm>=lo)&(sm<hi)
            if m.sum(): print("     s in [%5.1f,%5.1f): max|dOt| = %.4e  (%d nodes)"
                              % (lo,hi,p[m].max(),m.sum()))
        print()
    M.step(1e-3)
