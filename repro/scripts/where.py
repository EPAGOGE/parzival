import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pst=M("pst","polar_stability.py")
print("WHERE DO THE N=36 AND N=44 PROFILES DIFFER?  1/g grows like N^2 at the first node.\n")
sols={}
for N in (28,36,44):
    St,x,r,cl,cw=pst.converge(N)
    Ot,Bt=St.S.unpack(x[:-2])
    sols[N]=(St.C, Ot, Bt, cl, cw)
    C=St.C
    print("  N=%2d  first xi node = %.5f   g = %.5f   1/g = %6.1f   c_l=%.6f alpha=%.6f"
          %(N,C.x[1],C.g[1],1/C.g[1],cl,cw/cl))
print("\n  interpolating the N=44 profile onto the N=36 grid and comparing, by xi band:")
C36,O36,B36,_,_ = sols[36]; C44,O44,B44,_,_ = sols[44]
from scipy.interpolate import RegularGridInterpolator
f44=RegularGridInterpolator((C44.x,C44.b),O44,bounds_error=False,fill_value=None)
S,Bb=np.meshgrid(C36.x,C36.b,indexing="ij")
O44i=f44(np.stack([S,Bb],axis=-1))
d=np.abs(O44i-O36)/max(np.abs(O36).max(),1e-300)
for lo,hi in ((0,0.2),(0.2,1),(1,3),(3,8),(8,15),(15,25)):
    m=(C36.x>=lo)&(C36.x<hi)
    if m.sum(): print("     xi in [%5.2f,%5.2f)  r in [%9.3g,%9.3g)  max rel diff = %.4e  (%d nodes)"
                      %(lo,hi,np.exp(lo)-1,np.exp(hi)-1,d[m].max(),m.sum()))
print("\n  if the difference concentrates at the SMALLEST xi, the 1/g amplification at the")
print("  first node is the cause, and the fix is to absorb the corner vanishing into the")
print("  variables (Ot = g*Dt, Bt = g^2*Ct) so no 1/g ever appears.")
