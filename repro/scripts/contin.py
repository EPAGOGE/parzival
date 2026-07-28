import importlib.util, sys, time, numpy as np
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pst=M("pst","polar_stability.py")

def interp_onto(Csrc, Ot, Bt, Cdst):
    """Cubic interpolation of a converged profile from one Chebyshev grid onto another."""
    out=[]
    S,B=np.meshgrid(Cdst.x, Cdst.b, indexing="ij")
    pts=np.stack([S,B],axis=-1)
    for A in (Ot,Bt):
        f=RegularGridInterpolator((Csrc.x,Csrc.b),A,method="cubic",
                                  bounds_error=False,fill_value=None)
        out.append(f(pts))
    return out

print("CONTINUATION: seed Newton at N2 from the CONVERGED solution at N1,")
print("instead of from the reference interpolation.  If N=56 then converges, the")
print("problem was the seed's basin, not the finite-difference Jacobian.\n")

# converged base at N=44
t0=time.time()
St44,x44,r44,cl44,cw44 = pst.converge(44)
Ot44,Bt44 = St44.S.unpack(x44[:-2])
print("  base  N=44: ||F||=%.3e  c_l=%.6f  alpha=%.8f   (%.1f min)"
      %(r44,cl44,cw44/cl44,(time.time()-t0)/60),flush=True)

for N2 in (56,):
    t0=time.time()
    St=pst.Stability(N2, alpha=cw44/cl44)
    Oi,Bi = interp_onto(St44.C, Ot44, Bt44, St.C)
    Oi[0,:]=St.C.Ot0[0,:]; Bi[0,:]=St.C.Bt0[0,:]      # corner row stays exact
    x0=np.concatenate([St.S.pack(Oi,Bi),[cl44,cw44]])
    f0,_,_=St.S.F(x0)
    print("  seed  N=%d: ||F|| from CONTINUATION = %.3e   (reference-interp seed gave"
          " ~2e-2 after 8 Newton steps)"%(N2,np.linalg.norm(f0)/np.sqrt(f0.size)),flush=True)
    St.S.x0=x0
    x,f,r,r0=St.S.solve(steps=10,verbose=True)
    cl,cw=float(x[-2]),float(x[-1])
    print("  ->    N=%d: ||F||=%.3e  c_l=%.6f  alpha=%.8f   (%.1f min)"
          %(N2,r,cl,cw/cl,(time.time()-t0)/60),flush=True)
    print("        %s"%("CONVERGED -- continuation fixes it" if r<1e-9
                        else "still not converged -> the FD Jacobian is the blocker"))
