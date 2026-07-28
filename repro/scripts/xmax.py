import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pst=M("pst","polar_stability.py")
print("IS THE OUTER BOUNDARY THE RADIAL PROBLEM?  Nb is now known converged, so vary")
print("XMAX at fixed Nx, Nb. alpha moving with XMAX => the outer treatment is the cause.")
print("alpha moving only with Nx at fixed XMAX => it is radial RESOLUTION, not the edge.\n")
print("  %4s %4s %7s %6s %12s %14s %11s"%("Nx","Nb","XMAX","dim","||F||","alpha","vs ref"))
for Nx,Nb,X in ((36,48,20.0),(36,48,25.0),(36,48,30.0),
                (44,48,20.0),(44,48,25.0),(44,48,30.0),
                (52,48,25.0)):
    try:
        St,x,r,cl,cw=pst.converge_exact(Nx,XMAX=X)  # Nb tied inside; see note
        a=cw/cl
        print("  %4d %4d %7.1f %6d %12.3e %14.8f %10.3f%%"
              %(Nx,St.C.nb,X,St.n,r,a,100*(a+0.34240009)/0.34240009),flush=True)
    except Exception as e:
        print("  %4d %4d %7.1f  FAILED %s"%(Nx,Nb,X,str(e)[:44]),flush=True)
print("\n  reference alpha = -0.34240009")
print("  NOTE converge_exact ties Nb=Nx; the Nb sweep already showed alpha is converged")
print("  in Nb to 1e-5, so the Nx column here is a clean radial-resolution signal.")
