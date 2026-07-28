import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pst=M("pst","polar_stability.py")
St,x,r,cl,cw=pst.converge_exact(36)
C=St.C; Ot,Bt=St.S.unpack(x[:-2]); Pt=C.poisson(Ot)
eps=np.pi/2-C.b                      # distance to the axis
print("ORDER OF EACH FIELD AT THE SYMMETRY AXIS (converged profile, N=36, ||F||=%.1e)"%r)
print("  needed for the substitution Ot = cos(b) Ot_hat, Bt = cos^2(b) Bt_hat:")
print("  dividing the advection by cos(b) creates a tan(b) term, which is FINITE only")
print("  if Pt vanishes at least LINEARLY at the axis.\n")
print("  %6s %10s | %12s %12s %12s"%("i","eps","Ot","Bt","Pt"))
for i in range(1,9):
    j=-i-1
    print("  %6d %10.3e | %12.5e %12.5e %12.5e"%(i,eps[j],
          np.abs(Ot[C.nx//2,j]),np.abs(Bt[C.nx//2,j]),np.abs(Pt[C.nx//2,j])))
print()
for nm,A,pred in (("Ot",Ot,1),("Bt",Bt,2),("Pt",Pt,None)):
    v=np.abs(A[C.nx//2,-9:-1]); e=eps[-9:-1]
    m=v>0
    sl=np.polyfit(np.log(e[m]),np.log(v[m]),1)[0] if m.sum()>3 else np.nan
    tag = "" if pred is None else ("  (expected %d)"%pred)
    print("  %s ~ eps^%+.4f%s"%(nm,sl,tag))
print("\n  VERDICT on the tan(b) term:")
v=np.abs(Pt[C.nx//2,-9:-1]); e=eps[-9:-1]
slp=np.polyfit(np.log(e),np.log(v),1)[0]
print("     Pt ~ eps^%.3f  ->  tan(b)*Pt ~ eps^%.3f  ->  %s"
      %(slp,slp-1,"FINITE, substitution is safe" if slp>=0.95 else
        "SINGULAR -- Pt must be substituted too (Pt = cos(b) Pt_hat)"))
