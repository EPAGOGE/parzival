import importlib.util, sys, numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
sys.path.insert(0,".")
sp=importlib.util.spec_from_file_location("ps","polar_seed.py")
ps=importlib.util.module_from_spec(sp); sys.modules["ps"]=ps; sp.loader.exec_module(ps)
P=ps.load(); a=P["alpha"]
NB=600
b=np.linspace(1e-6, np.pi/2-1e-6, NB)
s=np.array([24.0])
Ot,Bt,_,_=ps.seed_on_grid(P,s,b); Ot=Ot[0]; Bt=Bt[0]
# Pt by velocity quadrature (as in polar_residual_gate)
d=loadmat(ps.MAT,squeeze_me=True,struct_as_record=False); sol=d["solu"]
w=np.asarray(d["w"],float)
u1=ps._grid_field(d,sol,"u1",w.shape); u2=ps._grid_field(d,sol,"u2",w.shape)
X,Y=P["X"],P["Y"]; XX,YY=np.meshgrid(X,Y,indexing="ij"); R=np.sqrt(XX**2+YY**2)
R=np.where(R>0,R,np.nan)
su1,su2=u1*R**(-(1+a)),u2*R**(-(1+a))
for A in (su1,su2): A[~np.isfinite(A)]=A[1,1]
f1=RegularGridInterpolator((X,Y),su1,method="cubic",bounds_error=False,fill_value=None)
f2=RegularGridInterpolator((X,Y),su2,method="cubic",bounds_error=False,fill_value=None)
bq=np.concatenate([[0.0],b]); r0=np.exp(24.0)
pts=np.stack([r0*np.cos(bq),r0*np.sin(bq)],axis=-1)
urs=f1(pts)*np.cos(bq)+f2(pts)*np.sin(bq)
Pt=-cumulative_trapezoid(urs,bq,initial=0.0)[1:]

print("MIXED-TRIG BASIS TEST -- can every beta condition be satisfied IDENTICALLY,")
print("so that order_b = 0 and the tau nullity (= order_s * order_b) VANISHES?\n")
print("required beta conditions, from the MEASURED edge behaviour:")
print("  Psi=Pt : 0 at beta=0 AND 0 at beta=pi/2            -> sin(2k b)")
print("  Om =Ot : free at beta=0, 0 at beta=pi/2 (linear)   -> cos((2k+1) b)")
print("  B  =Bt : free at beta=0, d_b B = 0 at beta=pi/2    -> cos(2k b)")
print()
def conv(name, field, gen, Ks=(4,8,16,32,64,96)):
    print("  %-26s" % name, end="")
    prev=None
    for K in Ks:
        M=np.stack(gen(K),axis=1)
        c,*_=np.linalg.lstsq(M,field,rcond=None)
        e=np.linalg.norm(M@c-field)/np.linalg.norm(field)
        print(" %9.2e" % e, end="")
    print()
print("  %-26s" % "basis \\ K", end="")
for K in (4,8,16,32,64,96): print(" %9d" % K, end="")
print()
conv("Pt  in sin(2k b)",  Pt, lambda K:[np.sin(2*k*b) for k in range(1,K+1)])
conv("Ot  in cos((2k+1)b)",Ot, lambda K:[np.cos((2*k+1)*b) for k in range(0,K)])
conv("Bt  in cos(2k b)",   Bt, lambda K:[np.cos(2*k*b) for k in range(0,K)])
print()
conv("[control] Ot in sin(2k b)", Ot, lambda K:[np.sin(2*k*b) for k in range(1,K+1)])
conv("[control] Pt in cos(2k b)", Pt, lambda K:[np.cos(2*k*b) for k in range(0,K)])
print("\n  endpoint sanity:")
print("    Pt(0)=%.3e  Pt(pi/2)=%.3e" % (Pt[0],Pt[-1]))
print("    Ot(0)=%.5f  Ot(pi/2)=%.3e" % (Ot[0],Ot[-1]))
print("    Bt(0)=%.5f  dBt/db(pi/2)=%.3e" % (Bt[0], np.gradient(Bt,b)[-1]))
