import importlib.util, sys, numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid
sys.path.insert(0, ".")
sp = importlib.util.spec_from_file_location("ps","polar_seed.py")
ps = importlib.util.module_from_spec(sp); sys.modules["ps"]=ps; sp.loader.exec_module(ps)
P = ps.load(); a=P["alpha"]; cl=P["cl"]; cw=P["cw"]; p2=2.0+a
d=loadmat(ps.MAT,squeeze_me=True,struct_as_record=False); sol=d["solu"]
w=np.asarray(d["w"],float)
u1=ps._grid_field(d,sol,"u1",w.shape); u2=ps._grid_field(d,sol,"u2",w.shape)
X,Y=P["X"],P["Y"]; XX,YY=np.meshgrid(X,Y,indexing="ij"); R=np.sqrt(XX**2+YY**2)
R=np.where(R>0,R,np.nan)
su1,su2=u1*R**(-(1+a)),u2*R**(-(1+a))
for A in (su1,su2): A[~np.isfinite(A)]=A[1,1]
f1=RegularGridInterpolator((X,Y),su1,method="linear",bounds_error=False,fill_value=None)
f2=RegularGridInterpolator((X,Y),su2,method="linear",bounds_error=False,fill_value=None)

def resid(S0,S1,NS,NB,pad=0.03):
    s=np.linspace(S0,S1,NS); b=np.linspace(pad,np.pi/2-pad,NB)
    ds,db=s[1]-s[0],b[1]-b[0]
    Ot,Bt,_,_=ps.seed_on_grid(P,s,b)
    bq=np.concatenate([[0.0],b]); S,Bq=np.meshgrid(s,bq,indexing="ij"); Rq=np.exp(S)
    pts=np.stack([Rq*np.cos(Bq),Rq*np.sin(Bq)],axis=-1)
    urs=f1(pts)*np.cos(Bq)+f2(pts)*np.sin(Bq)
    Pt=-cumulative_trapezoid(urs,bq,axis=1,initial=0.0)[:,1:]
    gs=lambda A: np.gradient(A,ds,axis=0); gb=lambda A: np.gradient(A,db,axis=1)
    Ot_s,Ot_b=gs(Ot),gb(Ot); Bt_s,Bt_b=gs(Bt),gb(Bt); Pt_s,Pt_b=gs(Pt),gb(Pt)
    E=np.exp(a*s)[:,None]; cb,sb=np.cos(b)[None,:],np.sin(b)[None,:]
    advO=(Pt_s+p2*Pt)*Ot_b-Pt_b*(Ot_s+a*Ot)
    srcO=cb*(Bt_s+(1+2*a)*Bt)-sb*Bt_b
    R1=cl*Ot_s+E*advO-E*srcO
    advB=(Pt_s+p2*Pt)*Bt_b-Pt_b*(Bt_s+(1+2*a)*Bt)
    R2=cl*Bt_s+E*advB
    I=(slice(3,NS-3),slice(6,NB-6))
    sc1=max(np.abs(cl*Ot_s[I]).max(),np.abs((E*advO)[I]).max(),np.abs((E*srcO)[I]).max())
    sc2=max(np.abs(cl*Bt_s[I]).max(),np.abs((E*advB)[I]).max())
    return (np.abs(R1[I]).max()/sc1, np.abs(R2[I]).max()/sc2,
            float(np.sqrt(np.mean(R1[I]**2)))/sc1, np.abs(cl*Ot_s[I]).max(),
            np.abs((E*advO)[I]).max(), np.abs((E*srcO)[I]).max())

print("DIAGNOSTIC A -- s-REFINEMENT at fixed window [20,30].")
print("  interpolation noise amplified by d/ds GROWS as ds shrinks; a real error does not.\n")
print("   %4s %6s %12s %12s %12s" % ("NS","ds","R1 max rel","R2 max rel","R1 rms rel"))
for NS in (20,40,80,160,320):
    r=resid(20.0,30.0,NS,320)
    print("   %4d %6.3f %12.4e %12.4e %12.4e" % (NS,10.0/(NS-1),r[0],r[1],r[2]))

print("\nDIAGNOSTIC B -- the STRONG test: the INNER region, where e^(a s) is O(1) and all")
print("  transport terms are comparable, so R1/R2 are actually discriminating.\n")
print("   %14s %10s %12s %12s   %11s %11s %11s" % ("window","e^(a s)","R1 max rel","R2 max rel","|cl Ot_s|","|E advO|","|E srcO|"))
for (s0,s1) in ((0.0,4.0),(2.0,8.0),(6.0,12.0),(10.0,16.0),(14.0,20.0),(20.0,30.0)):
    r=resid(s0,s1,120,320)
    print("   [%5.1f,%5.1f] %10.2e %12.4e %12.4e   %11.3e %11.3e %11.3e"
          % (s0,s1,np.exp(a*0.5*(s0+s1)),r[0],r[1],r[3],r[4],r[5]))
