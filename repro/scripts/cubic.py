import importlib.util, sys, numpy as np
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0,".")
sp=importlib.util.spec_from_file_location("ps","polar_seed.py")
ps=importlib.util.module_from_spec(sp); sys.modules["ps"]=ps; sp.loader.exec_module(ps)
P=ps.load(); a=P["alpha"]; cl=P["cl"]
X,Y=P["X"],P["Y"]; XX,YY=np.meshgrid(X,Y,indexing="ij"); R=np.sqrt(XX**2+YY**2)
R=np.where(R>0,R,np.nan)
Otg=P["w"]*R**(-a); Otg[~np.isfinite(Otg)]=Otg[1,1]
print("Does the far-field residual come from the INTERPOLATION FLOOR?")
print("If so, a higher-order interpolant must reduce Ot's spurious s-variation.\n")
print("  %-8s %14s %14s %14s" % ("method","max spread/mean","median","max|cl*Ot_s|"))
s=np.linspace(20.0,30.0,160); b=np.linspace(0.03,np.pi/2-0.03,320)
S,B=np.meshgrid(s,b,indexing="ij"); Rg=np.exp(S)
pts=np.stack([Rg*np.cos(B),Rg*np.sin(B)],axis=-1)
for meth in ("linear","cubic","quintic"):
    try:
        f=RegularGridInterpolator((X,Y),Otg,method=meth,bounds_error=False,fill_value=None)
        Ot=f(pts)
    except Exception as e:
        print("  %-8s unavailable (%s)" % (meth, str(e)[:50])); continue
    rel=np.nanmax(np.abs(Ot-Ot.mean(axis=0,keepdims=True)),axis=0)/np.maximum(np.abs(Ot.mean(axis=0)),1e-300)
    Ot_s=np.gradient(Ot,s[1]-s[0],axis=0)
    I=(slice(3,157),slice(6,314))
    print("  %-8s %14.4e %14.4e %14.4e" % (meth,np.nanmax(rel),np.nanmedian(rel),np.abs(cl*Ot_s[I]).max()))
print("\n  For reference, the equation in [20,30] demands |cl*Ot_s| ~ |E*bracket| ~ 1.1e-02.")
print("  A method whose spurious |cl*Ot_s| falls below that can resolve the far-field balance.")
