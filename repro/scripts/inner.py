import importlib.util, sys, numpy as np
sys.path.insert(0,".")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,f); m=importlib.util.module_from_spec(sp)
    sys.modules[n]=m; sp.loader.exec_module(m); return m
ps=M("ps","polar_seed.py"); gs=M("gs","polar_gauge_sweep.py")
P=ps.load(); a=P["alpha"]
print("INNER BOUNDARY from the CORNER TAYLOR EXPANSION.")
print("Near r=0: Om ~ w_x(0) y1 ~ r^1 ; B ~ th_xx(0)/2 y1^2 ~ r^2 ; Psi ~ r^2")
print("so in substituted variables the INNER behaviour is also a power law:")
print("   Ot ~ r^(1-a)   => d_s Ot = (1-a) Ot     predicted %+.5f" % (1-a))
print("   Bt ~ r^(1-2a)  => d_s Bt = (1-2a) Bt    predicted %+.5f" % (1-2*a))
print("   Pt ~ r^(-a)    => d_s Pt = -a Pt        predicted %+.5f\n" % (-a))
b=np.linspace(0.04,np.pi/2-0.04,240)
print("  measured d log F / d s  on a window just inside the proposed inner edge:")
print("  %-14s %10s %10s %10s   %8s" % ("window","Ot","Bt","Pt","cells@lo"))
h=float(P["X"][1]-P["X"][0])
for lo,hi in ((-3.0,-1.0),(-2.5,-1.5),(-2.0,-1.0),(-1.5,-0.5),(-1.0,0.0),(0.0,1.0),(1.0,2.0)):
    s=np.linspace(lo,hi,60)
    Om,B,Psi=gs.fields_on(ps,P,s,b)
    Ot=Om*np.exp(-a*s)[:,None]; Bt=B*np.exp(-(1+2*a)*s)[:,None]; Pt=Psi*np.exp(-(2+a)*s)[:,None]
    ds=s[1]-s[0]
    out=[]
    for F in (Ot,Bt,Pt):
        g=np.gradient(F,ds,axis=0)/np.where(np.abs(F)>1e-300,F,np.nan)
        out.append(np.nanmedian(g[10:-10,10:-10]))
    print("  [%5.1f,%5.1f] %10.5f %10.5f %10.5f   %8.1f"
          % (lo,hi,out[0],out[1],out[2], np.exp(lo)/h))
print("\n  predicted:     %10.5f %10.5f %10.5f" % (1-a,1-2*a,-a))
print("\n  (Ot exponent should -> 1-a going inward and -> 0 going outward, since")
print("   Ot ~ const in the far field. The crossover IS the transition region.)")
