import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
ps=M("ps","polar_seed.py"); gs=M("gs","polar_gauge_sweep.py")
P=ps.load(); a=P["alpha"]
WX  = 1.19620314      # omega_x(0)   (polar_gauge_gate.py)
THXX= 1.79819132      # theta_xx(0)
h   = float(P["X"][1]-P["X"][0])
print("ANALYTIC CORNER FORM vs INTERPOLATED SEED")
print("  Om = w_x(0) r cos b        w_x(0)  = %.8f" % WX)
print("  B  = th_xx(0)/2 r^2 cos^2b th_xx(0)= %.8f" % THXX)
print("  reference mesh spacing near origin h = %.8f\n" % h)
b=np.linspace(0.04, np.pi/2-0.04, 120)
rs=np.array([0.005,0.01,0.02,0.05,0.1,0.2,0.4,0.8,1.5,3.0])
print("  %8s %7s | %-28s | %-28s"%("r","cells","Om: interp vs analytic","B: interp vs analytic"))
for r0 in rs:
    s0=np.array([np.log(r0)])
    Om,B,Psi=gs.fields_on(ps,P,s0,b)
    Oa = WX*r0*np.cos(b)
    Ba = 0.5*THXX*r0**2*np.cos(b)**2
    eO = np.linalg.norm(Om[0]-Oa)/max(np.linalg.norm(Oa),1e-300)
    eB = np.linalg.norm(B[0]-Ba)/max(np.linalg.norm(Ba),1e-300)
    print("  %8.4g %7.1f | rel %8.4f   |Om|=%9.3e | rel %8.4f   |B|=%9.3e"
          %(r0,r0/h,eO,np.abs(Om[0]).max(),eB,np.abs(B[0]).max()))
print("\n  (cells = r/h, how many reference-mesh cells from the origin.")
print("   Below ~4 cells the interpolated seed is extrapolating and the ANALYTIC form")
print("   is the trustworthy one; far out the analytic Taylor form must fail.)")
