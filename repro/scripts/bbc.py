import importlib.util, sys, numpy as np
sys.path.insert(0,".")
sp=importlib.util.spec_from_file_location("ps","polar_seed.py")
ps=importlib.util.module_from_spec(sp); sys.modules["ps"]=ps; sp.loader.exec_module(ps)
P=ps.load(); a=P["alpha"]
print("What is theta's parity about the SYMMETRY AXIS (beta=pi/2, i.e. y1=0)?")
print("  even in y1  =>  d_b B -> 0   and  B(pi/2) != 0")
print("  odd  in y1  =>  B(pi/2) = 0  and  d_b B != 0\n")
eps=np.array([3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5])
for s0 in (20.0,25.0,30.0):
    _,Bt,_,_=ps.seed_on_grid(P,np.array([s0]),np.pi/2-eps)
    v=Bt[0]
    print("  s=%.0f  Bt at eps:" % s0, " ".join("%9.5f" % x for x in v))
    m=np.abs(v)>0
    sl=np.polyfit(np.log(eps[m][-6:]),np.log(np.abs(v[m][-6:])),1)[0]
    print("        -> Bt ~ eps^%+.4f near the axis; Bt(eps=1e-5)=%.6g" % (sl, v[-1]))
print("\nSame question at the WALL (beta=0, i.e. y2=0):")
for s0 in (25.0,):
    _,Bt,_,_=ps.seed_on_grid(P,np.array([s0]),eps)
    v=Bt[0]
    print("  s=%.0f  Bt at beta=eps:" % s0, " ".join("%9.5f" % x for x in v))
    sl=np.polyfit(np.log(eps[-6:]),np.log(np.abs(v[-6:])),1)[0]
    print("        -> Bt ~ eps^%+.4f ; Bt(0+)=%.6g" % (sl, v[-1]))
print("\nAnd Om for contrast (known: linear zero at axis, flat nonzero at wall):")
for s0 in (25.0,):
    Ot,_,_,_=ps.seed_on_grid(P,np.array([s0]),np.pi/2-eps)
    print("  axis Ot:", " ".join("%9.5f" % x for x in Ot[0]))
    Ot,_,_,_=ps.seed_on_grid(P,np.array([s0]),eps)
    print("  wall Ot:", " ".join("%9.5f" % x for x in Ot[0]))
