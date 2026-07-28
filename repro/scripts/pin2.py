import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pc=M("pc","polar_corner.py"); pm=M("pm","polar_march.py")
C=pc.Corner(64,64,25.0); Mp=pm.March(64,64,-2.0,25.0,filter_on=False)
Pc=C.poisson(C.Ot0); Pl=Mp.poisson.solve(Mp.Ot0)
k=C.nb//2
print("factor-by-factor at matched r (mid-beta).  adv = (Pt_x+mu Pt) Ot_b - Pt_b (Ot_x+a Ot)\n")
print("  %8s | %-9s %11s %11s %11s %11s"%("r","frame","Pt_x","Ot_b","Pt_b","Ot_x"))
for tgt in (1e6,1e10):
    i=int(np.argmin(np.abs(C.r-tgt))); j=int(np.argmin(np.abs(np.exp(Mp.s)-tgt)))
    a=(C.dx(Pc)[i,k], C.db(C.Ot0)[i,k], C.db(Pc)[i,k], C.dx(C.Ot0)[i,k])
    b=(Mp.ds(Pl)[j,k], Mp.db(Mp.Ot0)[j,k], Mp.db(Pl)[j,k], Mp.ds(Mp.Ot0)[j,k])
    print("  %8.0e | %-9s %11.4e %11.4e %11.4e %11.4e"%(tgt,"xi",*a))
    print("  %8s | %-9s %11.4e %11.4e %11.4e %11.4e"%("","log",*b))
    print("  %8s | actual r: xi-frame %.4e   log-polar %.4e"%("",C.r[i],np.exp(Mp.s[j])))
print("\n  Pt PROFILE in beta at r~1e10 (first 6 and last 3 nodes):")
i=int(np.argmin(np.abs(C.r-1e10))); j=int(np.argmin(np.abs(np.exp(Mp.s)-1e10)))
print("   xi : ", " ".join("%9.5f"%v for v in list(Pc[i,:6])+list(Pc[i,-3:])))
print("   log: ", " ".join("%9.5f"%v for v in list(Pl[j,:6])+list(Pl[j,-3:])))
print("\n  Ot0 PROFILE in beta at r~1e10:")
print("   xi : ", " ".join("%9.5f"%v for v in list(C.Ot0[i,:6])+list(C.Ot0[i,-3:])))
print("   log: ", " ".join("%9.5f"%v for v in list(Mp.Ot0[j,:6])+list(Mp.Ot0[j,-3:])))
