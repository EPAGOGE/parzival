import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pc=M("pc","polar_corner.py"); pm=M("pm","polar_march.py")
C=pc.Corner(64,64,25.0); Mp=pm.March(64,64,-2.0,25.0,filter_on=False)
i=int(np.argmin(np.abs(C.r-1e10))); j=int(np.argmin(np.abs(np.exp(Mp.s)-1e10))); k=C.nb//2
print("Ot0 last 3 beta nodes:  xi %s   log %s"
      %(np.array2string(C.Ot0[i,-3:],precision=5), np.array2string(Mp.Ot0[j,-3:],precision=5)))
print("beta grids identical: %s" % np.allclose(C.b, Mp.b))
print("Ot_b at mid-beta:  xi %.6e   log %.6e" % (C.db(C.Ot0)[i,k], Mp.db(Mp.Ot0)[j,k]))
print("Ot0 at mid-beta :  xi %.6e   log %.6e" % (C.Ot0[i,k], Mp.Ot0[j,k]))
print("\nFULL beta profile difference at r~1e10 (xi - log), max = %.3e"
      % np.abs(C.Ot0[i]-Mp.Ot0[j]).max())
d=np.abs(C.Ot0[i]-Mp.Ot0[j]); w=np.argsort(d)[::-1][:5]
for q in w: print("   beta=%.5f  xi=%.6f  log=%.6f  diff=%.3e"%(C.b[q],C.Ot0[i,q],Mp.Ot0[j,q],d[q]))
