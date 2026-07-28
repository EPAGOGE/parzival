import importlib.util, sys, time, numpy as np, numpy.linalg as la
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pst=M("pst","polar_stability.py")
print("IS N=56 AN ISOLATED BAD CASE, OR A THRESHOLD?  Each N is an INDEPENDENT system,")
print("so 56 can simply be skipped. If 60/64 converge, 56 is a one-off; if they fail too,")
print("something degrades systematically above N~44.\n")
print("  %4s %6s %12s %12s %12s %8s"%("N","dim","||F|| init","||F|| final","alpha","verdict"))
for N in (60,64):
    t0=time.time()
    try:
        St,x,r,cl,cw=pst.converge_exact(N)
        f0,_,_=St.S.F(St.S.x0)
        print("  %4d %6d %12.3e %12.3e %12.8f %8s   (%.1fm)"
              %(N,St.n,la.norm(f0)/np.sqrt(f0.size),r,cw/cl,
                "OK" if r<1e-9 else "FAIL",(time.time()-t0)/60),flush=True)
    except Exception as e:
        print("  %4d  EXCEPTION %s"%(N,str(e)[:60]),flush=True)
print("\n  reference alpha = -0.34240009")
