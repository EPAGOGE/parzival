import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
sp=importlib.util.spec_from_file_location("pm","/Users/epagogellc/parzival/boussinesq/polar_march.py")
pm=importlib.util.module_from_spec(sp); sys.modules["pm"]=pm; sp.loader.exec_module(pm)
print("DOES CLUSTERING IN THE TRANSITION BAND FIX THE DRIFT?  64x64, dt=1e-3, filter on.")
print("reference c_l = 3.006498\n")
print("  %7s | %s" % ("kte", "  ".join("%11s"%("tau=%g"%t) for t in (0,2,4,6))))
for k in (0.0,0.90,0.99,0.999):
    try:
        M=pm.March(64,64,-2.0,25.0,filter_on=True,kte=k)
    except Exception as e:
        print("  %7.4g | build failed: %s" % (k,str(e)[:60])); continue
    I=(slice(2,-2),slice(2,-2)); O0=M.Ot0.copy(); sc=np.abs(O0[I]).max()
    res=[];dr=[];cls=[]
    bad=False
    for j in range(6001):
        if j%2000==0:
            dO,_,_,cl,_,_=M.rhs(M.Ot,M.Bt)
            r=np.abs(dO[I]).max()
            if not np.isfinite(r) or r>1e3: bad=True
            res.append(r); dr.append(np.abs(M.Ot[I]-O0[I]).max()/sc); cls.append(cl)
        if bad: break
        M.step(1e-3)
    print("  %7.4g | res   %s" % (k,"  ".join("%11.4e"%v for v in res)),flush=True)
    print("          | drift %s" % "  ".join("%11.4e"%v for v in dr))
    print("          | c_l   %s" % "  ".join("%11.6f"%v for v in cls))
