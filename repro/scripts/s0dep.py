import importlib.util, sys, numpy as np, numpy.linalg as la
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
sp=importlib.util.spec_from_file_location("pm","/Users/epagogellc/parzival/boussinesq/polar_march.py")
pm=importlib.util.module_from_spec(sp); sys.modules["pm"]=pm; sp.loader.exec_module(pm)
print("DOES THE TRANSVERSE GROWTH RATE DEPEND ON THE INNER TRUNCATION S0?")
print("  Chen-Hou's stability holds on a space of perturbations vanishing QUADRATICALLY")
print("  at the ORIGIN. A domain cut at S0 admits perturbations that space excludes.")
print("  truncation-induced -> rate FALLS as S0 goes deeper.  intrinsic -> flat.\n")
print("  %6s %9s %11s %11s %11s %9s" % ("S0","r_inner","|d|@tau=2","|d|@tau=4","|d|@tau=6","rate"))
for S0 in (-4.0,-3.0,-2.0,-1.0,0.0,2.0):
    try:
        M=pm.March(64,64,S0,25.0,filter_on=True)
    except Exception as e:
        print("  %6.1f  build failed %s"%(S0,str(e)[:40])); continue
    I=(slice(2,-2),slice(2,-2)); O0,B0=M.Ot0.copy(),M.Bt0.copy()
    W=np.zeros_like(O0); W[I]=1.0
    dot=lambda X,Y: float(np.sum(W*X[0]*Y[0])+np.sum(W*X[1]*Y[1]))
    vA=(O0,2.0*B0); vT=(M.Ds@O0+M.a0*O0, M.Ds@B0+2.0*M.a0*B0)
    G=np.array([[dot(vA,vA),dot(vA,vT)],[dot(vT,vA),dot(vT,vT)]])
    nrm=lambda: (lambda d: (np.sqrt(dot(d,d)),
                 np.sqrt(max(dot(d,d)-dot(np.array([0]),np.array([0]))*0,0))))((M.Ot-O0,M.Bt-B0))
    ts=[]
    bad=False
    for k in range(6001):
        if k in (2000,4000,6000):
            d=(M.Ot-O0,M.Bt-B0)
            rhs=np.array([dot(d,vA),dot(d,vT)]); co=la.solve(G,rhs)
            pr=(co[0]*vA[0]+co[1]*vT[0], co[0]*vA[1]+co[1]*vT[1])
            tot=np.sqrt(dot(d,d)); al=np.sqrt(max(dot(pr,pr),0.0))
            tv=np.sqrt(max(tot**2-al**2,0.0))
            if not np.isfinite(tv) or tv>1e4: bad=True
            ts.append(tv)
        if bad: break
        M.step(1e-3)
    if len(ts)==3 and ts[0]>0:
        rate=np.log(ts[2]/ts[0])/4.0
        print("  %6.1f %9.4g %11.4e %11.4e %11.4e %9.4f"
              % (S0,np.exp(S0),ts[0],ts[1],ts[2],rate),flush=True)
    else:
        print("  %6.1f  diverged"%S0,flush=True)
