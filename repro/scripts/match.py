import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pc=M("pc","polar_corner.py"); pm=M("pm","polar_march.py")
C=pc.Corner(64,64,25.0); L=pm.March(64,64,-2.0,25.0,filter_on=False)

def gauge_window(obj, rlo, rhi, isC):
    _,pO,pB = obj.parts(obj.Ot0, obj.Bt0)
    KO,LO,MO=pO; KB,LB,MB=pB
    rr = obj.r if isC else np.exp(obj.s)
    W=np.zeros_like(obj.Ot0); m=(rr>=rlo)&(rr<=rhi)
    W[np.ix_(m,np.arange(2,obj.Ot0.shape[1]-2))]=1.0
    Ot,Bt=obj.Ot0,obj.Bt0
    d = obj.dx if isC else obj.ds
    vA=(Ot,2*Bt)
    if isC:
        G=obj.G; vT=(G*(d(Ot)+obj.a0*Ot), G*(d(Bt)+(1+2*obj.a0)*Bt)-Bt)
    else:
        vT=(d(Ot)+obj.a0*Ot, d(Bt)+2*obj.a0*Bt)
    dot=lambda X,Y: float(np.sum(W*X[0]*Y[0])+np.sum(W*X[1]*Y[1]))
    A=np.array([[dot((LO,LB),vA),dot((MO,MB),vA)],[dot((LO,LB),vT),dot((MO,MB),vT)]])
    r_=-np.array([dot((KO,KB),vA),dot((KO,KB),vT)])
    if m.sum()<3: return None
    cl,cw=np.linalg.solve(A,r_)
    return cl,cw,np.linalg.cond(A),int(m.sum())

print("SAME PHYSICAL r WINDOWS, BOTH FRAMES.  target alpha = -0.34240\n")
print("  %-22s | %-30s | %-30s"%("r window","corner (xi)","log-polar (s)"))
for rlo,rhi in ((0.135,7.2e10),(1,1e10),(148,3.3e6),(1e6,1e10),(1e8,1e10)):
    a=gauge_window(C,rlo,rhi,True); b=gauge_window(L,rlo,rhi,False)
    f=lambda z: ("cl=%+8.4f cw=%+8.4f a=%+8.5f"%(z[0],z[1],z[1]/z[0])) if z else "  (too few rows)"
    print("  [%8.3g,%8.3g] | %-30s | %-30s"%(rlo,rhi,f(a),f(b)))
print("\n  full-domain, each frame's own default weighting:")
for nm,obj,isC in (("corner",C,True),("log-polar",L,False)):
    _,pO,pB=obj.parts(obj.Ot0,obj.Bt0)
    cl,cw,cd=obj.gauge(obj.Ot0,obj.Bt0,pO,pB)
    print("    %-10s cl=%+9.5f cw=%+9.5f alpha=%+9.5f"%(nm,cl,cw,cw/cl))
