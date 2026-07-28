import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pc=M("pc","polar_corner.py"); pm=M("pm","polar_march.py")
C=pc.Corner(48,48,25.0)
_,pO,pB=C.parts(C.Ot0,C.Bt0)
KO,LO,MO=pO; KB,LB,MB=pB
print("Gauge solved on RESTRICTED xi windows (which region drives it?):")
print("  target c_l=%.5f c_w=%.5f  alpha=%.5f\n"%(C.P["cl"],C.P["cw"],C.a0))
def solve_on(lo,hi):
    W=np.zeros_like(C.Ot0)
    m=(C.x>=lo)&(C.x<=hi)
    W[np.ix_(m,np.arange(2,C.nb-2))]=1.0
    vA=(C.Ot0,2*C.Bt0)
    G=C.G
    vT=(G*(C.dx(C.Ot0)+C.a0*C.Ot0), G*(C.dx(C.Bt0)+(1+2*C.a0)*C.Bt0)-C.Bt0)
    dot=lambda X,Y: float(np.sum(W*X[0]*Y[0])+np.sum(W*X[1]*Y[1]))
    A=np.array([[dot((LO,LB),vA),dot((MO,MB),vA)],[dot((LO,LB),vT),dot((MO,MB),vT)]])
    r=-np.array([dot((KO,KB),vA),dot((KO,KB),vT)])
    try: cl,cw=np.linalg.solve(A,r)
    except Exception: return None
    return cl,cw,np.linalg.cond(A)
for lo,hi in ((0,25),(15,25),(20,25),(5,15),(1,5),(0,2)):
    out=solve_on(lo,hi)
    if out: print("  xi in [%4.1f,%4.1f]: c_l=%+10.5f c_w=%+10.5f alpha=%+9.5f cond=%.4g"
                  %(lo,hi,out[0],out[1],out[1]/out[0],out[2]))
print("\nFAR-FIELD CONSISTENCY: does the xi-frame RHS reduce to the log-polar one?")
Mp=pm.March(48,48,-2.0,25.0,filter_on=False)
_,qO,qB=Mp.parts(Mp.Ot0,Mp.Bt0)
# compare at matched physical r in the far field
for tgt in (1e6,1e8,1e10):
    i=int(np.argmin(np.abs(C.r-tgt))); j=int(np.argmin(np.abs(np.exp(Mp.s)-tgt)))
    k=C.nb//2
    print("  r~%.0e | xi-frame KO=%+11.4e LO=%+11.4e MO=%+11.4e  g=%.6f"
          %(tgt,KO[i,k],LO[i,k],MO[i,k],C.g[i]))
    print("          | log-polar KO=%+11.4e LO=%+11.4e MO=%+11.4e"
          %(qO[0][j,k],qO[1][j,k],qO[2][j,k]))
