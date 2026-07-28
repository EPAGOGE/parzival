import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pc=M("pc","polar_corner.py")
print("IS THE 1.5% SEED RESIDUAL A DISCRETIZATION ERROR OR THE INTERPOLATION FLOOR?")
print("  discretization error -> FALLS with N (spectral, on a smooth profile)")
print("  interpolation floor  -> FLAT in N\n")
print("  %5s %12s %12s %12s   %s"%("N","max|dOt|/|Ot|","max|dBt|/|Bt|","rms|dOt|/|Ot|","c_l"))
for N in (32,48,64,96,128):
    C=pc.Corner(N,N,25.0)
    I=(slice(2,-2),slice(2,-2))
    dO,dB,_,cl,cw,_=C.rhs(C.Ot0,C.Bt0)
    so=np.abs(dO[I]).max()/np.abs(C.Ot0[I]).max()
    sb=np.abs(dB[I]).max()/np.abs(C.Bt0[I]).max()
    rms=float(np.sqrt(np.mean(dO[I]**2)))/np.abs(C.Ot0[I]).max()
    print("  %5d %12.4e %12.4e %12.4e   %.6f"%(N,so,sb,rms,cl),flush=True)
print("\n  where does the seed residual LIVE? (N=64, max|dOt| by xi band)")
C=pc.Corner(64,64,25.0); dO,_,_,_,_,_=C.rhs(C.Ot0,C.Bt0)
p=np.abs(dO).max(axis=1)
for lo,hi in ((0,0.5),(0.5,2),(2,5),(5,10),(10,15),(15,20),(20,25)):
    m=(C.x>=lo)&(C.x<hi)
    if m.sum(): print("     xi in [%4.1f,%4.1f)  r in [%9.3g,%9.3g)  max|dOt|=%.4e  (%d nodes)"
                      %(lo,hi,np.exp(lo)-1,np.exp(hi)-1,p[m].max(),m.sum()))
