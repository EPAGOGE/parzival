import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
sp=importlib.util.spec_from_file_location("pm","/Users/epagogellc/parzival/boussinesq/polar_march.py")
pm=importlib.util.module_from_spec(sp); sys.modules["pm"]=pm; sp.loader.exec_module(pm)
M=pm.March(64,64,-2.0,25.0,filter_on=True)
I=(slice(2,-2),slice(2,-2))
O0,B0=M.Ot0.copy(),M.Bt0.copy()
W=np.zeros_like(O0); W[I]=1.0
def dot(X,Y): return float(np.sum(W*X[0]*Y[0])+np.sum(W*X[1]*Y[1]))
# orbit tangents evaluated at the SEED (the point we are measuring deviation from)
vA=(O0,2.0*B0)
vT=(M.Ds@O0+M.a0*O0, M.Ds@B0+2.0*M.a0*B0)
# orthonormalise the 2D orbit tangent space
import numpy.linalg as la
G=np.array([[dot(vA,vA),dot(vA,vT)],[dot(vT,vA),dot(vT,vT)]])
print("IS THE DRIFT ALONG THE GAUGE ORBIT, OR TRANSVERSE TO IT?")
print("  along  -> my projection is not holding  -> gauge bug, fix the projection")
print("  transverse -> a genuine mode the gauge cannot remove -> different problem\n")
print("  orbit-tangent Gram cond = %.4g   (angle %.2f deg)"
      % (la.cond(G), np.degrees(np.arccos(abs(G[0,1])/np.sqrt(G[0,0]*G[1,1])))))
print("\n  %7s %12s %12s %12s %10s" % ("tau","|d| total","along orbit","transverse","%along"))
for k in range(6001):
    if k%1000==0:
        dO=M.Ot-O0; dB=M.Bt-B0; d=(dO,dB)
        rhs=np.array([dot(d,vA),dot(d,vT)])
        coef=la.solve(G,rhs)
        proj=(coef[0]*vA[0]+coef[1]*vT[0], coef[0]*vA[1]+coef[1]*vT[1])
        tot=np.sqrt(dot(d,d)); al=np.sqrt(max(dot(proj,proj),0.0))
        tr=np.sqrt(max(tot**2-al**2,0.0))
        print("  %7.1f %12.4e %12.4e %12.4e %9.1f%%"
              % (k*1e-3,tot,al,tr,100*al/max(tot,1e-300)),flush=True)
    M.step(1e-3)
