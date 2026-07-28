import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pc=M("pc","polar_corner.py"); pm=M("pm","polar_march.py")
C=pc.Corner(64,64,25.0); L=pm.March(64,64,-2.0,25.0,filter_on=False)
RLO,RHI=1e8,1e10
def bits(obj,isC):
    _,pO,pB=obj.parts(obj.Ot0,obj.Bt0); KO,LO,MO=pO; KB,LB,MB=pB
    rr=obj.r if isC else np.exp(obj.s)
    m=(rr>=RLO)&(rr<=RHI)
    Ot,Bt=obj.Ot0,obj.Bt0
    d=obj.dx if isC else obj.ds
    if isC:
        G=obj.G; vT=(G*(d(Ot)+obj.a0*Ot), G*(d(Bt)+(1+2*obj.a0)*Bt)-Bt)
    else:
        vT=(d(Ot)+obj.a0*Ot, d(Bt)+2*obj.a0*Bt)
    vA=(Ot,2*Bt)
    W=np.zeros_like(Ot); W[np.ix_(m,np.arange(2,Ot.shape[1]-2))]=1.0
    dot=lambda X,Y: float(np.sum(W*X[0]*Y[0])+np.sum(W*X[1]*Y[1]))
    A=np.array([[dot((LO,LB),vA),dot((MO,MB),vA)],[dot((LO,LB),vT),dot((MO,MB),vT)]])
    r_=-np.array([dot((KO,KB),vA),dot((KO,KB),vT)])
    k=Ot.shape[1]//2; i=int(np.argmin(np.abs(rr-1e9)))
    return A,r_,int(m.sum()),(LB[i,k],MB[i,k],KB[i,k],vT[1][i,k],Bt[i,k])
for nm,o,isC in (("corner",C,True),("log-polar",L,False)):
    A,r_,n,pt=bits(o,isC)
    print("%-10s rows=%2d  A=[[%+11.4e %+11.4e],[%+11.4e %+11.4e]]  rhs=[%+11.4e %+11.4e]"
          %(nm,n,A[0,0],A[0,1],A[1,0],A[1,1],r_[0],r_[1]))
    print("%-10s  at r~1e9: LB=%+11.4e MB=%+11.4e KB=%+11.4e vT_B=%+11.4e Bt=%+11.4e"
          %("",*pt))
    cl,cw=np.linalg.solve(A,r_); print("%-10s  -> cl=%+9.5f cw=%+9.5f alpha=%+9.5f\n"%("",cl,cw,cw/cl))
