import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pc=M("pc","polar_corner.py"); pm=M("pm","polar_march.py")

def clencurt(n):
    """Clenshaw-Curtis weights on n Chebyshev-Lobatto nodes over [-1,1] (ascending)."""
    N=n-1
    th=np.pi*np.arange(N+1)/N
    w=np.zeros(N+1); v=np.ones(N-1)
    if N%2==0:
        w[0]=w[N]=1.0/(N*N-1)
        for k in range(1,N//2): v-=2*np.cos(2*k*th[1:N])/(4*k*k-1)
        v-=np.cos(N*th[1:N])/(N*N-1)
    else:
        w[0]=w[N]=1.0/(N*N)
        for k in range(1,(N-1)//2+1): v-=2*np.cos(2*k*th[1:N])/(4*k*k-1)
    w[1:N]=2*v/N
    return w

def gauge_w(obj, Ot, Bt, use_cc, xattr="x"):
    pO=pB=None
    _,pO,pB = obj.parts(Ot,Bt)
    KO,LO,MO=pO; KB,LB,MB=pB
    xg=getattr(obj,xattr); nx=xg.size; nb=obj.b.size
    if use_cc:
        wx=clencurt(nx)*(xg[-1]-xg[0])/2.0
        wb=clencurt(nb)*(obj.b[-1]-obj.b[0])/2.0
        W=np.abs(np.outer(wx,wb))
    else:
        W=np.ones((nx,nb))
    W=W.copy(); W[:2]=0; W[-2:]=0; W[:,:2]=0; W[:,-2:]=0
    dxf = obj.dx if hasattr(obj,"dx") else obj.ds
    G = obj.G if hasattr(obj,"G") else 1.0
    vA=(Ot,2*Bt)
    if hasattr(obj,"G"):
        vT=(G*(dxf(Ot)+obj.a0*Ot), G*(dxf(Bt)+(1+2*obj.a0)*Bt)-Bt)
    else:
        vT=(dxf(Ot)+obj.a0*Ot, dxf(Bt)+2*obj.a0*Bt)
    dot=lambda X,Y: float(np.sum(W*X[0]*Y[0])+np.sum(W*X[1]*Y[1]))
    A=np.array([[dot((LO,LB),vA),dot((MO,MB),vA)],[dot((LO,LB),vT),dot((MO,MB),vT)]])
    r=-np.array([dot((KO,KB),vA),dot((KO,KB),vT)])
    cl,cw=np.linalg.solve(A,r)
    return cl,cw,np.linalg.cond(A)

C=pc.Corner(64,64,25.0); Mp=pm.March(64,64,-2.0,25.0,filter_on=False)
print("Is the gauge's inner product the problem?  node-sum vs CLENSHAW-CURTIS quadrature")
print("  target: c_l=3.00650  c_w=-1.02943  alpha=-0.34240\n")
print("  %-26s %11s %11s %11s %10s"%("frame / inner product","c_l","c_w","alpha","cond"))
for lab,obj,attr in (("log-polar s in[-2,25]",Mp,"s"),("corner   xi in[0,25]",C,"x")):
    for cc in (False,True):
        try:
            cl,cw,cd=gauge_w(obj,obj.Ot0,obj.Bt0,cc,attr)
            print("  %-26s %11.5f %11.5f %11.5f %10.4g"
                  %(lab+(" +CC" if cc else " node"),cl,cw,cw/cl,cd))
        except Exception as e:
            print("  %-26s FAILED %s"%(lab,str(e)[:40]))
