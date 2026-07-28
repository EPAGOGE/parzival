import os, sys, logging
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np, dedalus.public as d3
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
from polar_tau2d_gate_b import build, MU
for nm in list(logging.root.manager.loggerDict): logging.getLogger(nm).setLevel(logging.ERROR)
Ns,Nb=32,24
coords=d3.CartesianCoordinates("s","beta"); dist=d3.Distributor(coords,dtype=np.float64)
sb=d3.ChebyshevT(coords["s"],size=Ns,bounds=(10.,25.)); bb=d3.ChebyshevT(coords["beta"],size=Nb,bounds=(0.,np.pi/2))
s=dist.local_grid(sb); b=dist.local_grid(bb)
sb2=sb.derivative_basis(2); bb2=bb.derivative_basis(2)
U=dist.Field(bases=(sb,bb)); F=dist.Field(bases=(sb,bb))
# B's variant V4: tau fields living ON THE DERIVATIVE BASIS
ts1=dist.Field(bases=bb2); ts2=dist.Field(bases=bb2)
tb1=dist.Field(bases=sb2); tb2=dist.Field(bases=sb2)
ns=dict(U=U,F=F,ts1=ts1,ts2=ts2,tb1=tb1,tb2=tb2,
        Ls=lambda A,n: d3.Lift(A,sb2,n), Lb=lambda A,n: d3.Lift(A,bb2,n),
        ds=lambda A: d3.Differentiate(A,coords["s"]), lap=lambda A: d3.Laplacian(A,coords),
        MU=MU,S0=10.,S1=25.,BE=np.pi/2)
shape=np.sin(2*b)+0.3*np.sin(4*b)
F["g"]=MU**2*shape-4.0*np.sin(2*b)-4.8*np.sin(4*b)
p=d3.LBVP([U,ts1,ts2,tb1,tb2],namespace=ns)
p.add_equation("lap(U)+2*MU*ds(U)+MU**2*U+Ls(ts1,-1)+Ls(ts2,-2)+Lb(tb1,-1)+Lb(tb2,-2) = F")
p.add_equation("U(beta=0)=0"); p.add_equation("U(beta=BE)=0")
p.add_equation("ds(U)(s=S0)=0"); p.add_equation("ds(U)(s=S1)=0")
print("built OK, calling build_solver...", flush=True)
sv=p.build_solver()
print("solver built, calling solve()...", flush=True)
sv.solve()
print("SOLVED, no crash. err:", np.abs(U["g"]-shape).max(), flush=True)
