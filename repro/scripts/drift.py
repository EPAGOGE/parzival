import importlib.util, sys, numpy as np, numpy.linalg as la
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pst=M("pst","polar_stability.py")
rng=np.random.default_rng(0)

print("DOES THE FIXED-POINT DRIFT ALIGN WITH THE ILL-CONDITIONED DIRECTIONS?")
print("  A fixed point responds to a perturbation as dx ~ -J^-1 dF, so it moves MOST")
print("  along the near-singular directions of J -- which is where kappa blows up.")
print("  If the alpha drift is funnelled into identifiable modes, it is not noise.\n")

sols={}
for N in (36,44,52):
    St,x,r,cl,cw=pst.converge_exact(N)
    Ot,Bt=St.S.unpack(x[:-2])
    sols[N]=(St,x,Ot,Bt,cl,cw,r)
    print("  N=%2d ||F||=%.2e alpha=%.8f"%(N,r,cw/cl),flush=True)

def onto(Csrc,A,Cdst):
    S,B=np.meshgrid(Cdst.x,Cdst.b,indexing="ij")
    f=RegularGridInterpolator((Csrc.x,Csrc.b),A,method="cubic",
                              bounds_error=False,fill_value=None)
    return f(np.stack([S,B],axis=-1))

REF=36
Sref,xref,Oref,Bref,_,_,_=sols[REF]; Cref=Sref.C
J=None
# operator + its conditioning at the reference resolution
A=Sref.A_exact(xref); B=Sref.exact_B(Oref,Bref); Cg=Sref.exact_Cg(); n=Sref.n
L=(np.eye(n)-B@la.solve(Cg@B,Cg))@A
U,sv,Vt=la.svd(L)
w,V=la.eig(L); wl,W=la.eig(L.T)
kap=np.empty(w.size)
for i in range(w.size):
    j=int(np.argmin(np.abs(wl-w[i])))
    xr=V[:,i]/la.norm(V[:,i]); yl=W[:,j]/la.norm(W[:,j])
    kap[i]=1.0/max(abs(np.vdot(yl,xr)),1e-300)
# EXCLUDE THE PROJECTION'S NULL SPACE. P has rank n-2 BY CONSTRUCTION, so L=PA has two
# exact zero singular values; sv[-1] ~ 2e-11 is those zeros in roundoff, not the operator.
# Reporting cond = sv[0]/sv[-1] measures MY GAUGE, not the physics -- the same artifact
# that made R(0) ~ 1e15 in polar_resolvent.py.
print("\n  reference N=%d:"%REF)
print("     raw            sigma_min=%.3e sigma_max=%.3e cond=%.3e  <- CONTAMINATED by"
      " the projection's 2 null directions"%(sv[-1],sv[0],sv[0]/max(sv[-1],1e-300)))
print("     null excluded  sigma_min=%.3e cond=%.3e"%(sv[-3],sv[0]/max(sv[-3],1e-300)))
print("     smallest 6 singular values: %s"%np.array2string(sv[-6:],precision=3))

print("\n  %-16s %10s %10s %10s %10s"
      %("drift","|dx|","smallest-SV","highest-kappa","random"))
for Na,Nb in ((36,44),(44,52),(36,52)):
    Sa,xa,Oa,Ba,_,_,_=sols[Na]; Sb,xb,Ob,Bb,_,_,_=sols[Nb]
    dO=onto(Sb.C,Ob,Cref)-onto(Sa.C,Oa,Cref)
    dB=onto(Sb.C,Bb,Cref)-onto(Sa.C,Ba,Cref)
    dx=np.concatenate([dO.ravel(),dB.ravel()])[Sref.S.idx]
    nx=la.norm(dx)
    if nx==0: continue
    u=dx/nx
    K=40
    smallSV=Vt[-K:,:]                       # K smallest right singular directions
    ordk=np.argsort(-kap)[:K]
    hik=np.stack([ (V[:,i]/la.norm(V[:,i])) for i in ordk],axis=0)
    Qs,_=la.qr(smallSV.conj().T); Qk,_=la.qr(hik.conj().T)
    rv=rng.standard_normal(n); rv/=la.norm(rv)
    Qr,_=la.qr(rng.standard_normal((n,K)))
    print("  N=%2d->%2d      %10.3e %10.4f %10.4f %10.4f"
          %(Na,Nb,nx,la.norm(Qs.conj().T@u),la.norm(Qk.conj().T@u),
            la.norm(Qr.conj().T@u)),flush=True)
print("\n  (overlap of the unit drift direction with a K=40 subspace; the random column")
print("   is the chance level for a 40-dim subspace of an n=%d space, ~sqrt(40/n)=%.3f)"
      %(n,np.sqrt(40/n)))
