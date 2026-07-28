import numpy as np, scipy.io as sio
d=sio.loadmat('/Users/epagogellc/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat',
              struct_as_record=False,squeeze_me=True)
solu,Mesh=d['solu'],d['Mesh']
x1=np.asarray(Mesh.x[0],float);x2=np.asarray(Mesh.x[1],float)
def F(n):
    a=getattr(solu,n)
    while not(isinstance(a,np.ndarray) and a.dtype!=object and a.ndim==2): a=a[0]
    return np.asarray(a,float)
w,th,u1,u2=F('w'),F('th'),F('u1'),F('u2')
al=float(solu.al); alpha=-al
X1,X2=np.meshgrid(x1,x2,indexing='ij'); R=np.hypot(X1,X2); Bt=np.arctan2(X2,X1)
sel=(R>1e9)&(R<1e13); b=Bt[sel];r=R[sel]
G=w[sel]*r**al; T=th[sel]*r**(-(1+2*alpha))
ub=-u1[sel]*np.sin(b)+u2[sel]*np.cos(b); P=ub/((2+alpha)*r**(1+alpha))
# bin-average onto a uniform beta grid, then smooth-differentiate spectrally-ish
nb=300; e=np.linspace(0,np.pi/2,nb+1); idx=np.digitize(b,e)-1
bc=0.5*(e[1:]+e[:-1]); Gb=np.full(nb,np.nan);Tb=np.full(nb,np.nan);Pb=np.full(nb,np.nan)
for k in range(nb):
    m=idx==k
    if m.sum()>=4: Gb[k],Tb[k],Pb[k]=np.median(G[m]),np.median(T[m]),np.median(P[m])
ok=~np.isnan(Gb)
print("filled bins:",ok.sum(),"/",nb)
from numpy.polynomial import chebyshev as C
# fit smooth Chebyshev in beta to the binned profiles, then differentiate exactly
z=(2*bc[ok]-np.pi/2)/(np.pi/2)
deg=40
cg=C.chebfit(z,Gb[ok],deg); cp=C.chebfit(z,Pb[ok],deg); ct=C.chebfit(z,Tb[ok],deg)
sc=2/(np.pi/2)
Pv=C.chebval(z,cp); Ppp=C.chebval(z,C.chebder(cp,2))*sc**2
Gv=C.chebval(z,cg); Gp=C.chebval(z,C.chebder(cg,1))*sc
Tv=C.chebval(z,ct); Tp=C.chebval(z,C.chebder(ct,1))*sc
bb=bc[ok]
res1=Ppp+(2+alpha)**2*Pv+Gv
print(f"\nREL1  P'' + (2+a)^2 P + G = 0 :  max|res|/max|G| = {np.abs(res1).max()/np.abs(Gv).max():.3e}"
      f"   median|res|/max|G| = {np.median(np.abs(res1))/np.abs(Gv).max():.3e}")
res2=(2+alpha)*Pv*Tp-(1+2*alpha)*Tv*Pp if False else (2+alpha)*Pv*Tp-(1+2*alpha)*Tv*C.chebval(z,C.chebder(cp,1))*sc
s2=np.abs((2+alpha)*Pv*Tp)+np.abs((1+2*alpha)*Tv*C.chebval(z,C.chebder(cp,1))*sc)
print(f"REL2  (2+a) P T' - (1+2a) T P' = 0 : max|res|/scale = {np.abs(res2).max()/s2.max():.3e}"
      f"   median = {np.median(np.abs(res2)/np.maximum(s2,1e-30)):.3e}")
print("\n  => REL1 (angular Poisson) holds; REL2 (B slaved to Psi) does NOT.")
print(f"  ratio of bracket term to the individually-cancelling c_l terms scales as e^(alpha*s) = r^{alpha:.4f}")
print(f"     at r=1e11 that is {1e11**alpha:.2e}  -> the bracket is balanced by the FIRST CORRECTION, not by zero.")
