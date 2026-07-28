import numpy as np, scipy.io as sio
d = sio.loadmat('/Users/epagogellc/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat',
                struct_as_record=False, squeeze_me=True)
solu,Mesh = d['solu'],d['Mesh']
x1=np.asarray(Mesh.x[0],float); x2=np.asarray(Mesh.x[1],float)
def F(n):
    a=getattr(solu,n)
    while not(isinstance(a,np.ndarray) and a.dtype!=object and a.ndim==2): a=a[0]
    return np.asarray(a,float)
w,th,u1,u2=F('w'),F('th'),F('u1'),F('u2')
al=float(solu.al); alpha=-al
X1,X2=np.meshgrid(x1,x2,indexing='ij')
R=np.hypot(X1,X2); B=np.arctan2(X2,X1)
sel=(R>1e9)&(R<1e13)
print("grid points in annulus 1e9..1e13:", sel.sum())
b=B[sel]; r=R[sel]
G = w[sel]*r**al
T = th[sel]*r**(-(1+2*alpha))
ub = -u1[sel]*np.sin(b)+u2[sel]*np.cos(b)
P  = ub/((2+alpha)*r**(1+alpha))
o=np.argsort(b); b,G,T,P=b[o],G[o],T[o],P[o]
# COLLAPSE QUALITY: bin in beta, measure spread within each bin
nb=60; edges=np.linspace(0,np.pi/2,nb+1); idx=np.digitize(b,edges)-1
sp={'G':[],'T':[],'P':[]}
for k in range(nb):
    m=idx==k
    if m.sum()<8: continue
    for nm,A in (('G',G),('T',T),('P',P)):
        a_=A[m]; mu=np.mean(a_)
        if abs(mu)>1e-9*max(np.abs(A).max(),1e-30): sp[nm].append(np.std(a_)/abs(mu))
for nm in ('G','T','P'):
    v=np.array(sp[nm]); print(f"  {nm}: self-similar collapse, median rel spread within beta-bin = {np.median(v):.2e}, 90th pct = {np.percentile(v,90):.2e}  ({len(v)} bins)")
print("  G(beta->0) =",G[b<0.02].mean() if (b<0.02).sum() else None,
      " T(beta->0) =",T[b<0.02].mean() if (b<0.02).sum() else None,
      " P(beta->0) =",P[b<0.02].mean() if (b<0.02).sum() else None)
# near-axis exponents from the annulus (grid points, no interpolation)
c=np.cos(b)
for nm,A in (('G',G),('T',T),('P',P)):
    s=(b>1.45)&(b<np.pi/2-1e-6)&(np.abs(A)>0)
    if s.sum()>20:
        print(f"  {nm} ~ cos(beta)^{np.polyfit(np.log(c[s]),np.log(np.abs(A[s])),1)[0]:.3f}   (n={s.sum()})")
# TEST the B relation without derivatives: T = C * P^kappa  <=>  ln T - kappa ln P = const
kap=(1+2*alpha)/(2+alpha)
g=(P>1e-9)&(T>1e-9)
q=np.log(T[g])-kap*np.log(P[g])
print(f"\n  PREDICTION T = C*P^kappa, kappa={kap:.6f}")
print(f"    ln T - kappa ln P  should be CONSTANT: spread = {q.std():.4f}, range = {q.max()-q.min():.4f} (in log units)")
print(f"    => T/P^kappa varies by a factor {np.exp(q.max()-q.min()):.2f} across beta")
kfit=np.polyfit(np.log(P[g]),np.log(T[g]),1)[0]
print(f"    measured d(lnT)/d(lnP) = {kfit:.4f}  vs predicted {kap:.4f}")
