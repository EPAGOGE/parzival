import numpy as np, scipy.io as sio
from scipy.interpolate import RegularGridInterpolator as RGI
d = sio.loadmat('/Users/epagogellc/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat',
                struct_as_record=False, squeeze_me=True)
solu, Mesh = d['solu'], d['Mesh']
x1 = np.asarray(Mesh.x[0],float); x2 = np.asarray(Mesh.x[1],float)
def F(n):
    a = getattr(solu,n)
    while not (isinstance(a,np.ndarray) and a.dtype!=object and a.ndim==2): a=a[0]
    return np.asarray(a,float)
w,th,u1,u2 = F('w'),F('th'),F('u1'),F('u2')
al = float(solu.al); alpha = -al
I = {n:RGI((x1,x2),A,bounds_error=False,fill_value=None) for n,A in
     [('w',w),('th',th),('u1',u1),('u2',u2)]}
for R in (1e9, 1e11, 1e13):
    b = np.linspace(1e-4, np.pi/2-1e-4, 4001)
    P_ = np.stack([R*np.cos(b), R*np.sin(b)],axis=-1)
    W,TH,U1,U2 = (I[n](P_) for n in ('w','th','u1','u2'))
    ub = -U1*np.sin(b) + U2*np.cos(b)
    G = W  * R**(al)            # Om ~ r^alpha
    T = TH * R**(-(1+2*alpha))  # B  ~ r^(1+2alpha)
    P = ub / ((2+alpha)*R**(1+alpha))   # u_beta = Psi_r = (2+a) r^(1+a) P
    db=b[1]-b[0]
    Pp=np.gradient(P,db); Tp=np.gradient(T,db); Gp=np.gradient(G,db); Ppp=np.gradient(Pp,db)
    # relation 1: P'' + (2+a)^2 P + G = 0
    r1 = Ppp + (2+alpha)**2*P + G
    # relation 2 (THE PREDICTION): (2+a) P T' - (1+2a) T P' = 0
    r2 = (2+alpha)*P*Tp - (1+2*alpha)*T*Pp
    sc2 = np.abs((2+alpha)*P*Tp) + np.abs((1+2*alpha)*T*Pp)
    # relation 3
    r3 = (2+alpha)*P*Gp - alpha*G*Pp - ((1+2*alpha)*np.cos(b)*T - np.sin(b)*Tp)
    sc3 = np.abs((2+alpha)*P*Gp)+np.abs(alpha*G*Pp)+np.abs((1+2*alpha)*np.cos(b)*T)+np.abs(np.sin(b)*Tp)
    m=slice(200,-200)
    print(f"r=10^{np.log10(R):.0f}  G(0)={G[0]:.5f} T(0)={T[0]:.4f} P(0)={P[0]:.4f}")
    print(f"   rel1 (Psi-Poisson)  max|res|/scale = {np.max(np.abs(r1[m]))/np.max(np.abs(G[m])):.3e}")
    print(f"   rel2 (B transport)  max|res|/scale = {np.max(np.abs(r2[m]))/np.max(sc2[m]):.3e}   <-- PREDICTION")
    print(f"   rel3 (Om transport) max|res|/scale = {np.max(np.abs(r3[m]))/np.max(sc3[m]):.3e}")
    # direct test of T = C P^kappa
    kap=(1+2*alpha)/(2+alpha)
    good=(P>1e-12)&(T>1e-12)
    lp,lt=np.log(P[good]),np.log(T[good])
    k_fit=np.polyfit(lp,lt,1)[0]
    print(f"   fitted d(lnT)/d(lnP) = {k_fit:.4f}   predicted kappa = {kap:.6f}")
    # near-axis exponents
    c=np.cos(b)
    for nm,A in (('G',G),('T',T),('P',P)):
        s=(b>1.50)&(b<np.pi/2-1e-3)&(np.abs(A)>0)
        print(f"     {nm} ~ cos(beta)^{np.polyfit(np.log(c[s]),np.log(np.abs(A[s])),1)[0]:.3f}")
