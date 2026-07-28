import numpy as np
rng=np.random.default_rng(1)
C0,T0,R0=1.23,1.7135,2.60
t=np.linspace(0.80,1.53,40); d0=C0*(T0-t)**R0
half=len(t)//2

def make(tt,Tgrid):
    X=np.log(Tgrid[:,None]-tt[None,:])          # (nT,n)
    Xc=X-X.mean(1,keepdims=True)
    Sxx=(Xc**2).sum(1)
    return X,Xc,Sxx

def fitmany(tt,Y,pre):
    """Y:(nrep,n) -> best (T*,rho) per rep via profile over T grid."""
    Tgrid,X,Xc,Sxx=pre
    L=np.log(Y); Lc=L-L.mean(1,keepdims=True)
    b=(Xc@Lc.T)/Sxx[:,None]                      # (nT,nrep) slope=rho
    sse=((Lc**2).sum(1)[None,:]-b**2*Sxx[:,None])# (nT,nrep)
    i=np.argmin(sse,0)
    return Tgrid[i], b[i,np.arange(Y.shape[0])]

def pre_for(tt):
    Tg=np.arange(tt.max()+0.005,4.0,0.001)
    X,Xc,Sxx=make(tt,Tg); return (Tg,X,Xc,Sxx)

P,Pa,Pb=pre_for(t),pre_for(t[:half]),pre_for(t[half:])
NR=3000
print("=== TEST 2: iid scatter -> match reported half-window spread |dT*| ~ 0.20 (1.63 vs 1.83) ===")
print(f"{'sigma':>7} {'med|dT*|':>9} {'rho 5-95%':>20} {'T* 5-95%':>20}")
for sig in (0.005,0.01,0.02,0.03,0.05,0.08,0.12):
    Y=d0*np.exp(rng.normal(0,sig,(NR,len(t))))
    Ta,_=fitmany(t[:half],Y[:,:half],Pa); Tb,_=fitmany(t[half:],Y[:,half:],Pb)
    Tf,Rf=fitmany(t,Y,P)
    print(f"{sig:>7.3f} {np.median(abs(Ta-Tb)):>9.3f} {np.percentile(Rf,5):>9.2f} -{np.percentile(Rf,95):>8.2f} {np.percentile(Tf,5):>9.3f} -{np.percentile(Tf,95):>8.3f}")

print("\n=== TEST 3: smooth/correlated systematic (k-window & filter bias drift in t) ===")
print(f"{'amp':>7} {'med|dT*|':>9} {'rho 5-95%':>20} {'T* 5-95%':>20}")
s=(t-t.mean())/((t.max()-t.min())/2)
for amp in (0.01,0.02,0.03,0.05,0.08):
    a=rng.normal(0,amp,(NR,2))
    Y=d0*np.exp(a[:,[0]]*s[None,:]+a[:,[1]]*(2*s**2-1)[None,:]+rng.normal(0,amp/3,(NR,len(t))))
    Ta,_=fitmany(t[:half],Y[:,:half],Pa); Tb,_=fitmany(t[half:],Y[:,half:],Pb)
    Tf,Rf=fitmany(t,Y,P)
    print(f"{amp:>7.3f} {np.median(abs(Ta-Tb)):>9.3f} {np.percentile(Rf,5):>9.2f} -{np.percentile(Rf,95):>8.2f} {np.percentile(Tf,5):>9.3f} -{np.percentile(Tf,95):>8.3f}")
