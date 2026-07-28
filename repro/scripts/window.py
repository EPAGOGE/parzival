import numpy as np
rng=np.random.default_rng(7)
C0,T0,R0=1.23,1.7135,2.60
def tmax_for(dmin): return T0-(dmin/C0)**(1/R0)
print("=== fit window vs resolution criterion (N=1024, 2/3 dealias) ===")
for dmin,lab in ((0.00498,'c=1.7  (claimed, <2 e-folds)'),(0.0270,'c=9.2  (spectrum to 1e-4)'),(0.1078,'c=36.8 (to roundoff, std practice)')):
    tm=tmax_for(dmin); dec=np.log10((T0-0.8)/(T0-tm))
    print(f"  delta_min={dmin:.5f} {lab:34s} -> usable t<{tm:.3f}, window = {dec:.2f} decades in (T*-t)")
print("  (the fit's own cutoff delta>0.015 lies BELOW the 1e-4 floor 0.0270 -> late points unresolved)\n")

def pre(tt):
    Tg=np.arange(tt.max()+0.005,6.0,0.001)
    X=np.log(Tg[:,None]-tt[None,:]); Xc=X-X.mean(1,keepdims=True)
    return Tg,Xc,(Xc**2).sum(1)
def fitmany(Y,pr):
    Tg,Xc,Sxx=pr; L=np.log(Y); Lc=L-L.mean(1,keepdims=True)
    b=(Xc@Lc.T)/Sxx[:,None]; sse=(Lc**2).sum(1)[None,:]-b**2*Sxx[:,None]
    i=np.argmin(sse,0); return Tg[i],b[i,np.arange(Y.shape[0])]

print("=== rho 90% CI vs window end, at the scatter (sigma=0.05) implied by the reported T* spread ===")
print(f"{'t_end':>7} {'decades':>8} {'rho 5-95%':>20} {'rho=1.5 excluded?':>19}")
for tend in (1.53,1.48,1.45,1.40,1.32):
    t=np.linspace(0.80,tend,40); d0=C0*(T0-t)**R0; P=pre(t)
    Y=d0*np.exp(rng.normal(0,0.05,(3000,len(t))))
    Tf,Rf=fitmany(Y,P)
    lo,hi=np.percentile(Rf,5),np.percentile(Rf,95)
    dec=np.log10((T0-0.8)/(T0-tend))
    print(f"{tend:>7.2f} {dec:>8.2f} {lo:>9.2f} -{hi:>9.2f} {'yes' if lo>1.5 else 'NO':>19}")
