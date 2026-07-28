import numpy as np, itertools
t=np.linspace(0.80,1.53,40); half=len(t)//2

def pre(tt):
    Tg=np.arange(tt.max()+0.005,6.0,0.0005)
    X=np.log(Tg[:,None]-tt[None,:]); Xc=X-X.mean(1,keepdims=True)
    return Tg,Xc,(Xc**2).sum(1)
P,Pa,Pb=pre(t),pre(t[:half]),pre(t[half:])
def fit(tt,y,pr):
    Tg,Xc,Sxx=pr; L=np.log(y); Lc=L-L.mean()
    b=(Xc@Lc)/Sxx; sse=(Lc**2).sum()-b**2*Sxx
    i=int(np.argmin(sse))
    lnC=L.mean()-b[i]*(Xc[i]+np.log(Tg[i]-tt).mean()).mean()
    return Tg[i],b[i],float(np.exp(np.mean(L-b[i]*np.log(Tg[i]-tt)))),sse[i]

# truth: delta = C*(Tt-t)^rho_true * (1 + a*(Tt-t)^s)   <-- standard correction-to-scaling
best=None
for rho_true in (1.0,1.5):
    for Tt in np.arange(1.56,1.86,0.01):
        for a in np.arange(0.2,12.01,0.2):
            for s in np.arange(0.5,6.01,0.25):
                x=Tt-t
                y=(x**rho_true)*(1+a*x**s)
                Tf,rf,Cf,sse=fit(t,y,P)
                if abs(rf-2.60)>0.06 or abs(Tf-1.7135)>0.02: continue
                Ta,_,_,_=fit(t[:half],y[:half],Pa); Tb,_,_,_=fit(t[half:],y[half:],Pb)
                score=abs(Ta-1.63)+abs(Tb-1.83)
                if best is None or score<best[0]:
                    best=(score,rho_true,Tt,a,s,Tf,rf,Ta,Tb,sse,y)
    if best:
        sc,rt,Tt,a,s,Tf,rf,Ta,Tb,sse,y=best
        # relative residual of the pure-power-law fit to this data
        resid=np.abs(np.exp(np.log(y)-(np.log(np.mean(np.exp(np.log(y)-rf*np.log(Tf-t))))+rf*np.log(Tf-t)))-1).max()
        print(f"rho_true={rt}: TRUE T*={Tt:.3f}, a={a:.1f}, s={s:.2f}")
        print(f"   -> pure power-law fit returns rho_eff={rf:.3f}, T*_fit={Tf:.4f}")
        print(f"   -> half-window T*: {Ta:.3f} / {Tb:.3f}   (reported: 1.63 / 1.83)")
        print(f"   -> max relative residual of the power-law fit: {100*resid:.2f}%  (looks like a clean fit)\n")
        best=None
