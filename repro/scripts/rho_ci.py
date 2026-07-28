import numpy as np
rng = np.random.default_rng(1)
C0,T0,R0 = 1.23,1.7135,2.60
t = np.linspace(0.80,1.53,40)
d0 = C0*(T0-t)**R0
Tgrid = np.arange(1.535,4.001,0.0005)

def fit(tt, yy):
    ly=np.log(yy); best=(np.inf,)
    for T in Tgrid:
        if T<=tt.max()+1e-9: continue
        x=np.log(T-tt); A=np.vstack([np.ones_like(x),x]).T
        c,*_=np.linalg.lstsq(A,ly,rcond=None); sse=float(((ly-A@c)**2).sum())
        if sse<best[0]: best=(sse,T,float(np.exp(c[0])),float(c[1]))
    return best  # sse,T*,C,rho

half=len(t)//2
print("=== TEST 2: calibrate scatter sigma so half-window T* spread matches reported 1.63 vs 1.83 (|dT*|=0.20) ===")
print(f"{'sigma_iid':>10} {'med|dT*_halves|':>16} {'rho 5-95%':>22} {'T* 5-95%':>20}")
for sig in (0.005,0.01,0.02,0.03,0.05,0.08):
    dT=[];rr=[];TT=[]
    for _ in range(400):
        y=d0*np.exp(rng.normal(0,sig,d0.shape))
        a=fit(t[:half],y[:half]); b=fit(t[half:],y[half:]); f=fit(t,y)
        dT.append(abs(a[1]-b[1])); rr.append(f[3]); TT.append(f[1])
    print(f"{sig:>10.3f} {np.median(dT):>16.3f} {np.percentile(rr,5):>10.2f} -{np.percentile(rr,95):>9.2f} {np.percentile(TT,5):>9.3f} -{np.percentile(TT,95):>8.3f}")

print()
print("=== TEST 3: CORRELATED error (realistic: k-window choice / filter bias drifts smoothly in t) ===")
print(f"{'amp':>8} {'med|dT*_halves|':>16} {'rho 5-95%':>22}")
for amp in (0.01,0.02,0.04,0.06):
    dT=[];rr=[]
    for _ in range(400):
        # smooth low-order drift in ln delta: 2 correlated modes
        a1,a2=rng.normal(0,amp,2)
        s=(t-t.mean())/(t.ptp()/2)
        y=d0*np.exp(a1*s + a2*(2*s**2-1) + rng.normal(0,amp/3,t.shape))
        A=fit(t[:half],y[:half]); B=fit(t[half:],y[half:]); f=fit(t,y)
        dT.append(abs(A[1]-B[1])); rr.append(f[3])
    print(f"{amp:>8.3f} {np.median(dT):>16.3f} {np.percentile(rr,5):>10.2f} -{np.percentile(rr,95):>9.2f}")
