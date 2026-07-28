import numpy as np
from scipy.optimize import least_squares
np.random.seed(0)

t = np.linspace(0.8, 1.53, 30)          # user's fit window

def fit_pow(t, d, log=True):
    def res(p):
        C, Ts, rho = p
        if Ts <= t.max()+1e-6 or C<=0 or rho<=0: return np.full_like(d, 1e3)
        m = C*(Ts-t)**rho
        return (np.log(m)-np.log(d)) if log else (m-d)
    best=None
    for Ts0 in [1.55,1.6,1.7,1.9,2.5,4.0]:
        for r0 in [0.8,1.5,2.5,4.0]:
            try:
                s=least_squares(res,[d[0]/ (Ts0-t[0])**r0, Ts0, r0],method='lm',maxfev=20000)
                if best is None or s.cost<best.cost: best=s
            except Exception: pass
    return best.x, 2*best.cost

def fit_exp(t, d, log=True):
    def res(p):
        A,tau=p
        if A<=0 or tau<=0: return np.full_like(d,1e3)
        m=A*np.exp(-t/tau)
        return (np.log(m)-np.log(d)) if log else (m-d)
    best=None
    for tau0 in [0.1,0.26,0.5,1.0]:
        s=least_squares(res,[d[0]*np.exp(t[0]/tau0),tau0],method='lm',maxfev=20000)
        if best is None or s.cost<best.cost: best=s
    return best.x, 2*best.cost

print("="*74)
print("TEST 3: GROUND TRUTH = PURE EXPONENTIAL (NO finite-time singularity).")
print("        delta = A exp(-t/tau), tuned to span delta(0.8)=0.25 -> delta(1.53)=0.015")
print("="*74)
tau = 0.73/np.log(0.25/0.015); A = 0.25*np.exp(0.8/tau)
d = A*np.exp(-t/tau)
for lab,lg in [("log-residual",True),("linear-residual",False)]:
    (C,Ts,rho),sp = fit_pow(t,d,lg); (_,_),se = fit_exp(t,d,lg)
    print(f"\n  [{lab}] fitting a POWER LAW to exactly-exponential data:")
    print(f"     recovered T* = {Ts:.4f}   rho = {rho:.3f}   C = {C:.4f}")
    print(f"     SSE(power)={sp:.3e}   SSE(exp)={se:.3e}   ratio = {sp/se:.2e}")
    print(f"     -> power law fits {'BETTER' if sp<se else 'worse'} than the TRUE model"
          if sp<se else f"     -> exponential correctly wins")

print("\n"+"="*74)
print("TEST 4: GROUND TRUTH = user's power law (T*=1.7135, rho=2.60, C=1.23).")
print("        How well does a 2-param EXPONENTIAL mimic it on this window?")
print("="*74)
d2 = 1.23*(1.7135-t)**2.60
for lab,lg in [("log-residual",True),("linear-residual",False)]:
    (C,Ts,rho),sp = fit_pow(t,d2,lg); (Ae,tau_e),se = fit_exp(t,d2,lg)
    print(f"  [{lab}] SSE(power)={sp:.3e}  SSE(exp)={se:.3e}  ratio={se/sp:.2e}"
          f"   (exp is {se/sp:.1e}x worse)")

print("\n"+"="*74)
print("TEST 5: DEGENERACY. True power law + 2% multiplicative noise on delta.")
print("        Spread of recovered T* and rho over 300 noise realisations.")
print("="*74)
for noise in [0.01,0.02,0.05]:
    Ts_s=[];rho_s=[]
    for _ in range(300):
        dn = d2*(1+noise*np.random.randn(len(t)))
        try:
            (C,Ts,rho),_=fit_pow(t,dn,True)
            if 1.53<Ts<10: Ts_s.append(Ts); rho_s.append(rho)
        except Exception: pass
    Ts_s=np.array(Ts_s);rho_s=np.array(rho_s)
    print(f"  noise={noise*100:.0f}%:  T* = {np.median(Ts_s):.3f}  "
          f"[16-84%: {np.percentile(Ts_s,16):.3f}, {np.percentile(Ts_s,84):.3f}]   "
          f"rho = {np.median(rho_s):.2f} [{np.percentile(rho_s,16):.2f}, {np.percentile(rho_s,84):.2f}]")
