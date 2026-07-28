import numpy as np
from scipy.optimize import least_squares
np.random.seed(0)
t = np.linspace(0.8, 1.53, 30)
TM = t.max()

def fit_pow(t,d,log=True):
    # T* = TM + exp(s) ; C = exp(c) ; rho = exp(r)  -> unconstrained
    def res(p):
        c,s,r = p
        m = np.exp(c)*(TM+np.exp(s)-t)**np.exp(r)
        return np.log(m)-np.log(d) if log else m-d
    best=None
    for s0 in np.log([0.005,0.02,0.05,0.18,0.5,2.0,10.0]):
        for r0 in np.log([0.7,1.0,1.5,2.6,4.0,8.0]):
            c0 = np.log(d[0]/ (TM+np.exp(s0)-t[0])**np.exp(r0))
            try:
                sol=least_squares(res,[c0,s0,r0],method='lm',max_nfev=40000)
                if best is None or sol.cost<best.cost: best=sol
            except Exception: pass
    c,s,r=best.x
    return (np.exp(c), TM+np.exp(s), np.exp(r)), 2*best.cost

def fit_exp(t,d,log=True):
    def res(p):
        a,l = p
        m=np.exp(a)*np.exp(-t/np.exp(l))
        return np.log(m)-np.log(d) if log else m-d
    best=None
    for tau0 in [0.05,0.1,0.26,0.5,1.0,3.0]:
        sol=least_squares(res,[np.log(d[0])+t[0]/tau0,np.log(tau0)],method='lm',max_nfev=40000)
        if best is None or sol.cost<best.cost: best=sol
    return (np.exp(best.x[0]),np.exp(best.x[1])), 2*best.cost

print("="*76)
print("TEST 3: GROUND TRUTH = PURE EXPONENTIAL decay (NO singularity at any finite T).")
print("        delta(0.8)=0.25 -> delta(1.53)=0.015, exactly A*exp(-t/tau). No noise.")
print("="*76)
tau=0.73/np.log(0.25/0.015); A=0.25*np.exp(0.8/tau)
d=A*np.exp(-t/tau)
print(f"  true tau = {tau:.4f}")
for lab,lg in [("log-residual",True),("LINEAR-residual",False)]:
    (C,Ts,rho),sp=fit_pow(t,d,lg); _,se=fit_exp(t,d,lg)
    print(f"\n  [{lab}]  power-law fit to exponential data:")
    print(f"      recovered  T* = {Ts:.4f}    rho = {rho:.3f}    C = {C:.4g}")
    print(f"      SSE(power)={sp:.3e}   SSE(exp,TRUE model)={se:.3e}")
    if lg: print(f"      NOTE: exp is the true model, so it should win; it does by {sp/se:.1e}x" if se<sp else "")

print("\n"+"="*76)
print("TEST 3b: same, but exponential ALSO given a mild curvature (delta ~ exp(-t/tau) *")
print("         (1+eps t)) i.e. a slightly-accelerating non-singular decay + 1% noise")
print("="*76)
for eps in [0.0,-0.15,-0.3]:
    dd = A*np.exp(-t/tau)*np.exp(eps*(t-0.8)**2/0.1)
    dd = dd*(1+0.01*np.random.randn(len(t)))
    (C,Ts,rho),sp=fit_pow(t,dd,True); _,se=fit_exp(t,dd,True)
    print(f"  curvature eps={eps:+.2f}: power-law gives T*={Ts:.3f}, rho={rho:.2f} | "
          f"SSE_pow/SSE_exp = {sp/se:.2e}  ({'POWER WINS' if sp<se else 'exp wins'})")

print("\n"+"="*76)
print("TEST 4: TRUE = user's law (T*=1.7135, rho=2.60, C=1.23). Exp fit quality?")
print("="*76)
d2=1.23*(1.7135-t)**2.60
for lab,lg in [("log-residual",True),("LINEAR-residual",False)]:
    (C,Ts,rho),sp=fit_pow(t,d2,lg); _,se=fit_exp(t,d2,lg)
    print(f"  [{lab}] SSE(exp)/SSE(power) = {se/sp:.2e}   "
          f"(user reports ~100x; recovered T*={Ts:.4f} rho={rho:.3f})")

print("\n"+"="*76)
print("TEST 5: DEGENERACY of (C,T*,rho). True law + multiplicative noise on delta.")
print("="*76)
for noise in [0.01,0.02,0.05]:
    TT=[];RR=[]
    for _ in range(200):
        dn=d2*(1+noise*np.random.randn(len(t)))
        if np.any(dn<=0): continue
        (C,Ts,rho),_=fit_pow(t,dn,True)
        if Ts<20: TT.append(Ts);RR.append(rho)
    TT=np.array(TT);RR=np.array(RR)
    print(f"  noise={noise*100:>2.0f}%: T*={np.median(TT):.3f} [16-84%: {np.percentile(TT,16):.3f},{np.percentile(TT,84):.3f}]"
          f"  rho={np.median(RR):.2f} [{np.percentile(RR,16):.2f},{np.percentile(RR,84):.2f}]")

print("\n"+"="*76)
print("TEST 6: SYSTEMATIC. True law + prefactor bias +n*b(t) that shrinks as the")
print("        k-window drifts to higher k with time (bias 0.008 -> 0.003).")
print("="*76)
for nprefac in [1.0, 4/3, 3.0]:
    bias = np.linspace(0.0079,0.0032,len(t))*nprefac/(4/3)
    dbias = d2 + bias
    (C,Ts,rho),_=fit_pow(t,dbias,True)
    print(f"  n={nprefac:.2f}: measured T*={Ts:.4f} (true 1.7135, err {100*(Ts-1.7135)/1.7135:+.1f}%), "
          f"rho={rho:.2f} (true 2.60, err {100*(rho-2.6)/2.6:+.1f}%)")
