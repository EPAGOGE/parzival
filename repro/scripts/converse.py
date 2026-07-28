import numpy as np, warnings
from scipy.optimize import least_squares
warnings.filterwarnings('ignore')
np.random.seed(11)
C0, TS0, RHO0 = 1.23, 1.7135, 2.60
t = np.linspace(0.8, 1.53, 13); TM = t.max()
d0 = C0*(TS0-t)**RHO0

def fitpow(t, d):
    tm = t.max()
    f = lambda p: np.log(np.exp(p[0])*(tm+np.exp(p[1])-t)**np.exp(p[2]))-np.log(d)
    best = None
    for s in np.log([.005,.02,.05,.18,.5,2.,10.]):
        for r in np.log([.7,1.,1.5,2.6,4.,8.]):
            try:
                sol = least_squares(f, [np.log(d[0]/(tm+np.exp(s)-t[0])**np.exp(r)), s, r],
                                    method='lm', max_nfev=40000)
                if best is None or sol.cost < best.cost: best = sol
            except Exception: pass
    c, s, r = best.x
    return np.exp(c), tm+np.exp(s), np.exp(r), 2*best.cost

def fitstretch(t, d):
    f = lambda p: np.log(np.exp(p[0])*np.exp(-np.exp(p[1])*t**np.exp(p[2])))-np.log(d)
    best = None
    for a in [.5,1.,3.,10.,30.]:
        for b in [1.,2.,4.,8.,16.]:
            try:
                sol = least_squares(f, [np.log(d[0])+a*t[0]**b, np.log(a), np.log(b)],
                                    method='lm', max_nfev=40000)
                if best is None or sol.cost < best.cost: best = sol
            except Exception: pass
    return np.exp(best.x[0]), np.exp(best.x[1]), np.exp(best.x[2]), 2*best.cost

print("="*78)
print("CONVERSE TEST. Ground truth = STRETCHED EXPONENTIAL, delta = A exp(-a t^b).")
print("This model NEVER reaches zero: there is NO finite-time singularity, ever.")
print("Fit it with the user's 3-param power law and see what T*, rho come out.")
print("="*78)
A, a, b, _ = fitstretch(t, d0)
print(f"  best stretched-exp mimic of the user's law: A={A:.4g}, a={a:.4g}, b={b:.4g}")
for sig in [0.0, 0.01, 0.03]:
    TT, RR, SS = [], [], []
    reps = 1 if sig == 0 else 60
    for _ in range(reps):
        d = A*np.exp(-a*t**b)
        if sig: d = d*np.exp(sig*np.random.randn(len(t)))
        Cf, Tf, rf, sp = fitpow(t, d)
        _, _, _, ss = fitstretch(t, d)
        if Tf < 50: TT.append(Tf); RR.append(rf); SS.append(sp/max(ss, 1e-30))
    print(f"  noise={sig*100:4.1f}%: power-law fit to NON-SINGULAR data gives "
          f"T*={np.median(TT):.4f}, rho={np.median(RR):.3f}")
    print(f"              and it fits {1/np.median(SS):.2f}x "
          f"{'BETTER' if np.median(SS)<1 else 'worse'} than the TRUE (stretched-exp) model")

print()
print("="*78)
print("EFFECTIVE SAMPLE SIZE. Residuals of a smooth 13-point curve are")
print("autocorrelated; dAIC scales with n_eff, not n.")
print("="*78)
for ratio, lab in [(115.4, 'vs 2-param pure exponential (the reported comparison)'),
                   (5.06, 'vs 3-param stretched exponential (equal complexity)')]:
    print(f"  SSE ratio {ratio:6.2f}  [{lab}]")
    for neff in [13, 8, 5, 3]:
        dA = neff*np.log(ratio) - (0 if ratio < 10 else 2)
        v = 'decisive' if dA > 10 else ('positive' if dA > 2 else 'NOT decisive')
        print(f"      n_eff={neff:2d}: dAIC={dA:6.1f}  -> {v}")
