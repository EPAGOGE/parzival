import numpy as np, warnings
from scipy.optimize import least_squares
warnings.filterwarnings('ignore')
np.random.seed(7)

C0, TS0, RHO0 = 1.23, 1.7135, 2.60
t = np.linspace(0.8, 1.53, 13); TM = t.max()
d0 = C0*(TS0-t)**RHO0

def fit(model, t, d):
    tm = t.max()
    if model == 'pow':                      # 3 params, SINGULAR
        f = lambda p: np.log(np.exp(p[0])*(tm+np.exp(p[1])-t)**np.exp(p[2]))-np.log(d)
        seeds = [[np.log(d[0]/(tm+np.exp(s)-t[0])**np.exp(r)), s, r]
                 for s in np.log([.005,.02,.05,.18,.5,2.,10.]) for r in np.log([.7,1.,1.5,2.6,4.,8.])]
    elif model == 'exp':                    # 2 params, NON-singular
        f = lambda p: np.log(np.exp(p[0])*np.exp(-t/np.exp(p[1])))-np.log(d)
        seeds = [[np.log(d[0])+t[0]/x, np.log(x)] for x in [.05,.1,.26,.5,1.,3.]]
    elif model == 'floor':                  # 3 params, NON-singular: delta -> delta_inf > 0
        f = lambda p: np.log(np.exp(p[0])+np.exp(p[1])*np.exp(-t*np.exp(p[2])))-np.log(d)
        seeds = [[np.log(a), np.log(d[0])+t[0]*np.exp(c), c] for a in [1e-5,1e-4,1e-3,5e-3,1e-2]
                 for c in np.log([.5,1.,2.,4.,8.])]
    elif model == 'stretch':                # 3 params, NON-singular: stretched exponential
        f = lambda p: np.log(np.exp(p[0])*np.exp(-np.exp(p[1])*t**np.exp(p[2])))-np.log(d)
        seeds = [[np.log(d[0])+a*t[0]**b, np.log(a), np.log(b)]
                 for a in [.5,1.,3.,10.] for b in [1.,2.,4.,8.]]
    best = None
    for s0 in seeds:
        try:
            sol = least_squares(f, s0, method='lm', max_nfev=40000)
            if best is None or sol.cost < best.cost: best = sol
        except Exception: pass
    return 2*best.cost

print("="*78)
print("CALIBRATED TEST. Truth = the user's power law. Add lognormal scatter to")
print("delta until SSE(2-param exponential)/SSE(power law) matches the REPORTED 100x.")
print("Then ask where EQUALLY-COMPLEX (3-param) NON-SINGULAR models land.")
print("="*78)
print(f"{'sigma':>7} | {'exp2p/pow':>10} | {'floor3p/pow':>12} | {'stretch3p/pow':>14}")
print("-"*78)
rows = {}
for sig in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
    r_e, r_f, r_s = [], [], []
    for _ in range(60):
        d = d0*np.exp(sig*np.random.randn(len(t)))
        sp = fit('pow', t, d)
        if sp <= 0: continue
        r_e.append(fit('exp', t, d)/sp)
        r_f.append(fit('floor', t, d)/sp)
        r_s.append(fit('stretch', t, d)/sp)
    rows[sig] = (np.median(r_e), np.median(r_f), np.median(r_s))
    print(f"{sig*100:6.1f}% | {np.median(r_e):10.1f} | {np.median(r_f):12.2f} | {np.median(r_s):14.2f}")

print()
print("READ-OFF: find sigma where exp2p/pow ~ 100 (the reported value), then read")
print("across to the 3-parameter non-singular models.")
best_sig = min(rows, key=lambda s: abs(np.log(rows[s][0]/100.0)))
e, f, s_ = rows[best_sig]
print(f"  sigma ~ {best_sig*100:.1f}%  ->  exp(2p)/pow = {e:.0f}x   "
      f"floor(3p)/pow = {f:.2f}x   stretched-exp(3p)/pow = {s_:.2f}x")
print()
print(f"  So of the reported {e:.0f}x, a factor ~{e/f:.0f}x is bought purely by giving the")
print(f"  NON-SINGULAR family a third parameter. The power law's genuine advantage")
print(f"  over an equally-complex model with NO finite-time singularity is ~{f:.1f}x.")
print()
print("  n=13, so dAIC for power law vs the 3-param non-singular null (equal k):")
print(f"     dAIC = 13*ln({f:.2f}) = {13*np.log(f):.1f}   "
      f"({'DECISIVE' if 13*np.log(f)>10 else 'WEAK/marginal - NOT decisive'})")
