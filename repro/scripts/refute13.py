import numpy as np
from scipy.optimize import least_squares
from scipy import stats
np.random.seed(1)

# ---- the user's reported fit, sampled at n=13 over the stated window ----
C0, TS0, RHO0 = 1.23, 1.7135, 2.60
t = np.linspace(0.8, 1.53, 13)
TM = t.max()
d_true = C0 * (TS0 - t) ** RHO0

RHO_RIG = 2.9205600      # Chen-Hou Part I: |c_l/c_omega|, rigorous asymptotic
RHO_LH  = 2.91           # Luo-Hou PNAS gamma_l


def sse_pow(t, d, log=True, fix_rho=None):
    """delta = C (T*-t)^rho, unconstrained via T* = tmax + exp(s)."""
    tm = t.max()
    def res(p):
        if fix_rho is None:
            c, s, r = p; rho = np.exp(r)
        else:
            c, s = p; rho = fix_rho
        m = np.exp(c) * (tm + np.exp(s) - t) ** rho
        return np.log(m) - np.log(d) if log else m - d
    best = None
    for s0 in np.log([0.005, 0.02, 0.05, 0.18, 0.5, 2.0, 10.0]):
        for r0 in np.log([0.7, 1.0, 1.5, 2.6, 4.0, 8.0]):
            rho0 = np.exp(r0) if fix_rho is None else fix_rho
            c0 = np.log(d[0] / (tm + np.exp(s0) - t[0]) ** rho0)
            x0 = [c0, s0, r0] if fix_rho is None else [c0, s0]
            try:
                sol = least_squares(res, x0, method='lm', max_nfev=40000)
                if best is None or sol.cost < best.cost: best = sol
            except Exception: pass
    if fix_rho is None:
        c, s, r = best.x; rho = np.exp(r)
    else:
        c, s = best.x; rho = fix_rho
    return (np.exp(c), tm + np.exp(s), rho), 2 * best.cost


def sse_exp(t, d, log=True):
    """delta = A exp(-t/tau)  [2 params]"""
    def res(p):
        a, l = p
        m = np.exp(a) * np.exp(-t / np.exp(l))
        return np.log(m) - np.log(d) if log else m - d
    best = None
    for tau0 in [0.05, 0.1, 0.26, 0.5, 1.0, 3.0]:
        sol = least_squares(res, [np.log(d[0]) + t[0]/tau0, np.log(tau0)],
                            method='lm', max_nfev=40000)
        if best is None or sol.cost < best.cost: best = sol
    return best.x, 2 * best.cost


def sse_expoff(t, d, log=True):
    """delta = a + b exp(-c t)  [3 params - the FAIR null]"""
    def res(p):
        a, b, c = p
        m = a + np.exp(b) * np.exp(-t * np.exp(c))
        m = np.maximum(m, 1e-12)
        return np.log(m) - np.log(d) if log else m - d
    best = None
    for a0 in [0.0, 1e-4, 1e-3, 5e-3]:
        for c0 in np.log([0.5, 1.0, 2.0, 4.0, 8.0]):
            b0 = np.log(max(d[0], 1e-9)) + t[0]*np.exp(c0)
            try:
                sol = least_squares(res, [a0, b0, c0], method='lm', max_nfev=40000)
                if best is None or sol.cost < best.cost: best = sol
            except Exception: pass
    return best.x, 2 * best.cost


print("="*78)
print("A.  IS 100x SSE FOR 3-PARAM vs 2-PARAM 'JUST PARAMETER COUNT'?  (n=13)")
print("="*78)
n, R = 13, 100.0
F = ((R - 1.0) / 1.0) / (1.0 / (n - 3))
p = 1 - stats.f.cdf(F, 1, n - 3)
dAIC = n*np.log(R) - 2*1
dBIC = n*np.log(R) - np.log(n)*1
print(f"  SSE ratio R = {R:.0f}, n = {n}, dof 10 vs 11")
print(f"  F(1,10) = {F:.0f}   p = {p:.3e}")
print(f"  Delta AIC = {dAIC:.1f}   Delta BIC = {dBIC:.1f}   (both overwhelmingly favour power law)")
print("  -> Under iid errors ONE extra parameter CANNOT buy 100x. The stated")
print("     '3-param vs 2-param artifact' objection FAILS on its own terms.")
print()
print("  BUT: how many effectively-independent residuals are there? Required n_eff")
print("  for Delta AIC to fall below the conventional decision thresholds:")
for thr, lab in [(10, 'AIC=10 (strong)'), (2, 'AIC=2 (marginal)'), (0, 'AIC=0 (tie)')]:
    neff = (thr + 2) / np.log(R)
    print(f"     {lab:22s}  needs n_eff <= {neff:.1f}")

print()
print("="*78)
print("B.  THE FAIR TEST: 3-PARAM POWER LAW vs 3-PARAM EXPONENTIAL-PLUS-OFFSET")
print("    Ground truth = the user's own power law, EXACT (no noise).")
print("="*78)
for lab, lg in [("log-residual", True), ("linear-residual", False)]:
    _, s3 = sse_pow(t, d_true, lg)
    _, s2 = sse_exp(t, d_true, lg)
    _, s3o = sse_expoff(t, d_true, lg)
    print(f"  [{lab}]")
    print(f"     SSE(power,3p)      = {s3:.4e}")
    print(f"     SSE(pure exp,2p)   = {s2:.4e}   ratio vs power = {s2/max(s3,1e-30):.2e}")
    print(f"     SSE(exp+offset,3p) = {s3o:.4e}   ratio vs power = {s3o/max(s3,1e-30):.2e}")
    if s2 > 0:
        print(f"     -> fraction of the 2p->3p gain explained purely by adding a 3rd")
        print(f"        parameter to the EXPONENTIAL family: {100*(1-np.log(s3o/s2)/np.log(max(s3,1e-30)/s2)):.1f}%")

print()
print("="*78)
print("C.  SUB-WINDOW DRIFT: is T*=1.63 -> 1.83 noise, or misspecification?")
print("="*78)
lo, hi = t[:7], t[6:]
for noise in [0.005, 0.01, 0.02, 0.05]:
    dT = []
    for _ in range(400):
        dn = d_true * (1 + noise*np.random.randn(len(t)))
        if np.any(dn <= 0): continue
        try:
            (_, T1, _), _ = sse_pow(lo, dn[:7], True)
            (_, T2, _), _ = sse_pow(hi, dn[6:], True)
            if T1 < 50 and T2 < 50: dT.append(T2 - T1)
        except Exception: pass
    dT = np.array(dT)
    frac = np.mean(dT > 0)
    print(f"  noise={noise*100:4.1f}%: median(T2-T1)={np.median(dT):+.3f} "
          f"[16-84%: {np.percentile(dT,16):+.3f},{np.percentile(dT,84):+.3f}]  "
          f"P(T2>T1)={frac:.2f}  |observed +0.20| at {np.mean(np.abs(dT)>=0.20)*100:.0f}th pct")
print("  (If the true law is an exact power law, T2-T1 is centred on 0 with NO")
print("   preferred sign. The observed split is one-signed and LATE-BIASED.)")

print()
print("  Honest error bar from TWO sub-window values {1.63, 1.83}, Student-t df=1:")
vals = np.array([1.63, 1.83])
m, s = vals.mean(), vals.std(ddof=1)
tcrit = stats.t.ppf(0.975, 1)
print(f"     mean={m:.3f} s={s:.4f} SE={s/np.sqrt(2):.4f} t(0.975,1)={tcrit:.3f}")
print(f"     95% CI = [{m - tcrit*s/np.sqrt(2):.2f}, {m + tcrit*s/np.sqrt(2):.2f}]")
print("     -> the quoted +/-0.10 is the HALF-RANGE of 2 numbers reported as if it")
print("        were a standard error. Proper 2-sample 95% CI spans a factor of ~6.")

print()
print("="*78)
print("D.  LEVERAGE: T* is set by the last (least trustworthy) points")
print("="*78)
for drop in [0, 1, 2, 3, 4]:
    tt, dd = t[:len(t)-drop], d_true[:len(t)-drop]
    (Cf, Tf, rf), _ = sse_pow(tt, dd, True)
    print(f"  drop last {drop}: n={len(tt):2d}  T*={Tf:.4f}  rho={rf:.3f}")
# now with a realistic 2% wobble
print("  with 2% noise (median of 200 realisations):")
for drop in [0, 1, 2, 3]:
    TT, RR = [], []
    for _ in range(200):
        dn = d_true*(1+0.02*np.random.randn(len(t)))
        tt, dd = t[:len(t)-drop], dn[:len(t)-drop]
        try:
            (_, Tf, rf), _ = sse_pow(tt, dd, True)
            if Tf < 50: TT.append(Tf); RR.append(rf)
        except Exception: pass
    print(f"  drop last {drop}: T*={np.median(TT):.3f} "
          f"[{np.percentile(TT,16):.3f},{np.percentile(TT,84):.3f}]  "
          f"rho={np.median(RR):.2f} [{np.percentile(RR,16):.2f},{np.percentile(RR,84):.2f}]")

print()
print("="*78)
print("E.  T*-rho DEGENERACY, AND WHAT HAPPENS IF rho IS PINNED TO THEORY")
print("="*78)
for rf, lab in [(None, 'free'), (1.0, 'rho=1 (Leray/self-similar)'),
                (2.60, "rho=2.60 (user)"), (RHO_LH, 'rho=2.91 (Luo-Hou gamma_l)'),
                (RHO_RIG, 'rho=2.92056 (Chen-Hou RIGOROUS)')]:
    (Cf, Tf, rr), s = sse_pow(t, d_true, True, fix_rho=rf)
    print(f"  {lab:34s} T*={Tf:.4f}  C={Cf:.4f}  SSE={s:.2e}")
print("  -> On EXACT data the fit is of course perfect. The point is the")
print("     conditional map T*(rho): with 2% noise, how far does T* travel?")
TTr = {}
for rf in [1.0, 2.0, 2.60, RHO_RIG, 3.5]:
    TT = []
    for _ in range(200):
        dn = d_true*(1+0.02*np.random.randn(len(t)))
        try:
            (_, Tf, _), _ = sse_pow(t, dn, True, fix_rho=rf)
            if Tf < 50: TT.append(Tf)
        except Exception: pass
    TTr[rf] = np.median(TT)
    print(f"     rho pinned {rf:7.4f} -> T* = {np.median(TT):.4f}")
print(f"  -> T* swings {TTr[1.0]:.3f} .. {TTr[3.5]:.3f} across plausible rho: "
      f"range {TTr[3.5]-TTr[1.0]:.3f}, i.e. +/-{(TTr[3.5]-TTr[1.0])/2:.3f}, "
      f"{(TTr[3.5]-TTr[1.0])/2/0.10:.1f}x the quoted +/-0.10.")

print()
print("="*78)
print("F.  THE TEST THAT WAS NOT RUN: tau(t) = -delta / (d delta/dt)")
print("    power law  -> tau = (T*-t)/rho : STRAIGHT LINE, root at T*, slope -1/rho")
print("    pure exp   -> tau = const")
print("    delta->floor -> tau curves UP and never reaches zero")
print("="*78)
dd = np.gradient(d_true, t)
tau = -d_true/dd
A = np.polyfit(t, tau, 1)
print(f"  on exact data: slope={A[0]:.4f} -> rho={-1/A[0]:.3f} ; root T*={-A[1]/A[0]:.4f}")
print("  This turns a 3-parameter NONLINEAR fit into a 2-parameter LINEAR")
print("  regression -> the 3-vs-2 parameter-count confound disappears entirely,")
print("  and curvature (a floor / no blowup) is visible by eye.")

print()
print("="*78)
print("G.  CHEBYSHEV MAP: raw decay rate alpha is NOT the strip width")
print("    Kolluru et al Eq.11: delta_r = (rho*-1/rho*)/2, rho*=e^alpha => delta_r=sinh(alpha)")
print("="*78)
d_cheb = np.sinh(d_true)
(Cc, Tc, rc), _ = sse_pow(t, d_cheb, True)
print(f"  if the reported delta is really alpha and the strip width is sinh(alpha):")
print(f"     refit gives T*={Tc:.4f} (vs {TS0}), rho={rc:.3f} (vs {RHO0})")
print(f"     shifts: dT*={Tc-TS0:+.4f}, drho={rc-RHO0:+.3f}")
print("  (small because sinh(x)~x for x<<1; matters only for the early large-delta points)")

print()
print("="*78)
print("H.  PREFACTOR OMISSION: community standard (Kolluru Eq.12) fits")
print("       ln S = C - n ln k - 2 delta k   (3 params).  User fit omits n ln k.")
print("="*78)
kwin = [(50, 341), (80, 341), (120, 341)]
print("  additive bias in delta_hat per unit n, and as % of the LAST fitted delta=0.015:")
for k1, k2 in kwin:
    k = np.arange(k1, k2+1.0)
    b = np.cov(np.log(k), k, bias=True)[0, 1]/np.var(k)
    print(f"    k in [{k1},{k2}]: bias = {b:.5f}*n  -> n=4/3: {4/3*b:.5f} ({100*4/3*b/0.015:.0f}% of 0.015)"
          f"   n=3: {3*b:.5f} ({100*3*b/0.015:.0f}%)")
print()
print("  A CONSTANT multiplicative bias in delta is absorbed by C and leaves")
print("  (T*,rho) untouched. Only a TIME-VARYING bias hurts. Here the bias is")
print("  ADDITIVE and nearly constant in delta, so it acts as a FAKE FLOOR:")
for nn in [4/3., 3.0]:
    k = np.arange(80, 342.0)
    b = np.cov(np.log(k), k, bias=True)[0, 1]/np.var(k)*nn
    (Cb, Tb, rb), _ = sse_pow(t, d_true + b, True)
    print(f"    n={nn:.2f}: bias=+{b:.5f} -> measured T*={Tb:.4f} ({100*(Tb-TS0)/TS0:+.1f}%), "
          f"rho={rb:.3f} ({100*(rb-RHO0)/RHO0:+.1f}%)")
print("  -> an unmodelled algebraic prefactor pushes T* LATER and rho DOWN,")
print("     exactly the direction that would turn a true rho=2.92 into 2.60.")

print()
print("="*78)
print("I.  RECONCILIATION: what rho does the data imply if the FLOOR is real?")
print("="*78)
for nn in [4/3., 2.0, 3.0]:
    k = np.arange(80, 342.0)
    b = np.cov(np.log(k), k, bias=True)[0, 1]/np.var(k)*nn
    # invert: true law + additive floor b, measured as pure power law
    dtrue_rig = C0*(TS0-t)**RHO_RIG
    dtrue_rig = dtrue_rig*(d_true[0]/dtrue_rig[0])   # renormalise
    (Cb, Tb, rb), _ = sse_pow(t, dtrue_rig + b, True)
    print(f"  TRUE rho={RHO_RIG:.4f} + additive prefactor bias (n={nn:.2f}, +{b:.5f})")
    print(f"     -> a naive pure-exponential strip fit MEASURES rho={rb:.3f}, T*={Tb:.4f}")
