import numpy as np
from scipy.optimize import curve_fit, least_squares

print("="*72)
print("TEST 1: BIAS IN delta FROM OMITTING THE ALGEBRAIC PREFACTOR k^-n")
print("="*72)
# truth: log|b_k| = logC - n*log k - delta*k. Fit only a - delta*k.
# LSQ slope bias = n * Cov(log k, k)/Var(k)
for (k1,k2) in [(20,120),(50,340),(100,340),(150,500),(200,680)]:
    k = np.arange(k1,k2+1.0)
    bias_per_n = np.cov(np.log(k),k,bias=True)[0,1]/np.var(k)
    print(f"  window k=[{k1:4d},{k2:4d}]  delta_hat-delta = {bias_per_n:.5f} * n   "
          f"(n=1 -> {bias_per_n:.5f}, n=4/3 -> {4/3*bias_per_n:.5f})")

print("\n  Compare to user's smallest fitted delta = 0.015")
k=np.arange(50,341.0); b=np.cov(np.log(k),k,bias=True)[0,1]/np.var(k)
print(f"  window [50,340], n=4/3 : bias={4/3*b:.5f} = {100*4/3*b/0.015:.1f}% of delta=0.015")
print(f"  window [50,340], n=3   : bias={3*b:.5f} = {100*3*b/0.015:.1f}% of delta=0.015")

print("\n"+"="*72)
print("TEST 2: HOU-LI FILTER rho(k)=exp(-36 (k/kmax)^36) AS APPARENT delta")
print("="*72)
print("  local log-slope d/dk[-36(k/kmax)^36] = -(1296/kmax)(k/kmax)^35")
print("  -> filter alone contributes apparent delta_fake = (1296/kmax)*(k/kmax)^35\n")
for kmax in [341, 512]:
    print(f"  kmax={kmax}:  (floor 1.7/kmax = {1.7/kmax:.5f}, tyger thresh 2pi/kmax = {2*np.pi/kmax:.5f})")
    for frac in [0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9]:
        dfake = (1296.0/kmax)*frac**35
        print(f"     k/kmax={frac:.2f}: delta_fake={dfake:.6f}  "
              f"= {dfake/(1.7/kmax):8.2f} x floor   filter amp={np.exp(-36*frac**36):.3e}")
    print()
