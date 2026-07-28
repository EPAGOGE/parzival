import numpy as np
Tstar, rho, C, A = 1.7135, 2.60, 1.23, 4.0
d = lambda t: C*(Tstar-t)**rho

print("== delta(t) from the fit ==")
for t in [0.8,1.2,1.45,1.50,1.53,1.55,1.60]:
    print(f"  t={t:.2f}  T*-t={Tstar-t:.4f}  delta={d(t):.5f}")

print("\n== delta*k_max (resolution quality of the fitted tail) ==")
for N in [256,512,1024]:
    for kmaxlab,kmax in [("N/2",N/2),("N/3 (2/3-dealias)",N/3)]:
        print(f"  N={N:5d} kmax={kmax:6.0f} ({kmaxlab:18s}) : "
              f"d*k @t=1.45 {d(1.45)*kmax:6.2f} | @t=1.53 {d(1.53)*kmax:6.2f}")

print("\n== cost of one decade in (T*-t) under uniform refinement ==")
print(f"  (T*-t)_min ~ N^(-1/rho), 1/rho = {1/rho:.4f}")
print(f"  N factor for 10x closer  = 10^rho = {10**rho:.1f}")
print(f"  compute ~ N^2 * Nt(~N) = N^3 -> {(10**rho)**3:.3e}")
print(f"  compute if Nt ~ N^2 (Cheb CFL)   -> {(10**rho)**4:.3e}")

print("\n== reachable t under different delta-floor criteria (N=1024) ==")
for kmax in [512, 341]:
    for crit in [1.7, 5.0, 10.0, 15.0]:
        dfloor = crit/kmax
        rem = (dfloor/C)**(1/rho)
        print(f"  kmax={kmax:4d} crit d*k>={crit:5.1f} -> dfloor={dfloor:.5f} "
              f"T*-t={rem:.4f} t_max={Tstar-rem:.4f}")

print("\n== front width / gradient consistency at t=1.53 ==")
dd = d(1.53); w = dd/(np.pi/2)
print(f"  delta={dd:.5f} -> tanh half-width w=2*delta/pi={w:.5f}")
for sup in [27.0]:
    print(f"  sup|grad b|={sup} implies jump  db = 2*sup*w = {2*sup*w:.3f}  "
          f"= {2*sup*w/A*100:.1f}% of A={A}")
print(f"  if instead the front carried the full A: sup would be A/(2w) = {A/(2*w):.0f}")

print("\n== what N-scaling of sup|grad b| can legitimately occur? ==")
print("  resolved analytic field           : N^0 (converged)")
print("  under-resolved, Fourier-limited   : sup ~ kmax        -> N^1")
print("  under-resolved, Chebyshev at wall : sup ~ kmax^2      -> N^2")
print("  jump discontinuity (Gibbs)        : |b_k|~1/k, sup~sum 1 -> N^1")
print("  observed at t=2.0                 : N^3  <-- above every legitimate ceiling")

print("\n== Hou-Li filter transfer function exp(-36 (k/kmax)^36) ==")
for r in [0.5,0.65,0.7,0.8,0.85,0.9,0.95,1.0]:
    print(f"  k/kmax={r:.2f} -> factor {np.exp(-36*r**36):.6f}")

print("\n== relative amplitude at the filter's active band, on the fit's own spectrum ==")
for t in [1.45,1.53]:
    for N in [256,1024]:
        kmax=N/3
        print(f"  t={t} N={N:5d}: |b_k|/|b_0| at 0.85kmax = exp(-delta*0.85kmax) = "
              f"{np.exp(-d(t)*0.85*kmax):.3e}")

print("\n== T* sub-window spread vs extrapolation distance ==")
print(f"  T* spread (1.63..1.83) = {1.83-1.63:.2f}")
print(f"  extrapolation distance from last datum: T*-1.53 = {Tstar-1.53:.3f}")
print(f"  ratio spread/distance = {(1.83-1.63)/(Tstar-1.53):.2f}  (>1 => T* unresolved)")
