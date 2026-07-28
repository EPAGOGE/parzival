import numpy as np
from scipy.optimize import brentq
Tstar,rho,C,A=1.7135,2.60,1.23,4.0
d=lambda t: C*np.maximum(Tstar-t,0)**rho

print("== IC gradient: b = A*0.5*(1-cos x)*exp(-30 (z/Lz)^4) ==")
u=np.linspace(0,1.5,2000001)
g=120*u**3*np.exp(-30*u**4)            # |d/du exp(-30u^4)|
i=g.argmax()
print(f"  max_u |d/du e^(-30u^4)| = {g[i]:.4f} at u={u[i]:.4f}  (z = {u[i]:.3f}*Lz)")
print(f"  => sup|dz b|(t=0) = A*{g[i]:.3f}/Lz = {A*g[i]:.2f}/Lz")
for Lz in [0.5,1.0,2.0,np.pi]:
    print(f"     Lz={Lz:5.3f} -> sup|grad b|(t=0) ~ {A*g[i]/Lz:6.2f}   "
          f"ratio 27/that = {27/(A*g[i]/Lz):5.2f}x")
print(f"  sup|dx b|(t=0) = A*0.5 = {A*0.5:.2f}  (subdominant unless Lz>~7)")
print("  NOTE argmax sits at z~0.40*Lz, x=pi -- NOT at the wall z=0.")

print("\n== the decoupling: delta collapses 65x, sup|grad b| grows <2x ==")
print(f"  delta(0.80)={d(0.80):.4f}  delta(1.53)={d(1.53):.5f}  ratio={d(0.80)/d(1.53):.1f}x")
print(f"  if a front of fixed jump sharpened with delta, sup would grow {d(0.80)/d(1.53):.0f}x")
print(f"  starting from ~14/Lz that predicts sup(1.53) ~ {14*d(0.80)/d(1.53):.0f}/Lz, measured 27")
print(f"  => implied residue/jump depletion factor ~ {14*(d(0.80)/d(1.53))/27:.0f}x")

print("\n== predicted convergence-loss time t_dep(N) from the SAME rho ==")
print("   (falsifiable: t_dep must obey delta(t_dep) = c/kmax)")
for c in [1.7,5.0]:
    row=[]
    for N in [256,512,1024,2048,4096]:
        kmax=N/3.0
        rem=(c/kmax/C)**(1/rho)
        row.append((N,Tstar-rem))
    print(f"  c={c:4.1f}: "+"  ".join(f"N={n}:t={t:.3f}" for n,t in row))
print("  observed: 5 sig figs to ~1.45, 3% at 1.53, 2x apart by 1.60  -> brackets c~1.7-5")

print("\n== ATTACK 2: is the Hou-Li filter faking convergence? ==")
print("  filter band is k/kmax > ~0.85. Relative amplitude exposed to it, exp(-delta*0.85kmax):")
for t in [1.45,1.53]:
    e={}
    for N in [256,512,1024]:
        kmax=N/3.0; e[N]=np.exp(-d(t)*0.85*kmax)
        print(f"   t={t}  N={N:5d}: {e[N]:.3e}")
    print(f"   -> N=256 exposes {e[256]/e[1024]:.3g}x MORE amplitude to the filter than N=1024")
print("  yet t=1.45 agrees to 5 sig figs across those runs.")
print("  => filter action on ~6e-2 relative amplitude changes sup|grad b| by <1e-5 relative.")
print("  A filter that suppressed real growth would be N-dependent (cutoff moves with N)")
print("  and would DESTROY that agreement. It does not. Attack 2 fails.")

print("\n== ATTACK 1: could N^3 at t=2.0 be physics? ==")
print(f"  under the study's own fit, t=2.0 > T*={Tstar}: (T*-t) = {Tstar-2.0:.3f} < 0 -> no solution exists.")
print("  reject the fit and posit T*>2.0? then delta(2.0)>0 but tiny; required kmax to see it:")
for dd in [1e-2,1e-3,1e-4]:
    print(f"     delta={dd:.0e} needs kmax>~{5/dd:.0f} i.e. N>~{3*5/dd:.0f}")
print("  ceilings on sup|grad b| N-scaling: resolved N^0 | Fourier-limited N^1 |")
print("  Chebyshev-at-wall N^2 | Gibbs/shock N^1.  Observed N^3 exceeds all of them.")
