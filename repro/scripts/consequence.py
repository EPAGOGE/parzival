import math
print("=== A. resolution floor: is delta_min ~ 1.7/k_max defensible? ===")
for N in (1024,):
    for kmax,lab in ((N/2,'k_max=N/2'),(N/3,'2/3 dealias'),(0.7*N/2,'Hou-Li knee ~0.7 k_max')):
        for c,crit in ((1.7,'~1.7 (<2 e-folds: very optimistic)'),(9.2,'spectrum down 1e-4'),(36.8,'down to 1e-16 (Shelley/Bustamante std)')):
            print(f"  N={N} {lab:24s} c={c:5.1f} {crit:38s} delta_min={c/kmax:.5f}")
    print()

print("=== B. cost of ONE decade in (T*-t), as a function of rho ===")
print(f"{'rho':>5} {'N factor=10^rho':>16} {'N from 1024':>12} {'grid pts':>11} {'cost N^3':>10} {'cost N^4':>10} {'verdict'}")
for rho in (1.0,1.5,2.0,2.6,2.9):
    f=10**rho; N=1024*f
    v = 'routine' if N<1e4 else ('large but done today' if N<5e4 else ('leadership-class' if N<2e5 else 'beyond any machine'))
    print(f"{rho:>5.1f} {f:>16.1f} {N:>12.0f} {N*N:>11.2e} {f**3:>10.2e} {f**4:>10.2e}  {v}")

print("\n=== C. sanity on 'at any budget' at rho=2.6 ===")
N=1024*10**2.6; pts=N*N
print(f"  N={N:.3e}, grid points={pts:.3e}, bytes/field={pts*8:.3e} ({pts*8/1e12:.2f} TB)")
print(f"  ~10 arrays -> {10*pts*8/1e12:.1f} TB  |  Frontier aggregate memory ~9.2 PB -> {10*pts*8/9.2e15*100:.2f}% of it. MEMORY FITS.")
print(f"  cost multiple over the N=1024 run (N^3): {10**(2.6*3):.2e}")
print("  If N=1024 run = 1 node-hour -> %.0f node-years; Frontier has 9408 nodes -> %.2f yr of the FULL machine."%(10**7.8/8766, 10**7.8/8766/9408))

print("\n=== D. Hou-Li filter, cumulative over a run (exp(-36 (k/kmax)^36) per stage) ===")
for n in (1,1e3,1e4,1e5):
    kk=(1/(36*n))**(1/36)
    print(f"  after {n:>7.0e} applications: e-fold knee at k/kmax={kk:.3f}  -> usable spectrum only below that")
