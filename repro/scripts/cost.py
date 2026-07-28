import numpy as np
print("="*74)
print("HOW THE 'IMPOSSIBLE AT ANY BUDGET' CONCLUSION DEPENDS ON rho")
print("  reachable (T*-t) ~ N^(-1/rho);  1 decade in (T*-t) costs N x 10^rho")
print("  compute ~ N^2 (2D) x N (CFL steps) = N^3   [LOWER bound: ignores u growth]")
print("="*74)
print(f"{'rho':>6} | {'N factor / decade':>18} | {'compute factor':>16} | {'verdict':>22}")
print("-"*74)
for rho,lab in [(1.0,'linear (radial/Cheb)'),(1.5,'Burgers preshock'),(2.0,''),
                (2.6,"user's fit"),(3.1,'+1 sigma of 2.6+-0.5')]:
    fN=10**rho; fC=fN**3
    v = "routine" if fC<1e4 else ("large cluster" if fC<1e7 else "out of reach")
    print(f"{rho:>6.1f} | {fN:>18.0f} | {fC:>16.3g} | {v:>22} {lab}")

print("\n" + "="*74)
print("USABLE WINDOW: user assumes floor delta=1.7/kmax. Literature thresholds:")
print("="*74)
for kmax,N in [(341,1024),(512,1024)]:
    print(f"  kmax={kmax}:")
    print(f"    user's assumed floor      1.7/kmax   = {1.7/kmax:.5f}")
    print(f"    tyger onset (Kolluru+)    2*pi/kmax  = {2*np.pi/kmax:.5f}   ({2*np.pi/1.7:.1f}x larger)")
    print(f"    'a few mesh widths' (3dx, L=2pi)     = {3*2*np.pi/N:.5f}   ({3*2*np.pi/N/(1.7/kmax):.1f}x larger)")
    print(f"    user's fit cutoff delta>0.015 -> {'BELOW' if 0.015 < 2*np.pi/kmax else 'above'} tyger threshold")
print("\n  Shrinking the usable delta-window by factor f costs f^(1/rho) in (T*-t):")
for f,nm in [(2*np.pi/1.7,'tyger'),(3*2*np.pi/1024/(1.7/341),'3 mesh widths')]:
    print(f"    f={f:.2f} ({nm}): reachable (T*-t) worsens by {f**(1/2.6):.2f}x")
