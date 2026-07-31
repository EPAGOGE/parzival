# SITUATION + QA  [SWAP THIS FILE ONLY — profiles above stay byte-identical]

Updated: 2026-07-24. Target: 2D Boussinesq blowup, `boussinesq/dedalus_bsq.py`
(physical) and `boussinesq/rescale.py` (dynamic rescaling).

## STATE (measured, not assumed)

**Physical solver — trustworthy result.** T* = 1.70 +/- 0.01, resolution-independent:
`(1/sup|grad b|)^(1/2)` is LINEAR in t (theory forces the -2 exponent) to R^2 =
0.9996 at BOTH N=1024 and N=2048, zero-crossings 1.7076 vs 1.7033. Two
self-consistency conditions agree: forcing the exponent to exactly -2 gives T*=1.696;
forcing gamma to the literature 2.9206 gives T*=1.7014. Independently
gamma = 2.94-3.00 from `L = ||omega||/||grad omega|| ~ (T*-t)^gamma`. b^2 drift
1.5e-11 throughout. `sup|grad b|` agrees 0.0-0.2% between N=1024 and N=2048 to t=1.505.

**Resolution honesty.** Data past t~1.55 is grid artifact: spectrum tail is a flat
~1e-3 bathtub that RISES at k/kmax=0.98 (spectral blocking), vs clean monotone decay
at t<=1.52. argmax|grad b| MIGRATES from interior (z/Lz~0.40) to the wall at t>=1.44.

**Corner geometry.** `argmax|grad omega| = 494.7` at EXACTLY x=pi, z=1.85e-6.
`|omega|(pi) = 2.2e-14` vs `||omega||inf = 5.027` -> omega is ODD about x=pi to 12
digits (Chen-Hou symmetry, confirmed by data). Structure half-widths at that point:
Lx = 1.2272e-02 = **2.0 uniform-x points** (RealFourier, dx=6.14e-3);
Lz = 5.3730e-02 = **86 Chebyshev z-points** (dz_min=1.48e-5).
Corner gauge inputs: omega_x = -494.738, b_xx = -3335.05, ratio +6.74103.

**Rescaling solver — stable but not converged.** Fatal bug found and fixed: U was
taken from a tau-CONTAMINATED gradient (`skew(grad(Psi)+lift(tau))`, copied from the
1-wall physical solver), injecting boundary-residual artifact into the advecting
velocity: |Psi|=0.049 gave |U|=18.0 (grid-scale). Clean `skew(grad(Psi))` gives
|U|=0.484 — factor 37 — and it now runs indefinitely.
Gauge: `c_l -> 3.0062` vs Chen-Hou 3.00649898 (4 digits, generic seed). But
`c_w`/gamma magnitudes wrong, and `c_l` OSCILLATES -33/-116/+178/+27 with the
algebraic seed.
Normalization in use: `c_l = 2 B_y1y1(0)/Om_y1(0)`, `c_w = c_l/2 + U1_y1(0)`.
Targets: U1_y1(0) = -2.5327, c_w = -1.0294, gamma = +2.9206.

**Ruled out by experiment (do not retry):** CFL (dt cut 100x, no change); aliasing
(Hou-Li filter made it WORSE, gamma 2.85->0.10, because damping Chebyshev
coefficients corrupts the boundary interpolation the gauge needs); gauge feedback
(froze c_l,c_w at Chen-Hou values -> still diverged, so not the loop);
ANALYTIC SEEDS (Gaussian gives U1_y1(0)=-3.148; algebraic seed with the CORRECT
r^alpha far field gives U1_y1(0)=+0.73 — opposite signs, neither near -2.5327).

## QA

**Q1.** What is the single best next step, and what number settles it?
**Q2.** What in the current setup is a formulation error rather than a tuning error?
**Q3.** What can be tested for free on data already on disk?
**Q4.** What should be abandoned?
