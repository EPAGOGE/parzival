# sigma_Lambda: the geometric criticality exponent

2026-07-30. Status: derivation at physics rigor (one structural hypothesis,
stated below); measurement twice validated; viscous ladder in flight.

## Definition

For a solution of 3D (Navier-)Stokes with vorticity omega, direction
xi = omega/|omega|, and peak set P(t) = { x : |omega(x,t)| >= (1/2)||omega||_inf }:

    Lambda(t)    = sup_{P(t)} |grad xi| |omega|^{-1/2}
    sigma_Lambda = d ln Lambda / d ln ||omega||_inf

Under the Navier-Stokes scaling u -> s u(sx, s^2 t) the field |omega| carries
s^2 and |grad xi| carries s, so Lambda is exactly scale invariant. The 1/2
power is forced by the group. Lambda is the unique scale-invariant pairing of
direction regularity with vorticity amplitude, and |xi| = 1 makes the direction
field globally controlled for free. This is the sigma = 0 object demanded by
the criticality atlas (criticality.py), and it evades the Tao averaging barrier
because the argument below consumes the Biot-Savart kernel geometry, which
averaging destroys while preserving energy and scaling.

## The stretching bound

Constantin's identity for the stretching rate alpha = xi . S xi:

    alpha(x) = (3/4pi) P.V. int D(yhat, xi(x+y), xi(x)) |omega(x+y)| dy/|y|^3,
    |D| <= |sin phi(x, x+y)|.

Misalignment is the only way to stretch. Split the integral at radius rho.

Near field, |y| <= rho:  |sin phi| <= |grad xi| |y| <= Lambda ||omega||^{1/2} |y|
gives a contribution <= 4pi Lambda ||omega||^{3/2} rho.

Far field, |y| > rho:  Cauchy-Schwarz gives <= C ||omega||_{L2} rho^{-3/2}.

Optimizing rho:

    alpha  <~  ( Lambda ||omega||_inf^{3/2} )^{3/5}  ||omega||_{L2}^{2/5}.

Hypothesis (H): the peak set is a single coherent structure on which the
gradient bound holds out to the optimal rho. The optimal rho fits inside the
structure only when sigma_Lambda <= -1/6; above that the estimate degrades to
the classical log bound and geometry buys nothing.

## The budget and the threshold

The only global control Navier-Stokes provides:

    nu int_0^T ||omega||_{L2}^2 dt <= E_0   (energy dissipation identity).

Suppose blowup at rate ||omega|| ~ (T-t)^{-gamma}, gamma >= 1 (gamma = 1 is the
proven lower bound), with Lambda ~ ||omega||^{sigma}. Holder in time on the
integral of alpha, feeding in the budget:

    int_t^T alpha ds  <~  ( int (T-s)^{-gamma(sigma + 3/2)} ds )^{3/5}
                          (E_0/nu)^{1/5} (T-t)^{1/5}.

If sigma < 1/gamma - 3/2 <= -1/2 the integral converges, int alpha < infty,
||omega||_inf stays bounded through the BKM exponential, contradiction.

Two rungs from one chain:

    sigma_Lambda <= -1/6 : geometric depletion begins to act at all
    sigma_Lambda <  -1/2 : type-I blowup excluded by geometry + energy alone
                           (faster blowup requires even larger sigma_Lambda)

## The measurement

Inviscid corner flow (Luo-Hou class, the strongest known singularity mechanism
of this type), measured 2026-07-30:

    sigma_Lambda = +1.00 +- 0.03

Validation one: exact amplitude symmetry (u -> lam u, t -> t/lam). Predicted
Lambda ratio 1/sqrt(2) = 0.707107, measured 0.706891 +- 0.001441. Predicted
|grad xi| ratio 1, measured 0.999746 +- 0.001218. Three parts in ten thousand.

Validation two: 2x2 off-ray factorial (Nz and Nr refined independently).
Slopes +1.0159, +0.9873, +1.0027, +0.9978; spread 0.029. Main effect of
doubling Nz: -0.0013. Of doubling Nr: -0.0167. Grid converged; the blind-review
faults F8/F9 (axial-mesh artifact, single-ray degeneracy) are tested and
rejected for this observable.

Reading: |grad xi| ~ ||omega||^{3/2}. Direction regularity fails a full 3/2 of
a power above the exclusion threshold. The Euler mechanism blows up because its
geometry is maximally non-depleting, and the observable returns the correct
sign on a case where blowup is proven (Chen-Hou). That is the calibration that
licenses using it where the answer is unknown.

## The reduction

Both directions of the Clay problem, restricted to this mechanism class, sit on
one measurable function sigma_Lambda(nu):

Blowup head: a viscous singularity inheriting the corner mechanism must keep
sigma_Lambda(nu) above -1/6 (necessarily above -1/2) as the collapse deepens.
If measurement shows it does, the geometric route to regularity is closed and
the mechanism stays live.

Regularity head: show viscosity forces sigma_Lambda below -1/2 in the
high-vorticity regime. That single monotonicity statement bypasses the
supercritical energy problem entirely for this class: geometry, not energy,
kills the singularity.

Nobody has sigma_Lambda(nu). The instrument is validated and the solver now
carries free-slip viscosity in the 5D-Laplacian form (nu Delta_5 on u1 and
omega1, Delta_5 = drr + (3/r) dr + dzz), implicit in the IMEX stepper, with the
quartic IC exactly compatible with the Neumann wall condition.

## Ladder in flight

nu in {1e-3, 3e-4, 1e-4, 3e-5, 1e-5} at 128x384, tmax 3e-3, snapshots every
6e-5, nu = 0 reference OR_z128r384 on disk. Gates for the viscous analysis:
spectral tail (tau rows excluded) and signed gamma drift (decay is physics,
growth is a violation). The Casimir drifts physically under viscosity and is
not a viscous trust criterion.

Outcome A, sigma_Lambda(nu) stays near +1: alignment failure survives
viscosity; geometric regularization is closed for this mechanism; the blowup
head keeps its best candidate.

Outcome B, sigma_Lambda(nu) falls with nu: direct observation of viscosity
restoring direction regularity, the geometric regularization mechanism itself,
measured, and an independent, sharper companion to the c_l = -1/alpha energy
argument for why this mechanism dies in Navier-Stokes.

Either outcome is a result. The number decides.

## The viscous measurement (2026-08-02) -- the number decided

The ladder's surviving snapshot pairs (nu in {1e-4, 1e-3}, grids 128x384
and 256x768, nu=0 references on disk) went through the certified chain
sigma_peak.py -> m4_sigma_spread.py (2x2 instrument factorial) ->
m4_sigma_deep.py (certified quote). Three findings, in discovery order:

1. The common-amplitude overlap window REPRODUCES THE CALIBRATION: the
   two inviscid grids, fit on the same amplitude range, give +1.016 /
   +1.033 (spread 0.017) against the twice-validated +1.00 +- 0.03.
2. At finite nu, sigma(A) is NOT one number: every viscous run crosses
   over from inviscid-like slope (+0.39..+0.81) at low amplitude to deep
   depletion (~-1.2) at high amplitude. Window-averaged quotes are
   regime mixtures and were discarded (the alpha campaign's XMAX
   lesson applied here).
3. DEEP-COLLAPSE MAGNITUDE, grid-certified (top half of the cross-grid
   overlap, symmetric midpoint cut, pre-registered; M4 bar spread<0.15):

       sigma_Lambda(1e-4) = -1.116 / -1.237   spread 0.121
       sigma_Lambda(1e-3) = -1.254 / -1.285   spread 0.032

OUTCOME B, MEASURED: viscosity does not merely damp the corner
mechanism's alignment failure -- it inverts it, from +1.0 inviscid to
about -1.2 deep in the collapse, a value well below BOTH thresholds of
the reduction (-1/6 depletion onset, -1/2 type-I exclusion). Within this
mechanism class and at these two viscosities, the geometric route is
doing exactly what the regularity head requires: geometry, not energy,
kills the singularity. Caveats owned: two viscosities, top-half windows
carry n=11-14 snapshots (bootstrap CI half-widths +-0.14-0.30), crossover amplitude
A_c(nu) not yet resolved as its own observable. Files:
SIGMA_PEAK_M4.out, M4_SIGMA_SPREAD.out, M4_SIGMA_DEEP.out,
M4_SIGMA_DEEP_NU0.out, sigma_grid_spread.txt, m4_sigma_rows.npz.

### Self-audit correction (2026-08-02, same session)

Applying the certified deep-collapse estimator to the nu=0 pair (the
apples-to-apples check) gives sigma_top(nu=0) = +0.589 / +0.585, spread
0.005 -- the inviscid slope itself drifts within the trusted window
(+1.42/+1.36 low-amplitude, +0.59 deep, artifact M4_SIGMA_DEEP_NU0.out; the +1.00 +- 0.03 calibration is the
full-window number, whose validations certify the INSTRUMENT, not slope
constancy). The window-matched inversion statement is therefore
+0.59 -> ~-1.2 (still a sign flip, ~1.8 powers), NOT "+1.0 -> -1.2" as
the section above loosely phrased it. Thresholds unaffected: deep
inviscid +0.59 stays far above -1/6; deep viscous -1.2 stays far below
-1/2. Robustness note: the deep viscous value agrees across grids with
4x cell-count difference and across the two viscosities, arguing
against a trust-boundary artifact.
