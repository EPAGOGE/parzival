# Viscosity inverts the direction-regularity scaling of the corner-flow blowup mechanism: a measured exponent crosses the type-I exclusion threshold

**Draft 0.1, 2026-08-02.** Every number in this paper regenerates from bytes
on disk with one command (Section 8). Claims are graded against a frozen
standard; the grading ledger, the confound engine, and the failed versions of
our own claims are part of the repository, not part of a memory hole.

## Abstract

For vorticity omega with direction xi = omega/|omega| and peak set P(t), the
quantity Lambda = sup_P |grad xi| |omega|^(-1/2) is scale invariant under the
Navier-Stokes rescaling, and its exponent

    sigma_Lambda = d ln Lambda / d ln ||omega||_inf

turns the direction-coherence regularity criteria of Constantin-Fefferman
and Beirao da Veiga-Berselli into a single measurable number. We measure it.

On the inviscid Luo-Hou corner flow, the strongest known singularity
mechanism of this type and one where blowup is proven in the Boussinesq
analogue by Chen-Hou, the instrument returns sigma_Lambda = +1.00 +- 0.03,
anchored by an exact amplitude symmetry of the equations (predicted Lambda
ratio 0.707107, measured 0.706891) and a 2x2 off-ray grid factorial (spread
0.029). Alignment failure grows with the collapse: the mechanism blows up
because its geometry is maximally non-depleting, and the observable returns
this verdict on a case where the answer is known. That calibration licenses
using it where the answer is not.

Under free-slip viscosity the same instrument, on the same scenario, shows
sigma(A) is not one number: every viscous run crosses over from
inviscid-like scaling at low amplitude to deep depletion at high amplitude.
Window-averaged slopes are regime mixtures and we retract our own earlier
quotes of that kind. Measured in the asymptotic window (top half of the
cross-grid amplitude overlap, symmetric midpoint cut, pre-registered), the
deep-collapse exponent is

    sigma_Lambda(nu = 1e-4) = -1.12 / -1.24   (grids 128x384 / 256x768, spread 0.121)
    sigma_Lambda(nu = 1e-3) = -1.25 / -1.29   (spread 0.032)

against a window-matched inviscid value of +0.589 / +0.585 (spread 0.005).
Viscosity does not merely damp the corner mechanism's alignment failure: it
inverts it, by roughly 1.8 powers of ||omega||, to a value far below both
thresholds of the exclusion chain (-1/6 for onset of geometric depletion,
-1/2 for exclusion of type-I blowup by geometry plus energy alone). As far
as a live literature kill-search reaches (Section 7), no direction-regularity
exponent has previously been measured as a function of viscosity on any
blowup candidate.

On the dynamical side, the collapse profile is an attractor in the rescaled
frame: a Lyapunov-certified march (definiteness by Cholesky only, no
eigenvalue enters any decision) shows perturbed orbits settle back to the
fixed point with cos_step = 0.986 and a contraction rate 0.270 +- 0.003 per
rescaled time unit, unanimous across five independent seeds at the largest
basin-validated amplitude.

## 1. Introduction

Two problems sit on one function. If a viscous singularity inheriting the
corner mechanism exists, it must keep sigma_Lambda(nu) above -1/6 as the
collapse deepens. If viscosity forces sigma_Lambda below -1/2 in the
high-vorticity regime, the geometric route closes the mechanism without
touching the supercritical energy problem: geometry, not energy, kills the
singularity. Nobody had sigma_Lambda(nu). This paper reports it, for the
mechanism class where the inviscid answer is proven, at two viscosities, on
two grids, with the disagreement between grids quantified and under a bar.

What is not new here, stated first. Direction-coherence regularity criteria
originate with Constantin and Fefferman [CF93] and were sharpened to the
1/2-Holder class by Beirao da Veiga and Berselli [BdVB02, BdVB09]: the
pairing of |grad xi| with the 1/2 power of amplitude lives in their sharp
exponent before it lives in our observable. Qualitative exclusion of type-I
blowup under direction continuity is likewise prior art: Giga-Miura [GM11]
for the infinite-energy class, and Barker-Prange [BP20] with scale-invariant
alignment estimates in the half-space with no-slip boundary, which is the
Luo-Hou geometry itself. The identity behind our instrument check
(Corollary 0 below) is Constantin's classical decomposition of the vorticity
equation into magnitude and direction. A 2026 analytical line by Grujic
[Gru26] pursues logarithmic depletion through direction classes; it defines
no observable and measures nothing.

What is new, correspondingly narrowed. First, the observable program:
treating the scale-forced pairing as a measurable scalar with an exponent,
and calibrating the instrument on a case where blowup is proven, so that its
verdict means something where blowup is unknown. Second, the numbers:
sigma_Lambda = +1.00 +- 0.03 on the inviscid corner mechanism, the first
direction-regularity exponent measured on any blowup candidate as far as our
search reaches. Third, the viscous inversion: sigma_Lambda(nu) measured at
two viscosities with cross-grid certification, crossing both thresholds deep
in the collapse. Fourth, the quantified exclusion chain (Section 4): the
known qualitative principle [GM11, BP20] sharpened to an explicit threshold
on the measured exponent by feeding the energy dissipation budget through
Constantin's kernel split, with the structural hypothesis carrying a
measured constant.

The laboratory operates under a discipline that we consider part of the
result (Section 8): a backwards proof of fifteen checks that failed its own
authors until the fit protocol was encoded, a confound engine enumerating
the ways each claim could be false, exact-symmetry anchors with
zero free parameters, and a standing watcher that reruns the certification
every twenty minutes. Several of our own earlier numbers died by these
instruments and are retained in the repository as corpses with names.

## 2. The observable

Definition. For a Navier-Stokes (or Euler, or Boussinesq) vorticity field,
xi = omega/|omega|, peak set P(t) = { x : |omega(x,t)| >= (1/2)||omega||_inf },

    Lambda(t) = sup_{P(t)} |grad xi| |omega|^(-1/2),
    sigma_Lambda = d ln Lambda / d ln ||omega||_inf.

Under u -> s u(sx, s^2 t) the field |omega| carries s^2 and |grad xi|
carries s, so Lambda is exactly scale invariant; the 1/2 power is forced by
the group, and |xi| = 1 makes the direction field globally controlled for
free. The same pairing appears, in criterion form, as the beta = 1/2
sharpness in [BdVB02, BdVB09]; the observable is that criterion turned into
a number a simulation can return.

Corollary 0 (of Constantin's |omega|-equation; stated as a corollary, used
as an instrument check). The magnitude of vorticity obeys
d_t|omega| + u.grad|omega| = alpha|omega| + nu(Delta|omega| - |omega||grad xi|^2),
where alpha = xi.S xi is the stretching rate. At any growing spatial
maximum of |omega|, the Laplacian term is nonpositive, hence

    nu |grad xi|^2 <= alpha    at the maximum.

This is a one-step corollary of a classical identity and we claim no
novelty for it; a literature kill-search retracted our earlier framing of
it as a theorem (Section 7). It earns its place as a validation: our data
satisfies it with three orders of magnitude of margin at every gated
snapshot, which any correct direction-field computation must.

## 3. Instruments

The DNS side is a Dedalus-based axisymmetric Boussinesq solver in the
Luo-Hou corner scenario, with free-slip viscosity in 5D-Laplacian form,
implicit in the IMEX stepper, and a quartic initial condition exactly
compatible with the Neumann wall condition. Trust gates per snapshot:
spectral tail at or below 1e-6 and signed Casimir drift (decay under
viscosity is physics; growth is a violation). Lambda is evaluated on the
2x-HWHM box at the vorticity peak; runs are trimmed at the first 10x
amplitude jump. The exponent is the slope of ln Lambda against
ln ||omega||_inf over gated snapshots, with pair-resampled bootstrap
confidence intervals (10,000 resamples) everywhere a slope is quoted.

The rescaled-frame side is a corner-frame s-march on the certified
self-similar profile: backward Euler on the index-2 DAE with the algebraic
manifold re-imposed at every iterate, a frozen reduced Jacobian whose exact
validity on ker(Cg) is a closed-form identity (verified at 4.6e-14 against
the exact marcher before being trusted), and a Lyapunov P-norm gate in
which definiteness is decided by Cholesky factorization only. No eigenvalue
of any operator enters any decision anywhere in this paper; that refusal is
standing and was never violated under pressure.

## 4. The exclusion chain, quantified

Constantin's identity for the stretching rate alpha = xi.S xi consumes the
Biot-Savart kernel geometry: misalignment is the only way to stretch. Split
the integral at radius rho; the near field is controlled by
Lambda ||omega||^(3/2) rho and the far field by the energy through
Cauchy-Schwarz. Optimizing rho,

    alpha  <~  ( Lambda ||omega||_inf^(3/2) )^(3/5) ||omega||_L2^(2/5).

Hypothesis (H): the peak set is a single coherent structure on which the
gradient bound holds out to the optimal rho. The hypothesis carries a
measured constant (lambda_0 = 5.0) and the chain inequality holds at 21 of
21 gated snapshot pairs with the stretching rate inferred independently.
Feeding the only global control Navier-Stokes provides (the energy
dissipation identity) through this bound, with blowup rate
||omega|| ~ (T-t)^(-gamma), gamma >= 1:

    sigma_Lambda <= -1/6 : geometric depletion begins to act at all
    sigma_Lambda <  -1/2 : type-I blowup excluded by geometry and energy alone.

The qualitative principle here is not ours: under a type-I condition,
direction continuity where vorticity is large already excludes blowup
[GM11], including in the half-space with no-slip boundary [BP20]. What the
chain above adds is the exponent: an explicit threshold on a measurable
number, so that a simulation, or eventually an experiment, can locate a
given flow on the right or wrong side of it.

## 5. The inviscid calibration

On the inviscid corner flow the instrument returns

    sigma_Lambda = +1.00 +- 0.03.

Two validations with zero free parameters between them. First, the exact
amplitude symmetry u -> lam u, t -> t/lam of the Euler equations predicts a
Lambda ratio of 1/sqrt(2) = 0.707107 between paired runs; measured
0.706891 +- 0.001441, three parts in ten thousand, with the predicted
|grad xi| ratio of 1 measured at 0.999746. Second, a 2x2 off-ray factorial
refining Nz and Nr independently returns slopes +1.0159, +0.9873, +1.0027,
+0.9978 (spread 0.029, main effects at or below 0.017), rejecting the
axial-mesh and single-ray confounds by test rather than by assertion.

Reading: |grad xi| ~ ||omega||^(3/2). Direction regularity fails a full 3/2
of a power above the exclusion threshold; the Euler mechanism blows up
because its geometry is maximally non-depleting. The observable returns the
correct sign on a case where blowup is proven in the Boussinesq analogue
(Chen-Hou), which is precisely what licenses pointing it at cases where the
answer is unknown.

Refinement recorded during the viscous campaign: the full-window +1.00 is
amplitude-composite. Fit on matched cross-grid amplitude windows, the
inviscid slope runs +1.4 in the lower half and +0.589 / +0.585 in the deep
half (cross-grid spread 0.005, the tightest certification in this paper).
The symmetry and factorial validations certify the instrument, not slope
constancy across amplitude; window-matched comparisons below therefore use
+0.59 as the inviscid contrast value.

## 6. The viscous measurement

Runs at nu in {1e-4, 1e-3} on grids 128x384 and 256x768 with inviscid
references on both grids. Three findings, in discovery order.

(i) The common-amplitude overlap window reproduces the calibration. Fit on
the same amplitude range, the two inviscid grids give +1.016 / +1.033
(spread 0.017) against the calibrated +1.00 +- 0.03. The cross-grid
disagreement of naive full-window fits (0.147) was a window artifact, not a
physics disagreement.

(ii) At finite nu, sigma(A) is not one number. Every viscous run crosses
over from inviscid-like slope (+0.4 to +0.8) at low amplitude to deep
depletion near -1.2 at high amplitude, on both grids, at both viscosities.
A window-averaged slope is a regime mixture; our own earlier quotes of that
kind (window averages near -0.5, and a single-grid ladder -0.37 to -0.57)
are retracted as measurements of a mixture, and the retraction is part of
the repository record.

(iii) Measured where the asymptotic regime actually lives (top half of the
cross-grid overlap in ln A, symmetric midpoint cut, pre-registered before
the cut-sensitivity scan), the deep-collapse exponent is

    nu = 1e-4:  sigma_Lambda = -1.116 (128x384) / -1.237 (256x768), spread 0.121
    nu = 1e-3:  sigma_Lambda = -1.254 (128x384) / -1.285 (256x768), spread 0.032

with bootstrap 95% intervals of order +-0.15 to +-0.35 on each fit and
cut-position sensitivity reported at 0.4 / 0.5 / 0.6 of the overlap span:
the magnitude sits in a band -1.0 to -1.3 at every cut, on every grid, at
both viscosities. Against the window-matched inviscid +0.59, viscosity
inverts the direction-regularity scaling by roughly 1.8 powers of
||omega||, to a value below both thresholds of Section 4.

Honest bounds on what this shows. Two viscosities, one mechanism class, one
scenario geometry, laptop resolution; the deep windows carry 11 to 14
snapshots each; the crossover amplitude A_c(nu) is visible but not yet
resolved as its own observable; and the exclusion chain runs through
Hypothesis (H) with its measured constant. What the measurement supports,
within those bounds, is exact: on the strongest known blowup mechanism of
this class, the geometric quantity that must stay above -1/6 for the
mechanism to survive viscosity is measured at -1.2 and below, deep in the
collapse, grid-certified, at both viscosities tried.

## 7. The kill-search

A live literature search (2026-08-02; arXiv, Springer, journal indices;
queries on direction-coherence criteria, Holder-1/2 direction results,
scale-invariant alignment estimates, type-I exclusion via direction,
Luo-Hou alignment measurements, and name collisions for sigma_Lambda)
graded every novelty-bearing claim. One claim died: our Theorem 0 is a
one-step corollary of Constantin's |omega|-equation and is demoted to
Corollary 0 above. Theorem-1-as-principle died too, as it should: the
qualitative type-I exclusion belongs to [GM11] and [BP20]; what survives is
the quantification. The observable program, the inviscid measurement, and
the viscous inversion survived the search outright, with the standing
caveat that survived means not found by this search, not proven absent.
Closest adjacent art in the numerical literature: Hou's dynamic-depletion
diagnostics (alignment with strain eigenvectors, vortex-line regularity;
qualitative) [Hou09, LH14], and Kerr's numerical tests of direction-derived
regularity conditions on antiparallel vortex tubes [Kerr13].

## 8. Reproducibility and discipline

    python verify.py        # the backwards proof: 15 checks, 3 tiers
    python mythos.py        # the confound engine
    ./done.sh               # the milestone ladder, M1-M5, all PASS
    cat VERIFY_STATUS       # what the standing watcher currently holds

verify.py holds at 15/15 with EXACT-tier worst deviation 3.05e-04 against
references forced by the equations before any measurement existed; input
checksums are in the verify output; a watcher reruns the full certification
every twenty minutes and files a regression report if anything drifts. The
laboratory was built by a human directing an AI system; the AI invented a
branch that did not exist, fit through windows that moved, and glossed a
mechanism its own certification refuted within the hour. Every failure was
caught the same day by exact symmetry anchors, context-free review, or the
backwards proof, and the corpses are kept on the page. The discipline that
survives contact with an AI that launders inference is part of what this
repository demonstrates.

## 9. Open

The strip -1/2 <= sigma <= 0: Corollary 0 forbids the top under viscosity
at a growing maximum; the chain of Section 4 needs the bottom. Close the
strip analytically and type-I Navier-Stokes blowup is finished for this
route. The crossover amplitude A_c(nu) and the sigma_Lambda(nu) curve
between 1e-4 and the inviscid limit are unmeasured (stream data exists at
nu = 3e-4, 3e-5, 1e-5; snapshots were not retained). The wandering sector
remains the measured frontier: generic near-wall data dies too young at
laptop resolution to read, and that failure is the recorded reason the
sector is unexplored.

## References

[CF93]  P. Constantin, C. Fefferman. Direction of vorticity and the problem
        of global regularity for the Navier-Stokes equations. Indiana Univ.
        Math. J. 42, 775-789 (1993). [VERIFIED 2026-08-02]
[Con94] P. Constantin. Geometric statistics in turbulence. SIAM Review 36,
        73-98 (1994). doi:10.1137/1036004 [VERIFIED 2026-08-02; source of
        the |omega|-equation decomposition]
[BdVB02] H. Beirao da Veiga, L.C. Berselli. On the regularizing effect of
        the vorticity direction in incompressible viscous flows. Diff.
        Integral Equations (2002).
[BdVB09] H. Beirao da Veiga, L.C. Berselli. Navier-Stokes equations:
        Green's matrices, vorticity direction, and regularity up to the
        boundary (2009). [beta = 1/2 sharp class]
[GM11]  Y. Giga, H. Miura. On vorticity directions near singularities for
        the Navier-Stokes flows with infinite energy. Comm. Math. Phys.
        (2011). doi:10.1007/s00220-011-1197-x
[BP20]  T. Barker, C. Prange. Scale-invariant estimates and vorticity
        alignment for Navier-Stokes in the half-space with no-slip
        boundary conditions. Arch. Ration. Mech. Anal. 235:881-926 (2020).
        arXiv:1906.08225
[Gru26] Z. Grujic. Logarithmic depletion of vortex stretching and
        singularity evasion in the 3D Navier-Stokes equations.
        arXiv:2607.08866 (2026).
[Hou09] T.Y. Hou. Blow-up or no blow-up? A unified computational and
        analytic approach to 3D incompressible Euler and Navier-Stokes
        equations. Acta Numerica (2009).
[LH14]  G. Luo, T.Y. Hou. Toward the finite-time blowup of the 3D
        axisymmetric Euler equations: a numerical investigation. Multiscale
        Model. Simul. (2014).
[CH22]  J. Chen, T.Y. Hou. Stable nearly self-similar blowup of the 2D
        Boussinesq and 3D Euler equations with smooth data I: Analysis.
        arXiv:2210.07191. [VERIFIED 2026-08-02]
[CH25]  J. Chen, T.Y. Hou. Stable nearly self-similar blowup of the 2D
        Boussinesq and 3D Euler equations with smooth data II: Rigorous
        Numerics. Multiscale Model. Simul. 23(1), 25-130 (2025).
        arXiv:2305.05660. [VERIFIED 2026-08-02]
[Kerr13] R.M. Kerr. Bounds for Euler from vorticity moments and line
        divergence. arXiv:1212.1106.
[Surv21] Survey: geometric constraints on the blowup of solutions of the
        Navier-Stokes equation. arXiv:2111.00040.

All venue data above was verified against primary sources in the
2026-08-02 live searches; no reference carries unverified detail.
