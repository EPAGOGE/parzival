# Numerics note — claims ledger

Every claim the note can make, graded by what actually backs it.  Nothing enters the
manuscript above its grade here.  Grades: **WITNESSED** (measured in this campaign,
reproducible from the logs) · **ENFORCED** (a gate in the code refuses violations) ·
**SPEC** (designed, not yet measured — must be labelled as such or cut).

Working title: *A sparse-Jacobian spectral Newton solver for the corner self-similar
profile, with a-posteriori gauge diagnostics.*

---

## Central result

| # | Claim | Grade | Backing |
|---|---|---|---|
| C1 | alpha_0 = **-0.34240 ± 4.4e-5** for the Chen–Hou corner profile | WITNESSED | 5-rung eps ladder on the CORRECTED abscissa (2.5e-5 had been printed as 3e-5 by a rounding format and fitted at the printed value); bar = model-class spread 3.93e-5 over three families, plus the corner-degree systematic 1.94e-5 in quadrature. No AICc: not computable on 5 points with 3 parameters |
| C2 | Independent of the reference in METHOD, not in DATA | WITNESSED (narrowed) | Discretization, formulation and solver are independent; but W_x and THXX are read from the reference profile as gauge targets and the production seed is interpolated from it. Freeing THXX moves alpha by 9.73e-5, about 2x the bar. Seed dependence separately closed at 9.65e-8 |
| C3 | Third method in existence for this object; first that solves it as a root problem with an explicit sparse Jacobian | WITNESSED | Prior art: adaptive-mesh march (CAP line), PINN + Gauss–Newton |
| C4 | Free gauge residual d_cl → **8.7e-7** | WITNESSED | Imposed by nothing; ~2500× better than any prior configuration here |

**Axes probed (worst measured effect on alpha; NOT all closed):**

| axis | worst effect on alpha | status |
|---|---|---|
| Nb (36 → 48) | 1.16e-8 | closed |
| seed provenance (Chen–Hou interp → 3 pure-analytic seeds, L=2/4/8) | 9.65e-8 | closed |
| axis-column pinning | 3.0e-8 | closed |
| corner-panel degree (24 → 28) | 1.94e-5 | in the quoted bar |
| corner-panel degree (16 → 24) | 6.98e-4 | 16 is not converged |
| XMAX (25 → 32), *on alpha* | 1.45e-10 | closed |
| middle panel edge (15 → 18), *on alpha* | 1.45e-10 | closed |
| eps_b wedge truncation | extrapolated, layer analysed | quoted, not hidden |
| Nb (36 → 28), *spectral abscissa* | 1.56e-6 | closed |
| XMAX (25 → 18), *spectral abscissa* | 3.36e-5 | closed |
| degs (16,40,12) → (24,56,12), *eps\** | 21.5% | quoted with band, NOT closed |

The seed and axis-pin rows were measured at a configuration where alpha itself is
~0.7% from the extrapolated value: they bound the sensitivity of the ROOT FOUND, not
of the extrapolated exponent.  XMAX = 25 and both panel edges are frozen in every run
of this solver; one run at XMAX = 32 would close that axis.

## Method contributions

| # | Claim | Grade | Backing |
|---|---|---|---|
| M1 | Keeping Pt as an unknown makes every operator local → sparse exact Jacobian (the eliminated form is dense at any grid) | WITNESSED (structure) / **unsourced (speed)** | Sparsity is structural and checkable. The "~20× faster" figure has NO timing log on disk — do not print it until one exists |
| M2 | **Corner dust**: wall-line transport rows below xi ≈ 0.025 are O(xi)-scaled and go dependent | WITNESSED | Consistent with the dense solver's N≈52–56 ceiling, the L=10 failure and the N=52 outlier (all satisfy the threshold; the successes do not). "Perfect separation" downgraded to "consistent with" — the sample is the failures we happened to record |
| M3 | **Global corner regularization** goes through that wall: Ot = xi A, Bt = xi² B, Pt = xi² P with analytic cancellation | WITNESSED | Corner degree 16→24 with d_cl falling monotonically 3.5e-4 → 2.6e-5 |
| M4 | **eps_b singular layer**: the truncated wedge shifts the corner exponent to k = pi/(pi/2 − 2 eps_b), so P carries a weakly singular xi^(k−2) layer no polynomial basis represents; deep-corner discretizations then admit *no root we could find* | WITNESSED | Floors ~1e-4 with every Newton step accepted, both solvers; vanishes at eps_b = 1e-4 (3–4 step convergence) |
| M5 | **Exact scaling symmetry**: A→sA, B→s²B, P→sP, c_l→s c_l, c_w→s c_w (residual degrees 2/3/1); **alpha is invariant** | WITNESSED | sympy unique grading + 896 covariant rows to 1.3e-15 on the live solver; Euler check J·v = d·F to 1.0e-15 |
| M6 | A-posteriori gauge residuals as solver diagnostics (`h_id = c_l − 2 THXX/WX`) | WITNESSED (not ENFORCED) | RETURNED in converge()'s info dict on success, integrated with alpha unchanged to 2.6e-9. Not attached to the zero_steps / outer_cap returns and no gate refuses a bad value, so the ENFORCED grade is not earned |

## Spectrum / linear stability

| # | Claim | Grade | Backing |
|---|---|---|---|
| S1 | The linearization is a **descriptor pencil (E, J)**, E a 0/1 diagonal mask, Hessenberg index 2; the solver's own Jacobian is the whole operator | WITNESSED | E = I on live transport rows, derived twice (sympy on the coded RO'/RB' to 5.8e-157; c_w-column identity exact); QZ 376 finite / 346 infinite = (N − rank E) + m closes exactly, matching the compressed generator to 3.7e-10 |
| S2 | The scaling symmetry is removed by the **DAE restriction itself**, not by deflation: sigma_min(J) = 2.5356e-6 -> sigma_min(L on ker Cg) = 3.9701e-3, a **1565.7x rise** | WITNESSED | Reproduced at three roots (1565.7x / 1724.0x / 1529.1x), ‖Cg w‖/‖w‖ = 3.9e-17; two independent resolvent routes agree to 3.0e-11 (bar 1e-8) |
| S3 | Deflating the grading direction would be **wrong, not merely cosmetic** — its restricted shadow is an ordinary stiff direction of the generator | WITNESSED | ‖L w1‖/‖w1‖ = 12.079 = 3043x sigma_min(L) = 1.09% of ‖L‖; deflating moves ‖R(+0.50)‖ by 19.6 / 17.1 / 20.2% |
| S4 | **Hurwitz on the corner-clamped admissible class, certified without trusting any eigenvalue** | WITNESSED | Lyapunov L^T P + P L = −I (Bartels–Stewart), relative residual 4.85e-16 / 4.88e-16 / 4.68e-16; cholesky(P) succeeds at all three roots |
| S5 | **Crossing verdict**: no eps-pseudospectrum reaches Re z > 0 below eps* = 3.93e-3 (A) / 3.08e-3 (B) — worst margin **8.8e5** above the round-off floor 1.175e-9. The *verdict* survives refinement; the *level* does not | WITNESSED (verdict) / quoted-with-band (level) | RHP minimum sits on the axis (3.946e-3 vs 8.984e-3 over Re > 0), as Davies–Shargorodsky requires; Nb 36->28 moves the abscissa 1.56e-6, XMAX 25->18 moves it 3.36e-5 (a COARSENING, not a refinement); eps* itself moves 21.5% across degs. Structural caveat: the untruncated essential spectrum sits on the axis, so eps* is expected to shrink as XMAX grows |
| S6 | **Transient growth is real**: 89.66 <= sup_t ‖e^{tL}‖ <= 5807.6 (A); 148.7 .. 7973.8 (B) — in the raw collocation norm, which is *not* similarity-invariant | WITNESSED | Kreiss K at an interior maximum z = +2.150; upper bound sqrt(kappa(P)), kappa(P) = 3.37e7 |
| S7 | **omega(L) is a grid scale, not a growth rate** — the single-dominant-block model predicts omega/‖L‖ = 1/2 exactly; we MEASURE 0.4952 / 0.4967 / 0.4952 | WITNESSED | 0.49516 / 0.49666 / 0.49523 across three roots; max Re W(S)/k -> e^{a0 xi} xi/2, measured 5.140972e-1 vs closed form 5.140981e-1; zeroing J_AB caps max Re at +15.68 |
| S8 | **W_e = C**, not the imaginary axis — the corner-regularized symbol's numerical range grows linearly in k, so pollution confinement is vacuous in this norm | WITNESSED (**retracts** the campaign's earlier closed-form essential-numerical-range claim) | max Re W(S) = +5.1409e4 at k = 1e5, +5.1410e5 at k = 1e6; the undivided far-field limit −i c_l k is what gives the imaginary axis |
| S9 | **The rightmost eigenvalue is unconverged as a discretization at the two resolutions computed** (not "unmeasurable in principle") — two routes agree on it to 7.2e-9 (worst of two roots) while two grids disagree by 2.14e-1 and it changes character (complex pair -> real) | WITNESSED (as a negative) | 7.5 orders between agreement and grid-disagreement; kappa(lambda) = 1.12e3 / 2.27e3 / 1.51e3; shift-invert Arnoldi at sigma = +0.5 would have found an RHP eigenvalue first, and found none |
| S10 | **Unconditional linear stability is NOT claimed** | — (refusal) | Lyapunov rate 8.82e-6 is 3.6e4x below the Schur abscissa 3.17e-1 ("stable" = no RHP spectrum, not "decays"); the class is corner-clamped; eps* unconverged |

*Note on S2/S3: the prerequisite recorded on arc `leray-ladder` node 4 predicted the
grading direction would need an explicit projector. Measurement corrected the premise —
the correct DAE realization makes it inadmissible by construction, and projecting it out
by hand would have injected a ~20% resolvent error. The prove-by (sigma_min rising to the
true operator scale) was met; the mechanism was not the one predicted.*

## The negative result (its own section)

| # | Claim | Grade | Backing |
|---|---|---|---|
| N1 | Deflated multistart at frozen alpha produced a root converging to **5.4e-14**, distinct (relative distance 1.70), eps-stable, morphologically coherent — and **false** | WITNESSED | Full anatomy in ALPHA_RESULT.md §7 |
| N2 | It is a **discretization ghost**: alpha drifts −0.42173 → −0.42555 → −0.43084 with *growing* steps; h_id violated by O(1) throughout (+2.39 → +0.88 vs ground −1.06e-3) | WITNESSED | Three independently-hunted converged grids |
| N3 | The corner identity **cannot serve as a closure**: the pinned family self-parallels it (dc_l/dTHXX = +1.677 vs 2/WX = +1.672, 99.7% cancellation) | WITNESSED | Freed-pin formulation built, Jacobian-verified 4.2e-11, failed its ground-recovery control — the failure is the measurement |
| N4 | Residual, distinctness, eps-stability and morphology are **jointly insufficient** to certify a root; a free residual is what separates real from false | WITNESSED | N1–N3 together |

**This is the note's most transferable content.**  Every group solving self-similar
profiles by collocation can manufacture N1; few report it.

## Explicitly NOT claimed

- No confirmation of DeepMind's alpha_1/alpha_2/alpha_3.  The one candidate this
  campaign produced was adjudicated a ghost.  (Route recorded as open, not closed:
  alpha as a Newton unknown with its own normalization, on the free problem.)
- No d(alpha)/d(theta).  The +1.40/rad figure is **retracted** — the slope is
  formulation-dependent (−2.8 in the panel frame vs ~−30 regularized).  It needs an
  angle-parameterized solve.
- No **unconditional** stability of the corner profile.  What is measured is
  Hurwitz-ness of the *corner-clamped* linearization at two resolutions, in the *raw
  collocation norm*, at *fixed eps_b* on a *truncated* wedge.  The corner pin deletes
  the continuum 2-D corner ODE; the axis column at beta = pi/2 − eps_b is an extra
  Dirichlet condition on the perturbation whose cost on the spectrum is unmeasured
  (its cost on alpha was 3e-8).
- No individual eigenvalue.  The rightmost moves 2.14e-1 between grids while two
  routes agree on it to 1.3e-9 — well-posed as linear algebra, unconverged as
  discretization.
- No decay rate.  The Lyapunov rate 8.82e-6 and the Schur abscissa 3.17e-1 are 3.6e4
  apart, and neither is extrapolated in XMAX or N.
- No claim that the quoted bar is smaller than 5e-5.  The three fit families
  straddle the reference; picking the closest one would be cherry-picking.

## Reproduction

Solver `polar_cornerreg.py`; campaign record `ALPHA_RESULT.md`; every number above
traceable to a logged run.  The engine ledger (witnesses, refusals, promotions,
resolved tensions) is a replayable audit trail of which claim was minted at which
strength at which moment.
