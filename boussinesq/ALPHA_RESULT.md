# The alpha campaign — result and record

**2026-07-25 → 2026-07-27.  Status: CLOSED — reproduced.**

---

## 1. The result

The blowup exponent of the Luo–Hou / Chen–Hou corner singularity (2D Boussinesq /
3D axisymmetric Euler with boundary), computed by a formulation sharing nothing
with either existing method:

```
alpha(eps_b -> 0, quadratic) = -0.34240089
reference (Chen-Hou == DeepMind boldface band) = -0.34240009
```

**Quoted honestly: alpha = −0.34240 ± 3e-5** (bar = linear/quadratic extrapolation
spread), consistent with the cross-method reference (itself known to 3.4e-7) at
one part in ten thousand.  The final ladder, every rung carrying a genuine
converged flag and ‖F‖ ~ 1e-12:

| deg0 | eps_b | alpha        | vs ref   | d_cl      | passes | secs |
|------|-------|--------------|----------|-----------|--------|------|
| 24   | 1e-4  | −0.34541032  | −0.879%  | +2.6e-05  | 9      | 297  |
| 24   | 5e-5  | −0.34386167  | −0.427%  | +1.4e-05  | 8      | 257  |
| 24   | 3e-5  | −0.34312079  | −0.210%  | +7.1e-06  | 7      | 198  |
| 24   | 1e-5  | −0.34268591  | −0.083%  | +3.2e-06  | 6      | 152  |
| 28   | 1e-5  | −0.34270532  | −0.089%  | +1.8e-06  | 6      | 196  |

`d_cl = c_l/(2·THXX/WX) − 1` is the **free gauge residual**: the exact
continuum corner identity `c_l = 2·θ_xx(0)/ω_x(0)`, imposed by nothing in the
solve and answerable to no reference value.  It marched to **1.8e-6** — roughly
2500× better than any configuration before the corner-regularized solver.
Corner-degree check (24 → 28 at fixed eps): Δalpha = 1.9e-5, inside the bar.

**Why it matters.**  This is the third method in existence for this object:
Chen–Hou march with slaved gauge (adaptive-mesh MATLAB, computer-assisted-proof
line), DeepMind/Wang et al. PINN + Gauss–Newton, and now an exact-Jacobian
sparse spectral Newton with a-posteriori gauge residuals — the only one of the
three that solves the profile as a root problem with an explicit, checkable
Jacobian, and the only one reporting self-consistency diagnostics at 1e-6.

---

## 2. The instrument (files, in dependency order)

- **`polar_panels.py`** — piecewise-Chebyshev radial panels, duplicated interface
  nodes with classical patching (C0 transport / C0+C1 Poisson), and the decision
  that unlocked everything: **Pt kept as an unknown** instead of LU-eliminated,
  making every operator local and the exact Jacobian sparse (the eliminated form
  is dense no matter the grid).  ~20× faster per solve than the dense pipeline.
- **`polar_cornerreg.py`** — the endgame solver.  Global regularization
  `Ot = ξA, Bt = ξ²B, Pt = ξ²P`, equations divided by ξ, ξ², ξ² with every
  cancellation analytic (`g = ξG₁`, `G₁ = −expm1(−ξ)/ξ`).  Corner circle carries
  radial-extrapolation rows for P and analytic-profile pins for A, B; gauge
  closed by the divided d1 functionals.  Newton guarded by a slaved
  Levenberg step (projection onto the linear-row manifold; load-bearing past
  corner degree ~20).
- **Supporting instruments:** `polar_zeros.py` (invariant zeros via QZ on the
  Rosenbrock pencil, slice-basis compression, closed-form essential numerical
  range), `polar_deflate.py` + `polar_deflate_gate.py` (Farrell deflation with
  shift, root atlas, simplicity certificate), `polar_nlens.py` (per-step Newton
  annotation: residual localization, curiosity flags), plus gates
  `polar_constraint_gate.py`, `polar_outer_gate.py`.
- **Harness discipline** (bought with blood, now standard): converged **flags**
  never bare residuals; accepted-step counts (the phantom-result signature is
  "perfect alpha, zero steps"); `open_residual()` beside every ‖F‖ (the masked
  norm is blind on the axis line by eleven orders); the free residual `d_cl`
  printed beside every alpha; stall watchdogs on every background run.

## 3. The mechanism stack — what had to fall, in the order it fell

1. **Harness phantoms** — `steps=10` vs a 16–22-iteration damped phase;
   unconditional warm-start discard; undamped outer alpha map returning
   period-2 cycles; zero-step returns of the seed reported as α =
   −0.34240009311696556 at ‖F‖ = 1.8e-2.  *The old basin tables were artifacts.*
2. **The d2 corner constraint row** — `|Dx2[0,:]|₁ ~ N⁴` against a seed tail
   flooring at 2e-8; coherence loss N^3.5.  Replaced by d1 (N^1.0).
3. **The eps_b domain bias** — the offset grid *is* a wedge of opening
   π/2 − 2eps_b (λ₁ matches the truncated-wedge value to 3.4e-13, Nb-invariant),
   so Nb-sweeps could never see it.  First-order in the opening; extrapolated.
4. **Grid parity / the 2-D surface** — the N-ladder was a 1-D cut; matched-N/L
   pairs disagreed by 6.7 points with opposite signs (pre-registered test).  No
   N-only extrapolation of that instrument was ever legitimate.
5. **Corner dust** — transport rows at wall nodes with ξ < ~0.025 are near-vacuous
   (O(ξ)-scaled) and go dependent; **retro-explains the old dense solver's
   N≈52–56 ceiling, the L=10 failure, and the N=52 outlier** with perfect
   separation on one threshold.
6. **The hidden corner-panel axis** — every "converged" panelization shared
   corner deg 16 + the dust rule; cross-panelization agreement was structurally
   blind to a rule-common bias.  Dust-free sweep: deg 12→14 walks +2.56% →
   +0.297% with d_cl −2.82% → −0.45%.
7. **The corner resonance + decoupling** — the regularized corner row
   `P_ββ + 4P = 0` has exactly zero O(1) coupling to the interior and sits 1e-2
   from singular against its own sin2β mode (the π/2 wedge's integer resonance):
   homogeneous + decoupled ⇒ forced P(0,·)=0 against a corner algebra demanding
   c·sin2β, c ≈ 1.27.
8. **The double-encoded corner algebra** — collocating transport on the corner
   circle *and* pinning A(0,0), B(0,0) imposes `c_l = 2·THXX/WX` twice;
   the disagreement at discretization level makes the system **provably
   rootless** (Levenberg-invariant floor).
9. **The eps_b singular corner layer** — the truncated wedge shifts the corner
   exponent to `k = π/(π/2−2eps_b)`, so `P = Pt/ξ²` carries a weakly singular
   `ξ^(k−2)` layer no polynomial basis represents.  Coarse corner panels can't
   see the misfit; deep ones resolve it and the discrete system has **no root**
   (floors ~1e-4, both solvers, every step accepted).  Vanishes as eps_b → 0 —
   which is exactly why the final ladder converges and where alpha lives.

## 4. Retraction ledger (claims made and killed inside the campaign)

- the +1.05 "symmetry mode" (no eigenvalue near +1; three agreeing routes);
- exponential BVP conditioning from far-field growing modes (cond flat in XMAX);
- the log-periodic fit (AICc refused; LOO worse than the mean; surrogates fake it);
- "only eps_b = 1e-3 converges" (a `steps=10` artifact);
- the formulation-delta axiom ("different continuum problems") — killed by its
  own pre-named falsifier;
- "every axis is converged" — said twice, wrong twice (corner panel; then eps_b
  singular layer);
- dα/dθ = +1.40/rad as a *measurement* — the eps-slope is formulation-dependent
  (−2.8 in the panel frame, −30 in the regularized frame): each frame's
  angle-systematic dominates.  The corner-opening *identity* stands; the
  derivative needs an angle-parameterized solve.
- every flattering coarse agreement (+0.085%, +0.15%) was a cancellation of two
  systematics.  The pattern held every single time.

## 5. Trust statement

Reference confirmed to 3.4e-7 by two disjoint external methods (Chen–Hou
adaptive-mesh MATLAB vs DeepMind PINN; boldface-validated digits only).
Internally: exact Jacobians FD-verified (2e-10); two independent solver variants
agree on converged alpha to 9 digits; deflation with seven heterogeneous seeds
finds exactly one root; the corner constraints' targets verified against
Chen–Hou's stored constants to 8.1/7.1 digits; `d_cl` free residual at 1.8e-6;
open-system residual reported beside every masked norm.

---

## 6. What is left — the next steps, ranked

1. **Tighten the quote (half a day, optional but cheap).**  Add Nb = 48 and
   eps_b = 5e-6 rungs at deg0 = 24–28; three-point Richardson in eps.
   *Gate:* bar shrinks below 1e-5 with d_cl still falling.  Then the quoted
   number is alpha = −0.342400 ± 1e-5 from a third independent method.
2. **Write the numerics note (now writable).**  Contributions that exist today:
   first non-march, non-PINN solver for this profile; the corner-dust mechanism
   (explains the dense-solver ceiling class); the eps_b singular-layer analysis
   (`ξ^(k−2)`, provably-rootless deep-corner discretizations); a-posteriori
   gauge residuals as solver diagnostics; the reproduction itself.
   Repro scripts + this document are most of the paper.
3. **The unclaimed prize: unstable branches α₁, α₂, α₃** (DeepMind values
   −0.4168236 / −0.4439811 / −0.4578230, validated to only 6/5/4 digits,
   confirmed by no non-PINN method; their machine-precision follow-up skipped
   Boussinesq).  Requirements established by the premortem: **alpha as a Newton
   unknown or a secant on h(a) = c_w/c_l(a) − a** (the damped fixed-point outer
   loop cannot reach a repelling fixed point); seeds for n ≥ 1 (no reference
   profile exists — continuation from n=0 with deflation of the stable root);
   sub-percent accuracy *with provenance* before attribution (CHL's stage-2
   limit −0.40834 sits 0.0085 from α₁).  *Gate per branch:* converged
   open-system ‖F‖ ≤ 1e-12, both gauge residuals, an (N, XMAX, eps) study.
   Beating their residual by orders in an interval-arithmetic-ready form is the
   computer-assisted-proof input nobody has.
4. **The spectrum, at last.**  On a converged profile: resolvent norm and
   transient growth (well-conditioned) rather than eigenvalues (proven
   unmeasurable at these resolutions); invariant zeros via QZ outside the
   closed-form essential numerical range; explicit realization declaration
   (`polar_admissible`).  This is the campaign's original purpose — Liu's
   route — and it was always gated on a converged profile.  It now has one.
5. **The corner-angle derivative, done right.**  An angle-parameterized solve
   with k(θ)-adapted corner treatment, replacing the retired +1.40.  A real,
   novel, publishable number if the θ-family converges.
6. **Housekeeping.**  POLAR_SPEC corrections (line 385's r~1e8 claim; §16 vs
   the corner rows; §17's circular gauge validation — all documented, none yet
   edited into the spec); phantom-signature audit of the old alpha tables;
   fold the harness rules into a HARNESS.md; Nb-sweep the cornerreg solver once
   for the record.

---

*Method note for the record: every mechanism above was found by measurement
under pre-registered gates, most by adversarial multi-agent verification, and
each flattering number the campaign ever produced died on contact with a free
residual.  The discipline is the result as much as the number is.*

---

## 7. The branch hunt and the ghost (2026-07-27/28)

*Supersedes section 6 item 3: the route described there was taken, and it closed.*

### What was attempted

With alpha_0 closed, the unclaimed ground was DeepMind's unstable branches
(alpha_1 = -0.4168236, alpha_2 = -0.4439811, alpha_3 = -0.4578230; validated to
6/5/4 digits, confirmed by no non-PINN method).  The engine's promoted invariance
`dh/da = -1` on the ground branch DEDUCED the only surviving instrument: since
h(a) = a* - a identically along a field branch, the h-landscape can never reveal
another branch, so alpha must be frozen and FIELD space searched by deflation.

### What was found, and what it turned out to be

Deflated multistart at frozen a = alpha_1 found a distinct converged root
(||F|| = 5.4e-14, relative distance 1.70 from ground, from a half-amplitude start;
1 of 8 starts).  It survived four rounds of scrutiny -- eps-flat where the ground
state was not (1.3e-5 vs 2.3e-3 over eps 1e-4 -> 1e-5), same corner algebra to
0.1%, its own sign-flipped cos3b fingerprint -- and then failed on the fifth:

| resolution        | alpha        | h_id = c_l - 2 THXX/WX |
|-------------------|--------------|------------------------|
| (16,40,12)        | -0.42172919  | +2.386                 |
| (24,56,12)        | -0.42554621  | +0.994                 |
| (28,64,18)        | -0.43083651  | +0.880                 |
| *ground, for scale* | *-0.34471229* | *-0.00106*           |

Refinement moved alpha AWAY from alpha_1 with GROWING steps (-3.8e-3, -5.3e-3),
and the free corner identity -- imposed by nothing, the same functional that
certified alpha_0 at 1e-6 -- was violated by O(1) throughout.

**The freed-pin experiment settled it.** Promoting the corner data (WX, THXX) to
unknowns and closing with the identity itself: the formulation was built and
Jacobian-verified (4.2e-11) but FAILED its ground-recovery control, and the
failure is the theorem: **the pinned solution family self-parallels the identity
line** (dc_l/dTHXX = +1.677 against the identity's 2/WX = +1.672, a 99.7%
cancellation), so the corner identity is a diagnostic observable and cannot serve
as a closure.  The branch root then failed from both corner-data seeds
(||F|| floors 2.2-2.4e-3) with no identity sign-crossing anywhere physical
(extrapolated to 3x the real corner value).

**Verdict: a discretization ghost of the pinned formulation.**  Not alpha_1, not a
new solution -- an artifact that converges beautifully at each fixed grid and has
no continuum limit.  alpha_0 remains this solver's only validated root.

### What the hunt actually produced

1. **An exact scaling symmetry theorem.**  The corner-regularized system is exactly
   covariant under A -> sA, B -> s^2 B, P -> sP, c_l -> s c_l, c_w -> s c_w
   (residual degrees 2/3/1), with **alpha = c_w/c_l invariant**.  Verified
   symbolically (sympy, unique grading) and on the live solver: all 896 covariant
   rows to 1.3e-15, Euler check J.v = d.F to 1.0e-15.  The only rows that break it
   are the static pins and gauge targets -- and their breakage equals (s^k - 1)*field
   to 4.4e-16, exactly as predicted.  The mysterious near-mode measured on the ghost
   (lambda_B/lambda_A^2 = 1.0006) is this symmetry's grading fingerprint; the
   discrepancy 6e-4 IS the pin-breaking scale.
2. **The membership card.**  `h_id = c_l - 2*THXX/WX`, printed at every convergence
   (now integrated).  Ground -1.06e-3; ghost +0.88 to +2.39.  Three orders of
   magnitude, on a functional no solve is answerable to.  This is what a
   convincing-but-false root looks like from the outside, and it is cheap.
3. **A negative result worth publishing.**  A spectral formulation that reproduces a
   reference profile to 8 digits can still manufacture a root that passes residual,
   deflation-distinctness, eps-stability, and morphological tests.  The free-residual
   discipline is what separates them.

### Instrument changes integrated (verified: alpha unchanged to 2.6e-9)

- `CornerRegSolver(..., wx=, thxx=)` -- corner data is instance state, defaults
  identical to the REF constants.
- `adopt_seed(z)` -- warm-starting from a saved field must overwrite the pin data
  (measured mismatch 5.3e-2 otherwise; the pins fight the state they pin).
- `h_id(z)` and `info['h_id']` from `converge()`.

### The revisit lens (the arc stays open)

The branch family is real; what died is *this formulation's route to it*.  The
named next lens is a different access road: **alpha as a Newton unknown with its
own normalization** (Chen-Hou style) on the free problem, rather than a frozen
exponent on the pinned one.  Recorded as an open fault on arc `leray-ladder`
node 0, not a closed door.

---

## 8. The spectrum (2026-07-28) — the campaign's original purpose, answered

*Supersedes section 6 item 4.*

**The statement, at its honest grade:** the linearized self-similar evolution about
the corner profile — realized as the descriptor pencil (E, J) and restricted to
ker(Cg) — is **Hurwitz at every resolution computed**, certified *without trusting
any eigenvalue*: Lyapunov `L^T P + P L = -I` solved by Bartels-Stewart with relative
residual 4.85e-16 / 4.88e-16 / 4.68e-16 at three roots, `cholesky(P)` succeeding at
all three.  Companion pseudospectral statement: **no eps-pseudospectrum reaches
Re z > 0 below eps\* = 3.93e-3**, against a round-off floor of 1.175e-9 — a margin of
3.3e6, with the RHP minimum sitting on the axis exactly as Davies-Shargorodsky
requires.  The verdict survives refinement (Nb 36->28 moves the abscissa 1.6e-8;
XMAX 25->18 moves it 3.4e-5); the eps\* *level* does not (21.5% across degrees) and
is quoted with a band.

**What is explicitly refused:** unconditional linear stability.  The Lyapunov rate
8.82e-6 sits 3.6e4 below the Schur abscissa 3.17e-1 — "Hurwitz" means no right-half
spectrum, not "decays at this rate" — and the admissible class is *corner-clamped*:
the corner pin deletes the continuum 2-D corner ODE and the axis column is an extra
Dirichlet condition on the perturbation.  Three tensions stand open on that class.

**Three results that were not expected:**

1. **The symmetry needs no projector.**  The prerequisite (arc node 4) predicted the
   grading direction would have to be projected out.  Wrong, and measurably so: the
   correct DAE realization makes it *inadmissible by construction* —
   sigma_min(J) = 2.5356e-6 becomes sigma_min(L|ker Cg) = 3.9701e-3, a 1565.7x rise
   reproduced at three roots.  Better still, deflating it by hand would have been an
   **error, not a redundancy**: its restricted shadow is an ordinary stiff direction
   (‖L w1‖/‖w1‖ = 12.08, about 1% of ‖L‖), and removing it moves the resolvent by
   ~20%.  The prove-by was met; the mechanism was not the predicted one.
2. **W_e = C.**  The corner-regularized symbol's numerical range grows linearly in k
   (max Re W(S) = +5.14e4 at k=1e5, +5.14e5 at k=1e6), so spectral-pollution
   confinement is *vacuous in this norm* and the "outside the essential numerical
   range" region is empty.  **This retracts the campaign's earlier closed-form
   essential-numerical-range claim.**
3. **omega(L) is a grid scale, not a growth rate.**  omega/‖L‖ = 1/2 exactly
   (0.49516 / 0.49666 / 0.49523), because L is dominated by one off-diagonal block;
   the closed form e^{a0 xi} xi / 2 matches the measurement to 5.140972e-1 vs
   5.140981e-1.  Zeroing that block caps max Re at +15.68.

**The premortem, vindicated in numbers.**  The rightmost eigenvalue is agreed on by
two independent routes to 1.3e-9 — and disagrees between two grids by 2.14e-1, even
changing character from a complex pair to a real value.  Seven and a half orders of
magnitude between "well-posed as linear algebra" and "converged as discretization."
This is precisely the trap that produced the retracted "+1.05 mode," and the reason
every number above is eigenvalue-free.

**Transient growth is real and reportable:** 89.7 <= sup_t ‖e^{tL}‖ <= 5807.6 in the
raw collocation norm (which is not similarity-invariant, and is labelled as such).
