# HANDOFF: the s-march, and why the wandering question stalled

Written 2026-07-30 late. Read this before touching march_s.py or interpreting
any orbit measurement. It exists because the answers below were already in the
record and were rediscovered the expensive way.

## The one-paragraph state

Marching in PHYSICAL variables cannot answer settling-vs-wandering: work scales
as (T-t)^{-2 c_l} ~ 10^5.84 per decade, every physical run bought 1-2 e-foldings
and died. The fix is to march in the SELF-SIMILAR frame, where one e-folding of
amplitude is O(1) of rescaled time s. polar_cornerreg.py already solves that
system -- c_l and c_omega are carried AS UNKNOWNS in its state vector, and its
RO/RB rows already contain the rescaling terms -- so F(z) = 0 already means
d_s A = d_s B = 0. The steady solver IS the fixed point of the rescaled
dynamics. Marching is therefore a small addition to validated machinery, NOT a
new solver. It is not yet correct; four defects are listed below, two settled
and two open.

## THE ROW THAT EXPLAINS THE WHOLE STALL

NOTE_CLAIMS.md **S6**: transient growth is real,
    89.66 <= sup_t ||e^{tL}|| <= 5807.6   (root A;  148.7 .. 7973.8 at root B)

The linearization is Hurwitz (S4) -- it DOES settle asymptotically -- but a
perturbation can grow by up to ~5800x before decaying. Every orbit measurement
made on 2026-07-30 spanned 0.4 to 2.1 e-foldings. **All of it was inside the
transient.** Settling and wandering are indistinguishable there as a matter of
operator structure, not resolution, not estimator noise, not seed count.

Consequences, and they are the actionable part:
  * The night's INCONCLUSIVE verdicts were correct and were not a failure of
    the instrument. They were the instrument reporting honestly on a window
    that cannot carry the question.
  * More seeds do not help. More physical resolution does not help. Only
    running FAR ENOUGH IN s helps.
  * Any future orbit readout must state where it sits relative to the
    transient. A settling-vs-wandering claim inside it is void by S6.

## The four defects in march_s.py, graded

T1  Is the residual the raw RHS or a weighted version? A steady residual is
    invariant under multiplication by any nonzero weight, so this is
    unfalsifiable from the steady problem, and RO carries E1 = exp(a0 xi)/G1 on
    its advection terms but coefficient 1 on the cl/cw terms.
    **RESOLVED, in our favour.** S1: "E = I on live transport rows, derived
    twice (sympy on the coded RO'/RB' to 5.8e-157; c_w-column identity exact)."
    The mass matrix is the identity on exactly the rows the mass term belongs
    on. W = 1. The (A - A_old)/ds = RO structure is correct.

T2  rT_pin rows overwrite field rows with A[r] - A0[r] = 0, pinning to the
    SEED. Correct as a steady gauge; in a march it nails those points to the
    initial condition forever. march_s.py excludes them from the mass term
    (right) but leaves them pinned to a stale A0 (wrong).
    **OPEN.** Either refresh A0 each step or replace the pins with the corner
    conditions they stand for.

T3  Sign: is d_s A = +RO or -RO?
    **RESOLVED.** S4: Hurwitz certified WITHOUT eigenvalues, via Lyapunov
    L^T P + P L = -I, relative residual 4.85e-16 / 4.88e-16 / 4.68e-16,
    cholesky(P) succeeding at three roots. Match that convention. Do NOT
    determine the sign by computing eigenvalues -- see the refusal below.

T4  The system is a DAE, not an ODE.
    **OPEN, and the serious one.** S1 grades the linearization as a descriptor
    pencil (E, J) with E a 0/1 diagonal mask, **Hessenberg index 2** (QZ: 376
    finite / 346 infinite, closing exactly). Naive backward Euler on an index-2
    DAE suffers order reduction and can be unstable. march_s.py treats it as an
    ODE with algebraic rows bolted on. This is the defect that yields smooth,
    plausible, wrong trajectories. It needs an index-aware integrator or an
    explicit index reduction on ker(Cg).

## Standing refusals that bind any future attempt

S9  **Do not quote an eigenvalue of this operator.** Two independent routes
    agree on the rightmost to 1.3e-9 while two grids disagree by 2.14e-1, and
    it changes character (complex pair -> real). Well-posed as linear algebra,
    unconverged as discretization. Eigenvalue-based sign or stability
    determination is refused.

S3  **Do not deflate the grading direction.** Its restricted shadow is an
    ordinary stiff direction of the generator (||L w1||/||w1|| = 12.079, ~1% of
    ||L||); deflating moves the resolvent by ~20%. It is an error, not a
    redundancy. S2: the DAE restriction to ker(Cg) removes the scaling symmetry
    by itself (sigma_min 2.5356e-6 -> 3.9701e-3, a 1565.7x rise).

S7  omega(L) is a grid scale, not a growth rate (measured 0.4952 against the
    single-dominant-block prediction of exactly 1/2). Do not read the numerical
    abscissa as physics.

S10 Unconditional linear stability is NOT claimed. Hurwitz-ness is on the
    corner-clamped class, raw collocation norm, fixed eps_b, truncated wedge.

## Build order when this resumes

1. Fix T2: refresh the pin reference each step, or replace pins with the
   corner conditions.
2. Fix T4: index-aware integration on ker(Cg), reusing polar_spectrum.py's
   projections (project / project_oblique / restrict / prolong already exist
   and already respect S2/S3). Do not hand-roll a projector.
3. Calibrate the sign against S4's Lyapunov convention, not by marching.
4. Only then march -- and run long enough in s to clear a 5800x transient
   before reading any settling-vs-wandering verdict.

## Instruments that are correct and gated (use these, do not rebuild)

  verify.py        backwards proof, 15/15, three tiers, watcher-held
  mythos.py        confound engine (claims + anchors)
  criticality.py   sigma atlas, 2D-NS calibrated
  lambda_geom.py   Lambda = sup|grad xi| |omega|^-1/2, dual-gated
  sigma_peak.py    Lambda in the peak box, spectral+gamma gated
  orbit_speed.py   V with transient exclusion, bootstrap CI, structure-identity
  features.py      ~20 scalars + cos_step / cos_aim orbit direction
  lr_triple.py     three independent radial-scale estimators
  faults_freshpass.py   six blind reviewers, 14 faults, 6 results invalidated

## Live results, graded (STANDARD.md S1-S7)

ESTABLISHED   sigma_Lambda(inviscid) = +1.00 +- 0.03 (exact symmetry 3e-4;
              off-ray 2x2 factorial spread 0.029)
              Theorem 0 (Batchelor cap), unconditional, data-satisfied
              alpha_0 = -0.34240, two disjoint methods
              viscous SIGN inversion of sigma_PEAK, reproduced across 4x grid
ABDUCED       c_l = -1/alpha = 2.92056; branch family closing at -0.4722
              radial scale more IC-variable than axial (direction only,
              ~2.5x on the robust estimator -- the 8.9x figure is RETRACTED)
DEAD          reciprocal branch; Batchelor saturation gloss; sigma_PEAK
              monotonicity in nu; three collision sweeps
OPEN          the strip -1/2 <= sigma <= 0; the wandering sector (blocked by
              S6 transient, not by compute)
