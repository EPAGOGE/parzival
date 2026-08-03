# NU-CURVE CAMPAIGN SPEC (pre-registered 2026-08-02, before any run)

EJA #76, all three items in one slice: the sigma_Lambda(nu) CURVE, the
crossover amplitude A_c(nu) as its own observable, and denser snapshots to
fatten the thin deep windows. Follow-up paper's spine; also strengthens the
posted study's weakest axis.

## Runs (10, sequential, both grids at every point)

Scenario identical to the certified campaign: --scenario --A 100
--ic-power 4 --zpow 1 --r0 0.4 --tmax 3e-3, free-slip viscosity via --nu.
ONE deliberate change, stated: --ckpt-sim-dt 3e-5 (was 6e-5) -- 2x snapshot
density, targeting deep-window n ~ 22-28 (was 11-14). --ckpt-max-writes 60.

  tags C128_{nu} at 128x384 (~1.5 min each) and C256_{nu} at 256x768
  (~6 min each), nu in {1e-3, 1e-4, 3e-4, 3e-5, 1e-5}.

The certified points (1e-3, 1e-4) are RERUN at the new cadence: replication
of the certified values by an independent run at different cadence is a
free regression test; disagreement beyond spread is a finding, not an
embarrassment. Existing runs/tags are NOT touched (vault rules; new tags).

## Pre-registered analysis (before any new number is seen)

1. Per nu: the CERTIFIED estimator, verbatim (G1 signed circulation-drift
   gate, sigma_peak row recipe, cross-grid overlap, top half, symmetric
   midpoint cut, OLS + 10k pair bootstrap rng 271828). Cross-grid spread
   quoted per nu; the 0.15 bar applies per point.
2. A_c(nu) DEFINITION (fixed now): per run, on the full gated+trimmed row
   set, fit the lower-window line (rows below the overlap midpoint) and the
   upper-window line (rows above); A_c = the amplitude where the two lines
   intersect. Cross-grid A_c spread quoted the same way. If the lines are
   near-parallel (slope gap < 0.5), A_c is UNANSWERABLE for that run --
   report, do not force.
3. Curve read-out: sigma_deep(nu) at 5 viscosities + the nu=0 matched
   anchor (+0.589/+0.585). Monotonicity in nu is a hypothesis to TEST, not
   assume. A_c(nu) scaling: fit ln A_c vs ln nu ONLY if >= 4 answerable
   points; otherwise table only.
4. Replication check: C-tags at 1e-3/1e-4 vs certified M4 values;
   |delta| <= max(spread, CI half-width) = replicated, else TENSION minted.

## Refusals (inherited, binding)

Cholesky-only definiteness (not used here but standing); no eigenvalue
decisions; windows pre-registered above, never tuned post-hoc; retracted
window-averages stay retracted (no full-window sigma quotes); honest wall
times; runs/ raw never deleted (vault); disk gate via tools/free_space.sh
before launch; laptop only.

## Outputs

runs/ C-tag streams+snaps; NU_CURVE.out (analysis); nu_curve_data.npz;
EJA: consume/resolve #76 items as they land; Token3 bank at completion.

## B-TEST: #77 discriminating test (pre-registered 2026-08-02, before runs)

Vary IC amplitude A0 at fixed nu=1e-4, both grids; watch the crossover band.
  runs: B128_A50/B256_A50 (--A 50, tmax 6e-3, ckpt-sim-dt 6e-5)
        B128_A200/B256_A200 (--A 200, tmax 1.5e-3, ckpt-sim-dt 1.5e-5)
  (times/cadence per the exact amplitude symmetry, as bootstrap's SYMa/SYMb;
   ~100 snapshots each; baseline A0=100 = existing C-tags at 1e-4.)

BAND ESTIMATOR (split-free, replacing the void chord A_c): sliding-window
local slope sigma_loc(A) over k consecutive rows in ln A; band center
A* = the amplitude where sigma_loc first crosses zero from above (linear
interpolation between window centers). Quoted per grid at k = 11/15/19;
k-sensitivity IS the invariance check -- if A* drifts with k beyond the
cross-grid spread, A* is unanswerable (report, do not force).

HYPOTHESES (fixed now): H_IC: A*(A0)/A*(100) = A0/100 (band is IC/Euler
heritage; the exact amplitude symmetry scales all field amplitudes).
H_intrinsic: ratio = 1 (band set by corner scale r0 or mechanism-internal
amplitude). Discrimination bar: factor 2 vs observed ~15% scatter; verdict
requires BOTH grids agreeing on the same hypothesis at BOTH A0 values.
If neither hypothesis fits (intermediate scaling), report the measured
power of A0 with no forcing.

Gates/trim/rng: unchanged (G1, 1e-6 tail, 10x trim, rng 271828).
Outputs: B_BAND.out, b_band_data.npz. EJA #77 updated per outcome.

## CROSS-TERM + FORM (pre-registered 2026-08-02 evening, before runs/fits)

(a) CROSS-TERM: A0 in {50, 200} at nu=1e-3 (second viscosity), both grids.
    Tags B128_A50n3/B256_A50n3 (tmax 6e-3, dt 6e-5), B128_A200n3/
    B256_A200n3 (tmax 1.5e-3, dt 1.5e-5). Baseline: C-tags at 1e-3.
    SHARP PREDICTION, fixed now: A*/A0 = 54 within +-10% (observed scatter
    0.6%; bar generous) in all four new measurements, by the certified
    zero-crossing estimator at k=15 (k=11/19 as invariance). All four in
    => growth-factor invariant certified on the (nu, A0) plane. Any out
    => the invariant is nu- or A0-conditional: report where it breaks.

(b) FUNCTIONAL FORM: fit ln Lambda vs x = ln A per run with the smooth
    two-slope model  y = c + (s1+s2)/2 (x-x0) + (s2-s1)/2 w ln cosh((x-x0)/w)
    (slope = tanh blend between s1 and s2; least squares, all rows in the
    gated+trimmed set). Observables per run: center x0 (report exp(x0)/A0),
    width w (e-folds of A), asymptotic slopes s1, s2. Consistency demands:
    fit-implied slope zero crossing matches the measured A*; s2 within the
    certified deep CI; exp(x0)/A0 and w consistent across runs = the form
    is universal. No forcing: poor fit (resid RMS above ~1.5x the linear
    deep-window resid) = report MODEL INADEQUATE.

## BRIDGE CLOCK FIT (pre-registered 2026-08-02 late, before fitting)

Per run (inviscid OR pair + viscous C pairs at 1e-4/1e-3): gated (t, A)
series; fit ln A = -gamma_A ln(1 - t/T*) + ln A_i by grid-scan over T*
(least squares in the linear-in-lnA sense). Report gamma_A, T*, A_i, and
s* = -ln(1 - t*/T*) where fitted A(t*) = 54 x A0 (the certified switch).
Sensitivity: quote gamma_A and s* across the T* values within 2x the
minimum residual -- if s* varies by more than a factor 2 over that set,
the clock fit is UNANSWERABLE on this data (report, no forcing).
March-side comparison targets (fixed): fast transient peak s=0.5, slow
mode e-folding 1/0.270 = 3.7 s-units, deep re-cross s_transient=13.5.
Hypothesis space stated in advance: s* << 3.7 = switch precedes lock-on
(viscosity owns the geometry during approach); s* ~ 3.7-13.5 = switch
coincides with lock-on (the original bridge hypothesis); s* >> 13.5 =
switch after settling (would contradict the 1.43x onset reading).

## LARGE-DEVIATION MARCH (pre-registered 2026-08-02 night, before running)

Close the bridge caveat from the march side: does the fast-transient peak
(s=0.5, linear-certified) survive at large deviation? Amps 3e-3, 1e-2,
3e-2, 1e-1 (30x past the validated basin), seed 0, ds=0.25, <=240 steps,
M2 stop rule (true residual < 1e-10 each accepted step; QN>15 or
unconverged => clean stop; the stall amplitude IS the measured basin/
instrument edge). Accepted steps are valid at ANY amplitude (acceptance
requires the TRUE nonlinear residual to converge; the frozen Jacobian
only cheapens the solve).

PRIMARY READOUT: s_peak(amp). BAR (fixed now): timescale robust =
s_peak in [0.25, 0.75] (within one ds-step of 0.5) at every amp with a
valid march past the peak. SECONDARY: peak factor (nonlinearity gauge;
was +0.25% at 1e-3), s_transient, P-norm V(v) violations WITH V-floor
adjudication (violation at V within 100x of the 5e-16 floor = artifact;
violation at V >> floor while residual-valid = GENUINE nonlinear P-norm
growth, the wandering sector's first signature -- report loudly).
Output: LD_MARCH.out. EJA: closes the bridge caveat or reports the edge.

## CLOSING SWEEP (pre-registered 2026-08-02 late night, before running)

(A) NU-GAP EXTENSION: C128_3e-6/C256_3e-6/C128_1e-6/C256_1e-6, recipe
    identical to the nu-curve runs (tmax 3e-3, dt 3e-5, A0=100).
    Certified estimator verbatim. Outcomes, fixed now: sigma_deep in the
    -0.99..-1.31 band with spread < 0.15 = flat continues, gap narrows
    below 1e-6; both grids agreeing outside the band toward the inviscid
    value = crossover FOUND (quote it); window/gate failure or spread
    over bar = unanswerable at that nu (report).
(B) WITHIN-RUN TURNAROUND TEST: for the six clock-fit runs, profile
    series U per gated snapshot (features.py-style amplitude-and-length
    normalized crop), deviation speed V_k = |U_{k+1}-U_k| per unit s on
    the run's own fitted clock; s_turn = position of the smoothed
    (5-point median) V peak. BAR: |s_turn - s*| <= 0.25 (one march step)
    counts as within-run coincidence; require >= 5 of 6 runs inside to
    claim it. Else report the offsets, no forcing.
(C) BASIN END: march amps 5e-2 and 7e-2 seed 0 (bisect the 3e-2..1e-1
    bracket), plus 3e-2 and 1e-1 with seed 1 (direction dependence).
    Same stop rule; stall = basin-or-instrument edge as before.

## 1E-6 DEEPENING ADJUDICATION (pre-registered before the replica runs)

Question: is the 128-grid 1e-6 value (-1.366, 4% below the band edge)
trajectory-real or row-sampling scatter? Instrument: cadence replicas
R128_1e-6/R256_1e-6 at ckpt-sim-dt 1.5e-5 (the trajectory is
deterministic; changing output cadence resamples the SAME trajectory
independently -- the campaign's established replication tool).
DECISION RULE, fixed now:
  DEEPENING REAL (at that grid) = replica 128 value also below -1.31
    AND the 0.4/0.5/0.6 cut-sensitivity keeps 128 below the band at
    >= 2 of 3 cuts on BOTH cadences.
  SAMPLING SCATTER = replica lands in-band, or cut positions scatter
    the value across the band edge.
  Secondary readout (report only, no verdict): slope of sigma vs ln nu
  over the three lowest nu per grid, as a trend indicator.
