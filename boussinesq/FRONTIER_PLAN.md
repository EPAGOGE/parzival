# FRONTIER PLAN -- the non-cheap truths (2026-08-03)

The cheap truths are in: two published papers, an addendum, and a ledger
where every open item now costs real instrument work. This is the plan,
in execution order, with costs and kill-criteria stated before anything
runs. Standing refusals inherit (cholesky-only definiteness, no
eigenvalue decisions, pre-registration before data, laptop unless stated,
honest wall times, vault rules).

## 1. Escape-vs-stall, settled by the exact marcher
COST: ~1 hour. STATUS: next up.
Rerun the stall points (seed 0 at 5e-2; seed 1 at 3e-2) with the
unmodified exact SMarcher (per-iteration dense correction; the frozen-J
quasi-Newton only cheapened the solve and is the suspect at stall).
Pre-registered outcomes: exact marcher converges where QN stalled =
instrument limit, the basin edge moves up and the QN region gets
re-bracketed; exact marcher also fails = first genuine dynamical-escape
observation of the campaign, quote it. Either way the
"basin-or-instrument" caveat resolves into one word.

## 2. Third grid: the resolution ladder
COST: one overnight batch (est. 3-5 laptop-hours). STATUS: queued behind 1.
Runs at 384x1152 (fine-side doubling; optionally 192x576 as an
intermediate rung) at nu in {1e-4 anchor, 3e-6, 1e-6}. Purpose: (a)
grid-convergence of sigma_deep beyond the single doubling the 0.15 bar
currently tests; (b) probe below the measured 128-grid floor at 1e-6 and
locate (or clear) the 256-grid's own floor. Bars: the existing
adjacent-pair spread discipline, applied per doubling; the cadence-replica
+ cross-grid artifact taxonomy from the addendum applies verbatim.

## 3. DNS-state march initialization -- the capstone build
COST: multi-day build. STATUS: the frontier.
Map a DNS snapshot (fields, DNS grid) into the corner-frame solver state
(spectral coefficients + gauge constants), then march the ACTUAL approach
trajectory in the certified frame. Validation gates pre-registered before
first scientific use: (a) round-trip interpolation residual bounded on a
settled state; (b) the E=I identity at the mapped state; (c) recover the
fixed point by marching a deep, settled DNS snapshot. Payoffs, in order
of value: the last bridge daylight closes (the approach measured in the
march's own coordinates); the turnaround observed directly in the
certified frame; escape-vs-stall for the DNS approach itself; the
trigger's dynamical identity read rather than inferred. Known risks,
stated now: inter-grid aliasing; gauge initialization (c_l, c_w) off the
profile; the DNS state begins outside the certified basin, so early-march
failure is a possible MEASUREMENT (where the approach enters the
certified neighborhood), not a build failure -- the gates distinguish.

## 4. IC-family universality of the trigger
COST: ~1-2 hours per family; runs in parallel with 2.
Vary the IC family (ic-power, r0) at one viscosity, both grids; re-measure
the trigger by the certified zero-crossing estimator. Hypothesis pair,
fixed now: H-universal = growth-factor triggering persists with
family-specific constants (the analog of 37.7 and 1.43 shift; the
STRUCTURE survives); H-special = the trigger is a quartic-family accident.
Either outcome scopes paper three's mechanism section.

## 5. Trigger identity (analysis; unlocked by 3)
With DNS-state marching in hand, split the approach into the march's
P-norm and raw-norm dynamics (no eigenvalues; cholesky discipline) and
test directly: does the switch coincide with the end of transient
amplification in the certified frame, run by run. This converts the
addendum's within-run coincidence into a mechanism measurement.

## 6. The strip (analytical; no schedule)
-1/2 <= sigma <= 0: Corollary 0 forbids the top under viscosity at a
growing maximum; the exclusion chain needs the bottom. The measured
deep exponent (~-1.2, three decades, grid-certified) says the mechanism
lives far below the strip whenever any tested viscosity acts -- the
analytical target is unchanged and now has a measured landscape under it.
