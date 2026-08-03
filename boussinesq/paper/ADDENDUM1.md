# Addendum 1 to paper two: the closing sweep

**2026-08-03. Applies to** "The viscous inversion of the corner-flow
blowup geometry fires at the collapse transient's turnaround" (Draft 0.1,
2026-08-02). Everything below was pre-registered before its data existed
(NU_CURVE_SPEC.md, CLOSING SWEEP and 1E-6 DEEPENING sections) and is
recorded in CLOSING_SWEEP.out and BASIN_END.out. Scope inherits the
paper's Section 1 unchanged. Nothing here modifies a published claim;
three are extended and one gains a located caveat.

## A. The nu-gap: no approach across three decades, and a measured
## resolution floor

Four new runs extend the curve to nu = 3e-6 and 1e-6 (recipe unchanged).
At 3e-6: sigma_deep = -1.211 / -1.153 (spread 0.058), flat continues,
letter-clean. At 1e-6: -1.366 / -1.257 (spread 0.109); the 128-grid
point sits 4% below the pre-registered band's deep edge, away from the
inviscid value.

The 128-grid excursion was adjudicated by a pre-registered
cadence-replica rule: a replica run at halved snapshot interval
resamples the same deterministic trajectory independently. The replica
reproduces the 128-grid value to 0.001 (-1.3644 vs -1.3657; below-band
at 2 of 3 cut positions on both cadences), so it is not sampling
scatter; the 256-grid stays in-band at both cadences and all cut
positions (-1.218 / -1.257, 0 of 3 below), with no low-nu trend where
the coarse grid's is monotone. Verdict: a RESOLUTION EFFECT. nu = 1e-6
is the 128-grid's resolution floor for this observable; the
continuum-facing value there is the fine grid's in-band value.

Net: the deep exponent shows no approach toward the inviscid value at
any tested viscosity down to nu = 1e-6, now three decades. Any
crossover toward inviscid lies below that. The curve is quoted on both
grids down to 3e-6 and on the fine grid at 1e-6, with the coarse-grid
floor stated.

## B. The turnaround coincidence holds within single runs

For each of the six clock-fit runs, the run's own profile-space speed
(amplitude-and-length-normalized profile, differenced per unit of the
run's fitted rescaled time) peaks within one march step of that run's
switch: |s_turn - s*| = 0.03-0.20, six of six inside the pre-registered
0.25 bar. The paper's cross-instrument consistency upgrades to a
within-run statement. Honest structure: the inviscid runs land
near-exactly (0.03-0.04); all four viscous runs show the turnaround
about 0.2 s-units BEFORE the switch, a consistent ordering (turnaround,
then inversion), and their turnarounds sit earlier (s ~ 0.28-0.38) than
the inviscid ones (0.57-0.65). Both patterns are reported as observed,
unmodeled.

## C. The basin edge, tightened and anisotropic

Bisection along the seed-0 direction: the march holds 240 valid steps
at amplitude 3e-2 and stalls at step one at 5e-2 and 7e-2; the seed-0
basin-or-instrument edge is 3e-2 < r < 5e-2. A second direction
(seed 1) stalls already at 3e-2: the edge is direction-dependent,
with the seed-1 edge below 3e-2 where seed 0 held 240 valid steps. No
edge-ratio is quantified by these brackets. All stalls are
step-one residual failures; the protocol still does not separate
dynamical escape from solver stall (the P-norm decreased on three of
the four stall steps). Records: BASIN_END.out.

## Method note

Two tests compose into an artifact taxonomy used throughout this
addendum: cadence replication separates sampling scatter from
trajectory properties; cross-grid comparison then separates
discretization from continuum. Applied in sequence they classified a 4%
anomaly exactly.
