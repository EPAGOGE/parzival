# parzival / boussinesq

A verification-first laboratory for the geometry of fluid blowup, built in one
day by a human directing an AI system that repeatedly caught its own errors on
camera, and kept the corpses on the page.

Every number below regenerates from bytes on disk with one command. Claims are
graded against a frozen standard. A watcher reruns the full certification every
20 minutes and files a regression report if anything drifts.

## Quickstart

    python verify.py        # the backwards proof: 15 checks, 3 tiers
    python mythos.py        # the confound engine: every way the claims could be false
    python orbit_speed.py   # settling vs wandering in profile space
    cat VERIFY_STATUS       # what the watcher currently holds

## The graded ledger (STANDARD.md, frozen 2026-07-30)

ESTABLISHED (provenance + tier + exact anchor + identity + two streams + confound pass)
- sigma_Lambda(inviscid corner flow) = +1.00 +- 0.03. Anchored by an exact
  amplitude symmetry of the equations (predicted 1/sqrt(2) = 0.707107, measured
  0.706891) and a 2x2 off-ray grid factorial (spread 0.029).
- Corollary 0 (novelty RETRACTED by kill-search, NOVELTY.md C3: one-step
  corollary of Constantin's |omega|-equation): at any growing vorticity
  maximum, nu |grad xi|^2 <= alpha. Unconditional; retained as an
  instrument check. Data satisfies it with three orders of margin.
- alpha_0 = -0.34240 for the stable profile, two disjoint methods.

ESTABLISHED GIVEN NAMED HYPOTHESES
- Theorem 1: direction decay sigma < -1/2 excludes type-I blowup, via
  Constantin's kernel and the energy budget. The structural hypothesis carries
  a measured constant (lambda_0 = 5.0); the chain inequality holds at 21/21
  snapshot pairs with the stretching rate inferred independently.

ABDUCED (one stream; second stream named)
- Viscosity inverts the blowup geometry. 2026-08-02: second stream
  DELIVERED (M4): deep-collapse sigma_Lambda = -1.12/-1.24 (nu=1e-4) and
  -1.25/-1.29 (nu=1e-3), cross-grid spreads 0.121/0.032 (M4_SIGMA_DEEP.out).
  The earlier window-averaged quotes (-0.37..-0.57) are RETRACTED as regime
  mixtures (sigma(A) crosses over within the trusted window). Awaits the
  STANDARD.md confound pass before promotion to ESTABLISHED.
- c_l = -1/alpha = 2.9206 physical identification (far-field matching).

DEAD (refuted by this laboratory's own machinery; kept because a refuted claim
that vanishes teaches nothing)
- The reciprocal branch lambda = 1/3. The Batchelor saturation gloss. Three
  collision sweeps (wrong field, wrong window, uncontrolled amplitude).

OPEN, STATED EXACTLY
- The strip -1/2 <= sigma <= 0: Theorem 0 forbids the top under viscosity,
  Theorem 1 needs the bottom. Close the strip analytically and type-I
  Navier-Stokes blowup is finished for this route.
- The wandering sector: no rescaling frame settles. NRS/Tsai kill the exact
  frame; orbit_speed.py measures settling vs wandering (anchored at 0.5
  percent noise; the known-attractor case reads SETTLING at -0.79 as it must).
  Generic near-wall data dies too young at laptop resolution to read; that
  failure is the measured reason this sector is unexplored.

NOVELTY: kill-search run 2026-08-02, per-claim verdicts in NOVELTY.md.
One kill (Theorem 0 -> Corollary 0, Constantin's identity). Theorem 1
survives as quantification only -- cite Giga-Miura 2011 and Barker-Prange
2020 beside it. The observable/exponent program (C1, C2) and the viscous
inversion measurement (C5, the strongest) survive the search. All
CONFIRMED verdicts mean "not found by this one-session search," not
"proven absent." This paragraph remains load-bearing.

## The AI part, stated plainly

This laboratory was built by an AI under direction. The AI invented a branch
that did not exist, fit through windows that moved, measured the wrong field
against a zero baseline, and glossed a mechanism that its own certification
refuted within the hour. Every one of those failures was caught the same day:
by exact symmetry anchors, by six context-free reviewers (faults_freshpass.py,
seven unanimous findings, two session-results invalidated), and by a backwards
proof that failed its own author until the fit protocol was encoded
(verify.py, 10/15 -> 15/15). The discipline that survives contact with an AI
that launders inference is the actual product of this repository.

## Provenance

verify.py: 15/15. EXACT-tier worst deviation 3.05e-04 against references
forced by the equations before any measurement existed. Input checksums in the
verify output. Watcher: watch_verify.sh, status in VERIFY_STATUS.
