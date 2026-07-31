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
- Theorem 0: at any growing vorticity maximum, nu |grad xi|^2 <= alpha. One
  page, unconditional. Data satisfies it with three orders of margin.
- alpha_0 = -0.34240 for the stable profile, two disjoint methods.

ESTABLISHED GIVEN NAMED HYPOTHESES
- Theorem 1: direction decay sigma < -1/2 excludes type-I blowup, via
  Constantin's kernel and the energy budget. The structural hypothesis carries
  a measured constant (lambda_0 = 5.0); the chain inequality holds at 21/21
  snapshot pairs with the stretching rate inferred independently.

ABDUCED (one stream; second stream named)
- Viscosity inverts the blowup geometry: sigma_PEAK = -0.37 to -0.57 across
  nu = 1e-5..1e-3, monotone, negative at every nu. Single grid; needs 256x768.
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

NOVELTY: UNVERIFIED. No literature kill-search has been run. Theorem skeletons
exist in Constantin-Fefferman 1993 and descendants. Treat every novelty
implication as open until that search closes. This paragraph is load-bearing.

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
