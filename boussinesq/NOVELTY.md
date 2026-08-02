# NOVELTY.md -- the literature kill-search (M5)

Run 2026-08-02, live web search (arXiv, Springer, journals), queries on:
vorticity-direction regularity criteria; Holder-1/2 direction results;
scale-invariant alignment estimates; type-I exclusion via direction;
Luo-Hou alignment measurements; Constantin |omega|-equation depletion;
name/definition collisions for sigma_Lambda. Each README ledger claim is
graded CONFIRMED (novelty survives, with closest prior art named) or
RETRACTED (prior art owns it). A kill-search that kills nothing was not
run hard enough: one claim died.

## The prior-art map (all items sighted in this search)

- Constantin-Fefferman 1993 and descendants: direction-coherence
  regularity criteria -- the founding territory. Survey: arXiv:2111.00040
  (geometric constraints on NS blowup).
- Beirao da Veiga & Berselli 2002 (Diff. Int. Eq.) and 2009: beta-Holder
  continuity of the vorticity direction; beta = 1/2 is the sharp class
  (omega in L-inf(L2) follows). The 1/2 power lives HERE first, in
  criterion form.
- Giga-Miura, Comm. Math. Phys. 2011: type-I + uniformly continuous
  direction where |omega| is large => no blowup (infinite-energy class).
- Barker-Prange, ARMA 235:881-926 (2020), arXiv:1906.08225:
  scale-invariant estimates + vorticity alignment, HALF-SPACE NO-SLIP,
  under type-I -- the Luo-Hou geometry, qualitative direction-continuity
  hypothesis.
- arXiv:2501.08976: double-cone / great-circle geometric
  characterization of potential NS singularities (local vorticity flux
  method).
- Grujic, arXiv:2607.08866 (2026): logarithmic depletion of vortex
  stretching via log-BMO direction classes -- analytical, no observable,
  no exponent, no measurement.
- Hou program (Acta Numerica 2009; Luo-Hou MMS 2014; PNAS 2025
  smooth-data-with-boundary): measures alignment with strain
  eigenvectors and vortex-line regularity on the corner candidate --
  qualitative/diagnostic, no direction-gradient exponent.
- Kerr (vorticity-moments line, arXiv:1212.1106 and companions): tested
  direction-derived conditions numerically on antiparallel-tube
  candidates -- criteria testing, not exponent measurement.

## Verdicts

### C1. Lambda = sup_P |grad xi| |omega|^{-1/2} as a scale-invariant
### observable; sigma_Lambda = d ln Lambda / d ln ||omega|| as its exponent
CONFIRMED, REFRAMED. The dimensional pairing is implicit in the
Beirao da Veiga-Berselli sharp 1/2-Holder class -- the claim that the
PAIRING is new is hereby narrowed: what survives is the OBSERVABLE
program (a measurable scalar, an exponent, calibration on a case where
blowup is proven). No prior definition or measurement of any
direction-regularity exponent was found; no name collision found.

### C2. sigma_Lambda(inviscid corner flow) = +1.00 +- 0.03, measured,
### symmetry-anchored, grid-factorial-certified
CONFIRMED. Closest art: Hou's depletion diagnostics (alignment with
strain eigenvectors, vortex-line regularity -- qualitative) and Kerr's
direction-condition tests. Nobody measured a direction-gradient scaling
exponent on the Luo-Hou class (or any blowup candidate) in anything
this search reached. Within-window refinement recorded 2026-08-02: the
full-window +1.00 is amplitude-composite (+1.4 low-amp, +0.59 deep,
grid-certified both) -- the calibration validations certify the
instrument, not slope constancy.

### C3. Theorem 0: nu |grad xi|^2 <= alpha at any growing vorticity max
RETRACTED AS NOVELTY -- this is the kill. The ingredient identity
(d_t|omega| = ... + alpha|omega| + nu(Delta|omega| - |omega||grad xi|^2),
the magnitude-direction decomposition with the viscous direction-gradient
depletion term) is Constantin's classical |omega|-equation; at a growing
spatial max, Delta|omega| <= 0 makes the inequality a one-step
corollary. Unconditional, yes; novel, no. What remains ours: using it as
a DATA-VALIDATED instrument check (satisfied with three orders of margin
across the lab's runs) -- a diagnostic role, not a theorem.

### C4. Theorem 1: sigma_Lambda < -1/2 excludes type-I blowup
### (Constantin kernel + energy budget, measured lambda_0)
CONFIRMED AS QUANTIFICATION, RETRACTED AS PRINCIPLE. Qualitative
type-I-plus-direction exclusions exist and must be cited: Giga-Miura
2011, and Barker-Prange 2020 in the half-space no-slip geometry --
the corner scenario's own setting. Not found anywhere: the
exponent-quantified version (an explicit threshold -1/2 on a measured
direction-regularity exponent, derived by feeding the energy
dissipation budget through the kernel split, with the structural
hypothesis carrying a measured constant and the two-rung structure
-1/6 / -1/2). The claim stands as a sharpening with a measurement
attached, not as a new exclusion principle.

### C5. sigma_Lambda(nu): the viscous inversion, measured
CONFIRMED -- the strongest novelty of the set. "Nobody has
sigma_Lambda(nu)" survives the search: no measurement of any
direction-regularity exponent as a function of viscosity exists in
anything reached, on any blowup candidate. The 2026-08-02 M4
certification (deep-collapse sigma_Lambda = -1.12/-1.24 at nu=1e-4,
-1.25/-1.29 at nu=1e-3, cross-grid spreads 0.121/0.032; window-matched
inviscid contrast +0.59 -> ~-1.2) is, as far as this search reaches,
the first quantitative observation of viscosity inverting a blowup
mechanism's direction-regularity scaling. Closest art: classical
qualitative knowledge that the nu|grad xi|^2 term depletes |omega|
growth (Constantin), turbulence alignment statistics (PDF-level, not
exponent-level), Grujic 2026 (analytical).

### C6. alpha_0 = -0.34240, two disjoint methods
NOT A NOVELTY CLAIM -- independent reproduction of published values
(the campaign's own records: DeepMind-line reference and Chen-Hou),
kept in the ledger as verification. No grading needed; recorded here so
the claim set is complete.

## Standing obligations created by this search

1. Any write-up MUST cite Giga-Miura 2011 and Barker-Prange 2020
   alongside Theorem 1, and Beirao da Veiga-Berselli at Lambda's
   definition. The (former) Theorem 0 is presented as a corollary-level
   diagnostic with Constantin's identity credited.
2. README's ESTABLISHED entry for Theorem 0 must be downgraded in
   language (done in the same session as this file).
3. Search horizon honesty: this was a one-session web sweep, not a
   systematic review; the CONFIRMED verdicts are "not found by this
   search," not "proven absent." The load-bearing status of this caveat
   is inherited from the README paragraph it replaces.
