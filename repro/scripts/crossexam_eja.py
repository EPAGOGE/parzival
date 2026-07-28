#!/usr/bin/env python3
"""Cross-examiner EJA objects for the 'new_object' adjudication (real numbers
from crossexam_new_object.py / crossexam_ascan.py this run)."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
from eja_bridge import *
import json

# X1: WITNESS -- the advocate's error-budget scale vs the branch's own measured
# single-axis refinement motion (deg16->deg24 cw/cl step, confirmed from disk).
w_scale = mk_witness("ground_transferred_prior_per_axis", 7.0e-4,
                     "branch_measured_refinement_motion", 3.817022e-3,
                     scenario={"axis": "deg (16,40,12)->(24,56,12)",
                               "eps_b": 1e-5, "quantity": "cw/cl"},
                     tags=("error_budget", "cross_exam"))
print("X1 witness scale-mismatch: divergence =", w_scale.divergence)

# X2: WITNESS -- advocate claim 'branch1_deg16_40 bit-identical to branch1_eps1e-05'
# vs measured max|dz| = 8.631e-8 (and da = 9.1e-14): an independent re-polish,
# not a file copy.  Claim FALSE as stated, substance (no new axis) intact.
w_bit = mk_witness("advocate_bit_identical_claim", 0.0,
                   "measured_max_dz", 8.631e-8,
                   scenario={"files": "branch1_deg16_40 vs branch1_eps1e-05"},
                   tags=("provenance", "cross_exam"))
print("X2 witness bit-identity: divergence =", w_bit.divergence)

# X3: REFUSAL (negative control) -- decline the merge 'loose residual floors ==
# alpha-mismatch of the save convention'.  If the 1.100e-3 floor of
# branch1_eps1e-4 were a-mismatch, the a-scan would drop it to ~Newton tol at
# a_min (||dF/da|| = 49.2 measured); instead the floor moved <0.1% while a
# shifted 2.2e-5.  The floors are FIELD non-convergence (loose polish).
r_floor = refuse("residual_floor_is_alpha_mismatch",
                 distance=1.100e-3,          # floor that failed to drop
                 scale=1e-11,                # where a converged solve would sit
                 why="a-scan: min ||F|| 1.100e-3 at a_min shift 2.2e-5 with "
                     "||dF/da||=49.2 -- an a-mismatch of that size predicts "
                     "floor ~1e-9; observed flat. Same for eps3e-05 (4.75e-4) "
                     "and deg24_56 (9.517e-5).")
print("X3 refusal:", json.dumps(r_floor))

# X4: REFUSAL -- decline the advocate's implied merge 'deg24_56 == unusable
# mid-run junk' AND the opposite merge 'deg24_56 == converged rung'.  It is a
# NEAR-converged loose-polish state: residual 9.517e-5 at eps=1e-5 (vs 9.506e-3
# at eps=1e-4 -> grid/eps identified), stored a == cw/cl to all printed digits
# (post-secant save convention), ladder-smoothness bounds cw/cl error ~1e-6
# per 1e-3 of residual.  Usable for DIRECTION (away from alpha_1), not for a
# converged deg24 alpha.
r_junk = refuse("deg24_56_is_midrun_junk",
                distance=9.517e-5, scale=1e-11,
                why="residual 9.5e-5 is 1e7x tol -- not converged -- but "
                    "empirical cw/cl-error-per-residual from the eps ladder "
                    "(~2e-3 ratio: rungs at 1.1e-3 floor sit within ~1e-6 of "
                    "the smooth ladder trend anchored by two tight rungs at "
                    "5.4e-7/8.6e-8) bounds the deg24 cw/cl error ~1e-5..1e-4, "
                    "far below the 3.817e-3 refinement motion.")
print("X4 refusal:", json.dumps(r_junk))

# X5: shared-constant audit of the configs supporting 'refinement moved AWAY
# from alpha_1' -- one datapoint, everything else held.
audit = shared_constant_audit([
    {"deg": (16, 40, 12), "eps_b": 1e-5, "Nb": 36, "edges": (0, 2, 15, 25),
     "seed": "chen-hou-lineage", "frozen_a_convention": "a:=cw/cl"},
    {"deg": (24, 56, 12), "eps_b": 1e-5, "Nb": 36, "edges": (0, 2, 15, 25),
     "seed": "chen-hou-lineage", "frozen_a_convention": "a:=cw/cl"},
])
print("X5 shared-constant audit (direction evidence):", audit)

# X6: the surviving tension, emitted as the next intervention family.
led = TensionLedger("/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad/crossexam_tensions.json")
led.add("decision-rule hole: advocate's success criterion 'alpha within ~1e-3 "
        "of -0.42174' fails the measured trajectory (deg16 -0.421729 -> deg24 "
        "-0.425546, motion 3.8e-3): a real distinct object whose alpha is not "
        "resolution-converged lands in NO branch of the stated rule. Correct "
        "criterion: alpha Cauchy in resolution AND far from alpha_1, plus "
        "fingerprint persistence. Next intervention: polish the ON-DISK "
        "deg24_56 state (residual 9.5e-5, already quadratic-regime) to tol -- "
        "cheapest converged rung 2; then (20,48,12) interp-seeded.",
        tags=("decision_rule", "resolution"))
for t in led.open():
    print("X6 open tension:", t)
