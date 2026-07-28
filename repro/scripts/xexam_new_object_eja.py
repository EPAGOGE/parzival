"""Cross-exam EJA ledger for the 'new_object' hypothesis: witnesses on measured
divergences, two refusals (one against my own attack, one against the advocate's
comparison), shared-constant audit of the surviving two-grid distinctness."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
from eja_bridge import *

# W1: alpha_1-identification prediction vs measured refinement motion
w1 = mk_witness("alpha1_identification_needs", -0.4168236,
                "measured_24_56_branch_alpha", -0.425546211,
                scenario={"axis": "degs (16,40,12)->(24,56,12)", "eps_b": 1e-5,
                          "rms_residual": 5.970e-12, "selfconsistency": 5.7e-10},
                tags=("resolution", "direction"))
print("W1 identification-vs-measured divergence:", w1.divergence)

# W2: the hypothesis' stated identity alpha vs the refined-grid measured alpha
w2 = mk_witness("hypothesis_identity_alpha", -0.42174207,
                "measured_24_56_alpha", -0.425546211,
                scenario={"eps_total_motion": 1.288e-5,
                          "resolution_step_motion": -3.817e-3,
                          "ratio_res_over_eps": 296.3},
                tags=("unconverged_alpha",))
print("W2 stated-alpha-vs-fine-grid divergence:", w2.divergence)

# W3: ground-family fingerprint prediction vs measured (24,56) fingerprint
w3 = mk_witness("ground_family_c31_at_xi1", -0.1805,
                "measured_24_56_c31_at_xi1", +0.161392,
                scenario={"xi": 1.0, "coarse_candidate_value": +0.0940,
                          "corner_dip": "A 1.123->0.814 then rise (present)"},
                tags=("fingerprint", "family_id"))
print("W3 ground-vs-fine-root fingerprint divergence:", w3.divergence)

# R1 (negative control on MYSELF): refuse my initial 'deg24_56 is unconverged
# mid-run debris' merge -- the naive fresh-pin instrument was wrong.
r1 = refuse("deg24_56_is_unconverged_midrun_iterate",
            distance=9.5e-5, scale=5.970e-12,
            why="Under the maker solver (constructed@a_prev=-0.42172919, "
                "set_alpha(-0.425546211), eps=1e-5) the file's RMS residual is "
                "5.970e-12, matching branch1_res.log digit-for-digit; the 9.5e-5 "
                "came from re-pinning A0/B0 at the stored alpha (pin-seed shift "
                "max|dA0|=0.1635 over da=4.9e-3). Log ends in 'done': study "
                "COMPLETE, rung final.")
print("R1 refusal recorded:", r1)

# R2 (negative control on the ADVOCATE): refuse the cross-file 'cleanest object
# on disk' comparison -- instrument-invalid under the two-alpha pin structure.
r2 = refuse("find_half_is_cleanest_object_on_disk",
            distance=7.784e-5, scale=3.273e-12,
            why="rung_00's 7.8e-5 and branch1_eps1e-4's 1.1e-3 fresh-pin "
                "residuals are pin-mismatch artifacts of the evaluating "
                "instrument, not looseness of the saved roots: under its maker "
                "(constructed@A1 + set_alpha) branch1_eps1e-4 evaluates at RMS "
                "3.273e-12 (log: 3.3e-12). find_half only looks special because "
                "the advocate rebuilt the solver at exactly its construction "
                "alpha. Existence at 7.2e-13 stands; the ranking does not.")
print("R2 refusal recorded:", r2)

# Shared-constant audit: what ALL distinctness-supporting configs still share
audit = shared_constant_audit([
    {"config": "coarse_root_16_40", "Nb": 36, "edges": "(0,2,15,25)",
     "deg_corner": 12, "pin_recipe": "axis+corner pinned to alpha-rescaled "
     "Chen-Hou seed", "seed": "chen-hou-interp", "hunt": "anchor+deflate+half"},
    {"config": "fine_root_24_56", "Nb": 36, "edges": "(0,2,15,25)",
     "deg_corner": 12, "pin_recipe": "axis+corner pinned to alpha-rescaled "
     "Chen-Hou seed", "seed": "chen-hou-interp", "hunt": "anchor+deflate+half"},
])
print("shared-constant audit:", audit)

# Emitted tension: the surviving residual as the next intervention family
led = TensionLedger()
led.add(mk_witness("branch_alpha_at_16_40", -0.42172919,
                   "branch_alpha_at_24_56", -0.425546211,
                   scenario={"next": "third converged rung ((28,64)-class, "
                             "interp-seeded warm start to dodge the dry basins) "
                             "+ eps re-check on the fine grid + alpha_2 guard "
                             "(drift is TOWARD -0.4439811, now 32.1% of a1->a2)"},
                   tags=("unconverged", "next_intervention")))
print("tension ledger:", led)
import json, pathlib
pathlib.Path(sys.path[0] + "/xexam_new_object_tensions.json").write_text(json.dumps({
    "tid0": {"open": "branch alpha resolution-unconverged: -0.42172919 (16,40) -> "
             "-0.425546211 (24,56), step -3.817e-3 = 296x total eps motion; "
             "direction AWAY from alpha_1 TOWARD alpha_2",
             "decisive_next": "third converged resolution rung with interp seeding; "
             "decision on alpha STABILIZATION, not proximity to -0.42174"}}, indent=1))
print("tensions written")
