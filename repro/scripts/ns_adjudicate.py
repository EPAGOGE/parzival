"""NS ADJUDICATION -- converging-answer mint (2026-07-27). Measurement-free:
adjudicates only objects already measured; seconds of compute."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
from eja_bridge import *

print("== ADJUDICATION MINTS ==")

# W_adj: pre-registered rung-3 discriminator -- the two surviving hypotheses
# PREDICT different |d(cw/cl)| at the next converged rung (28,64)-class.
# new_object (geometric contraction, ratio <=0.5 of the measured 3.817e-3 step)
# vs ghost/alpha_2-drift (non-contraction, step stays O(3.8e-3)).
w_adj = mk_witness("new_object_predicts_contracted_step", 1.9e-3,
                   "ghost_predicts_uncontracted_step", 3.8e-3,
                   scenario={"rung": "(28,64,18)", "eps_b": 1e-5},
                   tags=("prereg", "rung3"))
print(f"W_adj rung-3 step prediction: 1.9e-3 vs 3.8e-3  divergence {w_adj.divergence:.4f}")

# NEGATIVE CONTROL: refuse to merge 'distinct root of the PINNED formulation'
# with 'second member of the FREE problem's blowup family'. The operator itself
# depends on construction alpha through ground-baked pins: same converged field
# evaluates RMS 2.97e-6 vs 6.0e-12 under two construction alphas, and every
# confirming run shares {Nb, edges, deg_corner, pin recipe, seed lineage, hunt}.
r_free = refuse("pinned_formulation_root == free_problem_family_member",
                distance=2.97e-6, scale=6.0e-12,
                why="operator-provenance sensitivity 4.95e5x the certified residual; "
                    "all 6 pin/seed axes untested (shared_constant_audit)")
print(f"REFUSED free-problem identity: ratio {r_free['ratio']:.3g}x -- {r_free['why']}")

# Refusal 2 (carried, restated at adjudication level): no converged alpha may be
# quoted for the object -- one deg step moved it 296x its whole eps-ladder motion.
r_alpha = refuse("branch_alpha == -0.42174207 (any single-grid value)",
                 distance=3.817e-3, scale=1.288e-5,
                 why="deg step 296x total eps motion; quote alpha(grid) only")
print(f"REFUSED single-grid alpha: ratio {r_alpha['ratio']:.0f}x")

# THE ORGANIZING CLAIM, minted conditional with its falsifier armed.
ax = conditional_axiom(
    "cornerreg_boussinesq_supports_second_unstable_profile_family",
    statement=("The corner-regularized Chen-Hou system has, besides the closed ground "
               "profile (alpha_0=-0.34240+/-3e-5, cross-method 3.4e-7), a second "
               "twice-converged unstable-type root family -- same corner transport "
               "algebra P(0,b)=c sin2b (<=1e-3), sign structure, and eps_b scale-mode "
               "cancellation in cw/cl (1.3e-5 over the ladder), but outward-displaced "
               "amplitude lobe, opposite-sign corner boundary layer, and flipped "
               "cos3b/c1 fingerprint (+0.094/+0.161 vs a-invariant -0.1805) -- that is "
               "NOT alpha_1: the gap WIDENS under refinement (-4.906e-3 -> -8.723e-3)"),
    domain=("grids (16,40,12) and (24,56,12), Nb=36, edges (0,2,15,25), eps_b in "
            "[1e-5,1e-4], ground-pinned corner/axis rows, Chen-Hou seed lineage"),
    residual=("alpha not resolution-converged (3.8e-3/step, drifting toward alpha_2); "
              "seed independence REFUSED (from-scratch dry 3.5e-3/4.1e-3 on two grids); "
              "free-vs-pinned identity REFUSED above; eps-flatness coarse-grid-only"),
    falsifier=("rung-3 interpolation-seeded (28,64,18)-class converged solve: step "
               "reversal >=+7e-3 toward -0.4168236 kills 'not alpha_1'; non-contraction "
               "or root loss under warm seeding demotes to formulation ghost; "
               "contraction to a stable limit elsewhere confirms"),
    evidence={"gap_deg16": -4.905589e-3, "gap_deg24": -8.722611e-3,
              "eps_flat": 1.288e-5, "cos3b_flip_sigma": 6172.0})
print(f"CONDITIONAL AXIOM: {ax.name}")
print(f"  domain: {ax.domain[:72]}...")
print(f"  residual: {ax.residual[:72]}...")

# Family-walk predictions: CANDIDATE only (no invariance evidence transfers the
# half-amplitude hunt recipe to alpha_2/alpha_3 frozen points).
d2 = deduce("alpha2_hunt_target", "frozen substitution exponent a", -0.4439811,
            note="start = 0.5 x (new root fields); predict eps-flat cw/cl via scale "
                 "mode + own cos3b fingerprint if a root lands")
d3 = deduce("alpha3_hunt_target", "frozen substitution exponent a", -0.4578230,
            note="run only after alpha_2 outcome; same recipe")
print(f"DEDUCE {d2.name}: {d2.value}  [{d2.status}]")
print(f"DEDUCE {d3.name}: {d3.value}  [{d3.status}]")
