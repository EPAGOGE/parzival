"""ns_jump4.py -- EJA run over the adjudicated NS branch-root quantities.

Mechanical pass: witnesses, refusals, invariances, deductions (I2-capped),
conditional axiom, shared-constant audit, tension ledger -> ns_tensions.json.
All numbers are MEASURED values from the morphology / corner-layer / numbers
agents and the two cross-examinations (2026-07-27). No solves launched.
"""
import sys

SCRATCH = "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad"
sys.path.insert(0, SCRATCH)
from eja_bridge import *  # noqa: E402,F401,F403

# ---------------------------------------------------------------- measured record
ALPHA1 = -0.4168236          # DeepMind first unstable branch (6 validated digits)
ALPHA2 = -0.4439811
ALPHA3 = -0.4578230
CHL2 = -0.40834
A_DEG16 = -0.421729189       # branch self-consistent alpha, (16,40,12)/Nb36/eps 1e-5
A_DEG16_E4 = -0.42174207     # same grid, eps 1e-4 (headline polish)
A_DEG24 = -0.425546211       # branch alpha, (24,56,12)/eps 1e-5 -- CONVERGED (rms 6.0e-12)
GAP16 = A_DEG16 - ALPHA1     # -4.906e-3
GAP24 = A_DEG24 - ALPHA1     # -8.723e-3
SUBPCT = 0.01 * abs(ALPHA1)  # 4.168e-3: the sub-percent attribution gate
DH_BRANCH, DH_GROUND = -1.026, -1.0002
EPS_LADDER = [(1e-4, -0.421742073), (5e-5, -0.421737556),
              (2.5e-5, -0.421732821), (1e-5, -0.421729189)]
EPS_MOTION_BRANCH = abs(EPS_LADDER[-1][1] - EPS_LADDER[0][1])   # 1.288e-5
EPS_MOTION_GROUND = 2.3e-3
DEG_STEP_BRANCH = abs(A_DEG24 - A_DEG16)                        # 3.817e-3
DEG_PRIOR_GROUND = 7e-4
C3C1_GROUND = [-0.180545, -0.180501, -0.180496]                 # rungs 00/09/10, Da=0.078
C3C1_CAND16, C3C1_CAND24 = +0.093992, +0.161392
PIN_JUMP_ALPHA_EFFECT = 4.517e-6      # eps1e-4 rung carried the full 4.9e-3 pin jump
PIN_CONSISTENT_STEPS = [4.735e-6, 3.632e-6]
OP_PROV_OWN, OP_PROV_MAKER = 2.97e-6, 6.0e-12   # same field, two construction alphas
FRESH_EVAL_FINDHALF, FRESH_EVAL_RUNG00, MAKER_RMS = 7.2e-13, 7.8e-5, 3.3e-12

print("=" * 78)
print("NS JUMP CYCLE 4 -- EJA over the adjudicated branch-root record (2026-07-27)")
print("=" * 78)

# ------------------------------------------------------------------- 1. WITNESSES
print("\n[1] WITNESSES (tension = the drive signal)")

w1 = mk_witness("branch_alpha_deg16", A_DEG16, "deepmind_alpha_1", ALPHA1,
                scenario={"grid": 164012, "eps_b": 1e-5},
                tags=("identity", "alpha1_attribution"))
print(f"  W1 candidate-vs-alpha_1 (deg16):  {A_DEG16:.9f} vs {ALPHA1:.7f}"
      f"  gap {GAP16:+.6e}  divergence {w1.divergence:.4e}")

w1b = mk_witness("branch_alpha_deg24", A_DEG24, "deepmind_alpha_1", ALPHA1,
                 scenario={"grid": 245612, "eps_b": 1e-5},
                 tags=("identity", "alpha1_attribution", "resolution"))
print(f"  W1b candidate-vs-alpha_1 (deg24): {A_DEG24:.9f} vs {ALPHA1:.7f}"
      f"  gap {GAP24:+.6e}  divergence {w1b.divergence:.4e}")
print(f"      -> refinement WIDENS the gap: {GAP16:+.4e} -> {GAP24:+.4e}"
      f"  (x{GAP24/GAP16:.2f}); attribution predicts SHRINK")

w2 = mk_witness("branch_dh_da", DH_BRANCH, "ground_dh_da", DH_GROUND,
                scenario={"quantity": "d(cw/cl)/da"}, tags=("response_physics",))
print(f"  W2 branch dh/da vs ground dh/da:  {DH_BRANCH} vs {DH_GROUND}"
      f"  divergence {w2.divergence:.4e}  (own response physics, 100x d(cw/cl)/da)")

w3 = mk_witness("ground_transfer_prior_deg", DEG_PRIOR_GROUND,
                "branch_measured_deg_step", DEG_STEP_BRANCH,
                scenario={"axis": "deg (16,40,12)->(24,56,12)"},
                tags=("transfer_without_invariance", "NEW"))
print(f"  W3 NEW deg-axis transfer failure: prior {DEG_PRIOR_GROUND:.1e} vs measured"
      f" {DEG_STEP_BRANCH:.3e}  divergence {w3.divergence:.4e}  (5.5x; the 7x-gate died here)")

w4 = mk_witness("cos3b_over_c1_ground", C3C1_GROUND[0], "cos3b_over_c1_candidate",
                C3C1_CAND16, scenario={"xi": 1.0, "panel": "corner"},
                tags=("morphology", "branch_fingerprint"))
print(f"  W4 corner cos3b/c1 SIGN FLIP:     {C3C1_GROUND[0]:+.6f} vs {C3C1_CAND16:+.6f}"
      f"  divergence {w4.divergence:.4e}  (6172x the ground-walk spread)")

w5 = mk_witness("field_rms_own_alpha_operator", OP_PROV_OWN,
                "field_rms_maker_operator", OP_PROV_MAKER,
                scenario={"file": "branch1_deg24_56.npz"},
                tags=("instrument", "two_alpha_operator", "NEW"))
print(f"  W5 NEW operator-provenance:       {OP_PROV_OWN:.2e} vs {OP_PROV_MAKER:.1e}"
      f"  divergence {w5.divergence:.6f}  (pins baked at construction alpha;"
      f" set_alpha never refreshes A0/B0)")

# ------------------------------------------------------------------- 2. REFUSALS
print("\n[2] REFUSALS (negative controls -- merges declined, with numbers)")

r1 = refuse("candidate_root == alpha_1", distance=abs(GAP16), scale=SUBPCT,
            why="gap 4.906e-3 exceeds the sub-percent attribution gate 4.168e-3 "
                "(1.18%); eps measured 177x flat cannot close it; and the one "
                "converged deg step moved 3.82e-3 AWAY (gap -> 8.72e-3, ratio 2.09 "
                "at deg24). Attribution would need an unevidenced non-monotonic "
                "deg-convergence reversal of >7e-3.")
print(f"  R1 REFUSED {r1['refused']}: distance {r1['distance']:.3e} vs scale "
      f"{r1['scale']:.3e}  ratio {r1['ratio']:.2f}x (2.09x at deg24)")

r2 = refuse("alpha = -0.42174207 as the OBJECT's alpha (resolution-converged value)",
            distance=DEG_STEP_BRANCH, scale=EPS_MOTION_BRANCH,
            why="one deg step moved the branch alpha 296x its TOTAL eps-ladder "
                "motion; the object's continuum alpha is unknown -- quote "
                "alpha(grid), never a bare five-digit identity")
print(f"  R2 REFUSED {r2['refused'][:52]}...: distance {r2['distance']:.3e} vs "
      f"scale {r2['scale']:.3e}  ratio {r2['ratio']:.0f}x")

r3 = refuse("cross-file residual ranking via fresh-construction evaluator "
            "('find_half is the cleanest object on disk')",
            distance=FRESH_EVAL_RUNG00, scale=MAKER_RMS,
            why="fresh-eval RMS is a pin-seed-mismatch artifact of the two-alpha "
                "operator (W5): rung_00 reads 7.8e-5 fresh but 3.3e-12 under its "
                "maker; find_half only looked special because the evaluator was "
                "rebuilt at exactly its construction alpha. Killed in cross-exam.")
print(f"  R3 REFUSED fresh-eval residual ranking: distance {r3['distance']:.2e} vs "
      f"scale {r3['scale']:.2e}  ratio {r3['ratio']:.1e}x")

r4 = refuse("radial extrema COUNT discriminates the branches",
            distance=0.0, scale=1.0,
            why="both roots show 2 interior extrema and zero sign changes at frozen "
                "a; the classic extra-node test FAILS as a count -- LOCATION and "
                "the cos3b sign are the fingerprints (morphology agent, re-minted)")
print(f"  R4 REFUSED extrema-count test: distance {r4['distance']} at scale "
      f"{r4['scale']} (identical counts; count carries zero bits)")

# ---------------------------------------------------------------- 3. INVARIANCES
print("\n[3] INVARIANCES (cheap promotions -- and one refusal-by-raise)")

inv_eps = mk_invariance("branch_cwcl_ignores_eps_b_on_coarse_grid",
                        worst_effect=EPS_MOTION_BRANCH, eps=1e-4)
ax_eps = promote_invariance(inv_eps)
print(f"  I1 PROMOTED {inv_eps.name}: worst effect {inv_eps.max_distance:.3e} over "
      f"eps 1e-4->1e-5 (ground moves {EPS_MOTION_GROUND:.1e}; 177x flatter). "
      f"Mechanism measured: near-pure scale mode, lambda_B/lambda_A^2=1.000618, "
      f"cancels in cw/cl while cl moves +3.1%. DOMAIN: (16,40,12)/Nb36 only.")

spread = max(C3C1_GROUND) - min(C3C1_GROUND)
inv_c3 = mk_invariance("ground_corner_cos3b_ratio_ignores_alpha",
                       worst_effect=spread, eps=1e-3)
ax_c3 = promote_invariance(inv_c3)
print(f"  I2 PROMOTED {inv_c3.name}: spread {spread:.3e} across Da=0.078 "
      f"(rungs 00/09/10) -- the a-invariant fingerprint the candidate flips.")

inv_pin = mk_invariance("branch_alpha_ignores_pin_construction_alpha",
                        worst_effect=PIN_JUMP_ALPHA_EFFECT, eps=1e-4)
ax_pin = promote_invariance(inv_pin)
print(f"  I3 PROMOTED {inv_pin.name}: the eps1e-4 rung carried the full 4.9e-3 "
      f"pin-alpha jump yet moved alpha {PIN_JUMP_ALPHA_EFFECT:.3e}, in family with "
      f"pin-consistent steps {PIN_CONSISTENT_STEPS} -- stale-pin bias <~5e-6. "
      f"CAVEAT: single control; assumes no eps/pin cancellation in that step.")

inv_deg = mk_invariance("branch_alpha_ignores_deg_axis",
                        worst_effect=DEG_STEP_BRANCH, eps=1e-4)
try:
    promote_invariance(inv_deg)
    print("  I4 ERROR: this should not have promoted")
except ValueError as e:
    print(f"  I4 promotion REFUSED (engine raise, as it must): "
          f"{inv_deg.name} worst effect {inv_deg.max_distance:.3e} >= 1e-4 -- "
          f"the deg axis is LIVE at gap scale; refusal is the finding ({e})")

# ---------------------------------------------------------------- 4. DEDUCTIONS
print("\n[4] DEDUCTIONS (I2 discipline: NOVEL only with invariance evidence)")

d1 = deduce("gap_to_alpha1_widens_under_refinement",
            quantity="alpha - alpha_1 at (24,56,12)", value=GAP24,
            old_rule_value=GAP16,
            invariance_evidence=("two-grid fingerprint continuity: c3/c1 corner sign "
                                 "positive on BOTH converged rungs (+0.0940 deg16, "
                                 "+0.1614 deg24; ground family -0.1805 a-invariant "
                                 "to 4.45e-5); both rungs converged rms 3.3e-12 / "
                                 "6.0e-12 under maker operators; eps-invariance of "
                                 "cw/cl promoted (I1)"),
            note="the away-direction is a property of the OBJECT across grids, "
                 "not of one grid")
print(f"  D1 [{d1.status}] {d1.name}: {d1.old_rule_value:+.4e} -> {d1.value:+.4e}")

d2 = deduce("candidate_is_alpha_1", quantity="alpha", value=ALPHA1,
            old_rule_value=A_DEG16, invariance_evidence=None,
            note="kept alive only by hypothetical non-monotonic deg convergence; "
                 "refused at current gap (R1); cross-exam verdict WEAK")
print(f"  D2 [{d2.status}] {d2.name}: would require {abs(GAP24):.3e} reversal "
      f"toward {ALPHA1} -- worth testing at rung 3, not worth asserting")

d3 = deduce("next_family_root_hunt_target", quantity="frozen a for alpha_2 hunt",
            value=ALPHA2, old_rule_value=None, invariance_evidence=None,
            note="morphology licenses the START, not the outcome: the working find "
                 "came from fields x 0.5 of the PREVIOUS root (ground -> candidate; "
                 "amplitude grew: max|B| 1.07->4.51, cl 3.01->5.22). Predicted "
                 "recipe: freeze a=-0.4439811, seed from find_half fields x 0.5 "
                 "(and x 2.0 as control), anchor+deflate; expect lobe displaced "
                 "further out, corner cos3b/c1 sign flipping again or growing")
print(f"  D3 [{d3.status}] {d3.name}: a={d3.value} with half-amplitude scaling of "
      f"the NEW root as the start")

d4 = deduce("deepmind_family_accumulation_point", quantity="alpha_inf",
            value=-0.46832, old_rule_value=None, invariance_evidence=None,
            note="3-gap extrapolation, ratios 0.365/0.510 non-constant; fitted "
                 "r=0.4313 gives -0.46832, last-ratio gives -0.47221. CANDIDATE "
                 "only. NOTE: branch drift at deg24 (-0.42555) is TOWARD alpha_2; "
                 "an alpha_2-in-disguise guard belongs in rung 3")
print(f"  D4 [{d4.status}] {d4.name}: {d4.value} (alt -0.47221)")

# ------------------------------------------------- 5. AUDIT + CONDITIONAL AXIOM
print("\n[5] SHARED-CONSTANT AUDIT (mandatory before any 'converged' claim)")
audit = shared_constant_audit([
    # every configuration that supports 'distinct root, eps-flat, away-motion'
    dict(Nb=36, edges="(0,2,15,25)", deg_corner=12, pin_recipe="alpha-rescaled "
         "Chen-Hou pins", seed="chen-hou-interp-lineage", hunt="anchor+deflate+half",
         deg0=16, degmid=40, eps=1e-4),
    dict(Nb=36, edges="(0,2,15,25)", deg_corner=12, pin_recipe="alpha-rescaled "
         "Chen-Hou pins", seed="chen-hou-interp-lineage", hunt="anchor+deflate+half",
         deg0=16, degmid=40, eps=1e-5),
    dict(Nb=36, edges="(0,2,15,25)", deg_corner=12, pin_recipe="alpha-rescaled "
         "Chen-Hou pins", seed="chen-hou-interp-lineage", hunt="anchor+deflate+half",
         deg0=24, degmid=56, eps=1e-5),
])
print(f"  UNTESTED AXES (held constant by ALL agreeing runs): {sorted(audit)}")
print("  -> the two-grid agreement is structurally blind to Nb, edges, deg_corner,"
      "\n     the ground-pinned corner/axis data, seed lineage, and the hunt recipe;"
      "\n     a corner-FORMULATION object (real for the pinned problem, absent in"
      "\n     the free one) passes every test run so far.")

axc = conditional_axiom(
    name="candidate_is_distinct_root_family_not_alpha1",
    statement=("The deflated-multistart root family (corner cos3b/c1 > 0, displaced "
               "amplitude lobe, corner dip below the pinned profile) is a root of "
               "the corner-regularized discrete system DISTINCT from DeepMind "
               "alpha_1: alpha(grid) = -0.42173 (16,40,12) / -0.42555 (24,56,12), "
               "gap to alpha_1 widening under refinement while cw/cl ignores eps_b "
               "(1.3e-5 over a decade)."),
    domain=("grids (16,40,12) and (24,56,12), Nb=36, edges (0,2,15,25), eps_b in "
            "[1e-5, 1e-4], Chen-Hou-pinned corner/axis rows -- the audited family"),
    residual=("alpha NOT resolution-converged (3.8e-3/step, drifting toward "
              "alpha_2); seed independence REFUSED on record (from-scratch dry at "
              "3.5e-3); pinned-problem vs free-problem identity untested (W5 "
              "two-alpha operator); eps-flatness coarse-grid-only"),
    falsifier=("rung 3 -- interpolation-seeded converged (28,64)-class solve: if "
               "the self-consistent alpha reverses by >7e-3 to land within 1.5e-3 "
               "of -0.4168236, the family IS alpha_1 and this axiom dies; if the "
               "root vanishes or the step fails to contract toward ANY limit "
               "(with the alpha_2 = -0.4439811 guard armed), it dies as a ghost"),
    evidence={"gap_deg16": GAP16, "gap_deg24": GAP24,
              "eps_motion": EPS_MOTION_BRANCH, "deg_step": DEG_STEP_BRANCH,
              "c3c1_flip_deg16": C3C1_CAND16, "c3c1_flip_deg24": C3C1_CAND24})
print(f"\n  CONDITIONAL AXIOM MINTED: {axc.name}")
print(f"  domain:   {axc.domain}")
print(f"  residual: {axc.residual}")

# ---------------------------------------------------------------- 6. TENSIONS
print("\n[6] EMITTED TENSIONS -> ns_tensions.json (the next intervention families)")
led = TensionLedger(SCRATCH + "/ns_tensions.json")
tids = []
tids.append(led.emit(
    "RUNG 3 (decisive): interpolation-seeded (28,64)-class converged solve, panel-"
    "by-panel Chebyshev transfer (never across duplicated interface nodes), secant "
    "loop RECONSTRUCTING the solver at each alpha (or set_alpha refreshing A0/B0 "
    "pins -- W5); decide on alpha-step CONTRACTION with an explicit alpha_2 guard "
    "(drift covered 3.8e-3 of the 2.72e-2 a1->a2 gap in one step). Outcomes: "
    "contract to stable limit != alpha_1 => new_object confirmed; reverse >7e-3 to "
    "alpha_1 => is_alpha1; vanish/non-contract => ghost.", "ns_jump4"))
tids.append(led.emit(
    "SEED-PROVENANCE RUNG: one off-Chen-Hou-lineage start reaching the same root "
    "(seed independence currently REFUSED, from-scratch dry at 3.5e-3); plus "
    "pin/formulation perturbation -- vary Nb, edges, deg_corner, and the pinned "
    "corner/axis data (WX/THXX references) -- the ONLY axis separating 'real "
    "second family member' from 'well-converged object of the ground-pinned "
    "problem' (shared-constant audit: all agreeing runs hold all six).", "ns_jump4"))
tids.append(led.emit(
    "ALPHA_2/ALPHA_3 HUNTS with the predicted start (D3): freeze a=-0.4439811 "
    "(then -0.4578230), seed find_half fields x 0.5 and x 2.0, anchor+deflate; "
    "fingerprint check = corner cos3b/c1 sign and lobe displacement; success would "
    "convert the half-amplitude recipe into a family-walking OPERATOR and give the "
    "accumulation-point extrapolation (D4, CANDIDATE) its first out-of-sample "
    "test.", "ns_jump4"))
tids.append(led.emit(
    "EPS->0 MODEL CLASS on the FINE grid: the promoted eps-invariance (I1) is "
    "coarse-grid-only; run the eps ladder 1e-4->1e-5 at (24,56,12) and re-do the "
    "scale-mode decomposition. Falsifier of the standing eps_flat_via_scale_mode "
    "axiom remains UNRUN: the same decomposition on a GROUND-branch eps ladder "
    "(does ground lack the cancellation, or merely add non-scale response?). The "
    "WHY of the wedge-layer decoupling is still unmeasured mechanism-side.",
    "ns_jump4"))
tids.append(led.emit(
    "CORNER-LAYER RESOLUTION: the A-field corner layer (dA/dxi(0) rms 25->33, "
    "opposite-sign cos b coefficient) has ~3-5 Chebyshev nodes across its 0.05-0.1 "
    "width; A-tail non-decay through deg24 is EXPECTED for a real thin layer at "
    "these degs. Either raise deg_corner specifically (12->18+) or map the layer "
    "with a dedicated stretched sub-panel before reading any ghost verdict from "
    "tail behavior.", "ns_jump4"))
for t in led.open():
    print(f"  tid={t.tid} [{t.status}] {t.text[:96]}...")
print(f"\n  {len(tids)} tensions emitted this run; ledger at "
      f"{SCRATCH}/ns_tensions.json")

# ---------------------------------------------------------------- 7. VERDICT
print("\n" + "=" * 78)
print("ENGINE VERDICT (mechanical, from the objects above)")
print("=" * 78)
print(f"""\
  PROMOTED:  eps-invariance of branch cw/cl (I1, coarse grid); ground cos3b
             a-invariance (I2); pin-benignity (I3, single control).
  REFUSED:   alpha_1 attribution (R1, 1.18x sub-percent gate, 2.09x at deg24);
             any bare converged alpha value (R2, 296x); fresh-eval residual
             ranking (R3, instrument artifact); extrema-count test (R4);
             deg-invariance of branch alpha (I4 raise -- the live axis).
  NOVEL:     the gap to alpha_1 WIDENS under refinement on a fingerprint-
             continuous, twice-converged root family (D1).
  CANDIDATE: is_alpha1 (D2, needs an unevidenced >7e-3 reversal); alpha_2-hunt
             recipe (D3); accumulation point -0.468/-0.472 (D4).
  STANDING:  conditional axiom '{axc.name}'
             with its falsifier armed at rung 3.""")
