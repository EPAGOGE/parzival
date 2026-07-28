"""Distinct-object stance: measurements + EJA minting.

Measurement-only (one residual EVALUATION, no Newton). Verifies from disk:
  1. The candidate at frozen a=-0.4168236 is a genuine discrete root of the
     corner-regularized system (re-evaluate ||F|| on its exact config).
  2. The eps ladder cw/cl flatness (the 1.3e-5) directly from the npz files.
  3. The gap to alpha_1 and the deg24_56 refinement DIRECTION.
Then mints the stance's EJA objects: witnesses, an invariance promotion,
two refusals (negative controls), a conditional axiom with named falsifier,
and the shared-constant audit of everything that supports distinctness.
"""
import sys
import importlib.util
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad"
sys.path.insert(0, SCRATCH)
from eja_bridge import *  # noqa: E402,F401,F403

HF = SCRATCH + "/hunt_fields"
ALPHA_1 = -0.4168236
ALPHA_2 = -0.4439811
GROUND_REF = -0.34240009

spec = importlib.util.spec_from_file_location(
    "pc", "/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py")
pc = importlib.util.module_from_spec(spec)
sys.modules["pc"] = pc
spec.loader.exec_module(pc)


def cwcl(path):
    d = np.load(path)
    z = d["z"]
    return float(d["a"]), float(z[-2]), float(z[-1])   # a, cl, cw


print("== 1. cw/cl from disk ==")
rows = {}
for tag, f in [("find_half", "find_half.npz"),
               ("eps1e-4", "branch1_eps1e-4.npz"),
               ("eps5e-5", "branch1_eps5e-05.npz"),
               ("eps3e-5", "branch1_eps3e-05.npz"),
               ("eps1e-5", "branch1_eps1e-05.npz"),
               ("deg16_40", "branch1_deg16_40.npz"),
               ("deg24_56", "branch1_deg24_56.npz"),
               ("rung_00", "rung_00_a-0.344712.npz")]:
    a, cl, cw = cwcl(HF + "/" + f)
    rows[tag] = (a, cl, cw, cw / cl)
    print(f"  {tag:9s} a={a:+.7f}  cl={cl:+.6f}  cw={cw:+.6f}  cw/cl={cw/cl:+.9f}")

eps_vals = [rows[t][3] for t in ("eps1e-4", "eps5e-5", "eps3e-5", "eps1e-5")]
eps_motion = max(eps_vals) - min(eps_vals)
alpha_branch = rows["eps1e-4"][3]
gap_to_a1 = alpha_branch - ALPHA_1
frac_into_gap = (alpha_branch - ALPHA_1) / (ALPHA_2 - ALPHA_1)
deg_motion = rows["deg24_56"][3] - rows["deg16_40"][3]
print(f"  eps-ladder total cw/cl motion : {eps_motion:.3e}  (ground moved 2.3e-3 over same eps range)")
print(f"  branch alpha (eps=1e-4)       : {alpha_branch:+.8f}")
print(f"  gap to alpha_1                : {gap_to_a1:+.6e}  ({100*gap_to_a1/abs(ALPHA_1):+.3f}%)")
print(f"  position in a1->a2 gap        : {100*frac_into_gap:.1f}% (family slots are 0%/100% only)")
print(f"  deg (16,40)->(24,56) motion   : {deg_motion:+.3e}  toward alpha_1 requires {-gap_to_a1:+.3e}")

print("\n== 2. residual re-evaluation (root EXISTENCE check, no solve) ==")
d = np.load(HF + "/find_half.npz")
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12),
                       Nb=36, eps_b=1e-4, alpha=float(d["a"]))
F = S.residual(d["z"])
resid_cand = float(np.max(np.abs(F)))
print(f"  candidate ||F||_inf at frozen a={float(d['a']):+.7f}: {resid_cand:.3e}")
dg = np.load(HF + "/rung_00_a-0.344712.npz")
Sg = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12),
                        Nb=36, eps_b=1e-4, alpha=float(dg["a"]))
resid_ground = float(np.max(np.abs(Sg.residual(dg["z"]))))
print(f"  ground control ||F||_inf at a={float(dg['a']):+.7f}: {resid_ground:.3e}")

print("\n== 3. EJA objects ==")
# W1: the core NOT-alpha_1 tension -- self-consistent branch alpha vs alpha_1
w1 = mk_witness("branch_selfconsistent_alpha", alpha_branch, "deepmind_alpha_1", ALPHA_1,
                scenario={"eps_b": 1e-4, "degs": "16,40,12"}, tags=("distinct-object",))
print(f"  W1 alpha-gap witness: {alpha_branch:+.8f} vs {ALPHA_1:+.7f}  divergence={w1.divergence:.4e}")

# W2: refinement DIRECTION -- is-alpha_1 predicts motion toward -0.4168236; measured away
w2 = mk_witness("is_alpha1_predicted_refinement_target", ALPHA_1,
                "measured_deg24_56_cwcl", rows["deg24_56"][3],
                scenario={"axis": "degs 16,40->24,56"}, tags=("distinct-object", "direction"))
print(f"  W2 refinement-direction witness: needed {-gap_to_a1:+.3e} toward alpha_1, "
      f"measured {deg_motion:+.3e} (AWAY); divergence={w2.divergence:.4e}")

# W3: existence -- re-evaluated residual vs solver tolerance (a genuine discrete root)
w3 = mk_witness("reevaluated_candidate_residual", resid_cand, "newton_tol", 1e-11,
                scenario={"file": "find_half.npz"}, tags=("existence",))
print(f"  W3 existence: candidate ||F||={resid_cand:.3e} vs tol 1e-11 "
      f"(ground control {resid_ground:.3e})")

# Invariance: cw/cl ignores eps_b on this branch (recomputed from disk)
inv = mk_invariance("branch_cwcl_ignores_eps_b_1e-4_to_1e-5", worst_effect=eps_motion, eps=1e-4)
promote_invariance(inv)
print(f"  INV promoted: {inv}")

# Refusal 1 (negative control): the merge my stance declines
r1 = refuse("candidate_root == alpha_1", distance=abs(gap_to_a1), scale=7e-4,
            why="gap 4.9e-3 is 7.0x the largest measured resolution sensitivity "
                "(ground deg0 16->24 = 7e-4); eps measured 177x flat; measured "
                "refinement motion points AWAY from alpha_1")
print(f"  R1 refusal: {r1}")

# Refusal 2 (the stance's own weakness, on record): seed independence NOT established
r2 = refuse("candidate_root_is_seed_independent", distance=3.5e-3, scale=1e-11,
            why="best from-scratch residual on (20,48,12) went dry at 3.5e-3 vs "
                "converged tol 1e-11; every existing find descends from the single "
                "Chen-Hou-interpolation seed lineage")
print(f"  R2 refusal (own weakness): ratio={r2['ratio']:.1e}  {r2['refused']}")

# Conditional axiom: the stance itself, falsifier named at mint
ax = conditional_axiom(
    name="candidate_is_distinct_root_of_cornerreg_system",
    statement=("The converged object at cw/cl=-0.42174 (root of the corner-regularized "
               "system at frozen a=-0.4168236, ||F|| re-verified {:.1e}) is a solution "
               "object DISTINCT from DeepMind alpha_1, possibly missed by the PINN sweep "
               "or specific to this formulation's corner treatment".format(resid_cand)),
    domain=("edges=(0,2,15,25), degs (16,40,12)->(24,56,12), Nb=36, eps_b in [1e-5,1e-4], "
            "Chen-Hou-interp seed lineage; alpha gap to alpha_1 = -4.9e-3 with eps motion "
            "1.3e-5 and deg motion -3.8e-3 (away)"),
    residual=("no independent-seed find (from-scratch dry at 3.5e-3); mid-deg axis has "
              "one, possibly mid-run, datapoint; corner layer dA/dxi(0) not "
              "resolution-converged (25->33)"),
    falsifier=("interpolation-seeded polish of the (16,40,12) branch field on (20,48,12) "
               "and (24,56,12), then secant self-consistency: if alpha moves >= 4.9e-3 "
               "toward -0.4168236 (lands within 1e-3 of alpha_1), the object IS alpha_1 "
               "and this axiom dies; it also dies if the root fails to exist (Newton dry) "
               "under interpolation seeding on BOTH refined grids"),
    evidence={"gap_to_alpha1": gap_to_a1, "eps_motion": eps_motion,
              "deg_motion": deg_motion, "resid": resid_cand,
              "frac_into_a1_a2_gap": frac_into_gap})
print(f"  AX minted: {ax.name}")
print(f"     domain: {ax.domain}")

# Shared-constant audit: what EVERY distinctness-supporting config holds fixed
audit = shared_constant_audit([
    dict(seed="chen-hou-interp", Nb=36, edges="(0,2,15,25)", frozen_a=ALPHA_1,
         deg0=16, degmid=40, degcorner=12, eps=1e-4),    # find_half
    dict(seed="chen-hou-interp", Nb=36, edges="(0,2,15,25)", frozen_a=ALPHA_1,
         deg0=16, degmid=40, degcorner=12, eps=5e-5),    # eps ladder
    dict(seed="chen-hou-interp", Nb=36, edges="(0,2,15,25)", frozen_a=ALPHA_1,
         deg0=16, degmid=40, degcorner=12, eps=1e-5),
    dict(seed="chen-hou-interp", Nb=36, edges="(0,2,15,25)", frozen_a=ALPHA_1,
         deg0=24, degmid=56, degcorner=12, eps=1e-5),    # deg24_56 (eps assumed)
])
print(f"  SHARED-CONSTANT AUDIT (axes the agreement is blind to): {audit}")

# Emitted tension for the ledger: the next intervention family
led = TensionLedger(SCRATCH + "/distinct_object_tensions.json")
tid = led.emit(
    "Decisive axis for candidate_is_distinct_root_of_cornerreg_system: "
    "interp-seed the (16,40,12) branch field onto (20,48,12) and (24,56,12), "
    "Newton-polish at frozen a=-0.4168236, secant to self-consistency; then one "
    "off-lineage seed; then re-hunt at frozen a=-0.42174 (its own alpha)",
    source="distinct_object_case.py")
print(f"  LEDGER: emitted tension tid={tid} -> {led.path}")
print("\nDONE")
