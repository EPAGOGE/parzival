"""EJA APPLIED TO EJA: the engine's own three NS cycles as the experience corpus.

The engine's record, as MEASURED OUTCOMES (not self-report):
  C1 promoted eps_b==wedge (correct; survived everything) and DEDUCED
     d(alpha)/d(theta) = +1.40 as NOVEL -- later RETIRED: the slope is
     formulation-dependent (-2.8 in the panel frame vs ~-30 in the corner-
     regularized frame => +1.4 vs ~+15). The transferred quantity was not
     invariant under the identity's hidden parameter (the discretization frame).
  C2 minted a conditional axiom (formulation delta) WITH a named falsifier;
     the falsifier fired; the axiom died cleanly. Mechanism: PERFECT.
     But its premise 'every axis is converged' was false -- the corner panel
     was a SHARED CONSTANT across all agreeing configs, invisible to
     cross-config agreement.
  C3 promoted dh/da = -1 from 12 rungs of ONE walk (no reseeds -- the engine's
     own N_RESEED=5 discipline, dropped in translation to the NS instance) and
     DERIVED the unique next experiment (deflated multistart). Prediction
     pre-registered and confirmed 12/12. Refused the polynomial model class
     correctly (three-fit scatter 5.4e-5 proved it right).
"""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
from eja_bridge import *

print("="*74); print("WITNESSES (the engine vs its own downstream measurements)")
w1 = mk_witness("C1_deduction_dadtheta", 1.40, "measured_cornerreg_frame", 15.0,
                {"quantity": "d(alpha)/d(theta)"})
print(f"  W1 the C1 'novel deduction' vs the other frame's value: divergence {w1.divergence:.2f}")
print(f"     -> the transfer was FRAME-DEPENDENT; 'novel' status was unearned.")
w2 = mk_witness("C2_premise_axes_exhausted", 0.0, "hidden_corner_panel_axis", 1.0,
                {"claim": "intervention space exhausted"})
print(f"  W2 'axes exhausted' vs the shared-constant corner panel: binary miss.")
print(f"     -> exhaustion was asserted over KNOWN axes; the space itself was")
print(f"        never audited for parameters held constant across all configs.")

print(); print("="*74); print("INVARIANCE MINING (what the engine's history proves about the engine)")
promote_invariance(mk_invariance("falsifier_at_mint_kills_cleanly", 0.0, eps=0.5))
print("  PROMOTED: conditional axioms with a named falsifier die cleanly (1/1 -- C2;")
print("            no lingering zombie claims). Mechanism verified in the wild.")
promote_invariance(mk_invariance("preregistered_predictions_hold", 0.0, eps=0.5))
print("  PROMOTED: pre-registered predictions were confirmed 2/2 (h-landscape walk;")
print("            collapse-test decision rule). Pre-registration works here.")

print(); print("="*74); print("REFUSALS (the control)")
r = refuse("the engine's DEDUCTIONS are reliable as-is", distance=1.0, scale=2.0,
           why="1 of 2 novel deductions failed downstream (dalpha/dtheta was frame-"
               "dependent); 50% is a coin, not an instrument")
print(f"  REFUSED: {r['refused']} -- {r['why']}")

print(); print("="*74); print("THE JUMP (quotient over the engine's own failures)")
print("  Both failures are ONE equivalence class: TRANSFER WITHOUT AN INVARIANCE")
print("  CHECK. C1 transferred a number across an identity without measuring its")
print("  invariance to the identity's hidden parameter (the frame). C2 transferred")
print("  'converged on every axis we varied' to 'exhausted' without auditing what")
print("  all configs held constant. Same defect, two costumes.")
ax = mk_axiom("transfer_requires_invariance_evidence",
    "A deduction across a promoted identity, or an exhaustion claim over an "
    "intervention space, is admissible ONLY with measured invariance evidence: "
    "the transferred quantity shown invariant under the identity's hidden "
    "parameters, or the config-set audited for shared constants.",
    domain="all future engine cycles; mechanically enforced in code (see below)",
    residual="cannot enforce IMAGINATION -- an axis nobody has named stays "
             "invisible; the audit only surfaces constants present in configs",
    evidence={"failures_explained": 2.0, "successes_consistent": 3.0})
print(f"  MINTED: {ax.name}")
print(f"  residual: {ax.residual}")

print(); print("="*74); print("EMITTED IMPROVEMENTS (code, not prose)")
print("  I1 mk_axiom(kind='identity', conditional): FALSIFIER becomes a REQUIRED")
print("     field -- refuse to mint a conditional axiom without its kill condition.")
print("  I2 mk_deduction(): new constructor. status='NOVEL' REQUIRES")
print("     invariance_evidence (the measured frame/config-invariance of the")
print("     transferred quantity). Without it: status is capped at 'CANDIDATE'.")
print("     This single gate would have caught the C1 failure at mint time.")
print("  I3 shared_constant_audit(configs): given the agreeing configurations,")
print("     return every parameter held constant across ALL of them -- the")
print("     mandatory premise-check before any 'exhausted/converged everywhere'")
print("     jump. Would have surfaced the corner panel in one call.")
print("  I4 TensionLedger: emitted tensions persist with status")
print("     open->consumed->confirmed/refuted; no more prose-only tensions that")
print("     evaporate between cycles.")
print("  I5 mk_equivalence gains reseeds=: promotion refuses robust=True unless")
print("     the caller states the independent-reseed count (>=2) -- restoring the")
print("     N_RESEED discipline the NS translation dropped (C3's dh/da promotion")
print("     ran on one walk).")
