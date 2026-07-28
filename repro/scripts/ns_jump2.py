"""ABDUCTION CYCLE 2 on the Parzival campaign -- the post-exoneration evidence table.

Cycle 1 promoted the corner-opening identity and emitted T1 (Nb) and T2 (axis column).
Both interventions have since been RUN.  This cycle feeds the results back through the
engine's three idioms: tension (theory-vs-theory witnesses), invariance mining
(quotient-and-promote), and the Jump itself -- with the negative control the operator
must refuse.  All numbers measured 2026-07-26/27; provenance in comments.
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/epagogellc/epagoge/jump")
from jump.tension import RuleSystem, Witness
from jump.abduce import InvarianceResult
from jump.translate import Axiom, promote_invariance

REF = -0.34240009        # Chen-Hou == DeepMind boldface band, cross-method 3.4e-7
OURS = -0.338835         # panel instrument, every axis converged/extrapolated
CLS = 3.00649824

# ---------------------------------------------------------------------------
print("=" * 74)
print("WITNESSES  (theory-vs-theory contradictions: the drive signal)")
sysA = RuleSystem("panel_instrument_limit", lambda s: OURS, frozenset({"formulation"}))
sysB = RuleSystem("chenhou_deepmind_pair", lambda s: REF, frozenset({"reference"}))
w1 = Witness(scenario={"corner": np.pi / 2}, system_a=sysA.name, system_b=sysB.name,
             pred_a=OURS, pred_b=REF, tags=frozenset({"alpha"}))
print(f"  W1 alpha at the right-angle corner: {OURS:+.6f} vs {REF:+.6f}"
      f"   divergence = {100*w1.divergence:.3f}%")
# internal witness: our own solution vs the continuum-forced corner identity
d_cl = -0.0461           # (c_l - 2 THXX/WX)/(2 THXX/WX) at eps->0, Nb=36..48
w2 = Witness(scenario={"identity": 1}, system_a="discrete_corner_limit",
             system_b="continuum_corner_ODE", pred_a=CLS * (1 + d_cl), pred_b=CLS,
             tags=frozenset({"gauge"}))
print(f"  W2 corner identity c_l = 2*THXX/WX: ours {CLS*(1+d_cl):.4f} vs exact {CLS:.4f}"
      f"   divergence = {100*abs(d_cl):.2f}%   [REFERENCE-FREE]")
print("  W2 is the sharper witness: it accuses our discrete corner limit without")
print("  invoking anyone's published number.")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("INVARIANCE MINING  (what tonight's interventions proved alpha ignores)")
invs = [
    ("axis_column_content", 3.0e-8,  "pinned data x0..x2 (300x-corrupted -> zeroed)"),
    ("outer_bc_form",       2.5e-8,  "neumann vs exact DtN annihilators"),
    ("outer_edge_position", 6.0e-8,  "XMAX 15 vs 25 at matched mid-band resolution"),
    ("midband_degree",      9.0e-8,  "deg 40 -> 88, converged to 8 digits"),
    ("angular_resolution",  1.3e-4,  "Nb 36 -> 48 (bounded, 30x below the gap)"),
]
for name, worst, note in invs:
    inv = InvarianceResult(name, worst, worst < 1e-2)
    ax = promote_invariance(inv)
    print(f"  PROMOTED  alpha invariant to {name:22s} (worst {worst:.1e})  [{note}]")

print("\n  NOT an invariance -- alpha responds to the corner OPENING (cycle-1 axiom):")
print("  d(alpha)/d(theta) = +1.40/rad, and to nothing else we can vary.")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("CONTROL  (the merge the operator must refuse)")
print("  candidate: 'd_cl = -4.6% == numerical noise'.  REFUSED on three measurements:")
print("    - it is MONOTONE in the domain healing (-14.56 -> -4.61% along eps_b->0),")
print("    - it is INVARIANT to Nb (-4.61 -> -4.48%) and to the axis column,")
print("    - noise is neither monotone in one knob nor invariant to all others.")
print("  d_cl is a SYSTEMATIC of the surviving formulation. It is signal.")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("THE JUMP  (Peirce, executed on W1+W2)")
print("  Rule:   a convergent discretisation agrees with the continuum problem it")
print("          discretises, in the limit. (Every axis above is at its limit.)")
print("  Result: two converged computations disagree by 1.04%, and ours violates a")
print("          continuum-forced identity by 4.6% AT convergence.")
print("  Case:   THE TWO COMPUTATIONS DISCRETISE DIFFERENT CONTINUUM PROBLEMS.")
print("          The discrepancy is not IN the intervention space we have been")
print("          searching (grids, domains, resolutions). It is a FORMULATION")
print("          delta: some operator, constraint, or corner treatment of ours")
print("          has a different continuum limit than theirs.")
ax_jump = Axiom(
    name="formulation_delta",
    kind="identity",
    statement=("The panel instrument's limit object and the Chen-Hou/DeepMind limit "
               "object are DISTINCT continuum problems; alpha and d_cl differences "
               "are properties of the delta, not errors of the mesh."),
    corr_map=None,
    domain=("holds unless the dust-free control (running) moves alpha at the 1e-3 "
            "scale -- the one shared rule all agreeing panelizations inherited"),
    residual=("d_cl = -4.6%: the measured fingerprint of the delta at the corner"),
    evidence={"alpha_divergence": 1.04e-2, "d_cl": 4.6e-2, "axes_exhausted": 8.0},
)
print(f"\n  AXIOM(conditional): {ax_jump.statement}")
print(f"  domain:   {ax_jump.domain}")
print(f"  residual: {ax_jump.residual}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("DEDUCTION + EMITTED EXPERIMENTS  (ranked by information per hour)")
print("  P1 (signature test): a FORMULATION delta is CONFIG-INVARIANT; an error is not.")
print("      Measure r2 = c_w - c_l/2 - u_x(0) -- the OTHER half of Chen-Hou's corner")
print("      relation, never imposed and never measured on our solutions -- across")
print("      >= 3 configs. Config-invariant violation => formulation delta CONFIRMED")
print("      and r2+d_cl jointly fingerprint WHICH operator differs. Scattered =>")
print("      the axiom is refuted and an unfound discretisation error survives.")
print("  P2 (live falsifier): the dust-free run. If deg-12/no-dust alpha shifts")
print("      ~1e-2, the shared dust rule owned the gap and the Jump axiom dies")
print("      cleanly. [running now]")
print("  P3 (the prize if P1 confirms): the delta candidates, ranked --")
print("      (a) our d1/d2 corner CONSTRAINTS pin reference VALUES where Chen-Hou")
print("          SLAVE the gauge relations at every stage: at a true steady state")
print("          these coincide ONLY if the discrete corner limit of the transport")
print("          operator matches the continuum corner ODE -- derive ours")
print("          symbolically at the first panel nodes and compare;")
print("      (b) the wall line j=0 row structure (E*Ginv corner scaling);")
print("      (c) the Pt=0 corner identity rows (POLAR_SPEC section-16 caveat --")
print("          Otway says the closed problem is OVER-determined).")
print("  P4 (cycle-1 payoff, unlocked by P1): if the delta is found and closed and")
print("      alpha lands on -0.3424, then d(alpha)/d(theta) = +1.40/rad graduates")
print("      from 'engine curiosity' to the first measured corner-angle derivative")
print("      of this blowup -- a standalone note.")
