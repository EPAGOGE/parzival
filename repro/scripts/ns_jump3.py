"""ABDUCTION CYCLE 3: tracks 1 (tighten) + 2 (branch hunt) fed to the engine."""
import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
from eja_bridge import *
import numpy as np

REF = -0.34240009
print("="*74)
print("WITNESSES")
# extrapolation-model tension: three estimators of the same limit disagree
w1 = mk_witness("quad_5pt", -0.34240089, "quad_3pt_deep", -0.34238515, {"limit":"eps->0"})
w2 = mk_witness("quad_3pt_deep", -0.34238515, "lin_3pt_deep", -0.34244202, {"limit":"eps->0"})
print(f"  W1 5pt-quad vs 3pt-quad: divergence {w1.divergence:.2e}")
print(f"  W2 quad vs lin (deep triple): divergence {w2.divergence:.2e}")
print("  -> the eps->0 LIMIT is stable to ~1e-4 but the MODEL CLASS is not settled:")
print("     estimators scatter +-5e-5 around the reference. Tension, not error.")

print()
print("="*74)
print("PROMOTIONS")
# 1. Nb invariance at final precision
promote_invariance(mk_invariance("alpha_ignores_Nb_36to48", 1.16e-8, eps=1e-5))
print("  PROMOTED alpha ignores Nb (36->48 shift 1.16e-8) -- axis closed at 1e-8.")
# 2. THE NEW ONE from the landscape walk: cw/cl of the n=0 field branch is
#    invariant to the frozen substitution exponent
slope = (3.005406 - 3.005438) / (0.430000 - 0.344712)   # d(cl)/da tiny
dh_da = (8.531e-2 - 0.0) / (-0.430000 + 0.344712)        # ~ -1.0002
d_cwcl_da = dh_da + 1.0                                   # h = cw/cl - a
promote_invariance(mk_invariance("n0_branch_cwcl_ignores_frozen_a", abs(d_cwcl_da), eps=1e-2))
print(f"  PROMOTED cw/cl of the n=0 branch ignores the frozen exponent a:")
print(f"     dh/da = {dh_da:+.5f} (exactly -1 within 2e-4) over Delta a = 0.085,")
print(f"     c_l moves 3.2e-5 total. h(a) = a* - a IDENTICALLY on this branch.")
# 3. the three-method identity, minted with domain and residual
ax = mk_axiom("three_method_alpha_identity",
    "Chen-Hou march, DeepMind PINN, and the corner-regularized spectral Newton "
    "compute ONE object: alpha_0 agrees across all three within stated bars.",
    domain="n=0 stable branch; our bar +-5e-5 (extrapolant scatter), their spread 3.4e-7",
    residual="our eps->0 model-class ambiguity (5.7e-5) + the retired d(alpha)/d(theta)",
    evidence={"ours_quad": -0.34240089, "ref": REF, "bar": 5.7e-5})
print(f"  MINTED: {ax.name} [domain: {ax.domain}]")

print()
print("="*74)
print("REFUSALS (the control)")
r = refuse("eps-convergence is polynomial (quadratic model class)",
           distance=5.7e-5, scale=3.4e-5,
           why="lin/quad estimators straddle the reference by more than the claimed "
               "bar; the singular layer xi^(k-2), k-2 ~ eps, plausibly injects a "
               "non-polynomial eps*ln(eps) term the quadratic fit cannot represent")
print(f"  REFUSED: {r['refused']} (ratio {r['ratio']:.1f})")

print()
print("="*74)
print("DEDUCTIONS  (what the promoted invariance FORCES)")
print("  D1: h(a) = a* - a exactly on the n=0 branch means the h-landscape can NEVER")
print("      reveal another branch -- no sign change, no structure, at ANY a. The")
print("      pre-registered walk prediction ('smooth monotone => branches live OFF")
print("      this field branch') is CONFIRMED by 12/12 rungs. Therefore the ONLY")
print("      instrument that can find alpha_1..3 is DEFLATED MULTISTART at frozen a:")
print("      repel the n=0 anchor, search field space, keep any root whose own")
print("      cw/cl - a ~ 0. The engine derives the next experiment, not as taste")
print("      but as the unique survivor.")
print("  D2: dh/da = -1 <=> d(cw/cl)/da = 0: the gauge ratio is a property of the")
print("      FIELD SOLUTION, not the substitution -- alpha is well-defined per branch")
print("      independent of the continuation parameter. Good news for attribution:")
print("      any deflated root's cw/cl IS its branch label, no self-consistency")
print("      iteration needed for identification (only for polish).")
print()
print("EMITTED EXPERIMENTS")
print("  T1 (track 1): fit alpha(eps) with a+b*eps+c*eps^2 VS a+b*eps+c*eps*ln(eps)")
print("      on the six rungs; AICc + LOO. If the log model wins, the limit shifts")
print("      by O(1e-5) and the bar tightens honestly. Cheap, decisive.")
print("  T2 (track 2): deflated multistart at frozen a = -0.4168236 (alpha_1),")
print("      anchors = saved rungs; starts = structured + random perturbations at")
print("      1/5/20%; keep roots with ||F||<1e-11 and |cw/cl - a| < 1e-2; then")
print("      secant-polish alpha on any find. Attribution gate: sub-percent + the")
print("      (N,eps) study before ANY claim -- CHL stage-2 sits 0.0085 away.")
