"""NUMBERS/CONTEXT measurer: spacing analysis, bias plausibility, CHL comparison.

Seconds of compute. No field loading, no Newton solves.
"""
import sys

sys.path.insert(0, "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
from eja_bridge import *  # noqa: F401,F403
import numpy as np

# ---- inputs (all previously measured / published) --------------------------
a0_ref = -0.34240009          # Chen-Hou x DeepMind, agree to 3.4e-7
a1 = -0.4168236               # DeepMind, 6 validated digits
a2 = -0.4439811               # DeepMind, 5 validated digits
a3 = -0.4578230               # DeepMind, 4 validated digits
ours = -0.42172919            # eps=1e-5 rung of the polished branch root
ours_1e4 = -0.42174207        # secant-polished at eps=1e-4
chl2 = -0.40834               # CHL stage-2

# measured sensitivities (ground branch = the only prior available)
ground_deg0_sens = 7e-4       # deg0 16->24 at eps=1e-4
ground_eps_sens = 2.3e-3      # eps 1e-4 -> 1e-5
branch_eps_motion = 1.3e-5    # eps ladder 1e-4/5e-5/3e-5/1e-5, total motion
dh_da_ground = -1.0002
dh_da_branch = -1.026

print("=" * 72)
print("1. SPACING ANALYSIS")
print("=" * 72)
seq = [a0_ref, a1, a2, a3]
gaps = np.diff(seq)                       # g1, g2, g3 (all negative)
print(f"gaps g_n = alpha_n - alpha_(n-1): {gaps}")
ratios = gaps[1:] / gaps[:-1]
print(f"gap ratios r_n = g_(n+1)/g_n:     {ratios}")
# log-linear fit of |g_n| vs n -> single geometric ratio
n = np.arange(1, 4)
coef = np.polyfit(n, np.log(np.abs(gaps)), 1)
r_fit = float(np.exp(coef[0]))
resid = np.log(np.abs(gaps)) - np.polyval(coef, n)
print(f"log-linear fit ratio r = {r_fit:.4f}; log-resid max {np.abs(resid).max():.3f}"
      f" (ratio is NOT constant: {ratios[0]:.4f} vs {ratios[1]:.4f})")

# accumulation point, two estimates
acc_last = a3 + gaps[-1] * ratios[-1] / (1 - ratios[-1])   # continue with last ratio
acc_fit = a3 + gaps[-1] * r_fit / (1 - r_fit)              # continue with fitted ratio
print(f"accumulation point (continue last ratio {ratios[-1]:.4f}): {acc_last:.6f}")
print(f"accumulation point (fitted ratio {r_fit:.4f}):          {acc_fit:.6f}")

# where does OUR root fall
print(f"\nour root: {ours}")
print(f"ours - alpha_0 = {ours - a0_ref:+.7f}")
print(f"ours - alpha_1 = {ours - a1:+.7f}  ({(ours - a1)/abs(a1)*100:+.3f}% of alpha_1)")
print(f"ours - alpha_2 = {ours - a2:+.7f}")
frac = (ours - a1) / (a2 - a1)
print(f"fractional position between alpha_1 and alpha_2: {frac:.4f} "
      f"(i.e. {frac*100:.1f}% of the a1->a2 gap)")
# any geometric slot?  members live at a0, a1, a2, a3, ... ; midpoints don't exist
# in a geometric family.  Nearest member distance:
dists = {f"alpha_{i}": abs(ours - a) for i, a in enumerate(seq)}
print(f"distance to nearest family member: {min(dists.values()):.6f} ({min(dists, key=dists.get)})")
print(f"|ours - alpha_0| / |g1| = {abs(ours - a0_ref)/abs(gaps[0]):.4f} "
      f"(if ours were 'alpha_1 of a shifted family', its first gap is 6.6% larger)")

print()
print("=" * 72)
print("2. BIAS PLAUSIBILITY")
print("=" * 72)
needed = a1 - ours            # what discretization must supply for ours==alpha_1
print(f"needed bias (alpha_1 - ours) = {needed:+.6f}  |needed| = {abs(needed):.4e}")
print(f"ground deg0 16->24 sensitivity:      {ground_deg0_sens:.1e}"
      f"  -> needed/measured = {abs(needed)/ground_deg0_sens:.1f}x")
print(f"ground eps sensitivity (1e-4->1e-5): {ground_eps_sens:.1e}"
      f"  (branch measured {branch_eps_motion:.1e}: {ground_eps_sens/branch_eps_motion:.0f}x flatter -> eps RULED OUT)")
# error amplification through the self-consistency map h(a)=cw/cl(a)-a
amp_ground = 1.0 / abs(dh_da_ground)
amp_branch = 1.0 / abs(dh_da_branch)
print(f"amplification |1/h'(a)|: ground {amp_ground:.4f}, branch {amp_branch:.4f}"
      f" (branch amplification is {(amp_ground / amp_branch - 1) * 100:.1f}% smaller)")
# stacked plausible bias from measured priors: deg0 term + eps residual + assume an
# unmeasured mid-deg term as large as the deg0 term
plaus = ground_deg0_sens + branch_eps_motion + ground_deg0_sens
print(f"plausible stacked bias (deg0 + eps + equal-size unmeasured mid-deg term): "
      f"{plaus:.2e} -> needed is {abs(needed)/plaus:.1f}x the stack")

print()
print("=" * 72)
print("3. DEEPMIND'S BARS (arithmetic only; reasoning in the report)")
print("=" * 72)
# 6 validated digits on -0.4168236 => their claimed uncertainty <= ~5e-7 in the
# 7th digit position; a 1.2% error is:
dm_bar = 5e-7                # half-ulp of the last boldfaced digit
print(f"1.2% error on alpha_1 = {abs(needed):.4e} = {abs(needed)/dm_bar:.0f}x their "
      f"boldface half-ulp ({dm_bar:.0e})")
print(f"their alpha_0 (same pipeline) matches Chen-Hou to 3.4e-7 -> a {abs(needed):.1e} "
      f"branch error would be {abs(needed)/3.4e-7:.0f}x their demonstrated cross-method accuracy")

print()
print("=" * 72)
print("4. CHL STAGE-2")
print("=" * 72)
print(f"CHL2 = {chl2}")
print(f"CHL2 - alpha_1 = {chl2 - a1:+.6f}  (|{abs(chl2-a1):.4e}|)")
print(f"ours - CHL2    = {ours - chl2:+.6f}  (|{abs(ours-chl2):.4e}|)")
print(f"|ours-CHL2| / |ours-alpha_1| = {abs(ours-chl2)/abs(ours-a1):.2f}x")
print(f"ordering on the line: a0({a0_ref}) > CHL2({chl2}) > a1({a1}) > OURS({ours}) > a2({a2})")
print(f"CHL2 is on the OPPOSITE side of alpha_1 from our root.")

print()
print("=" * 72)
print("EJA OBJECTS (minted, real output)")
print("=" * 72)

# --- witnesses: the tensions, as computable scalars -------------------------
w1 = mk_witness("polished_branch_root_eps->0", ours, "DeepMind_alpha_1", a1,
                scenario={"frozen_a": a1, "eps_b": 1e-5, "degs": "(16,40,12)"},
                tags=("branch_attribution", "outside_subpercent_gate"))
print(f"WITNESS ours-vs-alpha_1: divergence = {w1.divergence:.6e} "
      f"(gate: sub-percent = 1e-2; this is {w1.divergence/1e-2*100:.0f}% of the gate... "
      f"i.e. 1.18% relative -> OUTSIDE)")

w2 = mk_witness("polished_branch_root_eps->0", ours, "CHL_stage2", chl2,
                scenario={"frozen_a": a1, "eps_b": 1e-5},
                tags=("branch_attribution",))
print(f"WITNESS ours-vs-CHL2:   divergence = {w2.divergence:.6e}")

# --- refusals: the negative controls ---------------------------------------
# scale = the instrument scale on this branch: eps-ladder total motion 1.3e-5 is the
# branch's own demonstrated repeatability; the largest measured single-axis
# resolution effect anywhere is 7e-4.
r1 = refuse("candidate_root == alpha_1", distance=abs(needed), scale=ground_deg0_sens,
            why="needed bias 4.9e-3 is 7.0x the largest measured single-axis "
                "resolution sensitivity (ground deg0 16->24 = 7e-4) with eps "
                "measured flat (1.3e-5); amplification |1/h'| differs only 2.5% "
                "between branches, so no branch-specific error amplifier is measured")
print(f"REFUSED merge ours==alpha_1: distance {r1['distance']:.3e}, "
      f"scale {r1['scale']:.0e}, ratio {r1['ratio']:.1f}x")

r2 = refuse("candidate_root == CHL_stage2", distance=abs(ours - chl2), scale=ground_deg0_sens,
            why="1.34e-2 away, 2.7x farther than alpha_1 and on the OPPOSITE side; "
                "19x the largest measured resolution sensitivity")
print(f"REFUSED merge ours==CHL2:    distance {r2['distance']:.3e}, "
      f"scale {r2['scale']:.0e}, ratio {r2['ratio']:.1f}x")

# --- shared-constant audit BEFORE any 'eps-flat everywhere' reading ---------
configs = [
    dict(name="branch1_eps1e-4", eps_b=1e-4, deg0=16, degmid=40, degcorner=12,
         Nb=36, edges="(0,2,15,25)", seed_lineage="chen-hou-interp"),
    dict(name="branch1_eps5e-5", eps_b=5e-5, deg0=16, degmid=40, degcorner=12,
         Nb=36, edges="(0,2,15,25)", seed_lineage="chen-hou-interp"),
    dict(name="branch1_eps3e-5", eps_b=3e-5, deg0=16, degmid=40, degcorner=12,
         Nb=36, edges="(0,2,15,25)", seed_lineage="chen-hou-interp"),
    dict(name="branch1_eps1e-5", eps_b=1e-5, deg0=16, degmid=40, degcorner=12,
         Nb=36, edges="(0,2,15,25)", seed_lineage="chen-hou-interp"),
]
shared = shared_constant_audit(configs)
shared.pop("name", None)
print(f"SHARED-CONSTANT AUDIT (the eps-flat agreement is blind to): {shared}")

# --- conditional axiom: names its own falsifier at mint ---------------------
ax = conditional_axiom(
    "candidate_root_is_distinct_from_alpha1",
    statement="The deflated root at frozen a=-0.4168236 (self-consistent "
              "alpha=-0.421727, eps-flat to 1.3e-5) is NOT DeepMind's alpha_1; "
              "it is a distinct object 1.18% away",
    domain="degs=(16,40,12), Nb=36, edges=(0,2,15,25), eps_b in [1e-5,1e-4], "
           "seeds descended from Chen-Hou interpolation",
    residual="mid-deg (40 vs 56) and corner-deg axes UNMEASURED on this branch; "
             "seed provenance untested (all runs share one lineage); WHY the wedge "
             "layer barely couples to this profile is unexplained",
    falsifier="the running resolution study: if interpolation-seeded higher-deg rungs "
              "move branch alpha by >= 4.9e-3 toward -0.4168236 (7x the ground "
              "branch's deg0 sensitivity), the distinct-root claim dies",
    evidence={"gap": needed, "gap_rel": needed / abs(a1),
              "eps_flatness": branch_eps_motion, "ground_deg0_sens": ground_deg0_sens})
print(f"CONDITIONAL AXIOM minted: {ax.name}")
print(f"  statement: {ax.statement[:110]}...")

# --- deduction: the accumulation point, honestly capped ---------------------
d = deduce("deepmind_family_accumulation_point", "alpha_inf", acc_last,
           note="3-gap extrapolation, ratio NOT constant (0.365 vs 0.510); "
                "no invariance evidence -> CANDIDATE by construction")
print(f"DEDUCE accumulation point: {d.value:.5f} status={d.status} (capped: no "
      f"invariance evidence for a 3-gap geometric extrapolation)")

# fraction-slot check as a number: does ours sit at a geometric slot?
# a hypothetical alpha_1.5 doesn't exist in a geometric family; the nearest
# constructible object is alpha_1 + g2*x for integer-power x.
print(f"\nslot check: ours sits {frac*100:.1f}% into the a1->a2 gap; a geometric "
      f"family has members only at 0% and 100%. No slot.")
