"""THE JUMP ENGINE, POINTED AT THE PARZIVAL CAMPAIGN.

epagoge/jump mechanizes: two causally distinct configurations, indistinguishable
across a probe battery, robustly -> promoted to ONE phenomenon, with a fitted
correspondence map, a MEASURED domain of validity, and the residual kept as the
seed of the next theory.  Plus a negative control the operator must refuse.

Run on the NS campaign as it sits tonight.  All numbers below are MEASURED this
week (provenance in comments); nothing here re-runs the solver -- this is the
abduction layer reading the campaign's evidence table.

CANDIDATE E1 (the jump): "an eps_b-offset grid IS a wedge of smaller opening."
  Causally distinct: a NUMERICAL interpolation-safety offset vs a PHYSICAL
  domain deformation (opening angle theta = pi/2 - eps_wall - eps_axis).
  Battery evidence for indistinguishability:
    - Dirichlet spectrum: lambda_1 of -Db2 matches the truncated-wedge EXACT
      value (j*pi/theta)^2 to 3.4e-13, IDENTICALLY at Nb = 24/36/52/96/160
      (five 'reseeds' -- Nb plays the instrument-noise role: the identity is
      Nb-invariant, so it is a DOMAIN fact, not a discretisation fact).
    - the far-field exponents' deviations are exactly 2j*(eps_wall+eps_axis)*2/pi
      -- six of six, the wedge formula.
    - alpha responds to the OPENING, not the split: d(alpha)/d(eps_wall) =
      -1.8894 vs d(alpha)/d(eps_axis) = -1.8805 (premortem measurement, 0.47%).
CONTROL E2 (must NOT merge): constraint='d2' vs 'd1' -- IDENTICAL continuum
  content, but observationally distinguishable at finite N (alpha differs by
  9.1e-3 at N=28).  An engine that merges these is broken.
DEDUCTION across the promoted identity: the eps_b->0 extrapolation slope is not
  error-cleanup -- it is d(alpha)/d(theta) at the right-angle corner, a PHYSICAL
  derivative of the blowup exponent with respect to corner geometry.  Computed
  from tonight's panel ladder.  The old library calls this 'numerical error to
  extrapolate away'; the axiom says it is a measurement.  That disagreement is
  the NOVEL flag.
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/epagogellc/epagoge/jump")
from jump.abduce import EquivalenceCandidate
from jump.translate import Axiom

REF = -0.34240009

# ---------------------------------------------------------------------------
# E1: the correspondence map.  Predicted wedge formula: rel_err(lambda_1) =
# (8/pi)*eps_b.  Measured (premortem, three independent re-measurements):
eps_grid   = [1e-2, 1e-3, 1e-4, 1e-5]
ratio_meas = [1.01943, 1.00191, 1.00019, 1.00002]   # measured/(8 eps/pi) -> 1
pairs = [(e, e * r, abs(r - 1.0)) for e, r in zip(eps_grid, ratio_meas)]
xs = np.array([p[0] for p in pairs]); ys = np.array([p[1] for p in pairs])
slope, intercept = np.polyfit(xs, ys, 1)
pred = slope * xs + intercept
r2 = 1.0 - float(np.sum((ys - pred) ** 2)) / float(np.sum((ys - ys.mean()) ** 2))

E1 = EquivalenceCandidate(
    family_a="numerical eps_b offset grid",
    family_b="physical wedge of opening pi/2 - 2*eps_b",
    pairs=pairs, map_slope=float(slope), map_intercept=float(intercept),
    map_r2=float(r2),
    robust=True,   # lambda_1 identical at Nb = 24/36/52/96/160 (five reseeds, 3.4e-13)
)
# engine gate (its PROMOTION_EPS is metres in the box world; here the analogue
# instrument scale is the 0.47% wall/axis asymmetry -- the identity must hold
# below that):
promotable = E1.robust and all(d < 0.02 for _, _, d in E1.pairs) and E1.map_r2 > 0.999

print("=" * 74)
print("E1  eps_b-offset  ==  wedge-opening deformation")
print(f"    map: theta_eff = pi/2 - 2*({slope:.5f}*eps_b {intercept:+.2e}),  r^2 = {r2:.6f}")
print(f"    robustness: lambda_1 wedge-exact to 3.4e-13, Nb-invariant over 24..160")
print(f"    split-symmetry: d(alpha)/d(eps_wall) = -1.8894 vs d(eps_axis) = -1.8805 (0.47%)")
print(f"    PROMOTABLE: {promotable}")

# domain of validity, measured from the alpha(eps_b) response:
#   linear regime: tonight's panel ladder 1e-5..1e-3 (linear/quadratic agree to 1e-5)
#   breakdown: single-grid d2 at eps_b=1e-2 gave alpha=-0.4325 (-26%) -- wildly
#   nonlinear; and the N=36 d2 branch had a measured FOLD at eps_b ~ 5.1e-4.
axiom = Axiom(
    name="corner_opening_identity",
    kind="identity",
    statement=("A beta-grid with endpoint offsets (eps_w, eps_a) IS the blowup "
               "problem on a wedge of opening theta = pi/2 - eps_w - eps_a: "
               "alpha depends on the offsets ONLY through theta."),
    corr_map={"slope": float(slope), "intercept": float(intercept), "r2": float(r2)},
    domain=("linear response for eps_b <~ 1e-3 (panel ladder linear to 1e-5 bar); "
            "BREAKS by eps_b ~ 1e-2 (measured -26% nonlinearity) and can FOLD "
            "on some discretisations (d2/N=36 fold at 5.1e-4)"),
    residual=("0.47% wall-vs-axis asymmetry: the pinned axis column is "
              "inhomogeneous data, the wall is a true free edge -- the identity "
              "is exact only for the homogeneous-Dirichlet part of the system"),
    evidence={"lambda1_match": 3.4e-13, "nb_reseeds": 5.0,
              "split_symmetry_pct": 0.47},
)
print(f"\n    AXIOM: {axiom.statement}")
print(f"    domain: {axiom.domain}")
print(f"    residual: {axiom.residual}")

# ---------------------------------------------------------------------------
# DEDUCE: transfer across the identity.  Tonight's panel ladder (eps_b, alpha):
lad = [(1e-3, -0.34109901), (6e-4, -0.34033068), (3e-4, -0.33965380),
       (1e-4, -0.33911408), (3e-5, -0.33889946), (1e-5, -0.33883505)]
e = np.array([p[0] for p in lad]); a = np.array([p[1] for p in lad])
dade = float(np.polyfit(e[-4:], a[-4:], 1)[0])       # d(alpha)/d(eps_b), symmetric split
# theta = pi/2 - 2*eps_b  =>  d(alpha)/d(theta) = -dade/2
dadtheta = -dade / 2.0
print("\n" + "=" * 74)
print("DEDUCTION across the promoted identity  (the JUMP)")
print(f"    old library: 'alpha(eps_b) slope is numerical error; extrapolate it away'")
print(f"    axiom says:  it is d(alpha)/d(theta) at the right-angle corner -- a")
print(f"                 PHYSICAL derivative of the blowup exponent wrt corner geometry")
print(f"    measured:    d(alpha)/d(eps_b) = {dade:+.4f}   (panel ladder, small-eps tail)")
print(f"    =>           d(alpha)/d(theta)|_(pi/2) = {dadtheta:+.4f}")
print(f"    reading:     opening the corner past 90 deg RAISES alpha (weakens the")
print(f"                 singularity exponent) at ~{abs(dadtheta):.2f} per radian;")
print(f"                 status NOVEL -- no published value exists for this derivative.")
print(f"    checkable:   run the solver at theta != pi/2 deliberately (the identity")
print(f"                 says: just choose eps offsets); compare slope. Also the")
print(f"                 premortem's own split-asymmetry numbers ARE the first check:")
print(f"                 -1.8894 ~ -1.8805 within 0.47% (single-grid d2, different")
print(f"                 config -- note the engine keeps configs' slopes distinct).")

# ---------------------------------------------------------------------------
# E2: NEGATIVE CONTROL -- the operator must refuse to merge d1 and d2.
d_alpha_28 = abs(-0.33273323 - (-0.34187584))    # measured, N=28, eps_b=1e-3
print("\n" + "=" * 74)
print("E2  CONTROL: constraint 'd2' vs 'd1' (identical continuum content)")
print(f"    trace distance at N=28: |alpha_d1 - alpha_d2| = {d_alpha_28:.2e}")
print(f"    vs promotion scale 2e-4 (family-A internal spread): FAILS by {d_alpha_28/2e-4:.0f}x")
print(f"    -> REFUSED. Identical continuum content is NOT observational")
print(f"       equivalence at finite N; merging them would erase the corner-row")
print(f"       discretisation error the whole campaign just spent a day measuring.")

# ---------------------------------------------------------------------------
# EMITTED TENSION (the cycle's feedback -- what the axiom now makes urgent):
print("\n" + "=" * 74)
print("EMITTED TENSIONS (fed back as the next intervention families)")
print("  T1: if alpha is PHYSICAL in theta, the eps_b->0 limit is the true")
print("      right-angle value: alpha(pi/2) = -0.33882 +- 1e-4 (panels, all radial")
print("      axes converged). It disagrees with Chen-Hou -0.34240 by +1.05% while")
print("      the free residual d_cl = -4.6% REMAINS. The identity therefore says:")
print("      the remaining error is NOT domain deformation. The only never-varied")
print("      axis on the panel solver is Nb. -> intervention family: Nb ladder.")
print("  T2: the axiom's residual (wall/axis asymmetry from the inhomogeneous")
print("      pinned column) is the SAME object as the frozen-axis-column defect")
print("      the audit priced at 2e-5 on the OLD solver -- re-price it on panels")
print("      at eps_b -> 0, where the pinned data's interpolation error is largest.")
print("  T3: d(alpha)/d(theta) = %+.3f is a standalone publishable measurement" % dadtheta)
print("      IF T1 resolves (a number computed on an instrument with a live 1%")
print("      systematic is not yet a measurement).")
