"""Follow-up: is the branch's eps-motion mostly a global rescaling (which cancels
in cw/cl and hence in alpha)?  Decompose F(eps_lo) = lambda*F(eps_hi) + resid.
Also: cw/cl and c=(cl-2cw)/4 along the ladder, and EJA minting with real numbers.
"""
import sys
import importlib.util
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad"
HF = SCRATCH + "/hunt_fields"
sys.path.insert(0, SCRATCH)

spec = importlib.util.spec_from_file_location(
    "pc", "/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py")
pc = importlib.util.module_from_spec(spec)
sys.modules["pc"] = pc
spec.loader.exec_module(pc)
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12),
                       Nb=36, eps_b=1e-4, alpha=-0.4168236)
Nx, Nb = S.Nx, S.Nb
n = Nx * Nb
pm = sys.modules["pm"]

from scipy.interpolate import BarycentricInterpolator


def beta_grid(e):
    b, _, _ = pm.grid(Nb - 1, e, np.pi / 2 - e)
    return b


def interp_beta(F, b_src, b_dst):
    out = np.empty((F.shape[0], len(b_dst)))
    for i in range(F.shape[0]):
        out[i, :] = BarycentricInterpolator(b_src, F[i, :])(b_dst)
    return out


def load(fn):
    d = np.load(f"{HF}/{fn}")
    z = d["z"]
    return dict(A=z[:n].reshape(Nx, Nb), B=z[n:2 * n].reshape(Nx, Nb),
                P=z[2 * n:3 * n].reshape(Nx, Nb),
                cl=float(z[3 * n]), cw=float(z[3 * n + 1]),
                a=float(d["a"]) if "a" in d else None)


LADDER = [("branch1_eps1e-4.npz", 1e-4), ("branch1_eps5e-05.npz", 5e-5),
          ("branch1_eps3e-05.npz", 3e-5), ("branch1_eps1e-05.npz", 1e-5)]
lad = [(load(fn), e) for fn, e in LADDER]

print("=== ladder invariants ===")
for d, e in lad:
    c = (d["cl"] - 2 * d["cw"]) / 4
    print(f"eps={e:.0e}: a_frozen={d['a']:+.9f}  cl={d['cl']:+.9f}  "
          f"cw={d['cw']:+.9f}  cw/cl={d['cw'] / d['cl']:+.9f}  c={c:+.9f}")
h_hi = lad[0][0]["cw"] / lad[0][0]["cl"]
h_lo = lad[-1][0]["cw"] / lad[-1][0]["cl"]
print(f"cw/cl motion 1e-4 -> 1e-5: {h_lo - h_hi:+.3e}")
cl_hi, cl_lo = lad[0][0]["cl"], lad[-1][0]["cl"]
print(f"cl motion (relative): {(cl_lo - cl_hi) / cl_hi:+.3e}")

print("\n=== global-rescaling decomposition of eps motion (1e-4 -> 1e-5) ===")
bref = beta_grid(1e-4)
d_hi, d_lo = lad[0][0], lad[-1][0]
lams = {}
for fld in ("A", "B", "P"):
    Fh = interp_beta(d_hi[fld], beta_grid(1e-4), bref)
    Fl = interp_beta(d_lo[fld], beta_grid(1e-5), bref)
    lam = float(np.sum(Fh * Fl) / np.sum(Fh * Fh))
    resid = float(np.linalg.norm(Fl - lam * Fh) / np.linalg.norm(Fh))
    raw = float(np.linalg.norm(Fl - Fh) / np.linalg.norm(Fh))
    lams[fld] = lam
    print(f"{fld}: lambda={lam:+.6f}  raw rel motion={raw:.3e}  "
          f"after-scale resid={resid:.3e}  scale-removed fraction="
          f"{1 - resid / raw:.3f}")
print(f"lambda spread across fields: A={lams['A']:.6f} B={lams['B']:.6f} "
      f"P={lams['P']:.6f}")
print(f"lambda_B / lambda_A^2 = {lams['B'] / lams['A'] ** 2:.6f}  "
      f"lambda_P / lambda_A = {lams['P'] / lams['A']:.6f}")
print(f"cl ratio (lo/hi) = {cl_lo / cl_hi:.6f}  vs lambda_A = {lams['A']:.6f}")

# ---------------------------------------------------------------- EJA minting
print("\n=== EJA engine objects ===")
from eja_bridge import (mk_witness, mk_invariance, promote_invariance, refuse,
                        conditional_axiom, shared_constant_audit)

# refuted-mechanism witness: hypothesis said corner P amplitude branch << ground;
# measured c_branch/c_ground:
w1 = mk_witness("hyp_small_corner_P_explains_eps_flat", 0.0,
                "measured_corner_c_ratio_cand_over_ground", 2.408036609 / 1.269365094,
                scenario={"eps_b": 1e-4, "corner_row": 0},
                tags=("mechanism", "refutation"))
print(f"w1 corner-amplitude witness: divergence={w1.divergence:.4f} "
      f"(hypothesis 'much smaller' vs measured 1.897x LARGER)")

# corner algebra P(0,b)=c sin(k(b-eps)) holds on BOTH roots
w2 = mk_witness("P0_fit_ground_rel_resid", 3.737e-06,
                "P0_fit_candidate_rel_resid", 2.763e-04,
                scenario={"harmonic": "sin(k(b-eps_b))", "k": 2.00025468},
                tags=("corner_algebra",))
print(f"w2 corner-fit witness: ground resid 3.7e-6 vs candidate 2.8e-4 "
      f"(divergence={w2.divergence:.4f}) -- algebra holds on both, candidate "
      f"75x noisier at same resolution")

# invariance: cw/cl ignores the eps intervention on the branch
inv = mk_invariance("branch_cwcl_ignores_eps_1e-4_to_1e-5",
                    worst_effect=abs(h_lo - h_hi), eps=1e-4)
print(f"inv: worst_effect={abs(h_lo - h_hi):.3e} promotable={inv.promotable}")
try:
    promote_invariance(inv)
    print("inv PROMOTED: cw/cl is eps-invariant on the branch at this resolution")
except Exception as ex:
    print(f"inv promotion REFUSED: {ex}")

# the fields do NOT ignore eps -- refuse the tempting merge
r1 = refuse("branch_fields_ignore_eps",
            distance=3.18e-2, scale=1.3e-5 / 0.42,
            why="fields move 3-6% over the eps ladder while alpha moves 1.3e-5; "
                "only the RATIO cw/cl is flat -- refusing to call the branch "
                "fields eps-invariant")
print(f"r1 REFUSAL: {r1['refused']} distance/scale={r1['ratio']:.1f}x")

r2 = refuse("corner_P_small_explains_eps_flat",
            distance=2.408036609 - 1.269365094, scale=1.269365094 * 0.1,
            why="mechanism required candidate corner amplitude << ground; measured "
                "1.897x LARGER (energy ratio 2.15). Merge refused; eps-flatness "
                "must come from the scale-mode cancellation in cw/cl, not from "
                "weak corner coupling")
print(f"r2 REFUSAL: {r2['refused']} distance/scale={r2['ratio']:.1f}x")

aud = shared_constant_audit([
    {"degs": "(16,40,12)", "Nb": 36, "edges": "(0,2,15,25)",
     "seed": "Chen-Hou interpolation", "update": "secant on cw/cl",
     "eps_b": e} for _, e in LADDER])
print(f"shared-constant audit (untested axes all ladder rungs hold fixed): {aud}")

ca = conditional_axiom(
    "eps_flat_via_scale_mode",
    "On the candidate branch, eps_b perturbs the root predominantly along a "
    "near-uniform amplitude rescaling (lambda_P/lambda_A=1.0015, "
    "lambda_B/lambda_A^2=1.0006), which cancels in the ratio cw/cl; alpha "
    "eps-flatness is a quotient of the scale mode, not weak corner coupling.",
    domain="branch root, eps_b in [1e-5,1e-4], degs=(16,40,12), Nb=36; "
           "scale-removed fraction 0.883/0.929/0.989 for A/B/P",
    residual="after-scale residual 3.6e-3 (A), 4.4e-3 (B), 3.6e-4 (P) is NOT "
             "zero -- the non-scale part of the eps response is the surviving "
             "unexplained structure; and WHY ground lacks this cancellation "
             "is unmeasured (no ground eps-ladder fields on disk here)",
    falsifier="a ground-branch eps ladder decomposed the same way: if ground "
              "shows the SAME scale-removed fraction (>0.88) while its alpha "
              "moves 2.3e-3, the cancellation story dies; also dies if "
              "lambda_B/lambda_A^2 leaves 1 by more than the after-scale "
              "residual at the next resolution rung",
    evidence={"raw_motion_P": 3.174e-2, "cwcl_motion": abs(h_lo - h_hi),
              "lambda_A": 1.030233, "lambda_B": 1.062037, "lambda_P": 1.031740,
              "cl_ratio": 1.031174, "alpha_motion_state": 1.3e-5})
print(f"conditional axiom minted: name={ca.name}")
print(f"  statement: {ca.statement}")
