"""Cross-examination of the 'new_object' hypothesis. Residual EVALUATIONS only,
no Newton solves. Verifies/attacks each advocate evidence item from disk."""
import importlib.util, sys, os
import numpy as np

SP = "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad"
HF = os.path.join(SP, "hunt_fields")

spec = importlib.util.spec_from_file_location(
    "pc", "/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py")
pc = importlib.util.module_from_spec(spec)
sys.modules["pc"] = pc
spec.loader.exec_module(pc)

A1 = -0.4168236
A2 = -0.4439811
A0REF = -0.34240009

def load(name):
    d = np.load(os.path.join(HF, name), allow_pickle=True)
    return {k: d[k] for k in d.files}

def clcw(z):
    return float(z[-2]), float(z[-1])

print("=" * 72)
print("[1] FILE INVENTORY: stored (a, cl, cw, cw/cl) per branch file")
files = ["find_half.npz", "branch1_eps1e-4.npz", "branch1_eps5e-05.npz",
         "branch1_eps3e-05.npz", "branch1_eps1e-05.npz",
         "branch1_deg16_40.npz", "branch1_deg24_56.npz",
         "rung_00_a-0.344712.npz"]
store = {}
for f in files:
    d = load(f)
    z = d["z"]
    cl, cw = clcw(z)
    a = float(d["a"]) if "a" in d else np.nan
    store[f] = d
    print(f"  {f:26s} n={z.size:6d} a={a:+.9f} cl={cl:+.9f} "
          f"cw={cw:+.9f} cw/cl={cw/cl:+.9f}")

print("=" * 72)
print("[2] BIT-IDENTICAL CHECK: branch1_deg16_40 vs branch1_eps1e-05")
za = store["branch1_deg16_40.npz"]["z"]; zb = store["branch1_eps1e-05.npz"]["z"]
same_shape = za.shape == zb.shape
ident = same_shape and bool(np.array_equal(za, zb))
maxdiff = float(np.max(np.abs(za - zb))) if same_shape else np.nan
aa = float(store["branch1_deg16_40.npz"]["a"]); ab = float(store["branch1_eps1e-05.npz"]["a"])
print(f"  shapes equal: {same_shape}; z arrays identical: {ident}; "
      f"max|dz|={maxdiff:.3e}; a identical: {aa == ab}")

print("=" * 72)
print("[3] RESIDUAL RE-EVALUATION (no solves)")
# 3a. candidate on its native grid, frozen a = alpha_1
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12),
                       Nb=36, eps_b=1e-4, alpha=A1)
z = store["find_half.npz"]["z"]
r = S.residual(z)
print(f"  find_half @ (16,40,12)/eps1e-4/a=alpha_1: ||F||_inf = "
      f"{np.max(np.abs(r)):.3e}  ||F||_2 = {np.linalg.norm(r):.3e}")

# 3b. ground control at its own alpha
d0 = store["rung_00_a-0.344712.npz"]
a0 = float(d0["a"])
S0 = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12),
                        Nb=36, eps_b=1e-4, alpha=a0)
r0 = S0.residual(d0["z"])
print(f"  rung_00 @ its own a={a0:+.6f}:            ||F||_inf = "
      f"{np.max(np.abs(r0)):.3e}")

# 3c. the polished branch rung at eps=1e-4 (branch1_eps1e-4) at its stored a
d4 = store["branch1_eps1e-4.npz"]
a4 = float(d4["a"])
S4 = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12),
                        Nb=36, eps_b=1e-4, alpha=a4)
r4 = S4.residual(d4["z"])
print(f"  branch1_eps1e-4 @ stored a={a4:+.9f}: ||F||_inf = "
      f"{np.max(np.abs(r4)):.3e}")

print("=" * 72)
print("[4] THE UNMEASURED POINT: is branch1_deg24_56 a CONVERGED root?")
d24 = store["branch1_deg24_56.npz"]
z24 = d24["z"]; a24 = float(d24["a"])
n24 = z24.size
Nx24 = (n24 - 2) // (3 * 36)
print(f"  n={n24} -> Nx={Nx24} (expect 95 for degs=(24,56,12), Nb=36)")
for eps in (1e-4, 1e-5):
    S24 = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(24, 56, 12),
                             Nb=36, eps_b=eps, alpha=a24)
    r24 = S24.residual(z24)
    print(f"  eps_b={eps:.0e}: ||F||_inf = {np.max(np.abs(r24)):.3e}   "
          f"||F||_2 = {np.linalg.norm(r24):.3e}")
cl24, cw24 = clcw(z24)
h24 = cw24 / cl24
print(f"  stored a={a24:+.9f}  cw/cl={h24:+.9f}  (gap cw/cl - a = {h24 - a24:+.3e})")
print(f"  |cw/cl - alpha_1| = {abs(h24 - A1):.4e}   |cw/cl - (-0.42174207)| = "
      f"{abs(h24 - (-0.42174207)):.4e}")

print("=" * 72)
print("[5] ARITHMETIC OF THE GAP AND THE ERROR BUDGET, re-derived")
alpha_star = -0.42174207
gap = alpha_star - A1
print(f"  gap to alpha_1: {gap:+.6e}  ({100*gap/abs(A1):+.3f}%)")
eps_ladder = []
for f in ["branch1_eps1e-4.npz", "branch1_eps5e-05.npz",
          "branch1_eps3e-05.npz", "branch1_eps1e-05.npz"]:
    cl, cw = clcw(store[f]["z"]); eps_ladder.append(cw / cl)
print(f"  eps ladder cw/cl: {['%+.9f' % v for v in eps_ladder]}")
print(f"  total eps motion: {eps_ladder[-1] - eps_ladder[0]:+.4e}")
# the measured branch-side refinement step
step_deg = h24 - eps_ladder[-1]
print(f"  deg (16,40)->(24,56) step in cw/cl: {step_deg:+.4e} "
      f"(advocate: -3.817e-3)")
print(f"  ratio |gap|/|deg step| = {abs(gap)/abs(step_deg):.2f}  "
      f"<-- the REAL error-budget ratio if the deg step is a genuine converged rung")
print(f"  ratio |gap|/7e-4 (ground-transferred prior) = {abs(gap)/7e-4:.2f}")
# family placement
frac = (alpha_star - A1) / (A2 - A1)
print(f"  placement into alpha_1->alpha_2 gap: {100*frac:.1f}%")
shift = alpha_star - A0REF
g1 = A1 - A0REF
print(f"  (ours - alpha_0)/g1 = {shift/g1:.4f}")
