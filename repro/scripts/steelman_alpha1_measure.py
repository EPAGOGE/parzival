"""Steelman is_alpha1: cheap measurements only (function evals, no Newton).

1. Convergence status of branch1_deg24_56.npz (residual eval at eps 1e-5 and 1e-4)
   -- is the 'moved AWAY from alpha_1' datum a converged root or an intermediate?
2. Corner-panel Chebyshev tail fractions: candidate vs ground at (16,40,12)
   -- is the candidate spectrally resolved on the corner panel where alpha is read?
3. Grid geometry: nodes inside the measured corner layer (width 0.05-0.1) at
   deg 16 vs deg 24; are the candidate's feature locations exact grid nodes?
4. EJA minting: witnesses, one refusal (negative control), conditional axiom.
"""
import importlib.util, sys
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad"
sys.path.insert(0, SCRATCH)

spec = importlib.util.spec_from_file_location(
    "pc", "/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py")
pc = importlib.util.module_from_spec(spec)
sys.modules["pc"] = pc
spec.loader.exec_module(pc)

HF = SCRATCH + "/hunt_fields/"

def load(name):
    d = np.load(HF + name)
    return {k: d[k] for k in d.files}

d2456 = load("branch1_deg24_56.npz")
d1e5  = load("branch1_eps1e-05.npz")
d1e4  = load("branch1_eps1e-4.npz")
g00   = load("rung_00_a-0.344712.npz")

print("== keys ==")
for nm, d in [("deg24_56", d2456), ("eps1e-05", d1e5), ("eps1e-4", d1e4), ("rung_00", g00)]:
    print(f"  {nm}: {sorted(d.keys())}  a={float(d['a']):+.9f}" if 'a' in d else f"  {nm}: {sorted(d.keys())}")

# ---------- 1. residual eval of the deg24_56 file ----------
z2456 = d2456["z"]; a2456 = float(d2456["a"])
print("\n== 1. branch1_deg24_56 convergence status (residual evals only) ==")
for eps in (1e-5, 1e-4):
    S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(24, 56, 12),
                           Nb=36, eps_b=eps, alpha=a2456)
    F = S.residual(z2456)
    print(f"  eps_b={eps:g}: ||F||_inf={np.max(np.abs(F)):.3e}  ||F||_2={np.linalg.norm(F):.3e}"
          f"  (Nx={S.Nx}, expected z len {3*S.Nx*S.Nb+2}, got {len(z2456)})")
A24, B24, P24, cl24, cw24 = None, None, None, float(z2456[-2]), float(z2456[-1])
print(f"  stored cl={cl24:+.9f} cw={cw24:+.9f} cw/cl={cw24/cl24:+.9f}  a_frozen={a2456:+.9f}")

# control: the (16,40,12) polished rung at eps=1e-5, same style
z1e5 = d1e5["z"]; a1e5 = float(d1e5["a"])
S165 = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12),
                          Nb=36, eps_b=1e-5, alpha=a1e5)
F = S165.residual(z1e5)
print(f"  CONTROL eps1e-05 @(16,40,12): ||F||_inf={np.max(np.abs(F)):.3e}  ||F||_2={np.linalg.norm(F):.3e}")

# ---------- 2. corner-panel Chebyshev tail fractions ----------
print("\n== 2. corner-panel spectral tails (deg 16 corner panel, nodes 0..16) ==")
S164 = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12),
                          Nb=36, eps_b=1e-4, alpha=float(d1e4["a"]))
n_c = S164.sizes[0]              # corner-panel node count (=17)
xc = S164.x[:n_c]                # corner nodes on [0,2]
t = 2.0 * xc / 2.0 - 1.0         # map [0,2] -> [-1,1]

def tails(z, S, label, ncp):
    n2 = S.Nx * S.Nb
    out = {}
    for fname, sl in (("A", slice(0, n2)), ("B", slice(n2, 2 * n2))):
        Ffield = z[sl].reshape(S.Nx, S.Nb)
        xcp = S.x[:ncp]
        tt = 2.0 * xcp / 2.0 - 1.0
        fr = []
        for j in range(S.Nb):
            c = np.polynomial.chebyshev.chebfit(tt, Ffield[:ncp, j], ncp - 1)
            fr.append(np.max(np.abs(c[-3:])) / np.max(np.abs(c)))
        fr = np.array(fr)
        out[fname] = (float(np.sqrt(np.mean(fr ** 2))), float(fr.max()))
        print(f"  {label} {fname}: tail(last3)/max rms={out[fname][0]:.3e}  max={out[fname][1]:.3e}")
    return out

t_g  = tails(g00["z"],  S164, "ground rung_00  ", n_c)
t_c  = tails(d1e4["z"], S164, "candidate eps1e-4", n_c)
S245 = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(24, 56, 12),
                          Nb=36, eps_b=1e-5, alpha=a2456)
t_24 = tails(z2456, S245, "candidate deg24  ", S245.sizes[0])

# ---------- 3. grid geometry vs the measured corner layer ----------
print("\n== 3. corner-layer grid coverage ==")
for deg in (16, 24):
    S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(deg, 40, 12),
                           Nb=36, eps_b=1e-4, alpha=float(d1e4["a"]))
    nodes = S.x[:S.sizes[0]]
    inlayer = nodes[(nodes > 0) & (nodes <= 0.1)]
    print(f"  corner deg {deg}: interior nodes with xi<=0.1 (layer width): {len(inlayer)}  -> {np.round(inlayer, 5)}")
nodes16 = S164.x[:n_c]
for feat, val in (("candidate corner dip", 0.076), ("ground A peak", 0.444),
                  ("candidate A peak", 1.383), ("candidate B peak", 1.556)):
    k = int(np.argmin(np.abs(nodes16 - val)))
    print(f"  {feat} at xi={val}: nearest deg16 node x[{k}]={nodes16[k]:.6f} (offset {abs(nodes16[k]-val):.2e})")

# ---------- 4. EJA ----------
print("\n== 4. EJA objects ==")
from eja_bridge import (mk_witness, refuse, conditional_axiom, mk_invariance)

# measured branch resolution motion (deg16_40 -> deg24_56 file): |cw/cl| change
h16 = -0.421729189          # branch cw/cl at (16,40,12), eps 1e-5 (state doc)
h24 = cw24 / cl24
dres_branch = abs(h24 - h16)
w1 = mk_witness("ground_transfer_prior_deg_sensitivity", 7e-4,
                "branch_measured_deg_motion", dres_branch,
                scenario={"axis": "deg (16,40,12)->(24,56,12)"},
                tags=("resolution", "transfer"))
print(f"  W1 transfer-prior vs branch measured deg motion: {7e-4:.1e} vs {dres_branch:.3e}"
      f"  divergence={w1.divergence:.4f}")

r1 = refuse("branch_res_sensitivity == ground_res_sensitivity(7e-4)",
            distance=dres_branch - 7e-4, scale=7e-4,
            why="the branch's own first resolution datum moved cw/cl "
                f"{dres_branch:.2e}, {dres_branch/7e-4:.1f}x the ground prior; the "
                "transfer invariance the <=20%-odds estimate rests on is refuted "
                "by direct measurement")
print(f"  REFUSAL r1: {r1['refused']}  distance={r1['distance']:.3e} scale={r1['scale']:.1e} ratio={r1['ratio']:.1f}x")

w2 = mk_witness("corner_layer_dAdxi_deg16", 25.01, "corner_layer_dAdxi_deg24", 33.11,
                scenario={"quantity": "rms dA/dxi(0,:)"}, tags=("unconverged", "corner"))
print(f"  W2 corner-layer under refinement 25.01 -> 33.11: divergence={w2.divergence:.4f} (NOT converged)")

w3 = mk_witness("ground_cornerA_tail", t_g["A"][0], "candidate_cornerA_tail", t_c["A"][0],
                scenario={"metric": "rms |c[-3:]|/max|c|, corner panel, A"},
                tags=("spectral", "resolution"))
print(f"  W3 corner-panel A spectral tail ground {t_g['A'][0]:.3e} vs candidate {t_c['A'][0]:.3e}"
      f": divergence={w3.divergence:.4f}")

ax = conditional_axiom(
    "candidate_root_is_alpha1_biased_by_coarse_grid",
    "The converged root at frozen a=-0.4168236 on (16,40,12)/Nb36 is the discrete "
    "image of DeepMind's alpha_1; its self-consistent alpha -0.42173 carries a "
    "-4.9e-3 discretization bias sourced in the resolution-unconverged corner "
    "boundary layer (dA/dxi(0) rms 25->33 under refinement) that ground lacks",
    domain="(16,40,12)/Nb36/edges(0,2,15,25), eps_b in [1e-5,1e-4], Chen-Hou-lineage "
           "seeds; branch deg-axis measured once (deg24_56 file, convergence status "
           "measured above)",
    residual="the deg24_56 cw/cl moved AWAY from alpha_1 (-0.42555); reconciled only "
             "if that file is an unconverged intermediate or the deg-convergence is "
             "non-monotonic while the layer resolves",
    falsifier="interpolation-seeded Newton on (24,56,12) (radial per-beta-column "
              "barycentric transfer of branch1_eps1e-05 fields, panel-by-panel) "
              "converging to ||F||<=1e-11 with secant-polished self-consistent "
              "alpha staying within 1e-3 of -0.42173 or moving away from -0.4168236",
    evidence={"gap_to_alpha1": -4.906e-3, "branch_deg_motion": dres_branch,
              "ground_deg_prior": 7e-4, "layer_growth_ratio": 33.11 / 25.01})
print(f"  CONDITIONAL AXIOM minted: {ax.name}")
print(f"    statement: {ax.statement[:120]}...")

inv_try = mk_invariance("branch_alpha_ignores_deg_axis", worst_effect=dres_branch, eps=1e-3)
print(f"  negative control 2: mk_invariance(branch_alpha_ignores_deg_axis, worst={dres_branch:.2e},"
      f" eps=1e-3) -> promotable={inv_try.promotable} (correctly NOT promotable; the deg axis is live)")
