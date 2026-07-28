"""Control: ground-branch beta-shape invariance on RAW rungs; then mint the
morphology findings as E-J-A engine objects (witnesses + invariance + refusal)."""
import importlib.util, sys
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad"
HF = SCRATCH + "/hunt_fields"
spec = importlib.util.spec_from_file_location(
    'pc', '/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py')
pc = importlib.util.module_from_spec(spec); sys.modules['pc'] = pc
spec.loader.exec_module(pc)
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12),
                       Nb=36, eps_b=1e-4, alpha=-0.344712)
x, b = S.x, S.b
Nx, Nb = len(x), len(b); n2 = Nx * Nb

def load(name):
    d = np.load(f"{HF}/{name}.npz")
    z = d['z']
    return dict(A=z[:n2].reshape(Nx, Nb), B=z[n2:2*n2].reshape(Nx, Nb),
                P=z[2*n2:3*n2].reshape(Nx, Nb), cl=z[3*n2], cw=z[3*n2+1],
                a=float(d['a']))

G0 = load('rung_00_a-0.344712'); G9 = load('rung_09_a-0.414493')
G10 = load('rung_10_a-0.422247'); C = load('find_half')

A_basis = np.stack([np.cos((2*m+1)*b) for m in range(5)], axis=1)
def c31(R, xs):
    i = int(np.argmin(np.abs(x - xs)))
    c, *_ = np.linalg.lstsq(A_basis, R['A'][i, :], rcond=None)
    return c[1] / c[0]

print("== CONTROL: raw-rung beta-shape a-invariance (c3/c1 of A at xi=1.0) ==")
vals = {}
for nm, R in (('rung00 a=-0.3447', G0), ('rung09 a=-0.4145', G9),
              ('rung10 a=-0.4222', G10), ('cand a=-0.4168', C)):
    v = c31(R, 1.0)
    vals[nm] = v
    print(f"  {nm:18s} c3/c1(xi=1.0) = {v:+.6f}")
spread = max(abs(vals['rung00 a=-0.3447']-vals['rung09 a=-0.4145']),
             abs(vals['rung09 a=-0.4145']-vals['rung10 a=-0.4222']))
gap = abs(vals['cand a=-0.4168'] - vals['rung09 a=-0.4145'])
print(f"  ground-branch spread over Da=0.078: {spread:.2e}; "
      f"candidate gap to branch: {gap:.4f}  ratio gap/spread = {gap/spread:.1f}")

# is the whole ground walk just a rescaling? relative field distance rung00->rung09
for f in ('A', 'B', 'P'):
    rel = np.linalg.norm(G9[f]-G0[f])/np.linalg.norm(G0[f])
    # best scalar fit
    s = float(np.vdot(G0[f], G9[f])/np.vdot(G0[f], G0[f]))
    rs = np.linalg.norm(G9[f]-s*G0[f])/np.linalg.norm(G9[f])
    print(f"  walk rung00->rung09 {f}: relL2={rel:.4f}, best-scalar s={s:.4f}, "
          f"resid after scaling={rs:.4f}")
# candidate vs scaled ground: can a scalar explain it?
for f in ('A', 'B'):
    s = float(np.vdot(G9[f], C[f])/np.vdot(G9[f], G9[f]))
    rs = np.linalg.norm(C[f]-s*G9[f])/np.linalg.norm(C[f])
    print(f"  cand vs s*rung09 {f}: best s={s:.4f}, resid after scaling={rs:.4f}")

print("\n== E-J-A ENGINE OBJECTS ==")
sys.path.insert(0, SCRATCH)
from eja_bridge import *

# Witness 1: same-a morphology divergence, ground branch vs candidate --
# peak location of near-wall A (internal, reference-free: two converged roots of
# the SAME system at the SAME frozen a disagree on where the amplitude lives).
w1 = mk_witness(
    "ground-branch near-wall A peak location at frozen a=-0.4168 (xi)", 0.444,
    "candidate-root near-wall A peak location at frozen a=-0.4168 (xi)", 1.383)
print("W1 peak-location witness:", w1)

# Witness 2: corner-panel beta-shape sign flip -- c3/c1 of A at xi=1.0.
w2 = mk_witness(
    "ground-branch c3/c1 of A at xi=1.0 (a-invariant to 2e-5 across walk)", -0.180489,
    "candidate-root c3/c1 of A at xi=1.0", +0.094043)
print("W2 cos3b sign-flip witness:", w2)

# Witness 3: outer decay direction at frozen a -- ground branch grows outward
# (+0.0216/xi), candidate decays (-0.0560/xi).
w3 = mk_witness(
    "ground-branch@a1 outer log-slope d(lnA)/dxi on [15,25]", +0.02155,
    "candidate outer log-slope d(lnA)/dxi on [15,25]", -0.05597)
print("W3 outer-decay witness:", w3)

# Invariance: the ground-branch beta shape (c3/c1 at xi=1.0) is invariant under
# the a-walk intervention (a: -0.3447 -> -0.4222, 10 rungs), motion <= measured
# spread; the candidate breaks it by ~0.27. Evidence: measured spread vs gap.
try:
    inv = mk_invariance(
        target="ground-branch beta shape c3/c1(A, xi=1.0)",
        intervention="frozen-alpha walk a=-0.3447 -> -0.4222 (Da=0.078)",
        motion=float(spread), scale=float(gap))
    print("INV beta-shape a-invariance:", inv)
    try:
        prom = promote_invariance(inv)
        print("PROMOTED:", prom)
    except Exception as e:
        print("promote_invariance REFUSED:", e)
except Exception as e:
    print("mk_invariance signature mismatch, trying alt:", e)
    try:
        inv = mk_invariance("ground-branch beta shape c3/c1(A, xi=1.0)",
                            "frozen-alpha walk Da=0.078", float(spread), float(gap))
        print("INV:", inv)
    except Exception as e2:
        print("mk_invariance failed:", e2)

# REFUSAL (negative control): the naive 'unstable branch carries one more radial
# extremum' COUNT fingerprint does NOT discriminate: at frozen a=-0.4168 the
# ground branch also shows 2 interior extrema on near-wall A (max@0.444 +
# shallow min@13.76) versus the candidate's 2 (min@0.076 + max@1.383).
# Count distance = 0 at instrument scale 1 extremum.
try:
    r = refuse(
        "extrema-COUNT fingerprint (ground 2 vs candidate 2 at same frozen a)",
        distance=0.0, scale=1.0)
    print("REFUSAL extrema-count:", r)
except Exception as e:
    print("refuse signature mismatch:", e)
    try:
        r = refuse("extrema-COUNT fingerprint: 2 vs 2 at frozen a=-0.4168 "
                   "-- count does not separate branches; only LOCATION does",
                   0.0, 1.0)
        print("REFUSAL:", r)
    except Exception as e2:
        print("refuse failed:", e2)
