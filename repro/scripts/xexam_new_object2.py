"""Cross-exam pass 2: (a) confirm the two-alpha pin structure explains all
large re-eval residuals (rescuing convergence of every branch file), (b) bound
the pin-alpha sensitivity from the ladder's accidental natural experiment,
(c) fingerprint the (24,56,12) root: is it THE candidate object or a third root?
Residual evaluations + linear algebra only."""
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
EDGES = (0.0, 2.0, 15.0, 25.0)

def load(name):
    d = np.load(os.path.join(HF, name), allow_pickle=True)
    return {k: d[k] for k in d.files}

def rms(f): return float(np.linalg.norm(f) / np.sqrt(f.size))

print("[A] TWO-ALPHA PIN STRUCTURE: re-evaluate under the SOLVER THAT MADE the file")
d = load("branch1_eps1e-4.npz")
S = pc.CornerRegSolver(edges=EDGES, degs=(16, 40, 12), Nb=36, eps_b=1e-4, alpha=A1)
S.set_alpha(float(d["a"]))
r = S.residual(d["z"])
print(f"  branch1_eps1e-4: constructed@A1 + set_alpha(stored a):"
      f"  ||F||_inf={np.max(np.abs(r)):.3e}  RMS={rms(r):.3e}")
S2 = pc.CornerRegSolver(edges=EDGES, degs=(16, 40, 12), Nb=36, eps_b=1e-4,
                        alpha=float(d["a"]))
r2 = S2.residual(d["z"])
print(f"  branch1_eps1e-4: constructed@stored a (fresh pins):        "
      f"  ||F||_inf={np.max(np.abs(r2)):.3e}  RMS={rms(r2):.3e}")

d16 = load("branch1_deg16_40.npz")
a_prev = float(d16["a"])          # what branch1_res.py passed as construction alpha
d24 = load("branch1_deg24_56.npz")
S24 = pc.CornerRegSolver(edges=EDGES, degs=(24, 56, 12), Nb=36, eps_b=1e-5,
                         alpha=a_prev)
S24.set_alpha(float(d24["a"]))
r24 = S24.residual(d24["z"])
print(f"  branch1_deg24_56: constructed@a_prev={a_prev:+.8f} + set_alpha(stored):"
      f"  ||F||_inf={np.max(np.abs(r24)):.3e}  RMS={rms(r24):.3e}")

# scale of the pin-data shift between construction alphas
Sa = pc.CornerRegSolver(edges=EDGES, degs=(16, 40, 12), Nb=36, eps_b=1e-4, alpha=A1)
Sb = pc.CornerRegSolver(edges=EDGES, degs=(16, 40, 12), Nb=36, eps_b=1e-4,
                        alpha=float(d["a"]))
dA0 = np.max(np.abs(Sa.A0 - Sb.A0)); dB0 = np.max(np.abs(Sa.B0 - Sb.B0))
print(f"  pin-seed shift A0/B0 between construction alphas (da=4.9e-3): "
      f"max|dA0|={dA0:.3e} max|dB0|={dB0:.3e}")

print()
print("[B] NATURAL EXPERIMENT: ladder step 1 carried a 4.9e-3 pin-alpha jump")
ladder = [("branch1_eps1e-4.npz", 1e-4), ("branch1_eps5e-05.npz", 5e-5),
          ("branch1_eps3e-05.npz", 2.5e-5), ("branch1_eps1e-05.npz", 1e-5)]
al = [float(load(f)["a"]) for f, _ in ladder]
steps = np.diff(al)
print(f"  ladder alphas: {['%+.9f' % v for v in al]}")
print(f"  steps: {['%+.3e' % v for v in steps]}")
print(f"  step1 (eps 1e-4->5e-5) carried pin-alpha jump A1->astar = 4.9e-3;"
      f" steps 2,3 pin-consistent (jumps {abs(al[1]-al[0]):.1e}, {abs(al[2]-al[1]):.1e})")
print(f"  step1 motion {steps[0]:+.3e} is NOT anomalous vs steps 2,3 "
      f"({steps[1]:+.3e}, {steps[2]:+.3e}) -> |dh/d(a_pin)| <~ "
      f"{abs(steps[0]) / 4.918e-3:.1e} (upper bound, assuming no cancellation)")
print(f"  filename check: f'{2.5e-5:.0e}' = '{2.5e-5:.0e}' "
      f"(the 3e-05 file is eps=2.5e-5)")
da_repro = abs(float(d16['a']) - float(load('branch1_eps1e-05.npz')['a']))
print(f"  deg16_40 rung vs eps1e-05 rung: |da| = {da_repro:.3e} "
      f"(warm re-secant reproducibility)")

print()
print("[C] FINGERPRINT THE (24,56,12) ROOT: same object or a third root?")
z24 = d24["z"]
Nx24, Nb = S24.Nx, S24.Nb
A24 = z24[:Nx24 * Nb].reshape(Nx24, Nb)
B24 = z24[Nx24 * Nb:2 * Nx24 * Nb].reshape(Nx24, Nb)
x24, b24 = S24.x, S24.b

def c31(A, x, b, xi_target):
    i = int(np.argmin(np.abs(x - xi_target)))
    v = A[i, :]
    M = np.column_stack([np.cos((2 * m + 1) * b) for m in range(5)])
    c, *_ = np.linalg.lstsq(M, v, rcond=None)
    return x[i], c[1] / c[0]

for xt in (0.7, 1.0, 1.4):
    xi, ratio = c31(A24, x24, b24, xt)
    print(f"  (24,56) c3/c1 at xi={xi:.3f}: {ratio:+.6f}")
# comparators on (16,40): candidate +0.0940 at xi=1.0, ground -0.1805
# amplitude anatomy
j_axis = None
iA = np.unravel_index(np.argmax(np.abs(A24)), A24.shape)
iB = np.unravel_index(np.argmax(np.abs(B24)), B24.shape)
print(f"  (24,56) max|A|={np.abs(A24)[iA]:.4f} @ xi={x24[iA[0]]:.3f}   "
      f"max|B|={np.abs(B24)[iB]:.4f} @ xi={x24[iB[0]]:.3f}")
print(f"  comparators (16,40): candidate max|A|=3.5679@1.383 max|B|=4.5057@1.556")
print(f"                       ground    max|A|=1.4344@0.293 max|B|=1.0749@0.444")
# corner dip below pinned profile? candidate signature: A dips to ~0.64 near xi~0.08
b_mid = np.argmin(np.abs(b24))     # not meaningful; use beta station nearest 0
jj = int(np.argmin(np.abs(b24 - b24[0])))  # first station (near b=eps)
# use the same test as morphology: min of A/(WX cos b) profile near corner, station nearest b=0.35
jsta = int(np.argmin(np.abs(b24 - 0.35)))
prof = A24[:8, jsta]
print(f"  (24,56) near-corner A profile (beta={b24[jsta]:.3f}), first 8 radial nodes:")
print("   xi:", " ".join(f"{v:7.4f}" for v in x24[:8]))
print("    A:", " ".join(f"{v:7.4f}" for v in prof))
print(f"  corner value A(0)={A24[0, jsta]:.4f} (pinned WX cos b)")
# gap of stored a to alpha_1 and to candidate coarse alpha
a24v = float(d24["a"])
print(f"  (24,56) alpha={a24v:+.9f}: gap to alpha_1 {a24v - A1:+.3e}; "
      f"placement in a1->a2 gap: {100 * (a24v - A1) / (-0.4439811 - A1):.1f}%")
