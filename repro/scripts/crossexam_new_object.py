#!/usr/bin/env python3
"""Cross-examination of the 'new_object' advocate case.

Re-verifies from disk (measurement only, no Newton):
 1. Candidate residual on a from-scratch solver (advocate: 7.2e-13).
 2. Ground rung_00 control residual (advocate: 7.8e-5).
 3. eps-ladder (a, cl, cw, cw/cl) straight from files.
 4. branch1_deg16_40 vs branch1_eps1e-05: bit-identity claim.
 5. branch1_deg24_56: is it a CONVERGED root (residual ~1e-11) or a
    mid-run intermediate?  This adjudicates the advocate's only
    branch-side refinement-direction evidence.
"""
import importlib.util, sys, numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad"
HF = SCRATCH + "/hunt_fields"

spec = importlib.util.spec_from_file_location(
    'pc', '/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py')
pc = importlib.util.module_from_spec(spec)
sys.modules['pc'] = pc
spec.loader.exec_module(pc)

def load(name):
    d = np.load(f"{HF}/{name}", allow_pickle=True)
    return {k: d[k] for k in d.files}

def zinfo(d):
    z = d['z']
    cl, cw = float(z[-2]), float(z[-1])
    a = float(d['a']) if 'a' in d else None
    return a, cl, cw, cw / cl, z

print("=" * 78)
print("1) FILE INVENTORY: (a, cl, cw, cw/cl) straight from disk")
print("=" * 78)
names = ["find_half.npz", "rung_00_a-0.344712.npz",
         "branch1_eps1e-4.npz", "branch1_eps5e-05.npz",
         "branch1_eps3e-05.npz", "branch1_eps1e-05.npz",
         "branch1_deg16_40.npz", "branch1_deg24_56.npz"]
data = {}
for n in names:
    d = load(n)
    data[n] = d
    a, cl, cw, r, z = zinfo(d)
    extra = {k: (v.shape if hasattr(v, 'shape') and v.shape else float(v))
             for k, v in d.items() if k not in ('z',)}
    print(f"  {n:26s} len(z)={len(z):6d}  a={a!r:>22}  cl={cl:+.9f}  "
          f"cw={cw:+.9f}  cw/cl={r:+.9f}")
    print(f"      keys: {extra}")

print()
print("=" * 78)
print("2) BIT-IDENTITY CHECK: branch1_deg16_40 vs branch1_eps1e-05")
print("=" * 78)
d1, d2 = data["branch1_deg16_40.npz"], data["branch1_eps1e-05.npz"]
same_z = np.array_equal(d1['z'], d2['z'])
same_a = ('a' in d1 and 'a' in d2 and float(d1['a']) == float(d2['a']))
print(f"  z arrays bit-identical: {same_z}")
print(f"  a values identical:     {same_a}  "
      f"({float(d1['a'])!r} vs {float(d2['a'])!r})")
print(f"  max|dz| = {np.max(np.abs(d1['z']-d2['z'])):.3e}"
      if d1['z'].shape == d2['z'].shape else "  shapes differ")

print()
print("=" * 78)
print("3) RESIDUAL RE-EVALUATION (from-scratch solver, no Newton)")
print("=" * 78)

def resid(name, degs, eps_b, a=None, Nb=36):
    d = data[name]
    a_use = float(d['a']) if a is None else a
    S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=degs,
                           Nb=Nb, eps_b=eps_b, alpha=a_use)
    z = d['z']
    if len(z) != 3 * len(S.x) * len(S.b) + 2:
        return None, a_use, (len(z), 3 * len(S.x) * len(S.b) + 2)
    F = S.residual(z)
    return float(np.max(np.abs(F))), a_use, None

# 3a. candidate on its native grid
r, a_use, mism = resid("find_half.npz", (16, 40, 12), 1e-4)
print(f"  find_half     (16,40,12) eps=1e-4  a={a_use:+.7f}  ||F||_inf = {r:.3e}"
      if r is not None else f"  find_half  SIZE MISMATCH {mism}")

# 3b. ground control
r, a_use, mism = resid("rung_00_a-0.344712.npz", (16, 40, 12), 1e-4)
print(f"  rung_00       (16,40,12) eps=1e-4  a={a_use:+.7f}  ||F||_inf = {r:.3e}"
      if r is not None else f"  rung_00  SIZE MISMATCH {mism}")

# 3c. eps ladder, each at its own eps and its own a
for n, eps in [("branch1_eps1e-4.npz", 1e-4), ("branch1_eps5e-05.npz", 5e-5),
               ("branch1_eps3e-05.npz", 3e-5), ("branch1_eps1e-05.npz", 1e-5)]:
    r, a_use, mism = resid(n, (16, 40, 12), eps)
    print(f"  {n:22s} (16,40,12) eps={eps:.0e}  a={a_use:+.9f}  ||F||_inf = {r:.3e}"
          if r is not None else f"  {n}  SIZE MISMATCH {mism}")

# 3d. THE decisive check: deg24_56 -- converged rung or mid-run junk?
print()
print("  --- branch1_deg24_56: convergence adjudication ---")
for eps in (1e-4, 1e-5):
    r, a_use, mism = resid("branch1_deg24_56.npz", (24, 56, 12), eps)
    if r is not None:
        print(f"  deg(24,56,12) eps={eps:.0e}  a={a_use:+.9f}  ||F||_inf = {r:.3e}")
    else:
        print(f"  deg(24,56,12) eps={eps:.0e}  SIZE MISMATCH z={mism[0]} vs need {mism[1]}")

print()
print("=" * 78)
print("4) GAP ARITHMETIC RE-CHECK")
print("=" * 78)
ALPHA1 = -0.4168236
d = data["branch1_deg24_56.npz"]
a24, cl24, cw24, r24, _ = zinfo(d)
d16 = data["branch1_eps1e-05.npz"]
a16, cl16, cw16, r16, _ = zinfo(d16)
print(f"  deg16 cw/cl = {r16:+.9f}   deg24 cw/cl = {r24:+.9f}")
print(f"  refinement motion deg16->deg24: {r24-r16:+.6e} "
      f"(advocate: -3.817e-3, AWAY from alpha_1)")
print(f"  gap deg16 branch alpha (-0.42172919) to alpha_1: "
      f"{ALPHA1-(-0.42172919):+.6e}")
print(f"  |refinement motion| / ground-transferred prior 7e-4 = "
      f"{abs(r24-r16)/7e-4:.2f}x")
print(f"  gap / |branch-side measured refinement motion| = "
      f"{abs(ALPHA1-(-0.42172919))/abs(r24-r16):.2f}x  "
      f"(advocate's 7.0x used the GROUND prior)")
