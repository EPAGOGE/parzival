"""Adjudication computations for the freed-pin spec (comparing two derivations).

Sections:
  S1  npz audit: lengths, cl/cw/alpha, free-identity defect d_cl per seed
  S2  axis-column decision: does branch npz axis data == solver-constructed A0/B0
      axis data (i.e. is 're-seed axis pins from branch npz' a no-op)?
  S3  scaling-symmetry re-verification on the real residual + Euler check on the
      real jacobian (weights A:1 B:2 P:1 cl:1 cw:1)
  S4  corner-identity elimination (sympy record)
"""
import importlib.util
import pathlib
import sys

import numpy as np

BOUS = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
HF = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/"
                  "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad/hunt_fields")

spec = importlib.util.spec_from_file_location("pc", str(BOUS / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(spec)
sys.modules["pc"] = pc
spec.loader.exec_module(pc)

WX_REF, THXX_REF = pc.CornerRegSolver.WX_REF, pc.CornerRegSolver.THXX_REF
CL_FREE = 2.0 * THXX_REF / WX_REF

print("=== S1: npz audit ===")
print(f"free identity value 2*THXX_REF/WX_REF = {CL_FREE:.8f}")
seeds = {
    "rung_00 (ground)": ("rung_00_a-0.344712.npz", (16, 40, 12), 1e-4),
    "branch1_deg24_56": ("branch1_deg24_56.npz", (24, 56, 12), 1e-5),
    "branch1_deg28_64_18": ("branch1_deg28_64_18.npz", (28, 64, 18), 1e-5),
}
Z = {}
for name, (fn, degs, eps) in seeds.items():
    d = np.load(HF / fn)
    z = d["z"]
    a = float(d["a"]) if "a" in d else None
    Nx = sum(p + 1 for p in degs)
    n2 = Nx * 36
    cl, cw = float(z[-2]), float(z[-1])
    Z[name] = (z, degs, eps, n2)
    print(f"{name}: len={z.size} (pred {3*n2+2}, match={z.size == 3*n2+2}) "
          f"keys={list(d.keys())}")
    print(f"    cl={cl:.6f} cw={cw:.6f} alpha=cw/cl={cw/cl:+.8f} a_key={a}")
    print(f"    d_cl = cl - {CL_FREE:.6f} = {cl - CL_FREE:+.6e}   "
          f"rel = {(cl - CL_FREE)/CL_FREE:+.6e}")
    print(f"    implied dTHXX' if identity absorbed there: WX*d_cl/2 = "
          f"{WX_REF*(cl-CL_FREE)/2.0:+.6e}")

print()
print("=== S2: axis-column decision (is branch axis data == constructed A0/B0?) ===")
for name in seeds:
    z, degs, eps, n2 = Z[name]
    S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=degs, Nb=36,
                           eps_b=eps)
    A = z[:n2].reshape(S.Nx, S.Nb)
    B = z[n2:2 * n2].reshape(S.Nx, S.Nb)
    print(f"{name}: b[0]={S.b[0]:.6e} b[-1]={S.b[-1]:.6e} (axis col j=Nb-1)")
    dA = np.max(np.abs(A[:, -1] - S.A0[:, -1]))
    dB = np.max(np.abs(B[:, -1] - S.B0[:, -1]))
    sA = np.max(np.abs(S.A0[:, -1])); sB = np.max(np.abs(S.B0[:, -1]))
    print(f"    axis col: max|A_npz - A0| = {dA:.3e} (scale {sA:.3e}), "
          f"max|B_npz - B0| = {dB:.3e} (scale {sB:.3e})")
    # corner circle for reference
    dAc = np.max(np.abs(A[0, :] - WX_REF * np.cos(S.b)))
    dBc = np.max(np.abs(B[0, :] - 0.5 * THXX_REF * np.cos(S.b) ** 2))
    print(f"    corner circle: max|A(0,:)-WXcos b| = {dAc:.3e}, "
          f"max|B(0,:)-THXX/2 cos^2 b| = {dBc:.3e}")
    # gauge functionals evaluated on the npz fields
    xw = S.x
    g1f = float((S.Dx[0, :] @ (xw * A[:, 0]))[0])
    vt = np.zeros(S.Nx); vt[1:] = xw[1:] * B[1:, 0] / S.G1c[1:]
    g2f = float((S.Dx[0, :] @ vt)[0])
    print(f"    gauge functionals on npz fields: Dx(xiA)(0,0)={g1f:.8f} "
          f"(WX_REF {WX_REF:.8f}), 2*Dx(xiB/G1)(0,0)={2*g2f:.8f} "
          f"(THXX_REF {THXX_REF:.8f})")

print()
print("=== S3: symmetry re-verification (weights A:1 B:2 P:1 cl:1 cw:1) ===")
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(8, 10, 6), Nb=12,
                       eps_b=1e-3)
n2 = S.Nx * S.Nb
rng = np.random.default_rng(7)
A = S.A0 * (1.0 + 0.3 * rng.standard_normal(S.A0.shape))
B = S.B0 * (1.0 + 0.3 * rng.standard_normal(S.B0.shape))
Pf = S.P0 * (1.0 + 0.3 * rng.standard_normal(S.P0.shape)) + 0.05 * rng.standard_normal(S.P0.shape)
cl, cw = 3.1, -1.05
z = S.pack(A, B, Pf, cl, cw)
s = 1.3
zs = S.pack(s * A, s ** 2 * B, s * Pf, s * cl, s * cw)
F, Fs = S.residual(z), S.residual(zs)

pin = set(int(r) for r in S.rT_pin)
c0 = set(int(r) for r in S.rT_c0)
corner = set(range(S.Nb))          # rid(0,j)
axis_only = pin - corner
covA_deg, covB_deg = 2.0, 3.0
err = {}
# A block
eA_int = max(abs(Fs[r] - s**covA_deg * F[r]) / (abs(F[r]) + 1e-30)
             for r in range(n2) if r not in pin and r not in c0)
eA_c0 = max(abs(Fs[r] - s * F[r]) for r in c0) if c0 else 0.0
eA_pin = max(abs((Fs[r] - s * F[r]) - (s - 1) * S.A0.ravel()[r]) for r in pin)
# B block
eB_int = max(abs(Fs[n2+r] - s**covB_deg * F[n2+r]) / (abs(F[n2+r]) + 1e-30)
             for r in range(n2) if r not in pin and r not in c0)
eB_c0 = max(abs(Fs[n2+r] - s**2 * F[n2+r]) for r in c0) if c0 else 0.0
eB_pin = max(abs((Fs[n2+r] - s**2 * F[n2+r]) - (s**2 - 1) * S.B0.ravel()[r])
             for r in pin)
# P block: every row weight 1
eP = max(abs(Fs[2*n2+r] - s * F[2*n2+r]) / (abs(F[2*n2+r]) + 1e-30)
         for r in range(n2))
# gauge rows
g1_brk = abs((Fs[-2] - s * F[-2]) - (s - 1) * WX_REF)
g2_brk = abs((Fs[-1] - s**2 * F[-1]) - (s**2 - 1) * 0.5 * THXX_REF)
print(f"A interior rows covariant deg2, max rel err   = {eA_int:.3e}")
print(f"A C0 rows covariant deg1, max abs err          = {eA_c0:.3e}")
print(f"A pin rows broken by exactly (s-1)*A0, err     = {eA_pin:.3e}")
print(f"B interior rows covariant deg3, max rel err    = {eB_int:.3e}")
print(f"B C0 rows covariant deg2, max abs err          = {eB_c0:.3e}")
print(f"B pin rows broken by exactly (s^2-1)*B0, err   = {eB_pin:.3e}")
print(f"P ALL rows covariant deg1, max rel err         = {eP:.3e}")
print(f"g1 broken by exactly (s-1)*WX_REF, err         = {g1_brk:.3e}  "
      f"-> g1 functional weight 1")
print(f"g2 broken by exactly (s^2-1)*THXX_REF/2, err   = {g2_brk:.3e}  "
      f"-> g2 functional weight 2")

# Euler check on the analytic jacobian: v = (A, 2B, P, cl, cw)*z pattern
J = S.jacobian(z)
v = np.concatenate([A.ravel(), 2.0 * B.ravel(), Pf.ravel(), [cl, cw]])
Jv = np.asarray(J @ v).ravel()
d_expected = np.empty(3 * n2 + 2)
for r in range(n2):
    d_expected[r] = 2.0 if (r not in pin and r not in c0) else 1.0
for r in range(n2):
    d_expected[n2 + r] = 3.0 if (r not in pin and r not in c0) else 2.0
d_expected[2*n2:3*n2] = 1.0
d_expected[-2:] = [1.0, 2.0]
# pin rows are inhomogeneous: R = field - const; Jv = field = F + const
resid = Jv - d_expected * F
for r in pin:
    resid[r] -= (1.0 - 2.0) * F[r] + (S.A0.ravel()[r] * 0)  # recompute exactly below
# redo cleanly: expected Jv per row class
expJ = d_expected * F
for r in pin:
    expJ[r] = F[r] + S.A0.ravel()[r]            # A pin: Jv = A = (A-A0)+A0
    expJ[n2 + r] = 2.0 * (F[n2 + r] + S.B0.ravel()[r])  # B pin: Jv = 2B
for r in c0:
    expJ[r] = F[r]                               # weight-1 linear homogeneous
    expJ[n2 + r] = 2.0 * F[n2 + r]
expJ[-2] = F[-2] + WX_REF                        # g1: Jv = functional = g1+REF
expJ[-1] = 2.0 * (F[-1] + 0.5 * THXX_REF)        # g2: Jv = 2*functional
eul = np.max(np.abs(Jv - expJ)) / max(1.0, np.max(np.abs(Jv)))
print(f"Euler check J.v vs degree-weighted F (all rows): max rel err = {eul:.3e}")

print()
print("=== S4: corner identity elimination (sympy) ===")
import sympy as sy
WX, THXX, cP, clv, cwv = sy.symbols("WX THXX c_P cl cw", positive=False)
eq1 = 2 * cP + cwv - clv + THXX / WX           # RO'(0,b) / (WX cos b)
eq2 = 4 * cP + 2 * cwv - clv                   # RB'(0,b) / (THXX cos^2 b)
sol = sy.solve([eq1, eq2], [cP, clv], dict=True)[0]
print(f"eliminating c_P: cl = {sy.simplify(sol[clv])}   "
      f"c_P = {sy.simplify(sol[cP])}")
