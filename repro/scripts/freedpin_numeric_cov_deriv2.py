"""Numeric covariance of the DISCRETE residual under T_s, per row class.

T_s: (A,B,P,cl,cw) -> (sA, s^2 B, sP, s cl, s cw).  Predicted: interior ro rows
scale s^2, interior rb rows s^3, all rp rows s^1 (every special rp row is
homogeneous-linear in P), C0 duplicate rows s^1 (ro) / s^2 (rb).  Broken rows:
transport pins (corner+axis, affine in seed data) and the two gauge rows.
"""
import importlib.util
import pathlib
import sys

import numpy as np

HERE = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
spec = importlib.util.spec_from_file_location("pc", str(HERE / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(spec)
sys.modules["pc"] = pc
spec.loader.exec_module(pc)

S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(8, 10, 6), Nb=12,
                       eps_b=1e-3)
n2 = S.Nx * S.Nb
cl0, cw0 = float(S.P["cl"]), float(S.P["cw"])
z0 = S.pack(S.A0, S.B0, S.P0, cl0, cw0)
F0 = S.residual(z0)
s = 1.3
zs = S.pack(s * S.A0, s**2 * S.B0, s * S.P0, s * cl0, s * cw0)
Fs = S.residual(zs)

pin = set(int(r) for r in S.rT_pin)
c0 = set(int(r) for r in S.rT_c0)
interior = np.array(sorted(set(range(n2)) - pin - c0), dtype=int)
c0a = np.array(sorted(c0), dtype=int)


def dev(rows, block, k):
    a = F0[block * n2 + rows] if block < 3 else None
    f0 = F0[block * n2 + rows]
    fs = Fs[block * n2 + rows]
    scale = np.linalg.norm(f0)
    return np.max(np.abs(fs - s**k * f0)) / scale, scale


classes = [
    ("ro interior  (k=2)", interior, 0, 2),
    ("ro C0 dup    (k=1)", c0a, 0, 1),
    ("rb interior  (k=3)", interior, 1, 3),
    ("rb C0 dup    (k=2)", c0a, 1, 2),
    ("rp ALL rows  (k=1)", np.arange(n2), 2, 1),
]
print(f"s = {s};  grid degs=(8,10,6) Nb=12 eps_b=1e-3;  n2={n2}")
for name, rows, blk, k in classes:
    d, sc = dev(rows, blk, k)
    print(f"  {name}: max|Fs - s^k F0| / ||F0||_class = {d:.3e}   (||F0||_class={sc:.3e})")

# broken rows: pins and gauges -- measure the breaking magnitude
pin_rows = np.array(sorted(pin), dtype=int)
bro = np.max(np.abs(Fs[pin_rows] - s**2 * F0[pin_rows]))
brb = np.max(np.abs(Fs[n2 + pin_rows] - s**3 * F0[n2 + pin_rows]))
# corner pins vs axis pins separately
corner_rows = np.array([j for j in range(S.Nb)], dtype=int)
axis_rows = np.array(sorted(pin - set(corner_rows.tolist())), dtype=int)
bro_c = np.max(np.abs(Fs[corner_rows] - s**2 * F0[corner_rows]))
bro_a = np.max(np.abs(Fs[axis_rows] - s**2 * F0[axis_rows]))
brb_c = np.max(np.abs(Fs[n2 + corner_rows] - s**3 * F0[n2 + corner_rows]))
brb_a = np.max(np.abs(Fs[n2 + axis_rows] - s**3 * F0[n2 + axis_rows]))
gg = np.abs(Fs[-2:] - s * F0[-2:])
print(f"  BROKEN ro pins: corner {bro_c:.3e}  axis {bro_a:.3e}")
print(f"  BROKEN rb pins: corner {brb_c:.3e}  axis {brb_a:.3e}")
print(f"  BROKEN gauge rows |Fs - s F0| = g1 {gg[0]:.3e}  g2 {gg[1]:.3e}")
print(f"  (corner-pin breaking prediction (s-1)*WX = {(s-1)*S.WX_REF:.3e},"
      f" (s^2-1)*THXX/2 = {(s**2-1)*S.THXX_REF/2:.3e})")
print(f"  (gauge breaking prediction (s-1)*WX = {(s-1)*S.WX_REF:.3e},"
      f" (s-1)*THXX/2 = {(s-1)*S.THXX_REF/2:.3e})")
# axis seed magnitude (the O(eps_b) breaking scale)
print(f"  axis |A0| max = {np.max(np.abs(S.A0[:, -1])):.3e}, "
      f"axis |B0| max = {np.max(np.abs(S.B0[:, -1])):.3e}  (eps_b={S.eps_b:g})")
