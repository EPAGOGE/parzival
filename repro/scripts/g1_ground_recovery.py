"""G1 GROUND RECOVERY gate for FreePinSolver (freed-pin formulation).

Seed: hunt_fields/rung_00_a-0.344712.npz  at (16,40,12)/Nb36/eps_b=1e-4.
Converge the freed system (damped outer alpha loop, step-counted).

Report: WX' (== WX_REF by normalization), THXX' vs THXX_REF, d_cl variants,
alpha vs the pinned ground value -0.34471229, ||F||_rms, total newton steps.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from freepin import FreePinSolver, pc  # noqa: E402

HF = pathlib.Path(__file__).parent / "hunt_fields"
SEED = HF / "rung_00_a-0.344712.npz"
ALPHA_PINNED = -0.34471229
CL_ID = 3.00649824                      # 2*THXX_REF/WX_REF reference value

S, z0 = FreePinSolver.from_seed(SEED, degs=(16, 40, 12), Nb=36, eps_b=1e-4)
n2 = S.Nx * S.Nb
print(f"grid Nx={S.Nx} Nb={S.Nb} n2={n2}  freed system {3*n2+3} x {3*n2+3}")
print(f"seed alpha={float(np.load(SEED)['a']):+.8f}  TH0={z0[3*n2]:.8f}")
F0 = S.residual(z0)
print(f"seed ||F||_rms = {np.linalg.norm(F0)/np.sqrt(F0.size):.3e}  "
      f"gI(seed) = {F0[-1]:+.6e}")

theta, outer, tol = 0.5, 80, 1e-11
a, hist, total_steps = None, [], 0
z = z0
converged = False
for k in range(outer):
    if a is not None:
        S.set_alpha(a)
    z, f, r, taken = S.newton(z0=z, tol=tol)
    total_steps += taken
    cl, cw = float(z[-2]), float(z[-1])
    an = cw / cl
    TH = float(z[3 * n2])
    hist.append((an, TH, cl, r, taken))
    print(f"pass {k:02d}: steps={taken:2d}  ||F||_rms={r:.3e}  "
          f"cl={cl:.8f}  alpha={an:+.10f}  THXX'={TH:.8f}")
    if taken == 0 and r > tol:
        print("ZERO STEPS with residual above tol -> stalled")
        break
    if a is not None and abs(an - a) < 1e-9 and r < tol:
        converged = True
        break
    a = an if a is None else a + theta * (an - a)

WX_REF, THXX_REF = S.WX_REF, S.THXX_REF
d_cl_free = cl - 2.0 * TH / WX_REF          # the freed closure residual (gI)
d_cl_ref = cl - 2.0 * THXX_REF / WX_REF     # identity vs the REF corner data
print("\n================ G1 GROUND RECOVERY REPORT ================")
print(f"converged           : {converged}  (passes={k+1}, total newton steps={total_steps})")
print(f"||F||_rms           : {r:.3e}")
print(f"WX'                 : {WX_REF:.8f}  (== WX_REF {WX_REF:.8f} by normalization, dev 0)")
print(f"THXX'               : {TH:.8f}")
print(f"THXX_REF            : {THXX_REF:.8f}")
print(f"THXX' - THXX_REF    : {TH - THXX_REF:+.6e}")
print(f"cl                  : {cl:.8f}   (2*THXX_REF/WX_REF = {2*THXX_REF/WX_REF:.8f})")
print(f"d_cl (freed gI)     : {d_cl_free:+.3e}")
print(f"d_cl (vs REF data)  : {d_cl_ref:+.3e}")
print(f"alpha               : {an:+.10f}")
print(f"alpha_pinned        : {ALPHA_PINNED:+.8f}")
print(f"alpha - pinned      : {an - ALPHA_PINNED:+.6e}")

g_th_1e4 = abs(TH - THXX_REF) <= 1e-4
g_th_65 = abs(TH - THXX_REF) <= 6.5e-4
g_dcl = abs(d_cl_free) < 1e-6
g_a_1e5 = abs(an - ALPHA_PINNED) <= 1e-5
g_a_1e6 = abs(an - ALPHA_PINNED) <= 1e-6
print("\ngates:")
print(f"  |dTHXX'| <= 1e-4   : {'PASS' if g_th_1e4 else 'FAIL'}  ({abs(TH-THXX_REF):.3e})")
print(f"  |dTHXX'| <= 6.5e-4 : {'PASS' if g_th_65 else 'FAIL'}")
print(f"  |d_cl|   <  1e-6   : {'PASS' if g_dcl else 'FAIL'}  ({abs(d_cl_free):.3e})")
print(f"  |dalpha| <= 1e-5   : {'PASS' if g_a_1e5 else 'FAIL'}  ({abs(an-ALPHA_PINNED):.3e})")
print(f"  |dalpha| <= 1e-6   : {'PASS' if g_a_1e6 else 'FAIL'}")
ok = converged and g_th_65 and g_dcl and g_a_1e5
print(f"\nG1 VERDICT: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
