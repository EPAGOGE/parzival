"""THE BRANCH RUN, step 2: freed-pin solve at (28,64,18)/Nb36/eps_b=1e-5,
seeded by barycentric per-panel per-beta-column interpolation of the freed
(24,56,12) solution (rung3.py pattern).  THE NUMBER THAT MATTERS: the freed
alpha step between the two grids.  Pinned steps were -3.8e-3 / -5.3e-3
(growing).  Freed step < 5e-4 => shadow hypothesis CONFIRMED.
"""
import pathlib
import sys
import time

import numpy as np
from scipy.interpolate import BarycentricInterpolator

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from freepin import FreePinSolver  # noqa: E402

SCR = pathlib.Path(__file__).parent
SEED = SCR / "hunt_fields/freed_branch1_deg24_56.npz"
OLD = dict(degs=(24, 56, 12), Nb=36, eps_b=1e-5)
NEW = dict(degs=(28, 64, 18), Nb=36, eps_b=1e-5)
A1REF, A2REF, AGROUND = -0.4168236, -0.4439811, -0.344712
NEWTON_OK = 1e-9

d = np.load(SEED)
zf, a24 = np.asarray(d["z"], dtype=float), float(d["a"])

So = FreePinSolver(edges=(0.0, 2.0, 15.0, 25.0), **OLD, alpha=a24)
n2o = So.Nx * So.Nb
assert zf.size == 3 * n2o + 3, f"freed seed len {zf.size} != {3*n2o+3}"
Ao = zf[:n2o].reshape(So.Nx, So.Nb)
Bo = zf[n2o:2 * n2o].reshape(So.Nx, So.Nb)
Po = zf[2 * n2o:3 * n2o].reshape(So.Nx, So.Nb)
TH24, cl0, cw0 = float(zf[3 * n2o]), float(zf[-2]), float(zf[-1])
print(f"seed: freed deg24 a={a24:+.8f} TH'={TH24:.8f} cl={cl0:.6f} cw={cw0:.6f}",
      flush=True)

Sn = FreePinSolver(edges=(0.0, 2.0, 15.0, 25.0), **NEW, alpha=a24)
n2n = Sn.Nx * Sn.Nb


def interp_field(F):
    out = np.empty((Sn.Nx, Sn.Nb))
    for k in range(So.K):
        xo = So.x[So.offs[k]:So.offs[k] + So.sizes[k]]
        lo, hi = Sn.offs[k], Sn.offs[k] + Sn.sizes[k]
        xn = Sn.x[lo:hi]
        for j in range(So.Nb):
            out[lo:hi, j] = BarycentricInterpolator(
                xo, F[So.offs[k]:So.offs[k] + So.sizes[k], j])(xn)
    return out


Ai, Bi, Pi = interp_field(Ao), interp_field(Bo), interp_field(Po)
# axis/corner pin data = the interpolated BRANCH fields (spec section 4)
Sn.A0, Sn.B0, Sn.P0 = Ai.copy(), Bi.copy(), Pi.copy()
z0 = np.concatenate([Ai.ravel(), Bi.ravel(), Pi.ravel(), [TH24, cl0, cw0]])
F0 = Sn.residual(z0)
print(f"interpolated to (28,64,18): n={z0.size}  seed ||F||_rms="
      f"{np.linalg.norm(F0)/np.sqrt(F0.size):.3e}  gI(seed)={F0[-1]:+.3e}",
      flush=True)


def h_eval(a, zseed):
    Sn.set_alpha(a)
    t0 = time.time()
    z, f, r, taken = Sn.newton(z0=zseed.copy(), tol=1e-11)
    secs = time.time() - t0
    if taken == 0 or r > NEWTON_OK:
        return None, zseed, r, taken, secs
    return float(z[-1]) / float(z[-2]) - a, z, r, taken, secs


a0, z = a24, z0
h0, z, r, tk, secs = h_eval(a0, z)
if h0 is None:
    print(f"SEED NEWTON FAILED ||F||={r:.2e} taken={tk} secs={secs:.0f}", flush=True)
    sys.exit(1)
TH = float(z[3 * n2n])
print(f"  a={a0:+.8f} h={h0:+.6f} TH'={TH:.6f} cl={z[-2]:.6f} ||F||={r:.1e} "
      f"steps={tk} secs={secs:.0f}", flush=True)
a1 = a0 + h0
for it in range(14):
    h1, z2, r, tk, secs = h_eval(a1, z)
    if h1 is None:
        print(f"  a={a1:+.8f} FAILED ||F||={r:.1e} taken={tk}", flush=True)
        sys.exit(1)
    z = z2
    TH = float(z[3 * n2n])
    print(f"  a={a1:+.8f} h={h1:+.6f} TH'={TH:.6f} cl={z[-2]:.6f} ||F||={r:.1e} "
          f"steps={tk} secs={secs:.0f}", flush=True)
    if abs(h1) < 1e-9:
        break
    a0, h0, a1 = a1, h1, a1 - h1 * (a1 - a0) / (h1 - h0)

cl, cw = float(z[-2]), float(z[-1])
F = Sn.residual(z)
step = a1 - a24
print("\n================ FREED BRANCH1 (28,64,18) REPORT ================", flush=True)
print(f"alpha_free(28,64,18) : {a1:+.10f}   ||F||_rms={r:.3e}", flush=True)
print(f"THXX'                : {TH:.8f}   (deg24 freed: {TH24:.8f}, "
      f"dTH={TH - TH24:+.6e})", flush=True)
print(f"cl                   : {cl:.8f}   2*THXX'/WX' = {2*TH/Sn.WX_REF:.8f}", flush=True)
print(f"cw                   : {cw:.8f}", flush=True)
print(f"identity row         : {float(F[-1]):+.3e}", flush=True)
print(f"FREED RESOLUTION STEP deg24->deg28: {step:+.6e}", flush=True)
print(f"  (pinned steps were -3.817e-3 deg16->24, -5.290e-3 deg24->28)", flush=True)
print(f"vs alpha_1 -0.4168236: {a1 - A1REF:+.6e}", flush=True)
print(f"vs alpha_2 -0.4439811: {a1 - A2REF:+.6e}", flush=True)
print(f"vs ground  -0.344712 : {a1 - AGROUND:+.6e}", flush=True)
if abs(step) < 5e-4:
    print("PRE-REGISTERED: |step| < 5e-4 -> SHADOW HYPOTHESIS CONFIRMED", flush=True)
elif abs(step) >= 3.0e-3:
    print("PRE-REGISTERED: step uncontracted -> NOT pin-explained", flush=True)
else:
    print("PRE-REGISTERED: intermediate contraction -> weak support; do not "
          "promote without another rung", flush=True)
np.savez(SCR / "hunt_fields/freed_branch1_deg28_64_18.npz", z=z, a=a1)
print("saved hunt_fields/freed_branch1_deg28_64_18.npz", flush=True)
