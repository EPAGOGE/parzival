"""FREED BRANCH, deg28 rung via the CONDITIONED route (nested secant).

Seed: freed_branch1_deg24_56.npz (freed layout 3n2+3) interpolated to
(28,64,18)/Nb36/eps_b=1e-5 (barycentric per-panel per-beta-column, rung3
pattern).  Solve: for each alpha, pin-mode t-secant on
h_id(t) = cl(t) - 2t/WX_REF; outer secant on alpha.
THE NUMBER: freed alpha step deg24->deg28 (pinned was -5.3e-3; pre-registered
bars: <5e-4 shadow-confirmed, >=3e-3 not-pin-explained).
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
WX = 1.19620314
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
TH24, cl24, cw24 = float(zf[3 * n2o]), float(zf[-2]), float(zf[-1])
print(f"seed: freed deg24 a={a24:+.8f} TH'={TH24:.8f} cl={cl24:.6f}", flush=True)

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
Sn.A0, Sn.B0, Sn.P0 = Ai.copy(), Bi.copy(), Pi.copy()
z = np.concatenate([Ai.ravel(), Bi.ravel(), Pi.ravel(), [TH24, cl24, cw24]])
F0 = Sn.residual(z)
print(f"interpolated to (28,64,18): n={z.size}  seed ||F||_rms="
      f"{np.linalg.norm(F0)/np.sqrt(F0.size):.3e}", flush=True)


def solve_at(t, a, zseed):
    Sn.TH_target = t
    Sn.set_alpha(a)
    t0 = time.time()
    zt, f, r, tk = Sn.newton(z0=zseed.copy(), tol=1e-11)
    secs = time.time() - t0
    if r > NEWTON_OK:
        print(f"    NEWTON FAILED t={t:.7f} a={a:+.8f} ||F||={r:.1e} tk={tk} "
              f"secs={secs:.0f}", flush=True)
        return None, zseed, r
    return float(zt[-2]), zt, r


def tsecant(a, t0_, z):
    t_prev, h_prev = None, None
    t = t0_
    for it in range(12):
        cl, z2, r = solve_at(t, a, z)
        if cl is None:
            return None, None, z, r
        z = z2
        h = cl - 2.0 * t / WX
        print(f"    t={t:.8f} cl={cl:.8f} h_id={h:+.3e} ||F||={r:.1e}", flush=True)
        if abs(h) < 1e-10:
            return t, cl, z, r
        if t_prev is None:
            t_new = t + h * WX / 2.0
        else:
            t_new = t - h * (t - t_prev) / (h - h_prev)
        t_prev, h_prev, t = t, h, t_new
    return t, cl, z, r


a, t_star = a24, TH24
an = None
for outer in range(12):
    print(f"  outer {outer}: a={a:+.10f}", flush=True)
    t_star, cl, z, r = tsecant(a, t_star, z)
    if t_star is None:
        print("  t-secant failed", flush=True)
        sys.exit(1)
    cw = float(z[-1])
    an = cw / cl
    print(f"  -> t*={t_star:.8f}  cl={cl:.8f}  alpha={an:+.10f}", flush=True)
    if abs(an - a) < 1e-9:
        break
    a = an

step = an - a24
print("\n================ FREED BRANCH1 (28,64,18), conditioned route ================",
      flush=True)
print(f"alpha_free(28) : {an:+.10f}   ||F||={r:.3e}", flush=True)
print(f"THXX'(28)      : {t_star:.8f}   (deg24 freed {TH24:.8f}, dTH={t_star-TH24:+.6e})",
      flush=True)
print(f"cl             : {cl:.8f}   2TH/WX = {2*t_star/WX:.8f}", flush=True)
print(f"FREED STEP deg24->deg28 : {step:+.6e}   (pinned was -5.290e-3)", flush=True)
print(f"vs alpha_1: {an - A1REF:+.6e}   vs alpha_2: {an - A2REF:+.6e}   "
      f"vs ground: {an - AGROUND:+.6e}", flush=True)
if abs(step) < 5e-4:
    print("PRE-REGISTERED: |step| < 5e-4 -> SHADOW HYPOTHESIS CONFIRMED", flush=True)
elif abs(step) >= 3.0e-3:
    print("PRE-REGISTERED: step uncontracted -> NOT pin-explained", flush=True)
else:
    print("PRE-REGISTERED: intermediate contraction -> weak support", flush=True)
np.savez(SCR / "hunt_fields/freed_branch1_deg28_64_18.npz", z=z, a=an)
print("saved hunt_fields/freed_branch1_deg28_64_18.npz", flush=True)
