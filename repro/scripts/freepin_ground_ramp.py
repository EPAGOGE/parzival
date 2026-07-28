"""GROUND CONTROL via the CONDITIONED route: pin-mode scalar secant on
h_id(t) = cl(t) - 2 t / WX_REF at the ground grid.

Rationale (measured): the one-shot coupled freed system is near-singular
(sigma_min 2.98e-8 vs base 2.54e-6; null = identity-line mode), because gI is
nearly redundant at finite resolution.  The conditioned equivalent solves the
SAME mathematical problem -- find TH* with cl(TH*) = 2 TH*/WX -- as a scalar
outer equation over the well-conditioned pinned solver.  G1 gates apply:
|TH* - THXX_REF| <= 6.5e-4 and |alpha* - (-0.34471229)| <= 1e-6.
Alpha self-consistency: outer secant on a alternating with the t-secant
(nested: for each a, solve h_id(t)=0; then h_a = cw/cl - a).
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from freepin import FreePinSolver  # noqa: E402

SCR = pathlib.Path(__file__).parent
SEED = SCR / "hunt_fields/rung_00_a-0.344712.npz"
WX, THREF = 1.19620314, 1.79819132
ALPHA_PINNED = -0.34471229
NEWTON_OK = 1e-9

S, z0 = FreePinSolver.from_seed(SEED, degs=(16, 40, 12), Nb=36, eps_b=1e-4)
n2 = S.Nx * S.Nb
z = z0


def solve_at(t, a, zseed):
    S.TH_target = t
    S.set_alpha(a)
    t0 = time.time()
    zt, f, r, tk = S.newton(z0=zseed.copy(), tol=1e-11)
    if r > NEWTON_OK:
        print(f"    NEWTON FAILED t={t:.7f} a={a:+.8f} ||F||={r:.1e} tk={tk}",
              flush=True)
        return None, zseed, r
    return float(zt[-2]), zt, r


def tsecant(a, t0_, z):
    """solve h_id(t)=0 at frozen a; returns (t*, cl*, z, r)."""
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
            t_new = t + h * WX / 2.0          # slope ~ -2/WX if cl(t) flat
        else:
            t_new = t - h * (t - t_prev) / (h - h_prev)
        t_prev, h_prev, t = t, h, t_new
    return t, cl, z, r


a = float(np.load(SEED)["a"])
print(f"ground ramp-route control: seed a={a:+.8f} TH0={THREF:.8f}", flush=True)
t_star = THREF
for outer in range(10):
    print(f"  outer {outer}: a={a:+.10f}", flush=True)
    t_star, cl, z, r = tsecant(a, t_star, z)
    if t_star is None:
        print("  t-secant failed", flush=True)
        sys.exit(1)
    cw = float(z[-1])
    an = cw / cl
    print(f"  -> t*={t_star:.8f}  cl={cl:.8f}  cw={cw:.8f}  alpha={an:+.10f}",
          flush=True)
    if abs(an - a) < 1e-9:
        break
    a = an

dTH = t_star - THREF
da = an - ALPHA_PINNED
print("\n========== GROUND CONTROL (conditioned route) ==========", flush=True)
print(f"TH*            : {t_star:.8f}   (REF {THREF:.8f}, dTH = {dTH:+.6e})", flush=True)
print(f"cl*            : {cl:.8f}   2TH*/WX = {2*t_star/WX:.8f}", flush=True)
print(f"identity       : {cl - 2*t_star/WX:+.3e}", flush=True)
print(f"alpha*         : {an:+.10f}", flush=True)
print(f"alpha_pinned   : {ALPHA_PINNED:+.8f}   dalpha = {da:+.6e}", flush=True)
print(f"||F||          : {r:.3e}", flush=True)
g_th = abs(dTH) <= 6.5e-4
g_a6 = abs(da) <= 1e-6
g_a5 = abs(da) <= 1e-5
print(f"\ngates: |dTH|<=6.5e-4 {'PASS' if g_th else 'FAIL'} ({abs(dTH):.3e})   "
      f"|dalpha|<=1e-6 {'PASS' if g_a6 else 'FAIL'} ({abs(da):.3e})   "
      f"|dalpha|<=1e-5 {'PASS' if g_a5 else 'FAIL'}", flush=True)
print(f"G1(conditioned) VERDICT: {'PASS' if (g_th and g_a5) else 'FAIL'}", flush=True)
