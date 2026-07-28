"""FALLBACK 1 (spec section 6): pinned-THXX ramp continuation at deg24.

Both direct seedings of the freed branch solve stalled (A: ||F||=2.35e-3,
B: ||F||=2.17e-3).  Pre-registration: the ghost/fault verdict requires this
ramp -- swap gI for the pin row THXX' - t = 0, ramp t from THXX_REF toward
2.45 warm-started at FROZEN alpha = a_seed, monitor
    h_id(t) = cl(t) - 2*t/WX_REF
for a sign change.  On bracket: switch gI back on and Newton from the
bracketing state; if that converges, secant on alpha for self-consistency.
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from freepin import FreePinSolver  # noqa: E402

SCR = pathlib.Path(__file__).parent
SEED = SCR / "hunt_fields/branch1_deg24_56.npz"
GRID = dict(degs=(24, 56, 12), Nb=36, eps_b=1e-5)
WX = 1.19620314
A1REF, A2REF, AGROUND = -0.4168236, -0.4439811, -0.344712
NEWTON_OK = 1e-9

d = np.load(SEED)
a_seed, cl_seed = float(d["a"]), float(d["z"][-2])

S, z0 = FreePinSolver.from_seed(SEED, **GRID, TH0=1.79819132)
n2 = S.Nx * S.Nb
S.set_alpha(a_seed)

ts = [1.79819132, 1.95, 2.10, 2.25, 2.32, 2.39283, 2.45]
print(f"RAMP at frozen alpha={a_seed:+.8f}  cl_seed={cl_seed:.6f}", flush=True)
print(f"{'t':>10} {'cl(t)':>12} {'h_id(t)':>12} {'||F||':>10} {'steps':>5} {'secs':>5}",
      flush=True)
z = z0
rows = []
for t in ts:
    S.TH_target = t
    t0 = time.time()
    zt, f, r, tk = S.newton(z0=z.copy(), tol=1e-11)
    secs = time.time() - t0
    if r > NEWTON_OK:
        print(f"{t:10.5f}  RAMP STEP STALLED ||F||={r:.2e} steps={tk} secs={secs:.0f}",
              flush=True)
        sys.exit(1)
    z = zt
    cl = float(z[-2])
    h_id = cl - 2.0 * t / WX
    rows.append((t, cl, h_id))
    print(f"{t:10.5f} {cl:12.6f} {h_id:+12.6f} {r:10.1e} {tk:5d} {secs:5.0f}",
          flush=True)

sign_change = [(rows[i], rows[i + 1]) for i in range(len(rows) - 1)
               if rows[i][2] * rows[i + 1][2] < 0]
if not sign_change:
    print("\nNO IDENTITY SIGN-CROSSING on the ramp t in "
          f"[{ts[0]:.3f}, {ts[-1]:.3f}] -> pre-registered fault flank complete",
          flush=True)
    sys.exit(2)

(tl, cll, hl), (tr, clr, hr) = sign_change[0]
print(f"\nBRACKET: h_id({tl:.5f})={hl:+.6f}  h_id({tr:.5f})={hr:+.6f}", flush=True)
# secant refine the crossing in pin mode (2 extra points), then free gI
t_star = tl - hl * (tr - tl) / (hr - hl)
for _ in range(2):
    S.TH_target = t_star
    z, f, r, tk = S.newton(z0=z.copy(), tol=1e-11)
    cl = float(z[-2])
    h = cl - 2.0 * t_star / WX
    print(f"refine: t={t_star:.6f} cl={cl:.6f} h_id={h:+.6f} ||F||={r:.1e} steps={tk}",
          flush=True)
    if abs(h) < 1e-7:
        break
    t_star = t_star - h * 0.5  # d h/dt ~ -2/WX + dcl/dt; crude damped step

print("\n=== gI ON from the crossing state (still frozen alpha) ===", flush=True)
S.TH_target = None
t0 = time.time()
z, f, r, tk = S.newton(z0=z.copy(), tol=1e-11)
print(f"gI-on newton: ||F||={r:.1e} steps={tk} secs={time.time()-t0:.0f}", flush=True)
if r > NEWTON_OK:
    print("gI-on newton FAILED from the bracketing state", flush=True)
    sys.exit(3)

TH = float(z[3 * n2])
print(f"frozen-alpha freed root: TH'={TH:.8f} cl={z[-2]:.8f} cw={z[-1]:.8f} "
      f"cw/cl={float(z[-1])/float(z[-2]):+.8f}", flush=True)

print("\n=== secant on alpha for self-consistency ===", flush=True)
a0 = a_seed
h0 = float(z[-1]) / float(z[-2]) - a0
a1 = a0 + h0
for it in range(14):
    S.set_alpha(a1)
    t0 = time.time()
    z2, f, r, tk = S.newton(z0=z.copy(), tol=1e-11)
    secs = time.time() - t0
    if r > NEWTON_OK:
        print(f"  a={a1:+.8f} FAILED ||F||={r:.1e} steps={tk}", flush=True)
        sys.exit(4)
    z = z2
    h1 = float(z[-1]) / float(z[-2]) - a1
    TH = float(z[3 * n2])
    print(f"  a={a1:+.8f} h={h1:+.6f} TH'={TH:.6f} cl={float(z[-2]):.6f} "
          f"||F||={r:.1e} steps={tk} secs={secs:.0f}", flush=True)
    if abs(h1) < 1e-9:
        break
    a0, h0, a1 = a1, h1, a1 - h1 * (a1 - a0) / (h1 - h0)

a_free = a1
cl, cw = float(z[-2]), float(z[-1])
print("\n================ RAMP-ROUTE FREED BRANCH1 (24,56,12) ================",
      flush=True)
print(f"alpha_free  : {a_free:+.10f}   (pinned was {a_seed:+.8f})", flush=True)
print(f"THXX'       : {TH:.8f}   cl={cl:.8f}   2TH/WX={2*TH/WX:.8f}", flush=True)
print(f"vs alpha_1  : {a_free - A1REF:+.6e}  vs alpha_2: {a_free - A2REF:+.6e}  "
      f"vs ground: {a_free - AGROUND:+.6e}", flush=True)
np.savez(SCR / "hunt_fields/freed_branch1_deg24_56.npz", z=z, a=a_free)
print("saved hunt_fields/freed_branch1_deg24_56.npz", flush=True)
