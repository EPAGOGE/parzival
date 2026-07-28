"""THE BRANCH RUN, step 1: freed-pin solve of branch1 at (24,56,12)/Nb36/eps_b=1e-5.

Seed: hunt_fields/branch1_deg24_56.npz (pinned branch root, alpha=-0.42554621,
cl=4.000688 -> identity violated +33%). Freed system: THXX' unknown, closure
gI = cl - 2*THXX'/WX_REF = 0, WX'==WX_REF normalization.

Seeding strategy (tension #5, identity shock):
  attempt A: TH0 = THXX_REF        (shock in the single gI row, +0.99)
  attempt B: TH0 = cl_seed*WX/2    (gI pre-satisfied; shock spread over the
                                    Nb B-corner pin rows, ~0.3 each)
Outer loop: secant on h(a) = cw/cl - a, set_alpha only (pins live, no rebuild).
Ghost verdict tripwires: fall-back to ground alpha (-0.3447) or divergence.
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
A1REF, A2REF, AGROUND = -0.4168236, -0.4439811, -0.344712
NEWTON_OK = 1e-9

d = np.load(SEED)
a_seed, cl_seed = float(d["a"]), float(d["z"][-2])


def h_eval(S, a, zseed):
    S.set_alpha(a)
    t0 = time.time()
    z, f, r, taken = S.newton(z0=zseed.copy(), tol=1e-11)
    secs = time.time() - t0
    if taken == 0 or r > NEWTON_OK:
        return None, zseed, r, taken, secs
    return float(z[-1]) / float(z[-2]) - a, z, r, taken, secs


def attempt(tag, TH0):
    print(f"\n--- attempt {tag}: TH0={TH0:.8f} ---", flush=True)
    S, z0 = FreePinSolver.from_seed(SEED, **GRID, TH0=TH0)
    n2 = S.Nx * S.Nb
    F0 = S.residual(z0)
    print(f"grid Nx={S.Nx} n={3*n2+3}  seed ||F||_rms={np.linalg.norm(F0)/np.sqrt(F0.size):.3e}"
          f"  gI(seed)={F0[-1]:+.6e}", flush=True)
    a0, z = a_seed, z0
    h0, z, r, tk, secs = h_eval(S, a0, z)
    if h0 is None:
        print(f"  SEED NEWTON FAILED ||F||={r:.2e} taken={tk} secs={secs:.0f}", flush=True)
        return None
    TH = float(z[3 * n2])
    print(f"  a={a0:+.8f} h={h0:+.6f} TH'={TH:.6f} cl={z[-2]:.6f} ||F||={r:.1e} "
          f"steps={tk} secs={secs:.0f}", flush=True)
    a1 = a0 + h0
    for it in range(14):
        h1, z2, r, tk, secs = h_eval(S, a1, z)
        if h1 is None:
            print(f"  a={a1:+.8f} FAILED ||F||={r:.1e} taken={tk}", flush=True)
            return None
        z = z2
        TH = float(z[3 * n2])
        print(f"  a={a1:+.8f} h={h1:+.6f} TH'={TH:.6f} cl={z[-2]:.6f} ||F||={r:.1e} "
              f"steps={tk} secs={secs:.0f}", flush=True)
        if abs(h1) < 1e-9:
            return S, z, a1, r
        a0, h0, a1 = a1, h1, a1 - h1 * (a1 - a0) / (h1 - h0)
    print("  outer cap hit without |h|<1e-9", flush=True)
    return None


res = attempt("A (TH0=THXX_REF)", 1.79819132)
if res is None:
    res = attempt("B (pre-satisfied)", cl_seed * 1.19620314 / 2.0)
if res is None:
    print("\nBOTH SEEDINGS FAILED -> freed system did not take the branch seed "
          "(ghost-verdict flank; report the failure numbers above)", flush=True)
    sys.exit(1)

S, z, a_free, r = res
n2 = S.Nx * S.Nb
TH = float(z[3 * n2])
cl, cw = float(z[-2]), float(z[-1])
F = S.residual(z)
gI_row = float(F[-1])
cl_id = 2.0 * TH / S.WX_REF
print("\n================ FREED BRANCH1 (24,56,12) REPORT ================", flush=True)
print(f"alpha_free     : {a_free:+.10f}", flush=True)
print(f"alpha_pinned   : {a_seed:+.8f}   (freed - pinned = {a_free - a_seed:+.6e})", flush=True)
print(f"WX'            : {S.WX_REF:.8f} (normalization)", flush=True)
print(f"THXX'          : {TH:.8f}   (REF {S.THXX_REF:.8f}, dev {TH - S.THXX_REF:+.6e})", flush=True)
print(f"cl             : {cl:.8f}   2*THXX'/WX' = {cl_id:.8f}", flush=True)
print(f"cw             : {cw:.8f}", flush=True)
print(f"identity row   : {gI_row:+.3e}   (d_cl by construction)", flush=True)
print(f"||F||_rms      : {r:.3e}", flush=True)
print(f"vs alpha_1     : {a_free - A1REF:+.6e}   vs alpha_2: {a_free - A2REF:+.6e}", flush=True)
print(f"vs ground      : {a_free - AGROUND:+.6e}", flush=True)
if abs(a_free - AGROUND) < 5e-3:
    print("TRIPWIRE: freed solve fell back to the GROUND root -> ghost verdict", flush=True)
np.savez(SCR / "hunt_fields/freed_branch1_deg24_56.npz", z=z, a=a_free)
print("saved hunt_fields/freed_branch1_deg24_56.npz", flush=True)
