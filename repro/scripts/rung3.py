"""RUNG 3 -- the armed falsifier. Interpolation-seeded (28,64,18)/Nb36/eps=1e-5 solve
of the candidate root, REBUILDING the solver at every secant alpha (kills the
stale-pin systematic W5: pins/A0/B0 are baked at construction alpha).
Pre-registered decision rule (tension #1):
  |alpha step vs deg24| <= ~1.9e-3   -> contracting: distinct-object flank strengthens
  ~3.8e-3 (uncontracted) or erratic  -> ghost flank strengthens
  >= +7e-3 reversal toward -0.41682  -> alpha_1 after all (axiom dies)
  cumulative drift toward -0.44398   -> alpha_2-in-disguise: compare before naming
Quote ONLY steps and the sequence -- never a single-grid alpha (refusal R2, 296x)."""
import importlib.util, pathlib, sys, time
import numpy as np
from scipy.interpolate import BarycentricInterpolator

SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
spec = importlib.util.spec_from_file_location("pc", "/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py")
pc = importlib.util.module_from_spec(spec); sys.modules["pc"] = pc; spec.loader.exec_module(pc)

d = np.load(SCR / "hunt_fields/branch1_deg24_56.npz")
a24, z24 = float(d["a"]), d["z"]
OLD = dict(edges=(0.0, 2.0, 15.0, 25.0), degs=(24, 56, 12), Nb=36, eps_b=1e-5)
NEW = dict(edges=(0.0, 2.0, 15.0, 25.0), degs=(28, 64, 18), Nb=36, eps_b=1e-5)
So = pc.CornerRegSolver(**OLD, alpha=a24)
n2o = So.Nx * So.Nb
Ao = z24[:n2o].reshape(So.Nx, So.Nb)
Bo = z24[n2o:2*n2o].reshape(So.Nx, So.Nb)
Po = z24[2*n2o:3*n2o].reshape(So.Nx, So.Nb)
cl0, cw0 = float(z24[-2]), float(z24[-1])
print(f"seed: deg24 field a={a24:+.8f} cl={cl0:.6f} cw={cw0:.6f}", flush=True)

Sn = pc.CornerRegSolver(**NEW, alpha=a24)   # grids only; rebuilt fresh per alpha below
def interp_field(F):
    out = np.empty((Sn.Nx, Sn.Nb))
    for k in range(So.K):
        xo = So.x[So.offs[k]:So.offs[k] + So.sizes[k]]
        lo, hi = Sn.offs[k], Sn.offs[k] + Sn.sizes[k]
        xn = Sn.x[lo:hi]
        for j in range(So.Nb):
            out[lo:hi, j] = BarycentricInterpolator(xo, F[So.offs[k]:So.offs[k]+So.sizes[k], j])(xn)
    return out
z0 = np.concatenate([interp_field(Ao).ravel(), interp_field(Bo).ravel(),
                     interp_field(Po).ravel(), [cl0, cw0]])
print(f"interpolated to (28,64,18): n={z0.size}", flush=True)

def h_eval(a, zseed):
    S = pc.CornerRegSolver(**NEW, alpha=a)          # REBUILD: fresh pins at this alpha
    z, f, r, taken = S.newton(z0=zseed.copy())
    if taken == 0 or r > 1e-9:
        return None, None, r, taken
    return float(z[-1]) / float(z[-2]) - a, z, r, taken

a0, z = a24, z0
t0 = time.time()
h0, z, r, tk = h_eval(a0, z)
if h0 is None:
    print(f"SEED NEWTON FAILED ||F||={r:.2e} taken={tk} -- interpolation seed did not take", flush=True)
    sys.exit(1)
print(f"  a={a0:+.8f} h={h0:+.6f} ||F||={r:.1e} steps={tk} secs={time.time()-t0:.0f}", flush=True)
a1 = a0 + h0
for it in range(12):
    t0 = time.time()
    h1, z2, r, tk = h_eval(a1, z)
    if h1 is None:
        print(f"  a={a1:+.8f} FAILED ||F||={r:.1e}", flush=True); break
    z = z2
    print(f"  a={a1:+.8f} h={h1:+.6f} ||F||={r:.1e} steps={tk} secs={time.time()-t0:.0f}", flush=True)
    if abs(h1) < 1e-9: break
    a0, h0, a1 = a1, h1, a1 - h1 * (a1 - a0) / (h1 - h0)

A1REF, A2REF = -0.4168236, -0.4439811
step = a1 - a24
print(f"\nRUNG 3 RESULT: alpha(28,64,18) = {a1:+.8f}   ||F||={r:.1e}", flush=True)
print(f"  step vs deg24: {step:+.3e}   (prior step deg16->24 was -3.817e-3)", flush=True)
print(f"  gap to alpha_1: {a1-A1REF:+.3e}   gap to alpha_2: {a1-A2REF:+.3e}", flush=True)
if abs(step) <= 1.9e-3:  verdict = "CONTRACTING -> distinct-object flank strengthens"
elif step >= 7e-3:       verdict = "REVERSAL -> alpha_1 after all; the axiom dies"
elif abs(step) >= 3.0e-3: verdict = "UNCONTRACTED -> ghost flank strengthens / alpha_2-in-disguise check"
else:                    verdict = "INTERMEDIATE -> weak contraction; extend one more rung before naming"
print(f"  PRE-REGISTERED VERDICT: {verdict}", flush=True)
np.savez(SCR / "hunt_fields/branch1_deg28_64_18.npz", z=z, a=a1)
print("done", flush=True)
