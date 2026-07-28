"""T2: deflated multistart at frozen a = alpha_1 = -0.4168236 (DeepMind's first
unstable branch).  The cycle-3 deduction D1: the h-landscape is blind to other
branches (h = a* - a identically on the n=0 field branch), so deflation is the
unique surviving instrument.  D2: any root's own cw/cl IS its branch label.
Keep: converged ||F|| < 1e-11, distinct from the n=0 anchor; report cw/cl and
h = cw/cl - a.  |h| < ~1e-2 marks a self-consistency candidate for secant polish."""
import importlib.util, pathlib, sys, time
import numpy as np
SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
def modp(n, p):
    sp_ = importlib.util.spec_from_file_location(n, str(p))
    m = importlib.util.module_from_spec(sp_); sys.modules[n] = m
    sp_.loader.exec_module(m); return m
bh = modp("bh", SCR / "branch_hunt.py")
pc = bh.pc
A1 = -0.4168236
S = pc.CornerRegSolver(**bh.CFG, alpha=A1)
# anchor: nearest saved rung, warm-converged at exactly a = alpha_1
rung = sorted(SCR.glob("hunt_fields/rung_*a-0.414493*.npz"))[0]
z0 = np.load(rung)["z"]
z_anchor, f, r, taken = S.newton(z0=z0.copy())
cl, cw = float(z_anchor[-2]), float(z_anchor[-1])
print(f"anchor (n=0 branch at frozen a={A1}): ||F||={r:.2e} steps={taken} "
      f"cw/cl={cw/cl:+.6f} (invariance predicts ~ -0.34471)", flush=True)
rng = np.random.default_rng(11)
n2 = S.Nx * S.Nb
def start(tag):
    z = z_anchor.copy()
    if tag == "r1":  z[:-2] += 0.01 * np.linalg.norm(z[:-2]) / np.sqrt(z.size) * rng.standard_normal(z.size - 2)
    if tag == "r5":  z[:-2] += 0.05 * np.linalg.norm(z[:-2]) / np.sqrt(z.size) * rng.standard_normal(z.size - 2)
    if tag == "r20": z[:-2] += 0.20 * np.linalg.norm(z[:-2]) / np.sqrt(z.size) * rng.standard_normal(z.size - 2)
    if tag == "node":            # inject a radial node into B (unstable modes add structure)
        B = z[n2:2*n2].reshape(S.Nx, S.Nb).copy()
        B *= (1.0 - 1.5 * np.sin(np.pi * S.x[:, None] / 4.0) * np.exp(-S.x[:, None] / 2.0))
        z[n2:2*n2] = B.ravel()
    if tag == "half": z[:-2] *= 0.5
    if tag == "x15":  z[:-2] *= 1.5
    if tag == "nodeA":           # radial node in A instead
        A = z[:n2].reshape(S.Nx, S.Nb).copy()
        A *= (1.0 - 1.2 * np.sin(np.pi * S.x[:, None] / 3.0) * np.exp(-S.x[:, None] / 2.0))
        z[:n2] = A.ravel()
    if tag == "r10b": z[:-2] += 0.10 * np.linalg.norm(z[:-2]) / np.sqrt(z.size) * np.random.default_rng(77).standard_normal(z.size - 2)
    return z
finds = []
for tag in ("r1", "r5", "r20", "node", "half", "x15", "nodeA", "r10b"):
    t0 = time.time()
    z, rr, taken, ok = bh.deflated_newton(S, start(tag), [z_anchor], steps=80)
    if not ok:
        print(f"  start {tag:>5}: DRY ||F||={rr:.2e} steps={taken} secs={time.time()-t0:.0f}", flush=True)
        continue
    cl2, cw2 = float(z[-2]), float(z[-1])
    dist = np.linalg.norm(z - z_anchor) / np.linalg.norm(z_anchor)
    amp = float(np.abs(z[:2*n2]).max())
    h = cw2 / cl2 - A1
    print(f"  start {tag:>5}: CONVERGED ||F||={rr:.2e} steps={taken} dist={dist:.2e} "
          f"max|field|={amp:.3g} cw/cl={cw2/cl2:+.6f} h={h:+.4f} secs={time.time()-t0:.0f}", flush=True)
    if dist > 1e-3 and amp > 1e-6:
        finds.append((tag, cw2/cl2, h, dist))
        np.savez(SCR / f"hunt_fields/find_{tag}.npz", z=z, a=A1, h=h)
print(f"\ndistinct nontrivial roots found: {len(finds)}", flush=True)
for tag, al, h, d in finds:
    print(f"  {tag}: cw/cl={al:+.6f}  h={h:+.4f}  dist={d:.2e}"
          f"   {'<-- SELF-CONSISTENCY CANDIDATE' if abs(h) < 2e-2 else ''}", flush=True)
print("done", flush=True)
