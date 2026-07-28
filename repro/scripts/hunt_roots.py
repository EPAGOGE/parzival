"""ROOT HUNT with deflation + informed multi-start.

The gate passed: pole repels, root is simple, deflated Jacobian exact, shift correct.
Now the actual question -- is alpha = -0.3316 the only root, or is Chen-Hou's -0.3424
sitting in a neighbouring basin that single-start Newton never reached?

SEED DIVERSITY IS THE POINT.  In a fractal basin, seed quality dominates iteration count
(this project's own history: six configurations, three different wrong roots). So the
starts are deliberately heterogeneous:
  0. the raw interpolated seed (what every run has used)
  1. the converged d2 root -- a genuinely different converged point
  2-5. random perturbations of the seed at 1%, 5%, 20%, 50%
  6. the seed with the amplitude scaled -- pushes the constraints hard off their targets
Each is tried against the CURRENT deflation set, so a start whose natural basin is an
already-found root gets pushed out of it rather than wasting the round.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import numpy.linalg as la

H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")


def mod(name, fn):
    sp = importlib.util.spec_from_file_location(name, str(H / fn))
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


pst = mod("pst", "polar_stability.py")
pdf = mod("pdf", "polar_deflate.py")
pnl = mod("pnl", "polar_nlens.py")

REF = -0.34240009
N = 28
CONS = "d1"

print(f"N={N} constraint={CONS}  reference alpha={REF}", flush=True)
St, x0d1, r, cl, cw, info = pst.converge_exact(N, constraint=CONS, strict=False,
                                               outer_steps=80)
CLS = 2.0 * St.S.THXX_REF / St.S.WX_REF
print(f"base d1 root: alpha={cw/cl:+.8f} c_l={cl:.6f} (c_l*={CLS:.6f}, "
      f"off by {100*(cl-CLS)/CLS:+.2f}%) ||F||={r:.2e}\n", flush=True)

# the d2 root as an independent informed start
St2, x0d2, r2, cl2, cw2, info2 = pst.converge_exact(N, constraint="d2", strict=False,
                                                    outer_steps=80)
print(f"d2 root for reference: alpha={cw2/cl2:+.8f} c_l={cl2:.6f} ||F||={r2:.2e}\n",
      flush=True)

seed = St.S.x0.copy()
rng = np.random.default_rng(7)
seeds = [seed.copy(), x0d2.copy()]
for frac in (0.01, 0.05, 0.20, 0.50):
    p = rng.standard_normal(seed.size)
    p *= frac * la.norm(seed) / la.norm(p)
    seeds.append(seed + p)
amp = seed.copy()
amp[:-2] *= 1.5
seeds.append(amp)

atlas = pdf.Atlas()
rec = atlas.add(St, x0d1, r)
print(f"root 0: alpha={rec['alpha']:+.8f} c_l={rec['cl']:.6f} [{rec['kind']}]", flush=True)

for k, s in enumerate(seeds):
    centres = [R["x"] for R in atlas.deflatable()]
    tag = ["raw seed", "d2 root", "seed+1%", "seed+5%", "seed+20%", "seed+50%",
           "seed x1.5 amplitude"][k]
    print(f"\nround {k+1}: start = {tag}, deflating {len(centres)} root(s)", flush=True)
    xn, fn, rn, taken, ok = pdf.deflated_newton(St, s.copy(), centres, verbose=False)
    if abs(float(xn[-2])) < 1e-30:
        print(f"  c_l collapsed to zero -- degenerate", flush=True)
        continue
    a = float(xn[-1]) / float(xn[-2])
    clr = float(xn[-2])
    Ot, Bt = St.S.unpack(xn[:-2])
    ampn = float(max(np.abs(Ot).max(), np.abs(Bt).max()))
    print(f"  -> alpha={a:+.8f} c_l={clr:.6f} ||F||={rn:.3e} steps={taken} "
          f"converged={ok} max|field|={ampn:.4g}", flush=True)
    if not ok:
        print(f"     DRY (did not converge)", flush=True)
        continue
    if not atlas.is_new(xn, a):
        print(f"     already known -- deflation did not escape", flush=True)
        continue
    rec = atlas.add(St, xn, rn)
    print(f"     *** NEW ROOT *** [{rec['kind']}] "
          f"vs ref {100*(a-REF)/abs(REF):+.3f}%  "
          f"c_l off {100*(clr-CLS)/CLS:+.2f}%", flush=True)
    pnl.render(pnl.flags_for(St, xn))

print("\n" + "=" * 74)
print("ROOT ATLAS")
print(f"  {'alpha':>14} {'vs ref':>9} {'c_l':>11} {'c_l vs *':>10} {'||F||':>10}  kind")
for R in atlas.summary():
    print(f"  {R['alpha']:+14.8f} {100*(R['alpha']-REF)/abs(REF):+8.3f}% "
          f"{R['cl']:11.6f} {100*(R['cl']-CLS)/CLS:+9.2f}% {R['F']:10.2e}  {R['kind']}")
print(f"\n  reference alpha={REF}   c_l*={CLS:.8f}")
print(f"  distinct roots found: {len(atlas.roots)}")
