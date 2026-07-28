#!/usr/bin/env python3
"""Cross-exam part 2: are the loose-residual files (branch1_eps1e-4, eps3e-05,
deg24_56) converged Newton states at a slightly different frozen a (benign:
save convention stores a := cw/cl of the fields, the next secant iterate), or
genuinely unconverged mid-run junk?

Method: hold z fixed, scan alpha, find a_min = argmin ||F(z; a)||_inf.
 - If min residual ~ Newton tol  -> fields ARE a converged root at a_min;
   stored cw/cl is exact field data; self-consistency error of the stored
   alpha is |stored - a_min| * |m/(1-m)| with m = d(cw/cl)/da = -0.026.
 - If min residual stays >> tol  -> mid-run intermediate; cw/cl unreliable.

Also measures ||dF/da|| scale on find_half for unit conversion.
"""
import importlib.util, sys, numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad"
HF = SCRATCH + "/hunt_fields"
spec = importlib.util.spec_from_file_location(
    'pc', '/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py')
pc = importlib.util.module_from_spec(spec)
sys.modules['pc'] = pc
spec.loader.exec_module(pc)

def load(name):
    d = np.load(f"{HF}/{name}", allow_pickle=True)
    return {k: d[k] for k in d.files}

print("A) ||dF/da|| scale on find_half (converged root, ||F||=7.2e-13)")
d = load("find_half.npz")
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12),
                       Nb=36, eps_b=1e-4, alpha=float(d['a']))
z = d['z']
a0 = float(d['a'])
for da in (1e-6, 1e-5, 1e-4):
    S.set_alpha(a0 + da)
    r = float(np.max(np.abs(S.residual(z))))
    print(f"   da={da:.0e}  ||F||_inf={r:.3e}  ->  ||dF/da||_inf ~ {r/da:.3f}")
S.set_alpha(a0)

def ascan(name, degs, eps_b, label, width=2e-2, ncoarse=17, refine_rounds=6):
    d = load(name)
    a_st = float(d['a'])
    z = d['z']
    S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=degs,
                           Nb=36, eps_b=eps_b, alpha=a_st)
    def r_of(a):
        S.set_alpha(a)
        return float(np.max(np.abs(S.residual(z))))
    lo, hi = a_st - width, a_st + width
    for rnd in range(refine_rounds):
        aa = np.linspace(lo, hi, ncoarse)
        rr = [r_of(a) for a in aa]
        i = int(np.argmin(rr))
        ilo, ihi = max(i - 1, 0), min(i + 1, ncoarse - 1)
        lo, hi = aa[ilo], aa[ihi]
    a_min, r_min = aa[i], rr[i]
    cwcl = float(z[-1]) / float(z[-2])
    m = -0.026  # d(cw/cl)/da measured on branch at deg16
    err_alpha = abs(a_st - a_min) * abs(m / (1 - m))
    print(f"\nB) {label}")
    print(f"   stored a = cw/cl = {a_st:+.9f}")
    print(f"   a_min (argmin ||F||) = {a_min:+.9f}   |stored - a_min| = {abs(a_st-a_min):.3e}")
    print(f"   min ||F||_inf = {r_min:.3e}   (residual at stored a was the earlier number)")
    print(f"   -> if converged at a_min: stored-alpha self-consistency error "
          f"~ {err_alpha:.2e}  (damping m/(1-m) = {m/(1-m):+.4f})")
    return a_st, a_min, r_min, err_alpha

# worst eps-ladder rung
ascan("branch1_eps1e-4.npz", (16, 40, 12), 1e-4,
      "branch1_eps1e-4 (residual at stored a: 1.1e-3)", width=1.5e-2)
# second-worst ladder rung
ascan("branch1_eps3e-05.npz", (16, 40, 12), 3e-5,
      "branch1_eps3e-05 (residual at stored a: 4.75e-4)", width=5e-3)
# THE refinement datapoint
ascan("branch1_deg24_56.npz", (24, 56, 12), 1e-5,
      "branch1_deg24_56 @ eps=1e-5 (residual at stored a: 9.5e-5)", width=5e-3)
