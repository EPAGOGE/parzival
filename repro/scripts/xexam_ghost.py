#!/usr/bin/env python3
"""Cross-examination of the GHOST hypothesis. Measurement only:
(1) residual of branch1_deg24_56.npz on its own grid -- converged root or
    unconverged intermediate?  (the advocate's items 2,3 rest on this file)
(2) eps identity of that file (1e-5 vs 1e-4) -- kills the 'assumed eps' hole
(3) control residuals: find_half (original undeflated residual!), deg16_40
(4) dc/da along the ground walk -- how much of the -25.5% corner-amplitude
    collapse is mere alpha motion?
(5) tail recheck incl. the B-tail DECREASE the advocate underplayed
"""
import importlib.util, sys, json
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad"
sys.path.insert(0, SCRATCH)

spec = importlib.util.spec_from_file_location(
    'pc', '/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py')
pc = importlib.util.module_from_spec(spec); sys.modules['pc'] = pc
spec.loader.exec_module(pc)

EDGES = (0.0, 2.0, 15.0, 25.0); NB = 36

def load(fname):
    d = np.load(f"{SCRATCH}/hunt_fields/{fname}")
    return d['z'], float(d['a'])

def rms_residual(z, degs, eps, a):
    S = pc.CornerRegSolver(edges=EDGES, degs=degs, Nb=NB, eps_b=eps, alpha=a)
    S.set_alpha(a)
    f = S.residual(z)
    return np.linalg.norm(f) / np.sqrt(f.size), np.max(np.abs(f))

print("=== (1)+(2)+(3) residual evaluations (RMS norm = newton's own norm) ===")
z, a = load('find_half.npz')
r, mx = rms_residual(z, (16, 40, 12), 1e-4, a)
print(f"find_half        (16,40,12) eps=1e-4 a={a:+.7f}: RMS={r:.3e} max={mx:.3e}"
      "   <- ORIGINAL undeflated residual")

z16, a16 = load('branch1_deg16_40.npz')
r16, mx16 = rms_residual(z16, (16, 40, 12), 1e-5, a16)
print(f"branch1_deg16_40 (16,40,12) eps=1e-5 a={a16:+.7f}: RMS={r16:.3e} max={mx16:.3e}")

z24, a24 = load('branch1_deg24_56.npz')
for eps in (1e-5, 1e-4):
    r24, mx24 = rms_residual(z24, (24, 56, 12), eps, a24)
    print(f"branch1_deg24_56 (24,56,12) eps={eps:.0e} a={a24:+.7f}: RMS={r24:.3e} max={mx24:.3e}")

print("\n=== (4) is the -25.5% c collapse just alpha motion?  dc/da on ground walk ===")
cs, aa = [], []
for f in ('rung_08_a-0.406740.npz', 'rung_09_a-0.414493.npz', 'rung_10_a-0.422247.npz'):
    zz, av = load(f)
    cl, cw = float(zz[-2]), float(zz[-1])
    cs.append((cl - 2 * cw) / 4); aa.append(av)
    print(f"  {f}: a={av:+.6f} c={(cl-2*cw)/4:+.6f}")
sl1 = (cs[1]-cs[0])/(aa[1]-aa[0]); sl2 = (cs[2]-cs[1])/(aa[2]-aa[1])
print(f"  ground dc/da: {sl1:+.4f} (08->09), {sl2:+.4f} (09->10)")
da_deg = a24 - a16
c16v = (float(z16[-2]) - 2*float(z16[-1]))/4
c24v = (float(z24[-2]) - 2*float(z24[-1]))/4
pred = sl2 * da_deg
obs = c24v - c16v
print(f"  branch deg16->24: da={da_deg:+.4e}, observed dc={obs:+.4f}, "
      f"alpha-motion-predicted dc={pred:+.4f} -> unexplained {obs-pred:+.4f} "
      f"({100*abs(obs-pred)/abs(c16v):.1f}% of c)")

print("\n=== (5) tail recheck: all three fields, corner panel, deg16 -> deg24 ===")
def panels(degs):
    out, i0 = [], 0
    for dg in degs:
        out.append(slice(i0, i0 + dg + 1)); i0 += dg + 1
    return out

def cheb_tail(field, sl, frac=0.2):
    n = sl.stop - sl.start
    xj = np.cos(np.pi * np.arange(n) / (n - 1))
    C = np.polynomial.chebyshev.chebfit(xj, field[sl, :], n - 1)
    e = np.sqrt(np.mean(C**2, axis=1))
    ktail = max(2, int(round(frac * n)))
    return float(np.sum(e[-ktail:]) / np.sum(e))

def fields(z, degs):
    nx = sum(dg + 1 for dg in degs); n = nx * NB
    return (z[0:n].reshape(nx, NB), z[n:2*n].reshape(nx, NB), z[2*n:3*n].reshape(nx, NB))

A16, B16, P16 = fields(z16, (16, 40, 12)); A24, B24, P24 = fields(z24, (24, 56, 12))
s16 = panels((16, 40, 12))[0]; s24 = panels((24, 56, 12))[0]
for nm, F16, F24 in (('A', A16, A24), ('B', B16, B24), ('P', P16, P24)):
    t16, t24 = cheb_tail(F16, s16), cheb_tail(F24, s24)
    print(f"  {nm} corner tail: {t16:.3e} -> {t24:.3e}  ({'GROWS' if t24>t16 else 'decays'} {100*(t24-t16)/t16:+.0f}%)")

# wedge exponent at each eps -- the KNOWN singular layer xi^(k-2)
for eps in (1e-4, 1e-5):
    k = np.pi / (np.pi/2 - 2*eps)
    print(f"  eps={eps:.0e}: wedge k-2 = {k-2:.3e} (xi^(k-2) singular layer, algebraic tail)")
