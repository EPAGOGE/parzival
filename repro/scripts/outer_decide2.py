"""Rerun with `outer` threaded through the REAL call chain.

The first attempt monkeypatched pn.NewtonSolver.__init__, but Stability.__init__ calls
_mod("pn", "polar_newton.py") which re-executes the module and builds a fresh class object,
so the patch was discarded -- which is exactly why alpha came out bit-identical (0.000e+00)
across all three variants. That was a broken harness, not a physical result.

PREDICTION now on the record: alpha is insensitive to the outer condition at XMAX=25,
because a boundary perturbation there reaches the corner attenuated by e^{a0*25} = 1.9e-4
and alpha is read from CORNER derivatives. If |d(alpha)| < 1e-5 relative, the outer boundary
is exonerated. If it is larger, the attenuation argument is wrong.
Separately: does any variant rescue XMAX = 32, where the seed is still inside the reference
data (which reaches s = 35.7) but Newton currently stalls at ||F|| = 3.9e-3?
"""
import sys, pathlib, numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
import importlib.util


def mod(name, fn):
    sp = importlib.util.spec_from_file_location(name, str(H / fn))
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m


pst = mod("pst", "polar_stability.py")
REF = -0.34240009

print(f"{'N':>3} {'XMAX':>6} {'outer':>8} {'||F||':>11} {'alpha':>14} {'vs ref':>9} "
      f"{'d(alpha) vs neumann':>21}")
base = {}
for X in (25.0, 20.0, 32.0):
    for v in ("neumann", "dtn", "dtn3"):
        try:
            St, x, r, cl, cw = pst.converge_exact(36, XMAX=X, outer=v)
            a = cw / cl
            if v == "neumann":
                base[X] = a
                d = ""
            else:
                d = f"{a - base.get(X, float('nan')):+.4e}"
            print(f"{36:3d} {X:6.1f} {v:>8} {r:11.3e} {a:+14.8f} "
                  f"{100*(a-REF)/abs(REF):+8.3f}% {d:>21}", flush=True)
        except Exception as ex:
            print(f"{36:3d} {X:6.1f} {v:>8}   FAILED {type(ex).__name__}: {ex}", flush=True)
print(f"\nreference alpha = {REF}")
