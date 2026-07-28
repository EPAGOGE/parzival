"""DOES THE OUTER CONDITION MATTER FOR alpha AT ALL?

The conditioning hypothesis is refuted (cond(Poisson) is flat in XMAX: +0.0049 per unit,
not the predicted +2.3475 -- exponential dichotomy is a SHOOTING pathology, not a two-point
BVP one).  And the attenuation argument now cuts the other way: a boundary perturbation at
xi = L reaches the corner damped by e^{a0 L} = 1.9e-4 at L = 25, and alpha = c_w/c_l is
read from CORNER derivatives.  So the prediction is now:

    alpha is INSENSITIVE to the outer condition at XMAX = 25.

If that holds, the outer boundary is exonerated and the radial story has to be told
somewhere else.  If alpha moves by more than ~1e-4 relative, the attenuation argument is
wrong and I want to know why.

Second question, independent: XMAX = 32 currently fails Newton at ||F|| = 3.9e-3 while the
reference profile extends to s = 35.7, so the seed is NOT extrapolated there.  Does any
outer variant converge at 32?
"""
import sys, pathlib, numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
import importlib.util


def mod(name, fn):
    sp = importlib.util.spec_from_file_location(name, str(H / fn))
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m


pn = mod("pn", "polar_newton.py")
pst = mod("pst", "polar_stability.py")

REF = -0.34240009


def patched(outer):
    """converge_exact, with the outer condition threaded in."""
    orig = pn.NewtonSolver.__init__

    def init(self, N=32, XMAX=25.0, alpha=None, Nb=None, eps_b=1e-3, _o=outer):
        pc = mod("pc", "polar_corner.py")
        self.C = pc.Corner(N, N if Nb is None else Nb, XMAX, filter_on=False,
                           eps_b=eps_b, outer=_o)
        if alpha is not None:
            self.C.a0 = float(alpha)
            self.C.mu = 2.0 + self.C.a0
            self.C.E = np.exp(self.C.a0 * self.C.x)[:, None]
            self.C._build_poisson()
        C = self.C
        m = np.ones((C.nx, C.nb), dtype=bool)
        m[0, :] = False
        m[:, -1] = False
        self.mask = np.concatenate([m.ravel(), m.ravel()])
        self.idx = np.where(self.mask)[0]
        self.n2 = C.nx * C.nb
        self.x0 = np.concatenate([self.pack(C.Ot0, C.Bt0), [C.P["cl"], C.P["cw"]]])

    pn.NewtonSolver.__init__ = init
    return orig


print(f"{'N':>3} {'XMAX':>6} {'outer':>8} {'||F||':>11} {'alpha':>14} {'vs ref':>9} "
      f"{'d(alpha) vs neumann':>20}")
base = {}
for X in (25.0, 32.0, 20.0):
    for v in ("neumann", "dtn", "dtn3"):
        orig = patched(v)
        try:
            St, x, r, cl, cw = pst.converge_exact(36, XMAX=X)
            a = cw / cl
            d = "" if v == "neumann" else f"{a - base.get(X, np.nan):+.3e}"
            if v == "neumann":
                base[X] = a
            print(f"{36:3d} {X:6.1f} {v:>8} {r:11.3e} {a:+14.8f} "
                  f"{100*(a-REF)/abs(REF):+8.3f}% {d:>20}", flush=True)
        except Exception as ex:
            print(f"{36:3d} {X:6.1f} {v:>8}   FAILED {type(ex).__name__}: {ex}", flush=True)
        finally:
            pn.NewtonSolver.__init__ = orig
print(f"\nreference alpha = {REF}")
