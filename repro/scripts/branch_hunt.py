"""BRANCH HUNT: machinery for the unstable Boussinesq branches alpha_1..3.

Targets (DeepMind 2509.14185 Fig 2f, boldface-validated digits only):
    alpha_1 = -0.4168236   alpha_2 = -0.4439811   alpha_3 = -0.4578230
No non-PINN method has confirmed any of them.  Premortem requirements implemented here:

1. ALPHA AS A ROOT, NOT A FIXED-POINT ITERATE.  h(a) = c_w/c_l(a) - a, where each
   evaluation freezes the substitution exponent a0 = a and converges the field+(cl,cw)
   Newton.  The stable branch is h's root at -0.3424; an UNSTABLE branch is a root the
   damped fixed-point map repels from -- but a SECANT on h does not care about stability.
2. WARM-FIELD CONTINUATION.  No reference profile exists for n >= 1; the field is walked
   along a, each rung warm-started from the last.
3. DEFLATION ON THE SPARSE SOLVER.  The n>=1 profiles are DIFFERENT field roots at the
   same a.  Deflated residual G = m(z) F(z), m = prod(||z-r_i||^-p + 1); the Jacobian is
   sparse-plus-rank-one (m J + F grad_m^T), solved by SHERMAN-MORRISON over the sparse LU:
       (mJ + f g^T)^-1 rhs = (1/m)[y - (g^T y)/(m + g^T w) * w],  y = J^-1 rhs, w = J^-1 f.
   Convergence judged on the UNDEFLATED ||F|| (the shift=1 lesson).

GATES before any hunt is believed:
  A. The secant, started badly (a = -0.30), must recover the known root -0.3424x.
  B. The deflated Newton, started AT the known root + 1%, must be repelled (not converge
     back), while the undeflated one returns.  (Ported from polar_deflate_gate.)
This file runs gate A, then the LANDSCAPE WALK a: -0.3424 -> -0.43 recording h(a) and
saving each rung's field (the deflation anchors for the hunt proper).

Config: (16,40,12)/Nb=36/eps_b=1e-4 -- fast rungs; branch separation 21.7% and the
alpha_1-vs-CHL-stage-2 gap (0.0085) both >> the ~2e-3 eps-bias at this config, so the
landscape is meaningful; precision configs come after existence.
"""
import importlib.util, pathlib, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
OUT = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad/hunt_fields")
OUT.mkdir(exist_ok=True)


def mod(n, f):
    sp_ = importlib.util.spec_from_file_location(n, str(H / f))
    m = importlib.util.module_from_spec(sp_)
    sys.modules[n] = m
    sp_.loader.exec_module(m)
    return m


pc = mod("pc", "polar_cornerreg.py")
CFG = dict(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12), Nb=36, eps_b=1e-4)
REF0 = -0.34240009


class H_of_a:
    """h(a) = cw/cl(a) - a with a persistent warm field."""

    def __init__(self):
        self.S = pc.CornerRegSolver(**CFG, alpha=None)
        self.z = None
        self.evals = []

    def __call__(self, a, tol=1e-11):
        self.S.set_alpha(a)
        z, f, r, taken = self.S.newton(z0=None if self.z is None else self.z.copy(),
                                       tol=tol)
        if taken == 0 or r > 1e-9:
            return None, dict(fail=True, F=float(r), taken=taken)
        self.z = z
        cl, cw = float(z[-2]), float(z[-1])
        h = cw / cl - a
        self.evals.append((a, h, cl))
        return h, dict(F=float(r), cl=cl, cw=cw, taken=taken)


def secant(hf, a0, a1, tol=1e-10, itmax=25, verbose=True):
    h0, i0 = hf(a0)
    h1, i1 = hf(a1)
    if h0 is None or h1 is None:
        return None, "seed eval failed"
    for it in range(itmax):
        if abs(h1 - h0) < 1e-15:
            return None, "flat secant"
        a2 = a1 - h1 * (a1 - a0) / (h1 - h0)
        h2, i2 = hf(a2)
        if h2 is None:
            return None, f"eval failed at a={a2:.6f}"
        if verbose:
            print(f"    secant it{it}: a={a2:+.8f} h={h2:+.3e} cl={i2['cl']:.6f}",
                  flush=True)
        if abs(h2) < tol:
            return a2, i2
        a0, h0, a1, h1 = a1, h1, a2, h2
    return None, "itmax"


# --------------- deflated Newton via Sherman-Morrison ------------------------
def _m_grad(z, centres, p=2.0, shift=1.0):
    m = 1.0
    acc = np.zeros_like(z)
    for r in centres:
        d = z - r
        nd = float(np.linalg.norm(d))
        if nd < 1e-14:
            return np.inf, acc
        mi = nd ** (-p) + shift
        m *= mi
        acc += (-p * nd ** (-p - 2.0) / mi) * d
    return m, m * acc


def deflated_newton(S, z0, centres, steps=60, tol=1e-11, verbose=False):
    z = z0.copy()
    f = S.residual(z)
    r = np.linalg.norm(f) / np.sqrt(f.size)
    prev_g, taken = np.inf, 0
    for it in range(steps):
        m, gm = _m_grad(z, centres)
        if not np.isfinite(m):
            return z, r, taken, False
        J = S.jacobian(z)
        lu = spla.splu(J)
        y = lu.solve(-m * f)
        w = lu.solve(f)
        denom = m + float(gm @ w)
        dz = (y - (float(gm @ y) / denom) * w) / m if abs(denom) > 1e-300 else y / m
        lam, best = 1.0, None
        for _ in range(14):
            zt = z + lam * dz
            mt, _ = _m_grad(zt, centres)
            if not np.isfinite(mt):
                lam *= 0.5
                continue
            ft = S.residual(zt)
            gt = mt * np.linalg.norm(ft) / np.sqrt(ft.size)
            if gt < prev_g:
                best = (zt, ft, np.linalg.norm(ft) / np.sqrt(ft.size), gt)
                break
            lam *= 0.5
        if best is None:
            break
        z, f, r, prev_g = best
        taken += 1
        if verbose and it % 5 == 0:
            print(f"      d-it{it}: ||F||={r:.3e}", flush=True)
        if r < tol:
            return z, r, taken, True
    return z, r, taken, r < tol


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("GATE A: secant from a bad start must recover the known stable root", flush=True)
    hf = H_of_a()
    t0 = time.time()
    a_star, info = secant(hf, -0.30, -0.32)
    if a_star is None:
        print(f"  GATE A FAILED: {info}", flush=True)
        sys.exit(1)
    print(f"  GATE A: recovered a = {a_star:+.8f} (known ~{REF0} + eps-bias ~ -0.345 at "
          f"this config)  cl={info['cl']:.6f}  secs={time.time()-t0:.0f}", flush=True)

    print("\nLANDSCAPE WALK: a from the stable root toward alpha_1 = -0.4168", flush=True)
    print(f"{'a':>12} {'h(a)':>12} {'c_l':>10} {'||F||':>9} {'steps':>5}", flush=True)
    walk = np.linspace(a_star, -0.43, 12)
    for k, a in enumerate(walk):
        h, i = hf(a)
        if h is None:
            print(f"{a:+12.6f}   FIELD NEWTON FAILED {i}", flush=True)
            break
        np.savez(OUT / f"rung_{k:02d}_a{a:+.6f}.npz", z=hf.z, a=a, h=h, cl=i["cl"])
        print(f"{a:+12.6f} {h:+12.3e} {i['cl']:10.6f} {i['F']:9.2e} {i['taken']:>5}",
              flush=True)
    print("\nfields saved to hunt_fields/ -- these are the deflation anchors.", flush=True)
    print("Read the h(a) column: a sign change or local structure away from the stable "
          "root is a second branch announcing itself; a smooth monotone h means the "
          "n>=1 roots are OFF this field branch and the deflated multistart at "
          "a ~ -0.4168 is the next move.", flush=True)
