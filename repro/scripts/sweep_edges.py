"""THE TWO KNOBS WE NEVER TURNED.

XMAX has been 25.0 and the beta-endpoint offset eps_b has been 1e-3 for the entire
construction. Section 34's Nb sweep held eps_b FIXED, so it structurally could not see
any error the offset causes -- refining Nb does not refine an offset away. And three
independent mathlit reports predict the far-field truncation, not the resolution, is what
moves alpha (2310.13770: in the co-exploding frame the non-symmetry eigenvalues carry O(1)
finite-DOMAIN oscillation while resolution is irrelevant; 2007.11828 sec 2: truncation and
resolution error must be BALANCED, a fixed domain with growing n is asymptotically wrong).

Recorded per run, all from the SLICE basis (polar_zeros), never from ambient P@A:
    alpha, ||F||                    -- the live blocker
    ||P|| = 1/sin theta_min          -- the real obliqueness number, not cond(Cg B)
    real zeros in (-2, 8)            -- the discrete spectrum, by compression not projection
    W_e max|Re|                      -- how far from asymptotically-constant this XMAX is
"""
import sys, pathlib, json, numpy as np, scipy.linalg as la
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
import importlib.util


def mod(name, fn):
    sp = importlib.util.spec_from_file_location(name, str(H / fn))
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m


pst = mod("pst", "polar_stability.py")
pz = mod("pz", "polar_zeros.py")

REF = -0.34240009
OUT = pathlib.Path("/Users/epagogellc/parzival/boussinesq/runs/sweep_edges.json")


def one(N, XMAX, eps_b):
    St, x, r, cl, cw = pst.converge_exact(N, XMAX=XMAX, eps_b=eps_b)
    _, _, cCB, A, B, Cg = St.spectrum(x)
    M, M_orth, Z = pz.compress(A, B, Cg)
    pn_, smin = pz.proj_norm(B, Cg, Z)
    w = la.eigvals(M)
    rl = np.sort(w[np.abs(w.imag) < 1e-6].real)[::-1]
    rl = [float(z) for z in rl if -2.5 < z < 9.0]
    th, h, ext = pz.essential_numerical_range(St, x, ks=np.concatenate(
        [-np.logspace(2, -2, 20), [0.0], np.logspace(-2, 2, 20)]), thetas=32)
    return dict(N=N, XMAX=XMAX, eps_b=eps_b, dim=St.n, F=float(r),
                alpha=float(cw / cl), cl=float(cl), cw=float(cw),
                condCB=float(cCB), Pnorm=float(pn_), theta_deg=float(np.degrees(np.arcsin(smin))),
                dep_A=pz.departure_from_normality(A), dep_M=pz.departure_from_normality(M),
                real_zeros=rl, We_maxRe=float(h[0]), We_minRe=float(-h[len(th) // 2]))


rows = []
jobs = ([(36, X, 1e-3) for X in (15.0, 20.0, 25.0, 32.0, 40.0)]
        + [(36, 25.0, e) for e in (1e-2, 1e-4, 1e-5)])
print(f"{'N':>3} {'XMAX':>6} {'eps_b':>7} {'||F||':>10} {'alpha':>13} {'vs ref':>8} "
      f"{'||P||':>10} {'th(deg)':>9} {'W_e Re':>9}  real zeros", flush=True)
for (N, X, e) in jobs:
    try:
        d = one(N, X, e)
    except Exception as ex:
        print(f"{N:3d} {X:6.1f} {e:7.0e}   FAILED  {type(ex).__name__}: {ex}", flush=True)
        continue
    rows.append(d)
    print(f"{d['N']:3d} {d['XMAX']:6.1f} {d['eps_b']:7.0e} {d['F']:10.3e} {d['alpha']:+13.8f} "
          f"{100*(d['alpha']-REF)/abs(REF):+7.3f}% {d['Pnorm']:10.3e} {d['theta_deg']:9.2e} "
          f"{d['We_maxRe']:9.2e}  "
          + " ".join(f"{z:+.4f}" for z in d["real_zeros"][:6]), flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
print(f"\nreference alpha = {REF}\nwrote {OUT}")
