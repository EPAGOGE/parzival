"""Extend the d1 + eps_b->0 ladder to N=60 and N=68 -- the third point in each parity
family, which is what turns the two-sided compiler from provisional (shared-rate assumed,
increment ratio -8.77 says that assumption is strained) into rigorous (each family's
falloff rate fit INDEPENDENTLY, then tested for a common limit).

Families by N mod 16:  {28,44,60} = class 12 (the 'above' end);  {36,52,68} = class 4
(the 'below' end).  A short eps_b ladder suffices: alpha(eps_b) was linear to 1e-7, so
5 rungs pin the eps_b->0 extrapolation.
"""
import importlib.util, json, pathlib, sys
import numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")


def mod(n, f):
    sp = importlib.util.spec_from_file_location(n, str(H / f))
    m = importlib.util.module_from_spec(sp); sys.modules[n] = m; sp.loader.exec_module(m); return m


pst = mod("pst", "polar_stability.py")
REF = -0.34240009
LADDER = [6e-4, 3e-4, 1e-4, 3e-5, 1e-5]
OUT = pathlib.Path(H / "runs" / "extend_6070.json")
prior = {28: -0.33133082, 36: -0.35108229, 44: -0.33156463, 52: -0.34903108}
results = {}

for N in (60, 68):
    print(f"\n=== N={N} " + "=" * 50, flush=True)
    # d2 anchor -> warm d1 at eps_b=1e-3
    St, x, r, cl, cw, info = pst.converge_exact(N, eps_b=1e-3, constraint="d2",
                                                strict=False, outer_steps=80)
    if not (info["converged"] and np.isfinite(cl)):
        print(f"  d2 anchor failed ||F||={r:.2e}", flush=True); continue
    St, x, r, cl, cw, info = pst.converge_exact(N, eps_b=1e-3, constraint="d1",
                                                x0=x.copy(), alpha=cw/cl,
                                                strict=False, outer_steps=80)
    if not (info["converged"] and np.isfinite(cl)):
        print(f"  d1 warm failed ||F||={r:.2e}", flush=True); continue
    branch = [(1e-3, cw/cl)]
    print(f"  d1 eps_b=1.0e-03 alpha={cw/cl:+.8f} ||F||={r:.2e}", flush=True)
    xw, aw = x.copy(), cw/cl
    for e in LADDER:
        St, xn, rn, cln, cwn, info = pst.converge_exact(N, eps_b=e, constraint="d1",
                                                        x0=xw.copy(), alpha=aw,
                                                        strict=False, outer_steps=80)
        if not (info["converged"] and np.isfinite(cln)):
            print(f"  d1 eps_b={e:.1e} FAILED ||F||={rn:.2e}", flush=True); continue
        branch.append((e, cwn/cln)); xw, aw = xn.copy(), cwn/cln
        print(f"  d1 eps_b={e:.1e} alpha={cwn/cln:+.8f} vs_ref={100*(cwn/cln-REF)/abs(REF):+.3f}% "
              f"||F||={rn:.2e}", flush=True)
    es = np.array([b[0] for b in branch]); al = np.array([b[1] for b in branch])
    lin = float(np.polyfit(es[-4:], al[-4:], 1)[-1])
    quad = float(np.polyfit(es[-5:], al[-5:], 2)[-1]) if es.size >= 5 else lin
    results[N] = dict(lin=lin, quad=quad)
    print(f"  N={N}: eps_b->0 lin={lin:+.8f} ({100*(lin-REF)/abs(REF):+.3f}%) "
          f"quad={quad:+.8f}  [err {abs(lin-quad):.1e}]", flush=True)
    OUT.write_text(json.dumps({str(k): v for k, v in results.items()}, indent=1))

# ---- the rate-matched compiler on all SIX points -----------------------
allpts = dict(prior)
for N, v in results.items():
    allpts[N] = v["lin"]
print("\n" + "=" * 62)
print("RATE-MATCHED TWO-SIDED COMPILER (independent rate per family)")
from scipy.optimize import least_squares
fams = {"above {28,44,60}": [28, 44, 60], "below {36,52,68}": [36, 52, 68]}
limits = {}
for name, Ns in fams.items():
    Ns = [n for n in Ns if n in allpts]
    if len(Ns) < 3:
        print(f"  {name}: only {len(Ns)} pts, need 3"); continue
    Na = np.array(Ns, float); ya = np.array([allpts[n] for n in Ns])
    def res(p): return p[0] + p[1]*p[2]**Na - ya
    s = least_squares(res, [-0.342, ya[0]+0.342, 0.95], bounds=([-0.4, -5, 0.5], [-0.28, 5, 1.2]))
    limits[name] = s.x[0]
    print(f"  {name}: alpha_inf={s.x[0]:+.7f} ({100*(s.x[0]-REF)/abs(REF):+.3f}%) "
          f"rate={s.x[2]:.4f} rms={np.sqrt(np.mean(s.fun**2)):.1e}", flush=True)
if len(limits) == 2:
    v = list(limits.values())
    print(f"\n  DO THE TWO ENDS MEET?  gap between family limits = {abs(v[0]-v[1]):.2e}")
    print(f"  mean of the two limits = {np.mean(v):+.7f}  ({100*(np.mean(v)-REF)/abs(REF):+.3f}% from Chen-Hou)")
    print(f"  -> if the gap is small AND the mean ~ -0.34240, the two ends BIND at Chen-Hou.")
print(f"\n  reference alpha = {REF}", flush=True)
