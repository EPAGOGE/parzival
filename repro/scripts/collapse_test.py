"""THE N/XMAX COLLAPSE TEST -- the premortem's designated next measurement.

DECISION RULE, pre-registered before running:
  If matched-N/L pairs agree to better than the local slope of the error in r = N/XMAX,
  the error envelope is a function of N/L ALONE (Bernstein: rho - 1 ~ 2d/L), the object is
  analytic, the domain is simply TOO LONG for affordable N, and the optimum XMAX (~12-15)
  buys 3-11x error reduction for free.  If they do NOT agree, the surface is genuinely
  2-D and every alpha ever quoted must be alpha(N, XMAX) with no extrapolation claim.

MATCHED PAIRS (r = N/XMAX):
  (N=44, L=25, r=1.76)  vs  (N=36, L=20, r=1.80)   [both known: +2.263% vs +3.239%]
  (N=28, L=12, r=2.33)  vs  (N=36, L=15, r=2.40)
  (N=52, L=25, r=2.08)  vs  (N=36, L=18, r=2.00)
  (N=52, L=12, r=4.33)  vs  (N=36, L=10, r=3.60)   [loosest; check only]

FIVE NUMBERS PER POINT (never alpha alone -- alpha is the small difference of two larger
gauge errors; a good alpha beside a bad c_l is a cancellation, not an answer):
  alpha | d_cl = c_l/c_l* - 1 | open-system axis RMS | outer passes | d(alpha)/d(ln q)
The last is the REFERENCE-FREE gate: in the continuum alpha can only read q = THXX/WX^2
and d(alpha)/d(ln q) must be EXACTLY zero; measured nonzero and co-oscillating with the
error.  One extra solve at THXX*1.1 (dln q = ln 1.1), warm-started from the base solution.
THXX_REF is set as an INSTANCE attribute on each freshly constructed solver -- the _mod()
re-execution trap makes class-level monkeypatching silently vanish.

Fixed: constraint='d1', eps_b=1e-3 (like-for-like; the eps_b->0 axis deliberately NOT
mixed in). Damped outer loop, 80-pass cap, genuine convergence tests.
"""
import importlib.util, json, pathlib, sys, time
import numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
OUT = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad/collapse_test.json")
REF = -0.34240009


def mod(n, f):
    sp = importlib.util.spec_from_file_location(n, str(H / f))
    m = importlib.util.module_from_spec(sp); sys.modules[n] = m; sp.loader.exec_module(m); return m


pst = mod("pst", "polar_stability.py")
CLS = pst.CL_STAR


def solve(N, XMAX, thxx_scale=1.0, x0=None, a=None, theta=0.5, cap=80, tol=1e-11):
    """Damped outer alpha loop with THXX_REF scaled on the instance each pass."""
    hist = []
    for k in range(cap):
        St = pst.Stability(N, alpha=a, XMAX=XMAX, eps_b=1e-3, constraint="d1")
        St.S.THXX_REF = St.S.THXX_REF * thxx_scale        # instance attr; survives F()
        start = St.S.x0 if x0 is None else x0
        x, f, r, taken = pst.newton_exact(St, start, tol=tol)
        if taken == 0:
            return None, dict(fail="zero_steps", F=float(r), passes=k)
        cl, cw = float(x[-2]), float(x[-1])
        an = cw / cl
        x0 = x; hist.append(an)
        if a is not None and abs(an - a) < 1e-9 and r < tol:
            od = St.S.open_residual(x)
            return x, dict(alpha=an, cl=cl, F=float(r), passes=k + 1,
                           axis_rms=float(od["axis_rms"]), conv=True)
        a = an if a is None else a + theta * (an - a)
    return None, dict(fail="outer_cap", passes=cap, hist=hist[-4:])


JOBS = [(36, 12.0), (36, 18.0), (36, 10.0), (36, 30.0),
        (28, 12.0), (28, 15.0), (52, 15.0), (52, 12.0)]
results = {}
print(f"{'N':>3} {'XMAX':>5} {'r=N/L':>6} {'alpha':>13} {'vs ref':>8} {'d_cl':>8} "
      f"{'axisRMS':>9} {'pass':>4} {'dA/dlnq':>9} {'secs':>6}", flush=True)
for N, L in JOBS:
    t0 = time.time()
    x, info = solve(N, L)
    key = f"N{N}_L{L:g}"
    if x is None:
        print(f"{N:3d} {L:5.0f}   FAILED {info}", flush=True)
        results[key] = info; OUT.write_text(json.dumps(results, indent=1)); continue
    # q-derivative: one warm extra solve at THXX*1.1
    xq, iq = solve(N, L, thxx_scale=1.1, x0=x.copy(), a=info["alpha"])
    dAdlnq = (iq["alpha"] - info["alpha"]) / np.log(1.1) if xq is not None else float("nan")
    d_cl = info["cl"] / CLS - 1.0
    secs = time.time() - t0
    results[key] = dict(**info, d_cl=d_cl, dAdlnq=dAdlnq, secs=secs)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"{N:3d} {L:5.0f} {N/L:6.2f} {info['alpha']:+13.8f} "
          f"{100*(info['alpha']-REF)/abs(REF):+7.3f}% {100*d_cl:+7.2f}% "
          f"{info['axis_rms']:9.2e} {info['passes']:4d} {dAdlnq:+9.4f} {secs:6.0f}",
          flush=True)

# ---- the matched-pair verdict, folding in the known points -----------------
known = {"N44_L25": -0.33465017, "N36_L20": -0.33131000, "N36_L15": -0.33840790,
         "N52_L25": -0.35563478, "N36_L25": -0.34936411}
# (N44/N52/N36_L25 at eps_b=1e-3 d1 from constraint_real; N36_L15/L20 from pm xmax_fixed)
allpts = {k: v["alpha"] for k, v in results.items() if isinstance(v, dict) and "alpha" in v}
allpts.update(known)
PAIRS = [("N44_L25", "N36_L20"), ("N28_L12", "N36_L15"),
         ("N52_L25", "N36_L18"), ("N52_L12", "N36_L10")]
print("\nMATCHED-PAIR VERDICT (agree => envelope is f(N/L) alone):", flush=True)
for a_, b_ in PAIRS:
    if a_ in allpts and b_ in allpts:
        da = 100 * (allpts[a_] - REF) / abs(REF); db = 100 * (allpts[b_] - REF) / abs(REF)
        print(f"  {a_:9s} {da:+7.3f}%   {b_:9s} {db:+7.3f}%   gap {abs(da-db):5.3f} pts",
              flush=True)
print(f"\nreference alpha = {REF}   c_l* = {CLS:.8f}", flush=True)
