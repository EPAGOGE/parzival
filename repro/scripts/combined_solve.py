"""THE COMBINED SOLVE: constraint='d1' AND eps_b -> 0, by continuation.

WHY BOTH AT ONCE. The d1 gate landed as the informative middle outcome: d1's converged
rows agree with EACH OTHER to 1.9e-3 (11.9x better than d2's 2.27e-2 spread) but sit at
-0.333, +2.5% from Chen-Hou. Combined with the eps_b ladder -- alpha swings -26.3%..+3.5%
around eps_b=1e-3 and passes through ~zero error exactly at the hard-coded value -- the
+0.085% 'canonical agreement' is two systematics CANCELLING: the d2 row error against the
eps_b domain bias. This run removes both and lets the discretisation say what it actually
converges to.

MECHANICS, each dictated by a measured failure mode:
  * d1 cold-fails at N=52 (zero steps) and stalls in the outer loop at N=36. FIX: solve d2
    cold first (4/4 known convergent), warm-start d1 from the d2 solution at the SAME N --
    identical unknown vector, only the constraint row differs.
  * eps_b below ~6e-4 cold-fails (zero steps at 1e-4/1e-5), and N=36/d2 has a measured
    FOLD at eps_b* ~ 5.1e-4. FIX: walk eps_b DOWN by warm continuation (x0 and alpha
    carried), with one midpoint retry on failure, then stop the ladder honestly.
  * The eps_b domain error is FIRST ORDER (opening angle pi/2 - 2 eps_b), so alpha(eps_b)
    should be linear near 0: extrapolate from the smallest converged rung, linear AND
    quadratic, and report both -- disagreement between them is the error bar.

CAVEAT recorded up front: the seed interpolation corrupts a band r < h/sin(eps_b) that
WIDENS as eps_b shrinks (cubic across theta's double zero). Warm continuation dodges the
seed for the FIELD, but the frozen axis column is re-pinned from the seed at each eps_b.
If alpha(eps_b) bends away from linear at small eps_b, that defect is the first suspect.
"""
import importlib.util
import json
import pathlib
import sys

import numpy as np

H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
OUT = pathlib.Path("/Users/epagogellc/parzival/boussinesq/runs/combined_d1_epsb.json")


def mod(name, fn):
    sp = importlib.util.spec_from_file_location(name, str(H / fn))
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


pst = mod("pst", "polar_stability.py")
REF = -0.34240009
CLS = pst.CL_STAR
CWS = REF * CLS

LADDER = [6e-4, 4e-4, 3e-4, 2e-4, 1.5e-4, 1e-4, 7e-5, 5e-5, 3e-5, 2e-5, 1e-5]
NS = (28, 36, 44, 52)

results = {}


def solve(N, eps_b, constraint, x0=None, alpha=None):
    # outer_steps=80: the theta=0.5 damped alpha map closes a gap at rate ~0.5/pass, so a
    # 1e-2 initial gap needs ~24 passes JUST to reach 1e-9 -- the default cap of 24 made
    # the N=28 d1 warm stage report NO at ||F||=7e-15 with the right alpha. Each warm pass
    # is a ~2-iteration Newton, so 80 is cheap insurance, not a cost.
    St, x, r, cl, cw, info = pst.converge_exact(
        N, eps_b=eps_b, constraint=constraint, x0=x0, alpha=alpha, strict=False,
        outer_steps=80)
    ok = bool(info.get("converged")) and np.isfinite(cl)
    return ok, St, x, r, cl, cw, info


def row(tag, N, e, ok, r, cl, cw, info):
    if not np.isfinite(cl):
        print(f"  {tag} N={N:2d} eps_b={e:9.2e}  FAILED zero Newton steps ||F||={r:.2e}",
              flush=True)
        return
    a = cw / cl
    d_l = (cl - CLS) / CLS
    d_w = (cw - CWS) / CWS
    print(f"  {tag} N={N:2d} eps_b={e:9.2e} {'ok ' if ok else 'NO '} ||F||={r:9.2e} "
          f"alpha={a:+.8f} vs_ref={100*(a-REF)/abs(REF):+7.3f}% d_cl={100*d_l:+7.3f}% "
          f"diff={100*(d_w-d_l):+8.4f}%", flush=True)


for N in NS:
    print(f"\n=== N = {N} " + "=" * 60, flush=True)
    branch = []

    # stage 1: d2 cold at eps_b = 1e-3 (the known-convergent anchor)
    ok, St, x, r, cl, cw, info = solve(N, 1e-3, "d2")
    row("d2", N, 1e-3, ok, r, cl, cw, info)
    if not ok:
        print(f"  N={N}: d2 anchor failed, skipping N", flush=True)
        continue

    # stage 2: d1 at eps_b = 1e-3, warm from d2 (same vector, only the row differs)
    ok, St, x1, r, cl, cw, info = solve(N, 1e-3, "d1", x0=x.copy(), alpha=cw / cl)
    row("d1", N, 1e-3, ok, r, cl, cw, info)
    if not ok:
        print(f"  N={N}: d1 warm-from-d2 did not converge, skipping N", flush=True)
        continue
    branch.append(dict(eps_b=1e-3, alpha=cw / cl, F=float(r), cl=float(cl)))
    xw, aw = x1.copy(), cw / cl

    # stage 3: continuation down the eps_b ladder
    last_good = 1e-3
    i = 0
    targets = list(LADDER)
    while i < len(targets):
        e = targets[i]
        ok, St, xn, r, cl, cw, info = solve(N, e, "d1", x0=xw.copy(), alpha=aw)
        row("d1", N, e, ok, r, cl, cw, info)
        if ok:
            branch.append(dict(eps_b=e, alpha=cw / cl, F=float(r), cl=float(cl)))
            xw, aw, last_good = xn.copy(), cw / cl, e
            i += 1
            continue
        mid = 0.5 * (last_good + e)
        if (last_good - e) / last_good < 0.15:
            print(f"  N={N}: step to {e:.2e} failed with nowhere to bisect -- "
                  f"branch ends at eps_b={last_good:.2e}", flush=True)
            break
        print(f"  N={N}: retrying via midpoint {mid:.2e}", flush=True)
        ok, St, xn, r, cl, cw, info = solve(N, mid, "d1", x0=xw.copy(), alpha=aw)
        row("d1", N, mid, ok, r, cl, cw, info)
        if ok:
            branch.append(dict(eps_b=mid, alpha=cw / cl, F=float(r), cl=float(cl)))
            xw, aw, last_good = xn.copy(), cw / cl, mid
            # do NOT advance i: retry the original target from closer in
        else:
            print(f"  N={N}: midpoint failed too -- FOLD or basin edge; "
                  f"branch ends at eps_b={last_good:.2e}", flush=True)
            break

    # stage 4: extrapolate to eps_b = 0
    es = np.array([b["eps_b"] for b in branch])
    al = np.array([b["alpha"] for b in branch])
    fit = {}
    if es.size >= 3:
        k = min(4, es.size)
        p1 = np.polyfit(es[-k:], al[-k:], 1)
        fit["linear"] = float(p1[-1])
        fit["slope"] = float(p1[0])
    if es.size >= 4:
        k = min(6, es.size)
        p2 = np.polyfit(es[-k:], al[-k:], 2)
        fit["quadratic"] = float(p2[-1])
    results[N] = dict(branch=branch, fit=fit)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    if fit:
        lin = fit.get("linear", float("nan"))
        quad = fit.get("quadratic", float("nan"))
        print(f"  N={N}: eps_b->0  linear={lin:+.8f} ({100*(lin-REF)/abs(REF):+.3f}%)  "
              f"quadratic={quad:+.8f} ({100*(quad-REF)/abs(REF):+.3f}%)  "
              f"[extrap err bar ~ {abs(lin-quad):.2e}]", flush=True)

print("\n" + "=" * 74)
print("SUMMARY  (d1 + eps_b -> 0)")
print(f"{'N':>4} {'reached eps_b':>14} {'alpha(min eps)':>15} {'lin extrap':>13} "
      f"{'quad extrap':>13} {'vs ref (lin)':>13}")
extr = []
for N in NS:
    if N not in results or not results[N]["branch"]:
        continue
    b = results[N]["branch"]
    f = results[N]["fit"]
    lin = f.get("linear", float("nan"))
    quad = f.get("quadratic", float("nan"))
    print(f"{N:4d} {b[-1]['eps_b']:14.2e} {b[-1]['alpha']:15.8f} {lin:13.8f} "
          f"{quad:13.8f} {100*(lin-REF)/abs(REF):+12.3f}%")
    if np.isfinite(lin):
        extr.append(lin)
if len(extr) >= 2:
    print(f"\n  N-spread of the eps_b->0 extrapolations: {max(extr)-min(extr):.3e}"
          f"   (d2/eps_b=1e-3 baseline was 2.27e-2; d1 alone was 1.9e-3 on 2 rows)")
print(f"  reference alpha = {REF}")
