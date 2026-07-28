"""SEED-INDEPENDENCE RUNG -- closing the last shared constant.

shared_constant_audit flagged: EVERY run of this campaign, on every rung of the
final ladder, seeds its interior field from the Chen-Hou interpolated profile
(seed='chenhou-interp').  Cross-configuration agreement is structurally blind to a
constant every configuration shares.  This rung replaces the interior seed with a
PURELY ANALYTIC profile carrying no Chen-Hou data at all -- only the corner limits
(A -> wx cos b, B -> thxx/2 cos^2 b, which are the gauge targets, not the profile)
times a decay of arbitrary length L.  If alpha lands on the same value from a seed
of different provenance, seed lineage is exonerated as an alpha axis.
GATE: |alpha(analytic) - alpha(chenhou)| < 1e-6 for at least one L, and no L that
converges lands anywhere else."""
import importlib.util, pathlib, sys, time
import numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
spec = importlib.util.spec_from_file_location("pc", str(H / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(spec); sys.modules["pc"] = pc; spec.loader.exec_module(pc)

CFG = dict(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12), Nb=36, eps_b=1e-4)
REF_ALPHA = -0.34471229          # this config, Chen-Hou-seeded (gate A, branch_hunt)

def outer_loop(S, theta=0.5, outer=80, tol=1e-11):
    """converge()'s damped alpha loop, but on a solver whose seed we control."""
    a, z0, hist = None, None, []
    for k in range(outer):
        if a is not None:
            S.set_alpha(a)
        z, f, r, taken = S.newton(z0=z0, tol=tol)
        if taken == 0:
            return None, r, dict(converged=False, reason="zero_steps", passes=k+1)
        cl, cw = float(z[-2]), float(z[-1]); an = cw / cl; z0 = z; hist.append(an)
        if a is not None and abs(an - a) < 1e-9 and r < tol:
            return an, r, dict(converged=True, cl=cl, h_id=S.h_id(z), passes=k+1)
        a = an if a is None else a + theta * (an - a)
    return None, r, dict(converged=False, reason="outer_cap", passes=outer)

print(f"{'seed':>22} {'alpha':>13} {'vs chenhou':>11} {'h_id':>11} {'||F||':>9} {'passes':>6} {'secs':>5}", flush=True)
print(f"{'chenhou-interp (ref)':>22} {REF_ALPHA:+13.8f} {0.0:11.1e} {'-1.06e-03':>11} {'-':>9} {'-':>6} {'-':>5}", flush=True)
results = []
for L in (1.0, 2.0, 4.0, 8.0):
    t0 = time.time()
    S = pc.CornerRegSolver(**CFG)
    xi = S.x[:, None]; cb = np.cos(S.b)[None, :]
    # PURE ANALYTIC seed: corner limit x decay.  No Chen-Hou field anywhere.
    S.A0 = S.wx * cb * np.exp(-xi / L)
    S.B0 = 0.5 * S.thxx * cb ** 2 * np.exp(-xi / L)
    for k in range(1, S.K):                      # duplicated interface nodes equal
        S.A0[S.lefts[k], :] = S.A0[S.rights[k-1], :]
        S.B0[S.lefts[k], :] = S.B0[S.rights[k-1], :]
    S.P0 = S._p_seed(S.A0)
    a, r, info = outer_loop(S)
    if a is None:
        print(f"{'analytic L=%.0f' % L:>22}  NOT CONVERGED  {info.get('reason')} ||F||={r:.1e}  secs={time.time()-t0:.0f}", flush=True)
        continue
    results.append((L, a))
    print(f"{'analytic L=%.0f' % L:>22} {a:+13.8f} {a-REF_ALPHA:+11.2e} {info['h_id']:+11.2e} "
          f"{r:9.1e} {info['passes']:>6} {time.time()-t0:5.0f}", flush=True)

print("", flush=True)
if not results:
    print("VERDICT: no analytic seed converged -- seed independence NOT demonstrated "
          "(basin too small for a crude seed; not evidence of dependence)", flush=True)
else:
    worst = max(abs(a - REF_ALPHA) for _, a in results)
    spread = max(a for _, a in results) - min(a for _, a in results)
    print(f"VERDICT: {len(results)} analytic seed(s) converged; worst |dalpha| vs "
          f"chenhou = {worst:.2e}; spread among analytic seeds = {spread:.2e}", flush=True)
    print(f"  GATE (1e-6): {'PASS -- seed lineage exonerated as an alpha axis' if worst < 1e-6 else 'FAIL -- seed provenance is LIVE'}", flush=True)
print("done", flush=True)
