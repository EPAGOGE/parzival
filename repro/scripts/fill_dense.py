"""Fill in-ceiling resolutions to (a) TEST the N mod 16 parity hypothesis by sampling the
never-measured classes, (b) give 3+ points per family for rate-matched extrapolation, and
(c) densely sample the log-periodic curve.

Prior (28,36,44,52) are class 12,4,12,4 mod 16.  New N:
   16->0   20->4   24->8   32->0   40->8   48->0
so this adds the UNSAMPLED classes 0 and 8, plus a third class-4 point (20).  If the sign
is truly locked to N mod 16, classes 0 and 8 carry their own offsets; if 16/24/32/40/48
instead lie on a smooth curve through 28..52, the parity story is refuted.  All <= 48, so
within the dense-direct Newton ceiling (~52).
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
OUT = pathlib.Path(H / "runs" / "fill_parity.json")
results = {}

for N in (18, 22, 26, 30, 34, 38, 42, 46, 50):
    cls = N % 16
    print(f"\n=== N={N}  (class {cls} mod 16) " + "=" * 40, flush=True)
    try:
        St, x, r, cl, cw, info = pst.converge_exact(N, eps_b=1e-3, constraint="d2",
                                                    strict=False, outer_steps=80)
        if not (info["converged"] and np.isfinite(cl)):
            print(f"  d2 anchor failed ||F||={r:.2e}", flush=True); continue
        St, x, r, cl, cw, info = pst.converge_exact(N, eps_b=1e-3, constraint="d1",
                                                    x0=x.copy(), alpha=cw/cl,
                                                    strict=False, outer_steps=80)
        if not (info["converged"] and np.isfinite(cl)):
            print(f"  d1 warm failed ||F||={r:.2e}", flush=True); continue
        branch = [(1e-3, cw/cl)]; xw, aw = x.copy(), cw/cl
        print(f"  d1 eps_b=1.0e-03 alpha={cw/cl:+.8f} ||F||={r:.2e}", flush=True)
        for e in LADDER:
            St, xn, rn, cln, cwn, info = pst.converge_exact(N, eps_b=e, constraint="d1",
                                                            x0=xw.copy(), alpha=aw,
                                                            strict=False, outer_steps=80)
            if not (info["converged"] and np.isfinite(cln)):
                print(f"  eps_b={e:.1e} FAILED ||F||={rn:.2e}", flush=True); continue
            branch.append((e, cwn/cln)); xw, aw = xn.copy(), cwn/cln
            print(f"  eps_b={e:.1e} alpha={cwn/cln:+.8f} vs_ref={100*(cwn/cln-REF)/abs(REF):+.3f}%",
                  flush=True)
        es = np.array([b[0] for b in branch]); al = np.array([b[1] for b in branch])
        if es.size >= 4:
            lin = float(np.polyfit(es[-4:], al[-4:], 1)[-1])
            results[N] = dict(cls=cls, lin=lin)
            print(f"  N={N}: eps_b->0 = {lin:+.8f}  ({100*(lin-REF)/abs(REF):+.3f}%)  class {cls}",
                  flush=True)
            OUT.write_text(json.dumps({str(k): v for k, v in results.items()}, indent=1))
    except Exception as ex:
        print(f"  N={N} RAISED {type(ex).__name__}: {ex}", flush=True)

print("\n" + "=" * 60 + "\nPARITY STRUCTURE (this fill)")
for N in sorted(results):
    v = results[N]
    print(f"  N={N:3d} class {v['cls']:2d}: alpha={v['lin']:+.8f} ({100*(v['lin']-REF)/abs(REF):+.3f}%)",
          flush=True)
print(f"\n  reference alpha = {REF}", flush=True)
