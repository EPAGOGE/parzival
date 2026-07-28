"""THE REAL GATE for the d2 -> d1 constraint change.

PASS: the alpha spread over N = 28/36/44/52 falls from 2.27328e-2 to below 2e-3 (10x), with
      every row carrying a true converged flag, ||F|| <= 1e-11, and c_l printed beside it.
      The sharpest single numbers are N=44 and N=52: post-fix the corner-functional channel
      has authority 0.00x there, so if those two come into line the cause is CONFIRMED and
      CLOSED.
INFORMATIVE FAILURE: dln q falls exactly as predicted (already verified solve-free, gains
      6.1x-39.8x, |dlnq| falling 153.8x over N=28..96) but the alpha spread stays at or
      above 1e-2.  That proves the corner functional had AUTHORITY but not ATTRIBUTION,
      closes the cause, and hands the next iteration XMAX as the only channel left above
      1e-3.
SECOND FAILURE SIGNATURE to check BEFORE reading anything as physics: the new row carries a
      pointwise 1/g whose first-node entry is ~N^2 (confirmed -- |d1|_1 also scales as N^4,
      only 3.7x smaller than |d2|_1).  If Newton degrades, compare cond(J) against the
      current 3.159e9 at N=36; an orders-of-magnitude jump means the 1/g has to be absorbed
      analytically (make Vt the unknown) rather than applied pointwise.
"""
import sys, pathlib, numpy as np, numpy.linalg as la
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
import importlib.util


def mod(name, fn):
    sp = importlib.util.spec_from_file_location(name, str(H / fn))
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m


pst = mod("pst", "polar_stability.py")
REF, CLS = -0.34240009, pst.CL_STAR
CWS = REF * CLS
NS = (28, 36, 44, 52)

print(f"{'cons':>5} {'N':>3} {'cnv':>4} {'||F||':>10} {'cond(J)':>10} {'alpha':>14} "
      f"{'vs ref':>9} {'d_cl':>9} {'diff':>10} {'steps':>13}")
out = {}
for v in ("d2", "d1"):
    got = []
    for N in NS:
        try:
            St, x, r, cl, cw, info = pst.converge_exact(N, constraint=v, strict=False)
        except Exception as ex:
            print(f"{v:>5} {N:3d}  RAISED {type(ex).__name__}: {str(ex)[:80]}", flush=True)
            continue
        if not np.isfinite(cl):
            print(f"{v:>5} {N:3d} {'NO':>4} {r:10.3e}   zero Newton steps", flush=True)
            continue
        A = St.A_exact(x); Ot, Bt = St.S.unpack(x[:-2])
        B, Cg = St.exact_B(Ot, Bt), St.exact_Cg()
        n = St.n
        J = np.zeros((n + 2, n + 2)); J[:n, :n] = A; J[:n, n:] = B; J[n:, :n] = Cg
        cj = float(la.cond(J))
        a = cw / cl
        d_l = (cl - CLS) / CLS
        d_w = (cw - CWS) / CWS
        if info["converged"] and r < 1e-11:
            got.append(a)
        print(f"{v:>5} {N:3d} {('yes' if info['converged'] else 'NO'):>4} {r:10.3e} "
              f"{cj:10.3e} {a:+14.8f} {100*(a-REF)/abs(REF):+8.3f}% {100*d_l:+8.3f}% "
              f"{100*(d_w-d_l):+9.4f}% {str(info['newton_steps'])[:13]:>13}", flush=True)
    out[v] = got

print()
for v in ("d2", "d1"):
    g = out.get(v, [])
    if len(g) >= 2:
        sp_ = max(g) - min(g)
        print(f"  {v}: {len(g)}/{len(NS)} converged, alpha spread = {sp_:.5e} "
              f"({'PASS' if sp_ < 2e-3 else 'above the 2e-3 gate'})")
    else:
        print(f"  {v}: only {len(g)} converged rows -- spread not meaningful")
if len(out.get("d2", [])) >= 2 and len(out.get("d1", [])) >= 2:
    s2 = max(out["d2"]) - min(out["d2"]); s1 = max(out["d1"]) - min(out["d1"])
    print(f"  improvement factor d2 -> d1: {s2/max(s1,1e-300):.2f}x")
print(f"\nreference alpha = {REF}   c_l* = {CLS:.8f}")
