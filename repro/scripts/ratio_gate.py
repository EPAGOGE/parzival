"""GATE the ratio constraint before believing any alpha from it.

The ratio constraint is the FIRST that couples c_l, so the full Newton Jacobian
J = [[A, B],[Cg, Cc]] now has a nonzero lower-right block.  If that block or the modified
Cg is wrong, Newton converges to a tiny ||F|| at the WRONG point -- exactly the failure
class this project has been bitten by.  So:

  1. FD-check the ENTIRE bordered J (A, B, Cg, Cc) against a finite-difference of the full
     residual F(x) including the c-derivatives.  This is the check that matters.
  2. Confirm parse/convergence at one N, and read what the ratio constraint DOES:
     it should drive the c_l* violation to ~0 by construction (that's its definition),
     and alpha to whatever self-consistent-c_l implies.  Report alpha vs Chen-Hou.
"""
import importlib.util, pathlib, sys
import numpy as np, numpy.linalg as la
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")


def mod(n, f):
    sp = importlib.util.spec_from_file_location(n, str(H / f))
    m = importlib.util.module_from_spec(sp); sys.modules[n] = m; sp.loader.exec_module(m); return m


pst = mod("pst", "polar_stability.py")

N = 28
print(f"1. FULL-JACOBIAN FD CHECK for constraint='ratio' at N={N}")
St = pst.Stability(N, constraint="ratio")
S = St.S
x = S.x0.copy()
x[:-2] += 1e-3 * np.random.default_rng(0).standard_normal(x.size - 2)   # off any special point
n = St.n
# analytic bordered J
A = St.A_exact(x); Ot, Bt = S.unpack(x[:-2])
B = St.exact_B(Ot, Bt); Cg = St.exact_Cg(cl=float(x[-2])); Cc = St.exact_Cc(Ot)
Jan = np.zeros((n + 2, n + 2))
Jan[:n, :n] = A; Jan[:n, n:] = B; Jan[n:, :n] = Cg; Jan[n:, n:] = Cc
# FD of the full residual (field-rows + g1,g2) wrt every unknown incl c_l,c_w
f0, _, _ = S.F(x)
h = 1e-7 * max(la.norm(x), 1.0)
cols = list(np.random.default_rng(1).choice(n, size=5, replace=False)) + [n, n + 1]
print(f"   {'col':>6} {'kind':>10} {'rel err':>12}")
worst = 0.0
for j in cols:
    e = np.zeros(x.size); e[j] = h
    fp, _, _ = S.F(x + e); fm, _, _ = S.F(x - e)
    num = (fp - fm) / (2 * h)
    err = la.norm(num - Jan[:, j]) / max(la.norm(num), 1e-300)
    worst = max(worst, err)
    kind = "c_l" if j == n else ("c_w" if j == n + 1 else "field")
    print(f"   {j:6d} {kind:>10} {err:12.3e}")
print(f"   worst = {worst:.3e}  -> {'EXACT' if worst < 1e-5 else 'MISMATCH -- do not run ratio'}\n")

print("2. CONVERGE and read the ratio constraint's effect (warm from d2):")
St2, xd2, r2, cl2, cw2, i2 = pst.converge_exact(N, constraint="d2", strict=False, outer_steps=80)
CLS = 2.0 * S.THXX_REF / S.WX_REF
print(f"   d2:    alpha={cw2/cl2:+.8f} c_l={cl2:.6f} c_l* viol={100*(cl2-CLS)/CLS:+.3f}% ||F||={r2:.2e}")
St3, xr, rr, clr, cwr, ir = pst.converge_exact(N, constraint="ratio", x0=xd2.copy(),
                                               alpha=cw2/cl2, strict=False, outer_steps=80)
if np.isfinite(clr):
    print(f"   ratio: alpha={cwr/clr:+.8f} c_l={clr:.6f} c_l* viol={100*(clr-CLS)/CLS:+.3f}% "
          f"||F||={rr:.2e} converged={ir['converged']}")
    print(f"          vs Chen-Hou -0.34240009: {100*(cwr/clr+0.34240009)/0.34240009:+.3f}%")
    od = S.open_residual(xr) if hasattr(S, "open_residual") else None
    if od:
        print(f"          open-system residual = {od['open_rms']:.2e} (||F|| blind-check)")
else:
    print(f"   ratio: did NOT converge (||F||={rr:.2e})")
