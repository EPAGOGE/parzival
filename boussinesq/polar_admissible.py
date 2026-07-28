"""
IS THE UNSTABLE EIGENFUNCTION ADMISSIBLE IN CHEN-HOU'S FUNCTION SPACE?

THE QUESTION THIS SETTLES
-------------------------
The constrained spectrum shows an unstable complex pair (+0.425 +- 0.605i at N=36), while
Chen-Hou report STABLE and Liu reports eigenvalues with negative real part. Eigenvalues do
not depend on the norm, so a weighted-norm argument cannot reconcile that. But they DO
depend on the SPACE, and Chen-Hou state their stability on a RESTRICTED one
(`eq:normal_vanish`): perturbations must vanish QUADRATICALLY at the origin,

    omega = O(|x|^2),   theta_x = O(|x|^2),   theta_y = O(|x|^2)

Note this is FASTER than the PROFILE's own vanishing -- the profile has `Om ~ w_x(0) y1`,
i.e. LINEAR. So the admissible perturbations are a strict subspace, and an eigenfunction
that vanishes only linearly is simply not an admissible perturbation for them.

If the unstable mode vanishes linearly, the discrepancy with both references is EXPLAINED
and no contradiction remains: it is an eigenvalue of a larger space than they consider.
If it vanishes quadratically or faster, the discrepancy is REAL and has to be confronted.

MEASUREMENT
-----------
In substituted variables the corner factor `e^(p xi) -> 1`, so the substituted and raw
fields share their leading power at the corner. Fit `|dOt| ~ xi^p` and `|dBt| ~ xi^q` on a
window just outside the corner node, on the wall ray where the profile is largest.

    profile:              Ot ~ xi^1,  Bt ~ xi^2       (measured, POLAR_SPEC section 16)
    admissible pert.:     dOt ~ xi^2 or faster
    inadmissible pert.:   dOt ~ xi^1  (same order as the profile)
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import numpy.linalg as la

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(HERE / fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def corner_power(xi, prof, lo_idx=1, hi_idx=8):
    """Leading power of |prof| in xi near the corner, from a log-log fit on nodes
    [lo_idx, hi_idx). Node 0 is the corner itself and is pinned."""
    x = xi[lo_idx:hi_idx]
    v = np.abs(prof[lo_idx:hi_idx])
    m = (v > 0) & (x > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    c = np.polyfit(np.log(x[m]), np.log(v[m]), 1)
    fit = np.polyval(c, np.log(x[m]))
    y = np.log(v[m])
    r2 = 1 - ((y - fit) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300)
    return float(c[0]), float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=36)
    ap.add_argument("--nmodes", type=int, default=4)
    a_ = ap.parse_args()

    pst = _mod("pst", "polar_stability.py")
    St, x, r, cl, cw = pst.converge(a_.N)
    w, V, cCB, A, B, Cg = St.spectrum(x)
    C = St.C
    n2 = C.nx * C.nb

    print(f"N={a_.N}  ||F||={r:.2e}  c_l={cl:.6f}  alpha={cw/cl:.6f}\n")
    print("PROFILE corner powers (reference, from POLAR_SPEC section 16):")
    Ot0, Bt0 = St.S.unpack(x[:-2])
    pO, r2O = corner_power(C.x, Ot0[:, 0])
    pB, r2B = corner_power(C.x, Bt0[:, 0])
    print(f"   Ot ~ xi^{pO:.3f} (R2={r2O:.4f})   Bt ~ xi^{pB:.3f} (R2={r2B:.4f})")
    print("   expected: Ot ~ xi^1 (Om ~ w_x(0) y1),  Bt ~ xi^2 (B ~ th_xx(0) y1^2/2)\n")

    print("CHEN-HOU ADMISSIBILITY (eq:normal_vanish): perturbations need omega = O(|x|^2),")
    print("i.e. dOt ~ xi^2 or FASTER -- strictly faster than the profile's own xi^1.\n")
    print(f"  {'eigenvalue':>22s} {'dOt ~ xi^p':>12s} {'R2':>7s} "
          f"{'dBt ~ xi^q':>12s} {'R2':>7s}   verdict")
    lo = [i for i in range(w.size)
          if abs(w[i].imag) < 3.0 and abs(w[i].real) > 1e-7][:a_.nmodes]
    for i in lo:
        v = V[:, i]
        full = np.zeros(2 * n2, dtype=complex)
        full[St.S.idx] = v
        dOt = full[:n2].reshape(C.nx, C.nb)
        dBt = full[n2:].reshape(C.nx, C.nb)
        p1, q1 = corner_power(C.x, dOt[:, 0])
        p2, q2 = corner_power(C.x, dBt[:, 0])
        adm = "ADMISSIBLE" if p1 >= 1.8 else "INADMISSIBLE (linear)"
        tag = "UNSTABLE" if w[i].real > 1e-6 else "stable  "
        print(f"  {w[i].real:+9.4f}{w[i].imag:+9.4f}i {p1:12.3f} {q1:7.4f} "
              f"{p2:12.3f} {q2:7.4f}   {tag} {adm}")

    print("\n  If the UNSTABLE mode is INADMISSIBLE (vanishes only linearly), then it is")
    print("  not a perturbation Chen-Hou admit, and there is NO contradiction with their")
    print("  stability result -- our spectrum is simply taken on a larger space.")
    print("  If it is ADMISSIBLE, the disagreement is real and must be confronted.")


if __name__ == "__main__":
    main()
