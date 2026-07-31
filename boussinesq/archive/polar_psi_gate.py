"""
GATE the Psi seed and the SIGN CONVENTION against Chen-Hou's stored VELOCITY.

WHY THIS IS A REAL, INDEPENDENT TEST
------------------------------------
`polar_ops_gate.py` verified the sign convention SYMBOLICALLY -- but only for internal
consistency with this lab's own choice `u = skew(grad Psi) = (-Psi_2, Psi_1)`
(`dedalus_bsq.py:62`). It cannot tell us whether that convention matches the DATA. And
the Psi seed is not measured at all: Psi is not stored in their .mat, so it has to be
constructed, and a constructed seed with a sign error is exactly the kind of thing that
sends Newton to the trivial solution.

Both gaps close with one observation. In the far field, Psi = e^{(2+alpha)s} Pt(beta),
so the corrected velocity relations

    u_r = -(1/r) Psi_b   =>   u_r * r^-(1+alpha) = -Pt'(beta)
    u_b = + Psi_r        =>   u_b * r^-(1+alpha) = (2 + alpha) * Pt(beta)

predict BOTH velocity components from Pt alone. Pt in turn comes from the angular ODE

    (-d_bb - (2+alpha)^2) Pt = Ot,      Pt(0) = Pt(pi/2) = 0

driven by the Ot measured in `polar_seed.py`. Their `.mat` stores u1, u2, which have been
used for NOTHING so far -- so projecting them onto (e_r, e_b) and comparing gives two
independent checks against untouched data.

WHAT EACH OUTCOME MEANS
  both match, sign +1   -> Psi seed correct AND our convention matches theirs.
  both match, sign -1   -> Psi seed correct in magnitude; their Psi is OURS NEGATED.
                           That is a fine outcome but MUST be known before seeding.
  only one matches      -> the seed or the far-field structure is wrong. Stop.
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.io import loadmat

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    sp = importlib.util.spec_from_file_location(name, str(HERE / fname))
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


def solve_angular(beta, rhs, p):
    """(-d_bb - p^2) f = rhs on [0,pi/2], f(0)=f(pi/2)=0. Non-uniform 2nd-order FD --
    same discretization angular_gate.py validated to 1.65e-13."""
    n = beta.size
    A = np.zeros((n, n))
    b = np.asarray(rhs, float).copy()
    A[0, 0] = 1.0; b[0] = 0.0
    A[-1, -1] = 1.0; b[-1] = 0.0
    for i in range(1, n - 1):
        hm, hp = beta[i] - beta[i - 1], beta[i + 1] - beta[i]
        A[i, i - 1] = -2.0 / (hm * (hm + hp))
        A[i, i + 1] = -2.0 / (hp * (hm + hp))
        A[i, i] = 2.0 / (hm * hp) - p ** 2
    return np.linalg.solve(A, b)


def main():
    ps = _mod("ps", "polar_seed.py")
    P = ps.load()
    alpha = P["alpha"]
    p = 2.0 + alpha
    print(f"alpha = {alpha:+.8f}   2+alpha = {p:.8f}")

    # --- Ot(beta) from the validated seed, at a large s (it is s-independent there) ---
    S_EVAL = 24.0
    NB = 400
    beta = np.linspace(0.0, np.pi / 2, NB)
    binner = beta.copy()
    binner[0] = 1e-9                      # keep off the exact axis for interpolation
    binner[-1] = np.pi / 2 - 1e-9
    Ot, Bt, _, _ = ps.seed_on_grid(P, np.array([S_EVAL]), binner)
    g = Ot[0]
    print(f"Ot(beta) from seed at s={S_EVAL} (r={np.exp(S_EVAL):.2e}), {NB} points: "
          f"wall={g[0]:.5f} max={g.max():.5f} axis={g[-1]:.3e}")

    # --- Pt from the angular ODE -----------------------------------------------
    Pt = solve_angular(beta, g, p)
    dPt = np.gradient(Pt, beta)
    print(f"Pt: min={Pt.min():.5g} max={Pt.max():.5g}  Pt(0)={Pt[0]:.2e} "
          f"Pt(pi/2)={Pt[-1]:.2e}")

    # --- their velocity, projected onto (e_r, e_b) and de-scaled ----------------
    d = loadmat(ps.MAT, squeeze_me=True, struct_as_record=False)
    s_ = d["solu"]
    w = np.asarray(d["w"], dtype=float)
    u1 = ps._grid_field(d, s_, "u1", w.shape)
    u2 = ps._grid_field(d, s_, "u2", w.shape)
    X, Y = P["X"], P["Y"]
    from scipy.interpolate import RegularGridInterpolator
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    R = np.sqrt(XX ** 2 + YY ** 2)
    R = np.where(R > 0, R, np.nan)
    # de-scale so the interpolated quantities are the O(1) angular functions
    su1 = u1 * R ** (-(1.0 + alpha))
    su2 = u2 * R ** (-(1.0 + alpha))
    for A in (su1, su2):
        A[~np.isfinite(A)] = A[1, 1]
    f1 = RegularGridInterpolator((X, Y), su1, method="linear",
                                 bounds_error=False, fill_value=None)
    f2 = RegularGridInterpolator((X, Y), su2, method="linear",
                                 bounds_error=False, fill_value=None)
    r0 = np.exp(S_EVAL)
    pts = np.stack([r0 * np.cos(binner), r0 * np.sin(binner)], axis=-1)
    U1, U2 = f1(pts), f2(pts)
    cb, sb = np.cos(binner), np.sin(binner)
    ur_meas = U1 * cb + U2 * sb            # already scaled by r^-(1+alpha)
    ub_meas = -U1 * sb + U2 * cb

    ur_pred = -dPt
    ub_pred = p * Pt

    # --- compare, allowing for an overall sign convention -----------------------
    interior = (beta > 0.06) & (beta < np.pi / 2 - 0.06)

    def cmp(name, meas, pred):
        num = np.linalg.norm((meas - pred)[interior])
        nump = np.linalg.norm((meas + pred)[interior])
        den = np.linalg.norm(meas[interior])
        same = num / max(den, 1e-300)
        flip = nump / max(den, 1e-300)
        sign = +1 if same < flip else -1
        best = min(same, flip)
        print(f"  {name}: rel L2 with sign +1 = {same:.4e}, with sign -1 = {flip:.4e}"
              f"   -> best sign {sign:+d}, err {best:.4e}"
              f"   {'PASS' if best < 0.05 else 'FAIL'}")
        return sign, best

    print(f"\nCOMPARISON on the interior (beta in [0.06, pi/2-0.06], "
          f"{interior.sum()} of {NB} points):")
    s_r, e_r = cmp("u_r  vs  -Pt'   ", ur_meas, ur_pred)
    s_b, e_b = cmp("u_b  vs (2+a)Pt ", ub_meas, ub_pred)

    print(f"\n  scale check: max|u_r_meas|={np.abs(ur_meas[interior]).max():.5g} "
          f"vs max|Pt'|={np.abs(ur_pred[interior]).max():.5g}")
    print(f"               max|u_b_meas|={np.abs(ub_meas[interior]).max():.5g} "
          f"vs max|(2+a)Pt|={np.abs(ub_pred[interior]).max():.5g}")

    ok = (e_r < 0.05) and (e_b < 0.05)
    consistent = (s_r == s_b)
    print(f"\nGATE: {'PASS' if ok else 'FAIL'}"
          f"  ({'consistent' if consistent else 'INCONSISTENT'} sign across components)")
    if ok and consistent:
        if s_r > 0:
            print("  => Psi seed CORRECT and our sign convention MATCHES Chen-Hou's.")
        else:
            print("  => Psi seed correct in magnitude, but their Psi is OURS NEGATED.")
            print("     Seed with Psi -> -Psi, or equivalently flip the skew convention.")
        print("  Both velocity components are reproduced from Pt alone, which also")
        print("  independently confirms Psi ~ r^(2+alpha) and the whole far-field")
        print("  structure -- using data (u1,u2) not previously used for anything.")
    elif ok and not consistent:
        print("  => The two components want OPPOSITE signs. That is not a convention")
        print("     difference, it is an error in the velocity relations. STOP and")
        print("     re-derive before writing the solver.")

    np.savez(pathlib.Path.home() / "parzival/runs/polar_psi_seed.npz",
             beta=beta, Ot=g, Pt=Pt, dPt=dPt, alpha=alpha,
             ur_meas=ur_meas, ub_meas=ub_meas, sign=s_r)
    print("\nsaved -> ~/parzival/runs/polar_psi_seed.npz")


if __name__ == "__main__":
    main()
