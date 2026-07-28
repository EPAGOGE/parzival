"""
RETIRE THE BIGGEST ARCHITECTURAL RISK: can the gauge survive the move to log-polar?

THE RISK
--------
Every scalar closure in both reference codes is evaluated at the corner r = 0 exactly
(Chen-Hou `cl = 4*vx1(1,1)/wx1(1,1); cw = u1dx1(1,1) + cl/2` at F_pertb_2lev.m:135-137;
Liu's (3.3.10) is algebraically equivalent). **Log-polar deletes that point**: r = 0 is
s = -infinity. `polar_gauge_gate.py` only showed the constants can be RECOVERED from
their stored data -- a different claim from being able to IMPOSE them on a domain that
excludes the corner.

This is load-bearing three times over: it makes c_l, c_w determinate; it is the
structural reason the zero field is inadmissible (the failure that already bit this lab);
and it is what turns the saddle into an attractor if the marching route is taken.

THE DIAGNOSTIC -- basis-free, so it does not depend on guessing good functionals
--------------------------------------------------------------------------------
The system has exactly two scaling symmetries. Derived by requiring invariance of the
substituted system (both verified below against the identities they imply):

  AMPLITUDE   lam:  Om -> lam Om,  Psi -> lam Psi,  B -> lam^2 B,
                    and c_l -> lam c_l, c_w -> lam c_w
  TRANSLATION sig:  s -> s + sig,  Om -> Om,  Psi -> e^-2sig Psi,  B -> e^-sig B,
                    with c_l, c_w UNCHANGED

Their tangent vectors, as functions on the domain, are

  v_amp   = ( Om,        Psi,          2 B        )
  v_trans = ( Om_s,      Psi_s - 2Psi, B_s - B    )

A gauge on a domain can determine the two scaling constants ONLY IF these two
directions are distinguishable on that domain. So the right question is not "which two
functionals" but the ANGLE between v_amp and v_trans in a weighted L2 over the domain.
That is basis-free: if the angle collapses, NO pair of functionals can work.

THE PREDICTION THIS TESTS
-------------------------
In the pure power-law far field, Om_s = a Om, Psi_s = (2+a) Psi, B_s = (1+2a) B, so

  v_trans -> ( a Om, a Psi, 2 a B ) = a * v_amp

EXACTLY PARALLEL. A dilation is indistinguishable from an amplitude change for a pure
power law. So the gauge signal must vanish as the domain retreats outward, and the angle
should decay like the leading correction to the power law -- i.e. like e^(a s) = r^-0.342.
The inner truncation S0 at which the angle becomes numerically useless is precisely the
inner boundary the solver cannot cross. That converts an unquantified architectural risk
into one number and one curve, using only data already on disk.
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

HERE = pathlib.Path(__file__).parent
NB = 240
S_TOP = 25.0                    # outer edge (recon: s_max ~ 25 is ample; 37 oversizes)


def _mod(name, fname):
    sp = importlib.util.spec_from_file_location(name, str(HERE / fname))
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


def fields_on(ps, P, s, b):
    """Om, B from the (cubic) seed; Psi by angular quadrature of their velocity."""
    Ot, Bt, Om, B = ps.seed_on_grid(P, s, b)
    a = P["alpha"]
    d = loadmat(ps.MAT, squeeze_me=True, struct_as_record=False)
    sol = d["solu"]
    w = np.asarray(d["w"], dtype=float)
    u1 = ps._grid_field(d, sol, "u1", w.shape)
    u2 = ps._grid_field(d, sol, "u2", w.shape)
    X, Y = P["X"], P["Y"]
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    R = np.sqrt(XX ** 2 + YY ** 2)
    R = np.where(R > 0, R, np.nan)
    su1, su2 = u1 * R ** (-(1.0 + a)), u2 * R ** (-(1.0 + a))
    for A in (su1, su2):
        A[~np.isfinite(A)] = A[1, 1]
    f1 = RegularGridInterpolator((X, Y), su1, method="cubic",
                                 bounds_error=False, fill_value=None)
    f2 = RegularGridInterpolator((X, Y), su2, method="cubic",
                                 bounds_error=False, fill_value=None)
    bq = np.concatenate([[0.0], b])
    S, Bq = np.meshgrid(s, bq, indexing="ij")
    Rq = np.exp(S)
    pts = np.stack([Rq * np.cos(Bq), Rq * np.sin(Bq)], axis=-1)
    urs = f1(pts) * np.cos(Bq) + f2(pts) * np.sin(Bq)
    Pt = -cumulative_trapezoid(urs, bq, axis=1, initial=0.0)[:, 1:]
    Psi = Pt * np.exp((2.0 + a) * s)[:, None]
    return Om, B, Psi


def angle_at(ps, P, S0, ns=90, nb=NB):
    """Angle between the amplitude and translation tangent directions on [S0, S_TOP]."""
    s = np.linspace(S0, S_TOP, ns)
    b = np.linspace(0.04, np.pi / 2 - 0.04, nb)
    Om, B, Psi = fields_on(ps, P, s, b)
    ds = s[1] - s[0]
    g = lambda A: np.gradient(A, ds, axis=0)
    va = (Om, Psi, 2.0 * B)
    vt = (g(Om), g(Psi) - 2.0 * Psi, g(B) - B)
    # weighted L2 over the domain; weight each component by its own RMS so the angle is
    # not dominated by whichever field happens to carry the largest numbers
    I = (slice(2, ns - 2), slice(2, nb - 2))
    ip = 0.0
    na = nb_ = 0.0
    for A, T in zip(va, vt):
        sc = max(float(np.sqrt(np.mean(A[I] ** 2))), 1e-300)
        A2, T2 = A[I] / sc, T[I] / sc
        ip += float(np.sum(A2 * T2))
        na += float(np.sum(A2 * A2))
        nb_ += float(np.sum(T2 * T2))
    c = ip / max(np.sqrt(na * nb_), 1e-300)
    c = max(-1.0, min(1.0, c))
    ang = float(np.degrees(np.arccos(abs(c))))
    cond = (1.0 + abs(c)) / max(1.0 - abs(c), 1e-300)
    return ang, cond, abs(c)


def main():
    ps = _mod("ps", "polar_seed.py")
    P = ps.load()
    a = P["alpha"]
    print(f"alpha = {a:+.8f}   outer edge S_TOP = {S_TOP}")
    print("PREDICTION: v_trans -> a * v_amp in the pure power law, so the angle must")
    print(f"            decay, and the leading correction goes like e^(a s) = r^{a:.4f}\n")
    print(f"  {'S0':>6s} {'r_inner':>10s} {'angle (deg)':>12s} {'cond(2x2)':>12s}"
          f" {'|cos|':>10s}   {'usable?':>10s}  {'seed data':>14s}")
    rows = []
    for S0 in (-6.0, -4.0, -2.0, 0.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0):
        try:
            ang, cond, c = angle_at(ps, P, S0)
        except Exception as e:
            print(f"  {S0:6.1f}  failed: {str(e)[:50]}")
            continue
        rows.append((S0, ang, cond))
        # DATA-VALIDITY GUARD: their mesh is uniform near the origin with spacing
        # h = 0.00390625, so an inner edge at r < a few h is interpolating BELOW their
        # resolution and the angle there is not a measurement of anything.
        h = float(P["X"][1] - P["X"][0])
        cells = np.exp(S0) / h
        if cells < 4.0:
            verdict = "UNRESOLVED"
        elif cond < 1e3:
            verdict = "yes"
        elif cond < 1e5:
            verdict = "marginal"
        else:
            verdict = "NO"
        print(f"  {S0:6.1f} {np.exp(S0):10.3g} {ang:12.4f} {cond:12.3e} {c:10.7f}"
              f"   {verdict:>10s}  ({cells:7.1f} cells)")

    if len(rows) > 3:
        A = np.array(rows)
        m = (A[:, 0] >= 2.0) & (A[:, 1] > 0)
        if m.sum() > 2:
            sl = np.polyfit(A[m, 0], np.log(A[m, 1]), 1)[0]
            print(f"\n  angle decay for S0 >= 2:  angle ~ e^({sl:+.4f} s)"
                  f"   (predicted alpha = {a:+.4f})")
            print(f"  ratio measured/predicted = {sl/a:.3f}"
                  f"   {'-> CONFIRMS the r^alpha correction' if 0.6 < sl/a < 1.6 else '-> does NOT match'}")

    print("\nREADING THIS:")
    print("  The angle is the gauge SIGNAL. It is basis-free -- if it collapses, no")
    print("  choice of two functionals can determine c_l and c_w on that domain.")
    print("  The smallest S0 with a healthy angle is the inner truncation the solver")
    print("  must reach. Placing the inner edge further out than that is not a")
    print("  resolution compromise, it is a WELL-POSEDNESS failure, and its symptoms")
    print("  are the ones already logged here: singular Jacobian, sign-flipping c_w,")
    print("  step norms of 1e6-1e7, or silent convergence to the zero field.")


if __name__ == "__main__":
    main()
