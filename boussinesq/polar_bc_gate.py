"""
GATE for the FAR-FIELD BOUNDARY CONDITIONS the log-polar Newton solve will impose.

POLAR_SPEC.md puts three homogeneous Robin conditions on the outer radial boundary
s = +S (that is the whole reason log-polar was chosen -- on the box the far field
could not be written at all):

    d_s Om  = alpha       * Om          i.e.  Om  ~ r^alpha
    d_s B   = (1 + 2alpha)* B           i.e.  B   ~ r^(1+2alpha)
    d_s Psi = (2 + alpha) * Psi         i.e.  Psi ~ r^(2+alpha)

with alpha = c_w/c_l. `angular_gate.py` already validated the ANGULAR content of the
formulation (ODE residual 1.65e-13, decay exponent -0.34240 on five rays). What it did
NOT check is the radial content -- the exponents on B and Psi. Those are two of the
three outer conditions, and if either is wrong the Newton solve is being handed a wrong
boundary and will converge to the wrong thing (or, more likely given the history, to
zero).

METHOD: measure the exponents in Chen-Hou's own 620x620 converged profile and compare
to the predictions. Fields are IDENTIFIED BY THEIR MEASURED EXPONENT, not by assuming
which array is which -- the .mat ships w, v, th, u1, u2 with no schema, and guessing
the mapping is exactly the kind of silent error this gate exists to catch.

Psi is not stored, but velocity is: Psi ~ r^(2+alpha) implies |u| ~ r^(1+alpha), so the
velocity exponent tests the Psi condition.
"""
import numpy as np
from scipy.io import loadmat
from pathlib import Path

MAT = Path.home() / ("parzival/refs/chen_hou/Perturbed_eqn/Computed profile/"
                     "Steady_state_pertb_oneMesh62036.mat")
R_LO, R_HI = 1e8, 1e15          # asymptotic window (the power law starts at r ~ 1e8)
TOL = 0.02                      # exponent must land this close to be called a match


def ray_exponent(R, B, F, b0, half=0.02):
    """Log-log slope of |F| against r along the ray beta = b0, inside the window."""
    sel = (np.abs(B - b0) < half) & (R > R_LO) & (R < R_HI) & (np.abs(F) > 0)
    if sel.sum() < 20:
        return np.nan, 0
    c = np.polyfit(np.log(R[sel]), np.log(np.abs(F[sel])), 1)
    return float(c[0]), int(sel.sum())


def main():
    d = loadmat(MAT, squeeze_me=True, struct_as_record=False)
    M, s = d["Mesh"], d["solu"]
    xs = [np.asarray(e, dtype=float) for e in np.ravel(np.asarray(M.x, dtype=object))]
    X, Y = xs[0], xs[1]
    al = float(np.ravel(np.asarray(s.al))[0])
    cl = float(np.ravel(np.asarray(s.cl))[0])
    cw = float(np.ravel(np.asarray(s.cw))[0])
    alpha = -al

    print(f"cl={cl:.8f}  cw={cw:.8f}  al={al:.8f}  =>  alpha = cw/cl = {cw/cl:+.8f}"
          f"   (stored -al = {alpha:+.8f})")
    # B IS theta, not theta_x. Read it off the spec's own B equation: the RHS
    # coefficient is (c_l + 2 c_w) = c_theta, so B carries theta's scaling exponent
    # c_theta/c_l = 1 + 2 alpha. Consistently, the Om equation's source is d_1 B,
    # and Boussinesq is omega_t + u.grad omega = theta_x -- so B = theta. theta_x is
    # then one power lower, r^(2 alpha), which gives a free FOURTH check.
    # NOTE: the field they store as `v` is theta/x1, NOT theta_x -- verified from their
    # own source (Build_profile_pertb_Nlev.m:125 is literally `th = x1 .* v;`) and by
    # measurement (max|th - x1*v| = 0.0 EXACTLY on the 620x620 profile, versus
    # max|v - d_1 th| = 0.295 against max|v| = 0.668). theta/x1 and theta_x share the
    # exponent 2*alpha, so this exponent check CANNOT distinguish them -- an exponent
    # match is necessary, never sufficient, for identifying a field.
    pred = {"Om  (vorticity)":  alpha,
            "B = theta":        1 + 2 * alpha,
            "theta/x1 (their v)": 2 * alpha,
            "u   (velocity)":   1 + alpha,
            "Psi (stream fn)":  2 + alpha}
    print("\nPREDICTED far-field exponents from POLAR_SPEC:")
    for k, v in pred.items():
        print(f"   {k:18s} r^{v:+.5f}")

    XX, YY = np.meshgrid(X, Y, indexing="ij")
    R = np.sqrt(XX ** 2 + YY ** 2)
    B = np.arctan2(YY, XX)

    fields = {}
    for name in ("w", "v", "th", "u1", "u2"):
        arr = d.get(name)
        if arr is None:
            arr = getattr(s, name, None)
        if arr is None:
            continue
        # solu entries may be cell arrays (one cell per mesh patch); take any element
        # whose shape matches the mesh and skip the rest rather than guessing.
        cands = [arr]
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            cands = list(np.ravel(arr))
        for c in cands:
            try:
                a = np.asarray(c, dtype=float)
            except (TypeError, ValueError):
                continue
            if a.shape == R.shape:
                fields[name] = a
                break

    print(f"\nMEASURED exponents (mean over rays, window r in [{R_LO:.0e},{R_HI:.0e}]):")
    rays = (0.2, 0.4, 0.8, 1.2, 1.4)
    measured = {}
    for name, F in fields.items():
        es = [ray_exponent(R, B, F, b0)[0] for b0 in rays]
        es = [e for e in es if np.isfinite(e)]
        if not es:
            print(f"   {name:5s}: no usable rays")
            continue
        m, sd = float(np.mean(es)), float(np.std(es))
        measured[name] = m
        print(f"   {name:5s}: r^{m:+.5f}   (ray spread {sd:.5f}, {len(es)} rays)")

    print("\nIDENTIFICATION -- match each stored field to the predicted exponent:")
    ok_all = True
    for name, m in measured.items():
        best, bd = None, 1e9
        for k, v in pred.items():
            if abs(m - v) < bd:
                best, bd = k, abs(m - v)
        hit = bd < TOL
        ok_all &= hit or True          # identification is informative, not pass/fail
        print(f"   {name:5s} r^{m:+.5f}  ->  {best:18s} (predicted r^{pred[best]:+.5f},"
              f" |diff|={bd:.5f})  {'MATCH' if hit else 'no match'}")

    print("\nGATE -- the three Robin conditions the solver will impose:")
    checks = [("d_s Om  = alpha      Om  ", "w",  alpha),
              ("d_s B   = (1+2alpha) B   ", "th", 1 + 2 * alpha),
              ("d_s Psi = (2+alpha)  Psi ", "u1", 1 + alpha),
              ("  [bonus] theta/x1 ~ r^2a  ", "v",  2 * alpha)]
    passed = 0
    for label, key, want in checks:
        if key not in measured:
            print(f"   {label}: field '{key}' unavailable -- CANNOT GATE")
            continue
        got = measured[key]
        ok = abs(got - want) < TOL
        passed += ok
        print(f"   {label}: want r^{want:+.5f}  got r^{got:+.5f}"
              f"  |diff|={abs(got-want):.5f}   {'PASS' if ok else 'FAIL'}")
    print(f"\n{passed}/{len(checks)} far-field conditions verified against Chen-Hou data.")
    if passed < len(checks):
        print("DO NOT write the Newton outer boundary until every one of these passes --")
        print("a wrong outer condition is exactly how the six Cartesian attempts died.")


if __name__ == "__main__":
    main()
