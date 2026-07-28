"""
GATE the SUBSTITUTED log-polar system by evaluating its residual on Chen-Hou's own
converged profile. This is "verify the residual AT INIT" done BEFORE the solver exists.

THE DERIVATION BEING TESTED
---------------------------
With Om = e^(a s) Ot, B = e^((1+2a) s) Bt, Psi = e^((2+a) s) Pt and a = c_w/c_l, every
exponential prefactor cancels, because c_l * a = c_w kills the c_w*Om term and
c_l*(1+2a) = c_l + 2 c_w kills the B source term:

  R1: c_l Ot_s + e^(a s)[(Pt_s+(2+a)Pt) Ot_b - Pt_b (Ot_s + a Ot)]
                 - e^(a s)[cos b (Bt_s+(1+2a)Bt) - sin b Bt_b]                = 0
  R2: c_l Bt_s + e^(a s)[(Pt_s+(2+a)Pt) Bt_b - Pt_b (Bt_s + (1+2a) Bt)]       = 0
  R3: Pt_ss + 2(2+a) Pt_s + (2+a)^2 Pt + Pt_bb + Ot                           = 0

Since a < 0 the bracket prefactor e^(a s) DECAYS outward, so R1/R2 degenerate to
c_l Ot_s = 0 and c_l Bt_s = 0 in the far field -- which is exactly the s-independence
already measured at 5.5e-4 in polar_seed.py. That consistency is a check on the
formulation, but a weak one, because it is what the far field trivially satisfies. This
gate is the strong version: evaluate ALL THREE residuals, INCLUDING the nonlinear
brackets and R3, on real data.

WHERE THE THIRD FIELD COMES FROM
--------------------------------
Psi is not stored, and getting it from R3 would need the very 2D machinery being gated
(circular). So take it from their VELOCITY instead: u_r = -(1/r) Psi_b with Psi = 0 at
the wall (beta = 0) gives a pure 1D angular quadrature at each s,

    Psi(s, b) = - e^s * INT_0^b u_r(s, b') db'

This makes all three substituted fields come from their data by independent routes, and
it yields a bonus cross-check: this Psi must agree with the Pt obtained in
polar_psi_gate.py from the ANGULAR ODE, which never touched the velocity.

WHAT A FAILURE WOULD MEAN
-------------------------
A large R3 means the Poisson substitution or a sign is wrong. A large R1/R2 with small
R3 means the transport terms or the bracket orientation are wrong -- note the Poisson
bracket sign was ALREADY caught once being -1x (polar_ops_gate.py), so this is the
independent numerical confirmation of that fix on real data.
"""
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

HERE = pathlib.Path(__file__).parent
NS, NB = 160, 320
S0, S1 = 20.0, 30.0            # inside the verified far-field window
BPAD = 0.03                    # keep off the exact edges for interpolation


def _mod(name, fname):
    sp = importlib.util.spec_from_file_location(name, str(HERE / fname))
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


def main():
    ps = _mod("ps", "polar_seed.py")
    P = ps.load()
    a = P["alpha"]
    cl, cw = P["cl"], P["cw"]
    p2 = 2.0 + a
    print(f"alpha={a:+.8f}  cl={cl:.8f}  cw={cw:.8f}")
    print(f"identity checks:  c_l*a - c_w = {cl*a-cw:+.3e}   "
          f"c_l*(1+2a) - (c_l+2c_w) = {cl*(1+2*a)-(cl+2*cw):+.3e}")

    s = np.linspace(S0, S1, NS)
    b = np.linspace(BPAD, np.pi / 2 - BPAD, NB)
    ds, db = s[1] - s[0], b[1] - b[0]

    # --- Ot, Bt from the validated seed ---------------------------------------
    Ot, Bt, _, _ = ps.seed_on_grid(P, s, b)

    # --- Psi by angular quadrature of their radial velocity -------------------
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
    f1 = RegularGridInterpolator((X, Y), su1, method="linear",
                                 bounds_error=False, fill_value=None)
    f2 = RegularGridInterpolator((X, Y), su2, method="linear",
                                 bounds_error=False, fill_value=None)

    # angular grid extended down to 0 so the quadrature starts at the wall
    bq = np.concatenate([[0.0], b])
    S, Bq = np.meshgrid(s, bq, indexing="ij")
    Rq = np.exp(S)
    pts = np.stack([Rq * np.cos(Bq), Rq * np.sin(Bq)], axis=-1)
    U1, U2 = f1(pts), f2(pts)
    ur_scaled = U1 * np.cos(Bq) + U2 * np.sin(Bq)         # = u_r * r^-(1+a)
    # Psi = -e^s INT u_r db  =>  Pt = Psi e^-((2+a)s) = -INT ur_scaled db
    from scipy.integrate import cumulative_trapezoid
    Pt_full = -cumulative_trapezoid(ur_scaled, bq, axis=1, initial=0.0)
    Pt = Pt_full[:, 1:]

    print(f"\nPt from velocity quadrature: min={Pt.min():.6g} max={Pt.max():.6g}")
    rel_s = np.max(np.abs(Pt - Pt.mean(axis=0, keepdims=True)), axis=0) / \
        np.maximum(np.abs(Pt.mean(axis=0)), 1e-300)
    print(f"  s-independence of Pt: max={np.nanmax(rel_s):.4e} "
          f"median={np.nanmedian(rel_s):.4e}")

    # bonus cross-check vs the ANGULAR-ODE Pt (never saw the velocity)
    npz = pathlib.Path.home() / "parzival/runs/polar_psi_seed.npz"
    if npz.exists():
        z = np.load(npz)
        Pt_ode = np.interp(b, z["beta"], z["Pt"])
        mid = Pt[NS // 2]
        num = np.linalg.norm(mid - Pt_ode)
        den = max(np.linalg.norm(Pt_ode), 1e-300)
        print(f"  CROSS-CHECK vs angular-ODE Pt (polar_psi_gate): rel L2 = {num/den:.4e}"
              f"   {'PASS' if num/den < 0.05 else 'FAIL'}")
        print("    (velocity quadrature vs angular ODE -- fully independent routes to Psi)")

    # --- residuals -------------------------------------------------------------
    gs = lambda A: np.gradient(A, ds, axis=0)
    gb = lambda A: np.gradient(A, db, axis=1)
    Ot_s, Ot_b = gs(Ot), gb(Ot)
    Bt_s, Bt_b = gs(Bt), gb(Bt)
    Pt_s, Pt_b = gs(Pt), gb(Pt)
    Pt_ss = np.gradient(Pt_s, ds, axis=0)
    Pt_bb = np.gradient(Pt_b, db, axis=1)
    E = np.exp(a * s)[:, None]
    cosb, sinb = np.cos(b)[None, :], np.sin(b)[None, :]

    adv_O = (Pt_s + p2 * Pt) * Ot_b - Pt_b * (Ot_s + a * Ot)
    src_O = cosb * (Bt_s + (1.0 + 2.0 * a) * Bt) - sinb * Bt_b
    R1 = cl * Ot_s + E * adv_O - E * src_O

    adv_B = (Pt_s + p2 * Pt) * Bt_b - Pt_b * (Bt_s + (1.0 + 2.0 * a) * Bt)
    R2 = cl * Bt_s + E * adv_B

    R3 = Pt_ss + 2.0 * p2 * Pt_s + p2 ** 2 * Pt + Pt_bb + Ot

    # interior only: np.gradient is one-sided at edges, and interpolation is weakest there
    I = (slice(3, NS - 3), slice(6, NB - 6))

    def report(nm, Rr, terms):
        sc = max(max(np.abs(t[I]).max() for t in terms), 1e-300)
        mx = np.abs(Rr[I]).max()
        rms = float(np.sqrt(np.mean(Rr[I] ** 2)))
        print(f"  {nm}: max|R|={mx:.4e}  rms={rms:.4e}  termscale={sc:.4e}"
              f"  ->  max rel = {mx/sc:.4e}   {'PASS' if mx/sc < 0.05 else 'CHECK'}")
        return mx / sc

    print("\nRESIDUALS of the substituted system on their profile"
          f" (interior {I[0].stop-I[0].start}x{I[1].stop-I[1].start}):")
    e1 = report("R1 (Om transport) ", R1, [cl * Ot_s, E * adv_O, E * src_O])
    e2 = report("R2 (B  transport) ", R2, [cl * Bt_s, E * adv_B])
    e3 = report("R3 (Poisson)      ", R3, [Pt_bb, p2 ** 2 * Pt, Ot])

    print(f"\n  far-field degeneracy check (a<0 so e^(a s) decays):")
    print(f"    |e^(a s)| at s={S0}: {np.exp(a*S0):.3e}   at s={S1}: {np.exp(a*S1):.3e}")
    print(f"    so R1,R2 -> c_l*Ot_s, c_l*Bt_s out there; measured "
          f"max|c_l Ot_s|={np.abs(cl*Ot_s[I]).max():.3e}")

    ok = (e3 < 0.05)
    print(f"\nGATE: Poisson equation {'PASS' if ok else 'FAIL'}"
          f" -- this is the discriminating one (it has no decaying prefactor).")
    if ok:
        print("  The substituted Poisson equation, its sign, and the Psi construction are")
        print("  all confirmed on real data. R1/R2 are dominated by the decaying bracket")
        print("  out here by construction, so a small R1/R2 is necessary but weak; the")
        print("  strong test of the transport terms needs the INNER region.")


if __name__ == "__main__":
    main()
