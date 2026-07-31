"""
GATE the LOG-POLAR OPERATOR IDENTITIES symbolically, before any solver is written.

POLAR_SPEC.md carries an explicit warning on its own velocity line -- "signs to be
re-derived carefully from u = skew(grad Psi) in polar" -- and that warning is
justified: the spec's u_r and u_beta both have the WRONG SIGN for the convention the
working engines use (`dedalus_bsq.py` line 62: `u = d3.skew(d3.grad(psi))`, i.e.
u = (-Psi_2, Psi_1)).

Derivation with y1 = r cos b, y2 = r sin b, e_r = (cos b, sin b), e_b = (-sin b, cos b):

    u_r = u . e_r = -Psi_2 cos b + Psi_1 sin b = -(1/r) Psi_b        [spec has +]
    u_b = u . e_b = +Psi_2 sin b + Psi_1 cos b = +Psi_r              [spec has -]

so with s = ln r (d_r = e^{-s} d_s) the advection collapses to a Poisson bracket:

    u.grad f = u_r d_r f + (u_b / r) d_b f = e^{-2s} ( Psi_s f_b - Psi_b f_s )

Three identities the solver depends on, gated here to machine precision on a
manufactured field -- the same discipline as G1/G2/G3 in the engines:

  A. advection      u.grad f          = e^{-2s} (Psi_s f_b - Psi_b f_s)
  B. Laplacian      Lap Psi           = e^{-2s} (Psi_ss + Psi_bb)
  C. wall-direction d_1 f             = e^{-s} (cos b f_s - sin b f_b)

C is the source term in the Om equation (Boussinesq forcing is theta_x = d_1 B), so a
sign error there flips the forcing and the profile does not exist. All three are
checked by SYMBOLIC differentiation of an explicit manufactured (Psi, f) in both
coordinate systems -- an exact identity check, not a numerical comparison.
"""
import sympy as sp


def main():
    r, b = sp.symbols("r beta", positive=True)
    s = sp.symbols("s", real=True)

    y1 = r * sp.cos(b)
    y2 = r * sp.sin(b)

    # Manufactured fields: explicit, smooth, no symmetry that could hide a sign error
    # (mixed parity in both arguments, non-polynomial part, no separability).
    Y1, Y2 = sp.symbols("y1 y2", real=True)
    Psi_c = Y1 ** 2 * Y2 + sp.sin(Y1) * sp.cos(2 * Y2) + Y1 * Y2 ** 3
    f_c = sp.exp(-Y1) * Y2 ** 2 + sp.cos(Y1 + 2 * Y2) + Y1 ** 3

    sub = {Y1: y1, Y2: y2}
    Psi = Psi_c.subs(sub)
    f = f_c.subs(sub)

    # --- Cartesian truth -----------------------------------------------------
    Psi_1 = sp.diff(Psi_c, Y1).subs(sub)
    Psi_2 = sp.diff(Psi_c, Y2).subs(sub)
    f_1 = sp.diff(f_c, Y1).subs(sub)
    f_2 = sp.diff(f_c, Y2).subs(sub)

    u1 = -Psi_2                      # u = skew(grad Psi) = (-Psi_2, Psi_1)
    u2 = Psi_1
    adv_cart = u1 * f_1 + u2 * f_2
    lap_cart = (sp.diff(Psi_c, Y1, 2) + sp.diff(Psi_c, Y2, 2)).subs(sub)
    d1_cart = f_1

    # --- log-polar candidates ------------------------------------------------
    # d_r = e^{-s} d_s with r = e^s, so d_s = r d_r on any expression in r.
    ds = lambda E: r * sp.diff(E, r)
    db = lambda E: sp.diff(E, b)
    em2 = 1 / r ** 2                 # e^{-2s}
    em1 = 1 / r                      # e^{-s}

    adv_polar = em2 * (ds(Psi) * db(f) - db(Psi) * ds(f))
    lap_polar = em2 * (ds(ds(Psi)) + db(db(Psi)))
    d1_polar = em1 * (sp.cos(b) * ds(f) - sp.sin(b) * db(f))

    # the SPEC's (wrong-sign) advection, to show the gate has discriminating power
    adv_spec = em2 * (db(Psi) * ds(f) - ds(Psi) * db(f))

    tests = [
        ("A  advection   u.grad f = e^-2s (Psi_s f_b - Psi_b f_s)", adv_polar, adv_cart),
        ("B  Laplacian   Lap Psi  = e^-2s (Psi_ss + Psi_bb)      ", lap_polar, lap_cart),
        ("C  wall deriv  d_1 f    = e^-s (cos b f_s - sin b f_b) ", d1_polar, d1_cart),
    ]

    print("SYMBOLIC GATE -- log-polar operator identities (exact, not numerical)\n")
    allok = True
    for label, got, want in tests:
        diff = sp.simplify(sp.expand_trig(sp.simplify(got - want)))
        ok = diff == 0
        allok &= ok
        print(f"  {label}: {'PASS (identically 0)' if ok else 'FAIL -> ' + str(diff)}")

    # discriminating-power check: the spec's sign must FAIL, else the gate is vacuous
    dspec = sp.simplify(adv_spec - adv_cart)
    spec_wrong = dspec != 0
    print(f"\n  control: POLAR_SPEC's stated signs (u_r=+Psi_b/r, u_b=-Psi_r) "
          f"{'FAIL as expected' if spec_wrong else 'also pass -- GATE IS VACUOUS'}")
    if spec_wrong:
        print(f"           they give exactly -1x the correct advection: "
              f"{sp.simplify(adv_spec + adv_cart) == 0}")

    print(f"\nRESULT: {'ALL IDENTITIES VERIFIED' if allok else 'SOMETHING IS WRONG'}"
          f"{' and the gate discriminates' if spec_wrong else ''}")
    if allok:
        print("\nCORRECTED lines for POLAR_SPEC.md:")
        print("    u_r = -(1/r) Psi_b = -e^{-s} Psi_b        (spec had +)")
        print("    u_b = +Psi_r       = +e^{-s} Psi_s        (spec had -)")
        print("    u.grad f = e^{-2s} ( Psi_s f_b - Psi_b f_s )")


if __name__ == "__main__":
    main()
