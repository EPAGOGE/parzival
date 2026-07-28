"""Derivation-1 verification for THE FREED-PIN PROBLEM.

(1) SYMPY grading: find all (p,q,c,w) such that A->sA, B->s^p B, P->s^q P,
    cl->s^c cl, cw->s^w cw maps RO',RB',RP' to s^{d}*themselves.
(2) CODE-LEVEL check: on the actual CornerRegSolver.residual with a RANDOM
    state, verify each row class scales with its derived exponent, and that
    ONLY the inhomogeneous rows (rT_pin value pins, g1, g2) break.
(3) EULER check on the actual jacobian(): for a degree-d homogeneous row,
    DR . v_grading = d * R at ANY state -- tests the analytic Jacobian too.
(4) Numbers for the design: axis-column field magnitude (how weakly the
    static axis pins anchor the amplitude), b-grid orientation.
"""
import importlib.util
import pathlib
import sys

import numpy as np

SOLVER = pathlib.Path("/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py")


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- (1) sympy
def sympy_grading():
    import sympy as sm

    s = sm.symbols("s", positive=True)
    p, q, c, w = sm.symbols("p q c w", real=True)
    E1, G1, cosb, sinb, x, mu = sm.symbols("E1 G1 cosb sinb x mu", positive=True)
    A, Ab, LAa = sm.symbols("A Ab LAa")
    B, Bb, LB2b = sm.symbols("B Bb LB2b")
    P, Pb, LPp, Px, Pxx, Pbb = sm.symbols("P Pb LPp Px Pxx Pbb")
    cl, cw = sm.symbols("cl cw")

    RO = (E1 * (-(LPp) * Ab + Pb * LAa + G1 * cosb * LB2b - sinb * Bb)
          - cl * G1 * LAa + cw * A)
    RB = (-E1 * (LPp * Bb - Pb * LB2b) + cl * (B - G1 * LB2b) + 2 * cw * B)
    RP = (G1 ** 2 * (x ** 2 * Pxx + (4 * x + 2 * mu * x ** 2) * Px
                     + (2 + 4 * mu * x + mu ** 2 * x ** 2) * P)
          + G1 * (1 - x * G1) * LPp + Pbb + x * G1 ** 2 * A)

    sub = {A: s * A, Ab: s * Ab, LAa: s * LAa,
           B: s ** p * B, Bb: s ** p * Bb, LB2b: s ** p * LB2b,
           P: s ** q * P, Pb: s ** q * Pb, LPp: s ** q * LPp,
           Px: s ** q * Px, Pxx: s ** q * Pxx, Pbb: s ** q * Pbb,
           cl: s ** c * cl, cw: s ** w * cw}

    def sdegs(expr):
        degs = set()
        for t in sm.expand(expr.subs(sub)).as_ordered_terms():
            d = t.as_powers_dict().get(s, sm.Integer(0))
            degs.add(sm.simplify(d))
        return degs

    eqs = []
    degsyms = {}
    for name, R in (("RO", RO), ("RB", RB), ("RP", RP)):
        ds = list(sdegs(R))
        d0 = ds[0]
        degsyms[name] = d0
        for d in ds[1:]:
            eqs.append(sm.Eq(d, d0))
    sol = sm.solve(eqs, [p, q, c, w], dict=True)
    print("[sympy] homogeneity conditions:", eqs)
    print("[sympy] solutions:", sol)
    assert len(sol) == 1, "grading not unique?"
    S = sol[0]
    print("[sympy] UNIQUE grading: p=%s q=%s c=%s w=%s" % (S[p], S[q], S[c], S[w]))
    degs = {k: sm.simplify(v.subs(S)) for k, v in degsyms.items()}
    print("[sympy] residual degrees:", degs)
    # exact verification of covariance at the solved grading
    for name, R, d in (("RO", RO, degs["RO"]), ("RB", RB, degs["RB"]),
                       ("RP", RP, degs["RP"])):
        lhs = sm.expand(R.subs(sub).subs(S))
        rhs = sm.expand(s ** d * R)
        assert sm.simplify(lhs - rhs) == 0, name
        print(f"[sympy] {name}'(s.z) == s^{d} {name}'(z)   EXACT")
    return {str(k): int(v) for k, v in S.items()}, {k: int(v) for k, v in degs.items()}


# ------------------------------------------------------------- (2)+(3) code
def code_check():
    pc = _mod("pc", SOLVER)
    S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(8, 10, 6),
                           Nb=12, eps_b=1e-3)
    Nx, Nb = S.Nx, S.Nb
    n2 = Nx * Nb
    print(f"[code] grid Nx={Nx} Nb={Nb}  b[0]={S.b[0]:.6f} b[-1]={S.b[-1]:.6f} "
          f"(pi/2={np.pi/2:.6f})  x[0]={S.x[0]}")
    rng = np.random.default_rng(7)
    A = rng.standard_normal((Nx, Nb))
    B = rng.standard_normal((Nx, Nb))
    P = rng.standard_normal((Nx, Nb))
    cl, cw = 1.37, -0.61
    z = S.pack(A, B, P, cl, cw)
    F = S.residual(z)

    s0 = 1.7
    zs = S.pack(s0 * A, s0 ** 2 * B, s0 * P, s0 * cl, s0 * cw)
    Fs = S.residual(zs)

    # per-row expected factor
    fac = np.empty(3 * n2 + 2)
    fac[:n2] = s0 ** 2                 # RO' rows
    fac[n2:2 * n2] = s0 ** 3           # RB' rows
    fac[2 * n2:3 * n2] = s0            # RP' rows (linear in P everywhere)
    ro_fac = fac[:n2]; rb_fac = fac[n2:2 * n2]
    ro_fac[S.rT_c0] = s0               # C0 duplicate rows are linear
    rb_fac[S.rT_c0] = s0 ** 2
    broken = np.zeros(3 * n2 + 2, bool)
    broken[S.rT_pin] = True            # value pins vs static A0/B0: inhomogeneous
    broken[n2 + S.rT_pin] = True
    broken[-2:] = True                 # g1,g2 static REF targets
    cov = ~broken

    scale_err = np.max(np.abs(Fs[cov] - fac[cov] * F[cov])) / np.max(np.abs(F[cov]))
    print(f"[code] covariant rows: max |F(s.z) - s^d F(z)| / max|F| = {scale_err:.3e}"
          f"   ({cov.sum()} rows)")
    # broken rows must break by exactly (s-1)*field at the pins
    pinA = Fs[S.rT_pin] - F[S.rT_pin] - (s0 - 1.0) * A.ravel()[S.rT_pin]
    pinB = (Fs[n2 + S.rT_pin] - F[n2 + S.rT_pin]
            - (s0 ** 2 - 1.0) * B.ravel()[S.rT_pin])
    print(f"[code] pin-row breakage identity: A {np.max(np.abs(pinA)):.3e}  "
          f"B {np.max(np.abs(pinB)):.3e}  (should be ~0: breakage == (s^k-1)*field)")

    # (3) Euler check on the analytic jacobian: J . v_grading = deg * F on
    # covariant rows, at this RANDOM state.
    J = S.jacobian(z)
    v = S.pack(A, 2.0 * B, P, cl, cw)          # grading vector (1,2,1,1,1)
    Jv = np.asarray(J @ v).ravel()
    deg = np.empty(3 * n2 + 2)
    deg[:n2] = 2.0; deg[n2:2 * n2] = 3.0; deg[2 * n2:] = 1.0
    deg[:n2][S.rT_c0] = 1.0; deg[n2:2 * n2][S.rT_c0] = 2.0
    eul_err = (np.max(np.abs(Jv[cov] - deg[cov] * F[cov]))
               / np.max(np.abs(F[cov])))
    print(f"[code] EULER (analytic J): max |J.v - d*F| / max|F| = {eul_err:.3e}")
    # what resists the grading direction at a hypothetical root: pins.
    corner = np.array([j for j in range(Nb)])           # rows rid(0,j)
    axis = np.array([i * Nb + (Nb - 1) for i in range(1, Nx)])
    print(f"[code] Jv on A-corner-pin rows (freed-> A(0,j), O(1) resist): "
          f"max {np.max(np.abs(Jv[corner])):.3e}")
    print(f"[code] Jv on A-axis-pin rows  (stay-static resist):          "
          f"max {np.max(np.abs(Jv[axis])):.3e}")

    # (4) seed-field axis magnitude: how strongly do static axis pins anchor
    # the amplitude?  (they are the ONLY inhomogeneous rows left after freeing)
    axA = np.max(np.abs(S.A0[:, Nb - 1])); gA = np.max(np.abs(S.A0))
    axB = np.max(np.abs(S.B0[:, Nb - 1])); gB = np.max(np.abs(S.B0))
    print(f"[code] seed axis-column magnitude: |A0_axis|/|A0| = {axA/gA:.3e}   "
          f"|B0_axis|/|B0| = {axB/gB:.3e}   (eps_b={S.eps_b:g}, "
          f"sin eps_b = {np.sin(S.eps_b):.3e})")
    return scale_err, eul_err, axA / gA, axB / gB


if __name__ == "__main__":
    grading, degs = sympy_grading()
    scale_err, eul_err, axA, axB = code_check()
    print("\n[SUMMARY] grading:", grading, " degrees:", degs)
    print(f"[SUMMARY] code scaling err {scale_err:.3e}, Euler err {eul_err:.3e}, "
          f"axis anchor A {axA:.3e} B {axB:.3e}")
