"""Q1 GATE: is the DIVIDED residual literally the tau-derivative of the state?

The claim to prove (not assume): in the corner-regularized DIVIDED variables the
dynamic-rescaling evolution reads

    d_tau A = RO'(A,B,P,cl,cw)          <- the coded RO', VERBATIM
    d_tau B = RB'(A,B,P,cl,cw)
    0       = RP'                        (elliptic constraint, NO time derivative)

i.e. the mass matrix on the two transported blocks is the IDENTITY -- no xi-power,
no exponential weight, no g-factor.  If true, the descriptor pencil is (E, J) with
E a 0/1 DIAGONAL selector and J the solver's own Jacobian.  If false, every
resolvent contour computed from J is in the wrong metric.

Method: continuum symbolic identity, evaluated to 50 digits at random points.
The dynamic-rescaling equations (POLAR_SPEC 'Rescaled steady system', with the
steady 0 replaced by d_tau):

    d_tau Om = -cl Om_s - e^{-2s}(Psi_s Om_b - Psi_b Om_s) + cw Om
               + e^{-s}(cos b Bf_s - sin b Bf_b)
    d_tau Bf = -cl Bf_s - e^{-2s}(Psi_s Bf_b - Psi_b Bf_s) + (cl + 2 cw) Bf
    0        = e^{-2s}(Psi_ss + Psi_bb) + Om

CORNER FRAME: xi = ln(1+r), so r = e^xi - 1, s = ln r, and d_s = g d_xi with
g = r/(1+r) = 1 - e^{-xi} = xi*G1.  SUBSTITUTION: Om = e^{a0 xi} xi A,
Bf = e^{(1+2a0) xi} xi^2 B, Psi = e^{mu xi} xi^2 P, mu = 2 + a0.

CLAIMS (the three lines the pencil rests on):
    RO' == e^{-a0 xi} xi^{-1} * (d_tau Om)        [ = d_tau A ]
    RB' == e^{-(1+2a0) xi} xi^{-2} * (d_tau Bf)   [ = d_tau B ]
    RP' == e^{-mu xi} xi^{-2} * (Psi_ss + Psi_bb + r^2 Om)
"""
import sympy as sp

xi, bt = sp.symbols("xi beta", positive=True)
a0, cl, cw = sp.symbols("a0 c_l c_w", real=True)

r = sp.exp(xi) - 1
g = 1 - sp.exp(-xi)
G1 = g / xi
E1 = sp.exp(a0 * xi) / G1
mu = 2 + a0

# manufactured NON-SEPARABLE smooth fields (identity check: any smooth trio works)
A = 1 + xi * sp.cos(bt) + xi**2 * sp.sin(2 * bt) + sp.exp(-xi / 2) * sp.cos(3 * bt)
B = 2 + bt * xi + xi**2 * sp.cos(bt) + sp.exp(-xi / 3) * sp.sin(bt)
P = bt + xi * sp.sin(bt) + xi**2 + sp.exp(-xi / 5) * sp.cos(2 * bt)

# ---- the coded divided residuals, transcribed from polar_cornerreg.residual ----
LAa = A + xi * (sp.diff(A, xi) + a0 * A)
LB2b = 2 * B + xi * (sp.diff(B, xi) + (1 + 2 * a0) * B)
LPp = 2 * P + xi * (sp.diff(P, xi) + mu * P)

RO = (E1 * (-(LPp) * sp.diff(A, bt) + sp.diff(P, bt) * LAa
            + G1 * sp.cos(bt) * LB2b - sp.sin(bt) * sp.diff(B, bt))
      + cl * (-(G1) * LAa) + cw * A)
RB = (-E1 * (LPp * sp.diff(B, bt) - sp.diff(P, bt) * LB2b)
      + cl * (B - G1 * LB2b) + 2 * cw * B)
RP = (G1**2 * (xi**2 * sp.diff(P, xi, 2) + (4 * xi + 2 * mu * xi**2) * sp.diff(P, xi)
               + (2 + 4 * mu * xi + mu * mu * xi**2) * P)
      + G1 * (1 - xi * G1) * LPp + sp.diff(P, bt, 2) + xi * G1**2 * A)

# ---- the continuum dynamic-rescaling right-hand sides -------------------------
Om = sp.exp(a0 * xi) * xi * A
Bf = sp.exp((1 + 2 * a0) * xi) * xi**2 * B
Ps = sp.exp(mu * xi) * xi**2 * P

ds = lambda f: g * sp.diff(f, xi)          # d/ds  (s = ln r), corner frame
db = lambda f: sp.diff(f, bt)

dOm = (-cl * ds(Om)
       - (ds(Ps) * db(Om) - db(Ps) * ds(Om)) / r**2
       + cw * Om
       + (sp.cos(bt) * ds(Bf) - sp.sin(bt) * db(Bf)) / r)
dBf = (-cl * ds(Bf)
       - (ds(Ps) * db(Bf) - db(Ps) * ds(Bf)) / r**2
       + (cl + 2 * cw) * Bf)
poisson = ds(ds(Ps)) + sp.diff(Ps, bt, 2) + r**2 * Om

claims = {
    "RO' - d_tau A": RO - sp.exp(-a0 * xi) / xi * dOm,
    "RB' - d_tau B": RB - sp.exp(-(1 + 2 * a0) * xi) / xi**2 * dBf,
    "RP' - (Poisson)": RP - sp.exp(-mu * xi) / xi**2 * poisson,
}

pts = [(sp.Rational(7, 10), sp.Rational(3, 7), sp.Rational(-3424, 10000),
        sp.Rational(30065, 10000), sp.Rational(-10295, 10000)),
       (sp.Rational(31, 10), sp.Rational(11, 9), sp.Rational(-41, 100),
        sp.Rational(27, 10), sp.Rational(-9, 8)),
       (sp.Rational(97, 10), sp.Rational(1, 50), sp.Rational(-1, 3),
        sp.Rational(5, 2), sp.Rational(-4, 5))]

print("Q1 REALIZATION -- continuum identity check (50-digit evaluation)", flush=True)
print(f"{'claim':<18s} " + "  ".join(f"pt{i}" for i in range(len(pts))), flush=True)
worst = 0.0
for name, expr in claims.items():
    vals = []
    for (xv, bv, av, lv, wv) in pts:
        v = expr.subs({xi: xv, bt: bv, a0: av, cl: lv, cw: wv})
        vals.append(abs(sp.N(v, 50)))
        worst = max(worst, float(vals[-1]))
    print(f"{name:<18s} " + "  ".join(f"{float(v):.3e}" for v in vals), flush=True)
print(f"\nworst |residual| over all claims and points = {worst:.3e}", flush=True)
print("VERDICT:", "IDENTITY HOLDS -- mass matrix is the identity on (A,B)"
      if worst < 1e-40 else "FAILED -- the realization is NOT d_tau(A,B) = (RO',RB')",
      flush=True)
