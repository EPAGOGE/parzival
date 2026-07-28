"""Derivation 2/2: exact scaling symmetry of the corner-regularized residuals.

Substitute A -> s*A, B -> s**p * B, P -> s**q * P, cl -> s**c * cl, cw -> s**c * cw
into the EXACT divided residuals of polar_cornerreg.py's docstring and demand each
residual maps to a pure power of s times itself.  Fully symbolic: A,B,P are
undetermined functions of (xi, b); E1, G1 undetermined functions of xi (their
alpha-dependence is untouched by the scaling, so they are inert coefficients).
"""
import sympy as sm

xi, b, s, cl, cw, a0, mu = sm.symbols("xi b s cl cw a0 mu", positive=True)
A = sm.Function("A")(xi, b)
B = sm.Function("B")(xi, b)
P = sm.Function("P")(xi, b)
G1 = sm.Function("G1")(xi)
E1 = sm.Function("E1")(xi)


def LA(F):
    return F + xi * (sm.diff(F, xi) + a0 * F)


def LB2(F):
    return 2 * F + xi * (sm.diff(F, xi) + (1 + 2 * a0) * F)


def LP(F):
    return 2 * F + xi * (sm.diff(F, xi) + mu * F)


def RO(A, B, P, cl, cw):
    return (E1 * (-LP(P) * sm.diff(A, b) + sm.diff(P, b) * LA(A)
                  + G1 * sm.cos(b) * LB2(B) - sm.sin(b) * sm.diff(B, b))
            - cl * G1 * LA(A) + cw * A)


def RB(A, B, P, cl, cw):
    return (-E1 * (LP(P) * sm.diff(B, b) - sm.diff(P, b) * LB2(B))
            + cl * (B - G1 * LB2(B)) + 2 * cw * B)


def RP(A, B, P, cl, cw):
    return (G1**2 * (xi**2 * sm.diff(P, xi, 2) + (4 * xi + 2 * mu * xi**2) * sm.diff(P, xi)
                     + (2 + 4 * mu * xi + mu**2 * xi**2) * P)
            + G1 * (1 - xi * G1) * LP(P) + sm.diff(P, b, 2) + xi * G1**2 * A)


# ---- Part 1: find all (p, q, c) with each residual a pure s-power multiple ----
p, q, c = sm.symbols("p q c")
print("== structural check: candidate p=2, q=1, c=1 ==")
sA, sB, sP = s * A, s**2 * B, s * P
scl, scw = s * cl, s * cw
checks = [
    ("RO' -> s^2 RO'", sm.simplify(RO(sA, sB, sP, scl, scw) - s**2 * RO(A, B, P, cl, cw))),
    ("RB' -> s^3 RB'", sm.simplify(RB(sA, sB, sP, scl, scw) - s**3 * RB(A, B, P, cl, cw))),
    ("RP' -> s^1 RP'", sm.simplify(RP(sA, sB, sP, scl, scw) - s * RP(A, B, P, cl, cw))),
]
for label, diff in checks:
    print(f"{label}: residual difference = {diff}")

# ---- Part 2: uniqueness -- general (p,q,c), term-weight equations ------------
# term weights: RO': {q+1 (advection), p (buoyancy), c+1 (cl,cw rows)}
#               RB': {q+p, c+p}     RP': {q, 1}
sol = sm.solve([sm.Eq(q + 1, p), sm.Eq(p, c + 1), sm.Eq(q + p, c + p),
                sm.Eq(q, 1)], [p, q, c], dict=True)
print("unique weight solution:", sol)

# ---- Part 3: corner algebra at xi=0 (identity cl = 2 THXX / WX) --------------
WX, THXX, cc = sm.symbols("WX THXX c_P")
Ac = WX * sm.cos(b)
Bc = THXX / 2 * sm.cos(b)**2
Pc = cc * sm.sin(2 * b)
# at xi=0: G1=1, E1=1, LA->A, LB2->2B, LP->2P
RO0 = (-2 * Pc * sm.diff(Ac, b) + sm.diff(Pc, b) * Ac + sm.cos(b) * 2 * Bc
       - sm.sin(b) * sm.diff(Bc, b) - cl * Ac + cw * Ac)
RB0 = (-(2 * Pc * sm.diff(Bc, b) - sm.diff(Pc, b) * 2 * Bc)
       + cl * (Bc - 2 * Bc) + 2 * cw * Bc)
RO0s = sm.simplify(RO0 / sm.cos(b))
RB0s = sm.simplify(RB0 / sm.cos(b)**2)
print("RO'(0,b)/cos b   =", sm.expand_trig(sm.simplify(RO0s)))
print("RB'(0,b)/cos^2 b =", sm.expand_trig(sm.simplify(RB0s)))
solc = sm.solve([RO0s, RB0s], [cl, cc], dict=True)
print("corner algebra solve for (cl, c_P):", solc)
print("=> cl - 2*THXX/WX =", sm.simplify(solc[0][cl] - 2 * THXX / WX))
