"""
GATE the DOUBLE-CHEBYSHEV TAU CONSTRUCTION -- the one piece of Dedalus machinery the
log-polar profile solver needs that no existing engine in this lab has ever exercised.

WHY THIS GATE
-------------
`dedalus_axisym.py` / `dedalus_bsq.py` are Fourier x Chebyshev: exactly ONE bounded
direction, so all tau terms are lifted out of a single basis and the tau bookkeeping is
the textbook case (dedalus_axisym.py:146-154, gate G2 at dedalus_axisym.py:240-262
inverts a manufactured Poisson problem that way).  `polar_radial_gate.py:76-100` does
the two-lift second-order reduction, but in ONE dimension only.

The log-polar profile problem is Chebyshev in BOTH (s, beta), with TWO conditions per
direction:

    Psi_ss + Psi_bb = F                     on [S0,S1] x [0, pi/2]
    Psi = 0                at beta = 0      (Dirichlet, wall)
    Psi = 0                at beta = pi/2   (Dirichlet, symmetry line)
    d_s Psi = mu Psi       at s = S0        (ROBIN, far-field power law)
    d_s Psi = mu Psi       at s = S1        (ROBIN, far-field power law)

Two bounded directions means the tau terms of the two directions OVERLAP in the corner
coefficients, and that overlap is the usual reason such systems come out singular.
This gate settles, with numbers, (a) whether the naive construction works, (b) exactly
how degenerate it is, (c) whether the degeneracy hurts, and (d) which formulation to
carry.

WHAT IT MEASURES (all against manufactured exact solutions, so the test is real)
-------------------------------------------------------------------------------
T1  single far-field mode      Psi = e^{mu s} sin(2b)
T2  two-mode, RAW vs SUBST     Psi = e^{mu s} (sin 2b + 0.3 sin 4b)
T3  INHOMOGENEOUS BC data      generic smooth Psi, nonzero data in all four conditions
T4  rank diagnosis             nullity, and WHERE the null space lives
T5  teeth                      deliberately-wrong tau setups must FAIL
T6  tau non-uniqueness         same Psi, different tau values, across matsolvers

Both manufactured solutions in T1/T2 satisfy all four conditions EXACTLY.  Note that a
sum of beta modes only satisfies one common Robin condition because the s-dependence
e^{mu s} is SHARED; sin(2b) and sin(4b) both vanish at b=0 and b=pi/2, so the Dirichlet
pair is satisfied mode by mode, but the Robin pair is satisfied only because mu is the
same for both.  That is exactly the structure of the real far field.

RUN
    ~/parzival/.venv-dedalus/bin/python3 ~/parzival/boussinesq/polar_tau2d_gate.py
"""
import numpy as np
import dedalus.public as d3

# ---------------------------------------------------------------------------- config
MU = 1.6576                  # far-field radial rate; = 2 + alpha to 4 digits for
                             # alpha = -0.34240009 (polar_bc_gate.py's measured value)
S0, S1 = 10.0, 25.0          # r = e^10 .. e^25, straddles the r ~ 1e8 law onset
SM = 0.5 * (S0 + S1)
B0, B1 = 0.0, np.pi / 2      # wall .. symmetry line
SPAN = np.exp(MU * (S1 - S0))          # ~6e10, the dynamic range the solver must eat


# ============================================================================
# THE TAU CONSTRUCTION.  This is the part to reproduce.
# ============================================================================
def build(Ns, Nb, subst=True, taus="2+2", matsolver=None):
    """Assemble the double-Chebyshev LBVP.

    TAU CONSTRUCTION ("2+2", the one that works) -- four tau fields:

        ts1, ts2  live on the BETA basis only   (functions of beta)
        tb1, tb2  live on the S    basis only   (functions of s)

        Ls(F, n) = d3.Lift(F, sb.derivative_basis(2), n)   n = -1, -2
        Lb(F, n) = d3.Lift(F, bb.derivative_basis(2), n)   n = -1, -2

    i.e. TWO lifts per bounded direction, both into that direction's
    `derivative_basis(2)` (second order equation -> second derivative basis), at lift
    indices -1 and -2 (the top two coefficients of that direction).  The tau field for
    a direction's conditions carries the dependence on the OTHER coordinate, which is
    why ts* live on bb and tb* live on sb.  Added to the equation as

        ... + Ls(ts1,-1) + Ls(ts2,-2) + Lb(tb1,-1) + Lb(tb2,-2) = F

    DOF count is square by construction:
        unknowns  = Ns*Nb (Psi) + 2*Nb (ts) + 2*Ns (tb)
        equations = Ns*Nb (main) + 2*Ns (two beta-Dirichlets, each a function of s)
                                 + 2*Nb (two s-Robins,        each a function of beta)

    CORNER OVERLAP: it is square but RANK DEFICIENT by exactly
    (#s lifts) x (#b lifts) = 4 -- see T4.  Nothing here removes that; T4 shows why it
    does not need removing.

    subst=True solves the SUBSTITUTED form Psi = e^{mu s} P (POLAR_SPEC's own line):
        P_ss + 2 mu P_s + mu^2 P + P_bb = e^{-mu s} F ,  Robin becomes  P_s = 0.
    subst=False solves for Psi directly and keeps the Robin as d_s Psi = mu Psi.
    """
    c = d3.CartesianCoordinates("s", "b")
    dist = d3.Distributor(c, dtype=np.float64)
    sb = d3.ChebyshevT(c["s"], size=Ns, bounds=(S0, S1))
    bb = d3.ChebyshevT(c["b"], size=Nb, bounds=(B0, B1))
    sg = dist.local_grid(sb)                      # shape (Ns, 1)
    bg = dist.local_grid(bb)                      # shape (1, Nb)

    P = dist.Field(name="P", bases=(sb, bb))
    ts1 = dist.Field(name="ts1", bases=bb)        # tau for s=S0 condition
    ts2 = dist.Field(name="ts2", bases=bb)        # tau for s=S1 condition
    tb1 = dist.Field(name="tb1", bases=sb)        # tau for b=0    condition
    tb2 = dist.Field(name="tb2", bases=sb)        # tau for b=pi/2 condition
    R = dist.Field(name="R", bases=(sb, bb))                       # source
    D0 = dist.Field(name="D0", bases=sb)          # data for P(b=0)
    D1 = dist.Field(name="D1", bases=sb)          # data for P(b=pi/2)
    RL = dist.Field(name="RL", bases=bb)          # data for s=S0  condition
    RR = dist.Field(name="RR", bases=bb)          # data for s=S1  condition

    ds = lambda F: d3.Differentiate(F, c["s"])
    db = lambda F: d3.Differentiate(F, c["b"])
    Ls = lambda F, n: d3.Lift(F, sb.derivative_basis(2), n)
    Lb = lambda F, n: d3.Lift(F, bb.derivative_basis(2), n)

    ns = dict(P=P, ts1=ts1, ts2=ts2, tb1=tb1, tb2=tb2, R=R, D0=D0, D1=D1, RL=RL, RR=RR,
              ds=ds, db=db, Ls=Ls, Lb=Lb, mu=MU, s=c["s"], b=c["b"])

    if subst:
        main = "ds(ds(P)) + 2*mu*ds(P) + mu**2*P + db(db(P))"
        rob = ("ds(P)(s='left') = RL", "ds(P)(s='right') = RR")
    else:
        main = "ds(ds(P)) + db(db(P))"
        rob = ("ds(P)(s='left')  - mu*P(s='left')  = RL",
               "ds(P)(s='right') - mu*P(s='right') = RR")

    if taus == "2+2":                       # the working construction
        tstr = " + Ls(ts1,-1) + Ls(ts2,-2) + Lb(tb1,-1) + Lb(tb2,-2)"
        vs = [P, ts1, ts2, tb1, tb2]
    elif taus == "same_index":              # WRONG: both lifts of a direction at -1
        tstr = " + Ls(ts1,-1) + Ls(ts2,-1) + Lb(tb1,-1) + Lb(tb2,-1)"
        vs = [P, ts1, ts2, tb1, tb2]
    elif taus == "s_only":                  # WRONG: no beta-direction taus at all
        tstr = " + Ls(ts1,-1) + Ls(ts2,-2)"
        vs = [P, ts1, ts2]
    elif taus == "db1":                     # first-derivative basis for a 2nd-order eqn
        L1s = lambda F, n: d3.Lift(F, sb.derivative_basis(1), n)
        L1b = lambda F, n: d3.Lift(F, bb.derivative_basis(1), n)
        ns.update(Ls=L1s, Lb=L1b)
        tstr = " + Ls(ts1,-1) + Ls(ts2,-2) + Lb(tb1,-1) + Lb(tb2,-2)"
        vs = [P, ts1, ts2, tb1, tb2]
    else:
        raise ValueError(taus)

    prob = d3.LBVP(vs, namespace=ns)
    prob.add_equation(main + tstr + " = R")
    prob.add_equation("P(b='left')  = D0")
    prob.add_equation("P(b='right') = D1")
    prob.add_equation(rob[0])
    prob.add_equation(rob[1])
    kw = {} if matsolver is None else {"matsolver": matsolver}
    solver = prob.build_solver(**kw)
    return dict(solver=solver, P=P, R=R, D0=D0, D1=D1, RL=RL, RR=RR,
                sg=sg, bg=bg, Ns=Ns, Nb=Nb, taus=(ts1, ts2, tb1, tb2), subst=subst)


# ------------------------------------------------------------------ error accounting
def errors(num, exact):
    """Return (global rel err, worst PER-S rel err).

    The per-s metric is the honest one for a 10-decade solution: for each s it
    normalises by that s's own magnitude, so a formulation that is only accurate near
    the large-s end cannot hide behind the global max.
    """
    e = np.abs(num - exact)
    glob = e.max() / max(np.abs(exact).max(), 1e-300)
    per_s = (e.max(axis=1) / np.maximum(np.abs(exact).max(axis=1), 1e-300)).max()
    return float(glob), float(per_s)


def solve_manufactured(Ns, Nb, modes, subst=True, matsolver=None, taus="2+2"):
    """Manufactured Psi = e^{mu s} * sum_k a_k sin(2k b).  Satisfies both Dirichlets
    (every sin(2k b) vanishes at 0 and pi/2) and both Robins (the s-dependence e^{mu s}
    is COMMON to all modes) exactly, so all four BC data are ZERO."""
    B = build(Ns, Nb, subst=subst, matsolver=matsolver, taus=taus)
    sg, bg = B["sg"], B["bg"]
    ang = sum(a * np.sin(2 * k * bg) for k, a in modes)
    lap_ang = sum(-(2 * k) ** 2 * a * np.sin(2 * k * bg) for k, a in modes)
    if subst:
        exact = np.ones((Ns, 1)) * ang               # P = ang, s-independent
        B["R"]["g"] = (MU ** 2) * ang + lap_ang      # P_ss=P_s=0
    else:
        exact = np.exp(MU * sg) * ang
        B["R"]["g"] = np.exp(MU * sg) * ((MU ** 2) * ang + lap_ang)
    # all four BC data are identically zero -> leave D0,D1,RL,RR at 0
    B["solver"].solve()
    B["P"].change_scales(1)
    num = np.asarray(B["P"]["g"]).copy()
    tmax = max(float(np.abs(t["c"]).max()) for t in B["taus"])
    return num, exact, tmax, B


# ------------------------------------------------------------- generic (inhomogeneous)
def generic_exact(sg, bg):
    """A generic smooth P with NONZERO data in all four conditions, and its
    derivatives, all analytic.  Used to test the corner compatibility question."""
    t = sg - SM
    P = (np.sin(2 * bg) + 0.3 * np.sin(4 * bg) + 0.13 * t * np.cos(3 * bg)
         + 0.07 * t ** 2 * (1.0 + 0.2 * np.cos(bg)))
    Ps = 0.13 * np.cos(3 * bg) + 0.14 * t * (1.0 + 0.2 * np.cos(bg))
    Pss = 0.14 * (1.0 + 0.2 * np.cos(bg)) * np.ones_like(t)
    Pbb = (-4 * np.sin(2 * bg) - 4.8 * np.sin(4 * bg) - 1.17 * t * np.cos(3 * bg)
           - 0.014 * t ** 2 * np.cos(bg))
    return P, Ps, Pss, Pbb


def solve_generic(Ns, Nb):
    """SUBST form with a generic manufactured P -> all four BC data NONZERO."""
    B = build(Ns, Nb, subst=True)
    sg, bg = B["sg"], B["bg"]
    Pe, Pse, Psse, Pbbe = generic_exact(sg, bg)
    B["R"]["g"] = Psse + 2 * MU * Pse + (MU ** 2) * Pe + Pbbe
    B["D0"]["g"] = generic_exact(sg, np.array([[B0]]))[0]
    B["D1"]["g"] = generic_exact(sg, np.array([[B1]]))[0]
    B["RL"]["g"] = generic_exact(np.array([[S0]]), bg)[1]
    B["RR"]["g"] = generic_exact(np.array([[S1]]), bg)[1]
    B["solver"].solve()
    B["P"].change_scales(1)
    num = np.asarray(B["P"]["g"]).copy()
    tmax = max(float(np.abs(t["c"]).max()) for t in B["taus"])
    return num, Pe, tmax


# ============================================================================
# T4 support: rank / null-space anatomy at arbitrary (order_s, order_b)
# ============================================================================
def rank_anatomy(Ns, Nb, order_s=2, order_b=2):
    """Build a WELL-POSED problem of order `order_s` in s and `order_b` in b, with that
    many conditions per direction and that many lifts per direction, then report the
    numerical nullity and how the null space splits across the variables."""
    c = d3.CartesianCoordinates("s", "b")
    dist = d3.Distributor(c, dtype=np.float64)
    sb = d3.ChebyshevT(c["s"], size=Ns, bounds=(S0, S1))
    bb = d3.ChebyshevT(c["b"], size=Nb, bounds=(B0, B1))
    P = dist.Field(name="P", bases=(sb, bb))
    tsf = [dist.Field(name=f"ts{i+1}", bases=bb) for i in range(order_s)]
    tbf = [dist.Field(name=f"tb{j+1}", bases=sb) for j in range(order_b)]
    ds = lambda F: d3.Differentiate(F, c["s"])
    db = lambda F: d3.Differentiate(F, c["b"])
    ns = dict(P=P, ds=ds, db=db, mu=MU, s=c["s"], b=c["b"])
    terms = []
    for i, f in enumerate(tsf):
        ns[f.name] = f
        ns[f"Ls{i}"] = (lambda F, n=-(i + 1), o=order_s:
                        d3.Lift(F, sb.derivative_basis(o), n))
        terms.append(f"Ls{i}({f.name})")
    for j, f in enumerate(tbf):
        ns[f.name] = f
        ns[f"Lb{j}"] = (lambda F, n=-(j + 1), o=order_b:
                        d3.Lift(F, bb.derivative_basis(o), n))
        terms.append(f"Lb{j}({f.name})")
    sop = "ds(ds(P))" if order_s == 2 else "ds(P)"
    bop = "db(db(P))" if order_b == 2 else "db(P)"
    prob = d3.LBVP([P] + tsf + tbf, namespace=ns)
    prob.add_equation(f"{sop} + {bop} + " + " + ".join(terms) + " = 0")
    prob.add_equation("P(b='left')  = 0")
    if order_b >= 2:
        prob.add_equation("P(b='right') = 0")
    prob.add_equation("ds(P)(s='left')  - mu*P(s='left')  = 0")
    if order_s >= 2:
        prob.add_equation("ds(P)(s='right') - mu*P(s='right') = 0")
    solver = prob.build_solver()
    solver.build_matrices(solver.subproblems, ["L"])
    sp = solver.subproblems[0]
    M = sp.L_min.toarray()
    U, S, Vt = np.linalg.svd(M)
    nn = int((S / S[0] < 1e-12).sum())
    out = dict(dim=M.shape[0], nullity=nn, cond=S[0] / max(S[-1], 1e-300), split={})
    if nn:
        Nfull = sp.pre_right.toarray() @ Vt[-nn:].T      # compressed -> full var vector
        off = 0
        for v in solver.problem.LHS_variables:
            sz = sp.field_size(v)
            out["split"][v.name] = float(np.linalg.norm(Nfull[off:off + sz]))
            off += sz
    return out


# ============================================================================
# REPORT
# ============================================================================
def main():
    hdr = lambda t: print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)
    rows = []
    print(f"mu = {MU}   s in [{S0}, {S1}]   beta in [0, pi/2]")
    print(f"e^(mu s) spans {SPAN:.3e}  ({MU*(S1-S0)/np.log(10):.1f} decades) -- the same "
          f"reach that killed all six Cartesian attempts")

    # ---------------------------------------------------------------- T1
    hdr("T1  single far-field mode   Psi = e^{mu s} sin(2 beta)   [2+2 tau, SUBST]")
    print(f"{'Ns':>5s} {'Nb':>5s} {'glob rel':>12s} {'per-s rel':>12s} {'max|tau|':>11s}"
          f" {'|num|max':>11s} {'|exact|max':>11s}")
    w1 = []
    for Ns, Nb in [(24, 16), (32, 24), (48, 32), (64, 48), (96, 64)]:
        num, ex, tm, _ = solve_manufactured(Ns, Nb, [(1, 1.0)])
        g, p = errors(num, ex)
        w1.append(p)
        print(f"{Ns:5d} {Nb:5d} {g:12.3e} {p:12.3e} {tm:11.3e} "
              f"{np.abs(num).max():11.6f} {np.abs(ex).max():11.6f}")
    rows.append(("T1 single mode, SUBST", max(w1) < 1e-10, max(w1), "<1e-10"))

    # ---------------------------------------------------------------- T2
    hdr("T2  two modes   Psi = e^{mu s}(sin 2b + 0.3 sin 4b)   RAW vs SUBST")
    print("   both modes vanish at b=0 and b=pi/2; the Robin pair holds because the\n"
          "   s-dependence e^{mu s} is COMMON to both modes.")
    print(f"\n{'Ns':>5s} {'Nb':>5s} | {'RAW glob':>11s} {'RAW per-s':>11s} "
          f"| {'SUB glob':>11s} {'SUB per-s':>11s} | verdict")
    wR, wS = [], []
    for Ns, Nb in [(32, 24), (48, 32), (64, 48), (96, 64), (128, 96)]:
        nR, eR, _, _ = solve_manufactured(Ns, Nb, [(1, 1.0), (2, 0.3)], subst=False)
        gR, pR = errors(nR, eR)
        nS, eS, _, _ = solve_manufactured(Ns, Nb, [(1, 1.0), (2, 0.3)], subst=True)
        gS, pS = errors(nS, eS)
        wR.append(pR); wS.append(pS)
        v = "SUBST" if pS < pR / 10 else ("RAW" if pR < pS / 10 else "tie")
        print(f"{Ns:5d} {Nb:5d} | {gR:11.3e} {pR:11.3e} | {gS:11.3e} {pS:11.3e} | {v}")
    print(f"\n   worst per-s:  RAW {max(wR):.3e}   SUBST {max(wS):.3e}   "
          f"-> SUBST better by {max(wR)/max(max(wS),1e-300):.3g}x")
    print("   RAW's GLOBAL error is fine and its PER-S error is not: solving for Psi\n"
          "   directly is accurate only near the large-s end.  Same verdict as\n"
          "   polar_radial_gate.py -- the solver must carry Psi = e^{(2+alpha)s} P.")
    rows.append(("T2 two modes, SUBST", max(wS) < 1e-10, max(wS), "<1e-10"))
    rows.append(("T2 two modes, RAW (expected poor)", max(wR) > 1e-6, max(wR), ">1e-6"))

    # ---------------------------------------------------------------- T3
    hdr("T3  INHOMOGENEOUS BC data (generic manufactured P; all four data nonzero)")
    print("   This is the case a Newton solve actually produces.  It also probes the\n"
          "   4 corner compatibility conditions  d_s P|(s edge, b edge) consistent\n"
          "   between the Dirichlet line and the Robin line.")
    print(f"{'Ns':>5s} {'Nb':>5s} {'rel err':>12s} {'max|tau|':>11s} {'|num|max':>11s}"
          f" {'|exact|max':>11s}")
    w3 = []
    for Ns, Nb in [(32, 24), (48, 32), (64, 48), (96, 64)]:
        num, ex, tm = solve_generic(Ns, Nb)
        g, _ = errors(num, ex)
        w3.append(g)
        print(f"{Ns:5d} {Nb:5d} {g:12.3e} {tm:11.3e} "
              f"{np.abs(num).max():11.6f} {np.abs(ex).max():11.6f}")
    print("   max|tau| is O(1..10) here and NOT converging: those are the undetermined\n"
          "   null-space components (T4).  The FIELD is still clean.")
    rows.append(("T3 inhomogeneous BC data", max(w3) < 1e-11, max(w3), "<1e-11"))

    # ---------------------------------------------------------------- T4
    hdr("T4  RANK ANATOMY -- how degenerate is the double-Chebyshev tau system?")
    print("   claim: numerical nullity == (#s lifts) x (#b lifts) == order_s * order_b,\n"
          "   and the null space lies ENTIRELY in the tau block (so Psi is unique).")
    print(f"\n{'ord_s':>6s} {'ord_b':>6s} {'dim':>6s} {'nullity':>8s} {'predicted':>10s}"
          f" {'cond':>10s} | null-space energy per variable")
    ok4 = True
    for os_, ob_ in [(2, 2), (1, 2), (2, 1), (1, 1)]:
        a = rank_anatomy(24, 16, os_, ob_)
        pred = os_ * ob_
        split = "  ".join(f"{k}={v:.1e}" for k, v in a["split"].items())
        # null vectors are unit norm, so a P-block energy of ~1e-15..1e-9 means the
        # null space is (numerically) entirely in the tau block
        ok4 &= (a["nullity"] == pred) and (a["split"].get("P", 1.0) < 1e-6)
        print(f"{os_:6d} {ob_:6d} {a['dim']:6d} {a['nullity']:8d} {pred:10d} "
              f"{a['cond']:10.2e} | {split}")
    print("\n   MECHANISM: Ls(ts_i,-i) writes into s-coefficient (Ns-i) for every b mode;\n"
          "   Lb(tb_j,-j) writes into b-coefficient (Nb-j) for every s mode.  The\n"
          "   coefficients (Ns-i, Nb-j) are therefore reachable BOTH ways -- one\n"
          "   redundancy per (i,j) pair, hence exactly order_s*order_b of them.\n"
          "   The (2,2) row is the real Psi problem: nullity 4, cond ~1e17-1e20.\n"
          "   The (1,1) row is the Om / B first-order transport equations: nullity 1.")
    print("   CONSEQUENCE: rank deficiency does NOT contaminate Psi (null-space energy in\n"
          "   the P block is ~1e-15), it only leaves order_s*order_b tau combinations\n"
          "   undetermined.  Do not try to remove it -- with two bounded directions there\n"
          "   is no square Dedalus formulation without it: dropping tau DOF to kill the\n"
          "   overlap leaves the system non-square, and Dedalus requires squareness.")
    rows.append(("T4 nullity == order_s*order_b, null space in taus only", ok4,
                 0.0 if ok4 else 1.0, "exact"))

    # ---------------------------------------------------------------- T5
    hdr("T5  TEETH -- deliberately wrong tau setups must FAIL")
    teeth = []
    for label, kwargs, why in [
        ("same_index  (both lifts of a direction at -1)", dict(taus="same_index"),
         "duplicate tau column -> exactly singular"),
        ("s_only      (no beta-direction taus)", dict(taus="s_only"),
         "unknowns short by 2*Ns -> non-square"),
    ]:
        try:
            num, ex, tm, _ = solve_manufactured(32, 24, [(1, 1.0)], **kwargs)
            g, p = errors(num, ex)
            failed = not (p < 1e-10)
            print(f"  {label:48s} -> solved, per-s rel {p:.3e}  "
                  f"{'(correctly BAD)' if failed else '*** WRONGLY OK ***'}")
        except Exception as ex_:
            failed = True
            print(f"  {label:48s} -> {type(ex_).__name__}: {str(ex_)[:60]}")
            print(f"  {'':48s}    expected: {why}")
        teeth.append(failed)
    # db1: derivative_basis(1) for a second-order equation -- report what actually happens
    try:
        num, ex, tm, _ = solve_manufactured(32, 24, [(1, 1.0)], taus="db1")
        g, p = errors(num, ex)
        print(f"  {'db1         (derivative_basis(1) lifts)':48s} -> per-s rel {p:.3e}"
              f"  (INFORMATIONAL: also works here)")
    except Exception as ex_:
        print(f"  {'db1         (derivative_basis(1) lifts)':48s} -> "
              f"{type(ex_).__name__}: {str(ex_)[:50]}")
    rows.append(("T5 wrong tau setups all rejected", all(teeth),
                 0.0 if all(teeth) else 1.0, "exact"))

    # ---------------------------------------------------------------- T6
    hdr("T6  TAU NON-UNIQUENESS -- direct consequence of T4, and a Newton trap")
    print("   Same problem, different matsolver pivoting.  If the taus are genuinely\n"
          "   undetermined, they will disagree by O(1) while Psi agrees to roundoff.")
    tvals = {}
    for msl in ["SuperluNaturalSpsolve", "SuperluColamdSpsolve",
                "SuperluNaturalFactorizedTranspose"]:
        num, ex, tm, B = solve_generic_ms(32, 24, msl)
        tvals[msl] = (num, [np.asarray(t["c"]).copy() for t in B["taus"]])
        print(f"   {msl:34s}  rel err {errors(num, ex)[0]:.3e}   max|tau| {tm:.4e}")
    ks = list(tvals)
    dP = max(np.abs(tvals[ks[0]][0] - tvals[k][0]).max() for k in ks[1:])
    dT = max(max(np.abs(a - b).max() for a, b in zip(tvals[ks[0]][1], tvals[k][1]))
             for k in ks[1:])
    print(f"\n   max |Psi_i - Psi_j| across matsolvers : {dP:.3e}   <- unique")
    print(f"   max |tau_i - tau_j| across matsolvers : {dT:.3e}   <- NOT unique")
    print("\n   *** NEWTON TRAP ***  Dedalus's NLBVP puts the tau fields in\n"
          "   `solver.perturbations` (core/solvers.py:446, :480-493), and the standard\n"
          "   Dedalus convergence measure sums |pert['c']| over ALL perturbations.  With\n"
          "   an O(1) undetermined tau block that sum NEVER goes to zero -- Newton will\n"
          "   look permanently stalled while the field is already converged.  Measure\n"
          "   the step norm over the PHYSICAL fields (Om, B, Psi, c_l, c_w) ONLY.\n"
          "   Corollary of the lab rule 'always report ||field|| beside the residual'.")
    rows.append(("T6 Psi unique across matsolvers", dP < 1e-11, dP, "<1e-11"))
    rows.append(("T6 taus NOT unique (expected)", dT > 1e-6, dT, ">1e-6"))

    # ---------------------------------------------------------------- T7
    hdr("T7  MATSOLVER ROBUSTNESS -- the degeneracy breaks several of them")
    print("   The 4-dim null space means a zero pivot is reachable.  Whether the solve\n"
          "   survives depends on the factorisation ORIENTATION and pivoting, and some\n"
          "   matsolvers fail SILENTLY (NaN) rather than raising.  Two resolutions are\n"
          "   used because some failures are resolution dependent -- the worst kind.")
    cands = ["SuperluColamdFactorizedTranspose",   # LBVP  default (MATRIX_FACTORIZER)
             "SuperluColamdSpsolve",               # NLBVP default (MATRIX_SOLVER)
             "SuperluNaturalSpsolve",
             "SuperluNaturalFactorizedTranspose",
             "SuperluColamdFactorized",
             "SuperluNaturalFactorized",
             "SparseInverse",
             "ScipyDenseLU"]
    dflt = {"SuperluColamdFactorizedTranspose": "  <- LBVP default",
            "SuperluColamdSpsolve": "  <- NLBVP default"}
    print(f"\n{'matsolver':36s} {'32x24':>22s} {'64x48':>22s}")
    robust = []
    for msl in cands:
        cells = []
        good = True
        for Ns, Nb in [(32, 24), (64, 48)]:
            try:
                num, ex, tm, _ = solve_generic_ms(Ns, Nb, msl)
                g, _ = errors(num, ex)
                if not np.isfinite(g) or g > 1e-9:
                    cells.append(f"BAD rel={g:.2e}"); good = False
                else:
                    cells.append(f"ok  rel={g:.2e}")
            except Exception as ex_:
                cells.append(f"{type(ex_).__name__}"); good = False
        print(f"{msl:36s} {cells[0]:>22s} {cells[1]:>22s}{dflt.get(msl,'')}")
        if good:
            robust.append(msl)
    print(f"\n   ROBUST: {', '.join(robust)}")
    print("   Pattern: the *Transpose and *Spsolve variants survive; the plain\n"
          "   `...Factorized` variants hit an exact zero pivot.  ScipyDenseLU returns\n"
          "   NaN at 32x24 and is fine at 64x48 -- silent, resolution-dependent garbage.\n"
          "   BOTH Dedalus defaults are in the robust set (dedalus.cfg:90 MATRIX_SOLVER =\n"
          "   SuperLUColamdSpsolve for NLBVP; dedalus.cfg:93 MATRIX_FACTORIZER =\n"
          "   SuperLUColamdFactorizedTranspose for LBVP), so DO NOT pass an explicit\n"
          "   `matsolver=` to the profile solver.  Take the default.")
    need = {"SuperluColamdFactorizedTranspose", "SuperluColamdSpsolve"}
    rows.append(("T7 both Dedalus defaults robust", need <= set(robust),
                 float(len(robust)), ">=2 incl. defaults"))

    # ---------------------------------------------------------------- T8
    hdr("T8  max|tau| IS THE CANARY -- accuracy loss is sporadic, not monotone")
    print("   T1/T2 use manufactured solutions that satisfy all four conditions\n"
          "   EXACTLY, so the true taus are ~0.  When the LU's pivot draw happens to\n"
          "   deposit part of the 4-dim null vector into the tau block, max|tau| jumps\n"
          "   to O(1e-2..10) and the field loses precision by roughly 1e-12 * max|tau|.\n"
          "   It is NOT a conditioning trend in N: 128x96 is bad and 160x96 is clean.")
    print(f"\n{'Ns':>5s} {'Nb':>5s} {'max|tau|':>11s} {'per-s rel':>12s}  regime")
    tab, law = [], True
    for Ns, Nb in [(32, 24), (48, 32), (64, 48), (96, 64), (128, 96),
                   (160, 96), (192, 128), (64, 16)]:
        num, ex, tm, _ = solve_manufactured(Ns, Nb, [(1, 1.0), (2, 0.3)], subst=True)
        g, p = errors(num, ex)
        clean = tm < 1e-10
        if clean and p > 1e-14:
            law = False
        print(f"{Ns:5d} {Nb:5d} {tm:11.3e} {p:12.3e}  "
              f"{'clean pivot draw' if clean else 'tau POLLUTED'}")
        tab.append((tm, p))
    pol = [(t, p) for t, p in tab if t >= 1e-10]
    if pol:
        print(f"\n   polluted draws: max|tau| {min(t for t,_ in pol):.1e}..{max(t for t,_ in pol):.1e}"
              f"  ->  err {min(p for _,p in pol):.1e}..{max(p for _,p in pol):.1e}")
    print("\n   The pivot draw is not even reproducible: SuperLU's ordering inside the\n"
          "   null space varies run to run, so max|tau| and the RAW per-s error in T2\n"
          "   jitter between identical invocations (e.g. 3.1e-4 vs 1.6e-4).  The PASS/FAIL\n"
          "   verdicts are stable; the last digits of the polluted rows are not.")
    print("\n   RULE FOR THE SOLVER: log max|tau| every Newton step next to ||field||.\n"
          "   If max|tau| is not small compared with the field, the step is degraded --\n"
          "   nudge Ns or Nb by a few modes and it goes away.  A silently large max|tau|\n"
          "   is the double-Chebyshev analogue of the 'residual falling by exactly the\n"
          "   damping factor' tell for Newton collapsing onto the zero field.")
    rows.append(("T8 clean pivot draw (max|tau|<1e-10) => err<1e-14", law,
                 0.0 if law else 1.0, "exact"))

    # ---------------------------------------------------------------- summary
    hdr("SUMMARY")
    allok = True
    for name, ok, val, tol in rows:
        allok &= bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:56s} {val:.3e}  ({tol})")
    print(f"\nGATE: {'PASS' if allok else 'FAIL'}")
    return allok


def solve_generic_ms(Ns, Nb, matsolver):
    """solve_generic with an explicit matsolver, for T6."""
    B = build(Ns, Nb, subst=True, matsolver=matsolver)
    sg, bg = B["sg"], B["bg"]
    Pe, Pse, Psse, Pbbe = generic_exact(sg, bg)
    B["R"]["g"] = Psse + 2 * MU * Pse + (MU ** 2) * Pe + Pbbe
    B["D0"]["g"] = generic_exact(sg, np.array([[B0]]))[0]
    B["D1"]["g"] = generic_exact(sg, np.array([[B1]]))[0]
    B["RL"]["g"] = generic_exact(np.array([[S0]]), bg)[1]
    B["RR"]["g"] = generic_exact(np.array([[S1]]), bg)[1]
    B["solver"].solve()
    B["P"].change_scales(1)
    num = np.asarray(B["P"]["g"]).copy()
    tmax = max(float(np.abs(t["c"]).max()) for t in B["taus"])
    return num, Pe, tmax, B


if __name__ == "__main__":
    import logging
    # dedalus names its loggers by bare module ("subsystems", "solvers", ...), not
    # under a "dedalus." prefix, so silence every already-registered logger.
    for _n in list(logging.root.manager.loggerDict):
        logging.getLogger(_n).setLevel(logging.ERROR)
    raise SystemExit(0 if main() else 1)
