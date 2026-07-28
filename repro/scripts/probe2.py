import numpy as np, dedalus.public as d3, logging
logging.getLogger('dedalus').setLevel(logging.ERROR)
np.set_printoptions(precision=3)
MU = 1.6576; S0, S1 = 10.0, 25.0

def build(Ns, Nb, variant, subst=True):
    c = d3.CartesianCoordinates("s", "b")
    dist = d3.Distributor(c, dtype=np.float64)
    sb = d3.ChebyshevT(c["s"], size=Ns, bounds=(S0, S1))
    bb = d3.ChebyshevT(c["b"], size=Nb, bounds=(0.0, np.pi/2))
    sg = dist.local_grid(sb); bg = dist.local_grid(bb)
    P = dist.Field(name="P", bases=(sb, bb))
    ts1 = dist.Field(name="ts1", bases=bb); ts2 = dist.Field(name="ts2", bases=bb)
    tb1 = dist.Field(name="tb1", bases=sb); tb2 = dist.Field(name="tb2", bases=sb)
    R = dist.Field(name="R", bases=(sb, bb))
    ds = lambda F: d3.Differentiate(F, c["s"]); db = lambda F: d3.Differentiate(F, c["b"])
    Ls = lambda F, n: d3.Lift(F, sb.derivative_basis(2), n)
    Lb = lambda F, n: d3.Lift(F, bb.derivative_basis(2), n)
    Ls1 = lambda F, n: d3.Lift(F, sb.derivative_basis(1), n)
    Lb1 = lambda F, n: d3.Lift(F, bb.derivative_basis(1), n)
    ang = np.sin(2*bg) + 0.3*np.sin(4*bg)
    exact = (np.ones((Ns,1))*ang) if subst else np.exp(MU*sg)*ang
    Rg = (MU**2-4)*np.sin(2*bg) + 0.3*(MU**2-16)*np.sin(4*bg)
    R["g"] = Rg if subst else np.exp(MU*sg)*Rg
    ns = dict(P=P, ts1=ts1, ts2=ts2, tb1=tb1, tb2=tb2, R=R, ds=ds, db=db,
              Ls=Ls, Lb=Lb, Ls1=Ls1, Lb1=Lb1, mu=MU, s=c["s"], b=c["b"])
    main = ("ds(ds(P)) + 2*mu*ds(P) + mu**2*P + db(db(P))" if subst
            else "ds(ds(P)) + db(db(P))")
    rob = (("ds(P)(s='left') = 0", "ds(P)(s='right') = 0") if subst else
           ("ds(P)(s='left')  - mu*P(s='left')  = 0",
            "ds(P)(s='right') - mu*P(s='right') = 0"))
    if variant == "A":       # 4 taus, deriv_basis(2), -1/-2 each direction
        taus = " + Ls(ts1,-1) + Ls(ts2,-2) + Lb(tb1,-1) + Lb(tb2,-2)"
        vs = [P, ts1, ts2, tb1, tb2]
    elif variant == "A_swapidx":  # -2/-1 order swapped (should be identical)
        taus = " + Ls(ts1,-2) + Ls(ts2,-1) + Lb(tb1,-2) + Lb(tb2,-1)"
        vs = [P, ts1, ts2, tb1, tb2]
    elif variant == "db1":   # WRONG: derivative_basis(1) for both lifts
        taus = " + Ls1(ts1,-1) + Ls1(ts2,-2) + Lb1(tb1,-1) + Lb1(tb2,-2)"
        vs = [P, ts1, ts2, tb1, tb2]
    elif variant == "s_only":  # WRONG: no b-direction taus
        taus = " + Ls(ts1,-1) + Ls(ts2,-2)"
        vs = [P, ts1, ts2]
    elif variant == "same_index":  # WRONG: both lifts at -1 in each dir
        taus = " + Ls(ts1,-1) + Ls(ts2,-1) + Lb(tb1,-1) + Lb(tb2,-1)"
        vs = [P, ts1, ts2, tb1, tb2]
    elif variant == "deep":  # -3/-4 instead of -1/-2
        taus = " + Ls(ts1,-3) + Ls(ts2,-4) + Lb(tb1,-3) + Lb(tb2,-4)"
        vs = [P, ts1, ts2, tb1, tb2]
    prob = d3.LBVP(vs, namespace=ns)
    prob.add_equation(main + taus + " = R")
    prob.add_equation("P(b='left')  = 0"); prob.add_equation("P(b='right') = 0")
    prob.add_equation(rob[0]); prob.add_equation(rob[1])
    solver = prob.build_solver()
    return solver, P, exact, dict(ts1=ts1,ts2=ts2,tb1=tb1,tb2=tb2)

def report(Ns, Nb, variant, subst=True):
    try:
        solver, P, exact, taus = build(Ns, Nb, variant, subst)
        solver.build_matrices(solver.subproblems, ['L'])
        sp = solver.subproblems[0]
        M = sp.L_min.toarray()
        cond = np.linalg.cond(M); rank = np.linalg.matrix_rank(M)
        solver.solve(); P.change_scales(1)
        num = np.asarray(P["g"]).copy()
        e = np.abs(num-exact)
        glob = e.max()/np.abs(exact).max()
        pers = (e.max(axis=1)/np.maximum(np.abs(exact).max(axis=1),1e-300)).max()
        tn = max(float(np.abs(t['c']).max()) for t in taus.values()) if taus else 0.0
        print(f"  {variant:10s} Ns={Ns:3d} Nb={Nb:3d} shape={M.shape} rank={rank} "
              f"cond={cond:.3e} glob={glob:.3e} per_s={pers:.3e} |num|={np.abs(num).max():.6e} maxtau={tn:.3e}")
    except Exception as ex:
        print(f"  {variant:10s} Ns={Ns:3d} Nb={Nb:3d}  EXC {type(ex).__name__}: {str(ex)[:120]}")

def main():
    print("== SUBST form, tau variants (32x24) ==")
    for v in ["A","A_swapidx","db1","s_only","same_index","deep"]:
        report(32,24,v)
    print("== variant A across shapes (subst) ==")
    for (a,b) in [(16,12),(24,24),(32,32),(48,32),(64,48),(96,64),(128,96),(64,64),(20,20),(12,8)]:
        report(a,b,"A")
    print("== variant A, RAW form ==")
    for (a,b) in [(32,24),(48,32),(64,48),(96,64),(128,96)]:
        report(a,b,"A",subst=False)

if __name__ == "__main__":
    main()
