"""(a) nullity == (#s-lifts)*(#b-lifts)?   (b) corner-incompatible data -> garbage?"""
import numpy as np, dedalus.public as d3, logging
logging.getLogger('dedalus').setLevel(logging.ERROR)
MU = 1.6576; S0, S1 = 10.0, 25.0

def nullity(Ns, Nb, ns_lift, nb_lift, order_s=2):
    c = d3.CartesianCoordinates("s", "b")
    dist = d3.Distributor(c, dtype=np.float64)
    sb = d3.ChebyshevT(c["s"], size=Ns, bounds=(S0, S1))
    bb = d3.ChebyshevT(c["b"], size=Nb, bounds=(0.0, np.pi/2))
    P = dist.Field(name="P", bases=(sb, bb))
    tsf = [dist.Field(name=f"ts{i}", bases=bb) for i in range(ns_lift)]
    tbf = [dist.Field(name=f"tb{i}", bases=sb) for i in range(nb_lift)]
    ds = lambda F: d3.Differentiate(F, c["s"]); db = lambda F: d3.Differentiate(F, c["b"])
    dbs = max(order_s, 1)
    ns = dict(P=P, ds=ds, db=db, mu=MU, s=c["s"], b=c["b"])
    terms = []
    for i, f in enumerate(tsf):
        ns[f.name] = f; ns[f"Ls{i}"] = (lambda F, n=-(i+1): d3.Lift(F, sb.derivative_basis(dbs), n))
        terms.append(f"Ls{i}({f.name})")
    for j, f in enumerate(tbf):
        ns[f.name] = f; ns[f"Lb{j}"] = (lambda F, n=-(j+1): d3.Lift(F, bb.derivative_basis(2), n))
        terms.append(f"Lb{j}({f.name})")
    main = ("ds(ds(P))" if order_s == 2 else "ds(P)") + " + db(db(P))"
    prob = d3.LBVP([P] + tsf + tbf, namespace=ns)
    prob.add_equation(main + " + " + " + ".join(terms) + " = 0")
    prob.add_equation("P(b='left')  = 0")
    if nb_lift >= 2: prob.add_equation("P(b='right') = 0")
    if ns_lift >= 1: prob.add_equation("ds(P)(s='left')  - mu*P(s='left')  = 0")
    if ns_lift >= 2: prob.add_equation("ds(P)(s='right') - mu*P(s='right') = 0")
    solver = prob.build_solver()
    solver.build_matrices(solver.subproblems, ['L'])
    M = solver.subproblems[0].L_min.toarray()
    S = np.linalg.svd(M, compute_uv=False)
    n = int((S/S[0] < 1e-12).sum())
    return M.shape[0], n, S[0]/S[-1] if S[-1] > 0 else np.inf

print("=== nullity vs (#s-lifts x #b-lifts) ===")
for (nsl, nbl, ordr, label) in [(2,2,2,"2nd order both (the real Psi problem)"),
                                (1,2,1,"1st order in s, 2nd in b"),
                                (2,1,2,"2nd in s, 1st in b"),
                                (1,1,1,"1st order both (Om/B transport)")]:
    dim, n, cond = nullity(24, 16, nsl, nbl, ordr)
    print(f" s-lifts={nsl} b-lifts={nbl} {label:38s} dim={dim} nullity={n} "
          f"predicted={nsl*nbl} cond={cond:.2e}  {'MATCH' if n==nsl*nbl else 'MISMATCH'}")

# ---------------------------------------------------------------- (b)
import probe4
def run_broken(Ns, Nb, bump):
    c = d3.CartesianCoordinates("s", "b")
    dist = d3.Distributor(c, dtype=np.float64)
    sb = d3.ChebyshevT(c["s"], size=Ns, bounds=(S0, S1))
    bb = d3.ChebyshevT(c["b"], size=Nb, bounds=(0.0, np.pi/2))
    sg = dist.local_grid(sb); bg = dist.local_grid(bb)
    P = dist.Field(name="P", bases=(sb, bb))
    ts1 = dist.Field(name="ts1", bases=bb); ts2 = dist.Field(name="ts2", bases=bb)
    tb1 = dist.Field(name="tb1", bases=sb); tb2 = dist.Field(name="tb2", bases=sb)
    R = dist.Field(bases=(sb, bb)); D0 = dist.Field(bases=sb); D1 = dist.Field(bases=sb)
    RL = dist.Field(bases=bb); RR = dist.Field(bases=bb)
    ds = lambda F: d3.Differentiate(F, c["s"]); db = lambda F: d3.Differentiate(F, c["b"])
    Ls = lambda F, n: d3.Lift(F, sb.derivative_basis(2), n)
    Lb = lambda F, n: d3.Lift(F, bb.derivative_basis(2), n)
    Pe, Pse, Psse, Pbbe = probe4.exact_fields(sg, bg)
    R["g"] = Psse + 2*MU*Pse + MU**2*Pe + Pbbe
    D0["g"] = probe4.exact_fields(sg, np.array([[0.0]]))[0]
    D1["g"] = probe4.exact_fields(sg, np.array([[np.pi/2]]))[0]
    RL["g"] = probe4.exact_fields(np.array([[S0]]), bg)[1] + bump   # BREAK the corner match
    RR["g"] = probe4.exact_fields(np.array([[S1]]), bg)[1]
    ns = dict(P=P, ts1=ts1, ts2=ts2, tb1=tb1, tb2=tb2, R=R, D0=D0, D1=D1, RL=RL, RR=RR,
              ds=ds, db=db, Ls=Ls, Lb=Lb, mu=MU, s=c["s"], b=c["b"])
    prob = d3.LBVP([P, ts1, ts2, tb1, tb2], namespace=ns)
    prob.add_equation("ds(ds(P)) + 2*mu*ds(P) + mu**2*P + db(db(P))"
                      " + Ls(ts1,-1) + Ls(ts2,-2) + Lb(tb1,-1) + Lb(tb2,-2) = R")
    prob.add_equation("P(b='left')  = D0"); prob.add_equation("P(b='right') = D1")
    prob.add_equation("ds(P)(s='left')  = RL"); prob.add_equation("ds(P)(s='right') = RR")
    solver = prob.build_solver(); solver.solve(); P.change_scales(1)
    num = np.asarray(P["g"]).copy()
    # measure how well the IMPOSED conditions are actually satisfied
    r1 = float(np.abs((P(b='left')).evaluate()['g'] - D0['g']).max())
    r3 = float(np.abs((ds(P)(s='left')).evaluate()['g'] - RL['g']).max())
    return np.abs(num).max(), np.abs(num - Pe).max()/np.abs(Pe).max(), r1, r3

print("\n=== corner-INCOMPATIBLE Robin data (RL += bump, breaks RL(0)=D0'(S0)) ===")
for bump in [0.0, 1e-8, 1e-3, 1.0]:
    try:
        nm, rel, r1, r3 = run_broken(48, 32, bump)
        print(f" bump={bump:8.1e}  |P|max={nm:.6e} rel_vs_unbroken_exact={rel:.3e} "
              f"BCresid(b=left)={r1:.3e} BCresid(s=left)={r3:.3e}")
    except Exception as ex:
        print(f" bump={bump:8.1e}  EXC {type(ex).__name__}: {str(ex)[:90]}")
