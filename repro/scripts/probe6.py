"""Correctly-posed mixed-order nullity check + resolution stress sweep."""
import numpy as np, dedalus.public as d3, logging, time
logging.getLogger('dedalus').setLevel(logging.ERROR)
MU = 1.6576; S0, S1 = 10.0, 25.0

def nullity(Ns, Nb, ns_lift, nb_lift):
    """order in s = ns_lift, order in b = nb_lift; that many BCs each. Well posed."""
    c = d3.CartesianCoordinates("s", "b")
    dist = d3.Distributor(c, dtype=np.float64)
    sb = d3.ChebyshevT(c["s"], size=Ns, bounds=(S0, S1))
    bb = d3.ChebyshevT(c["b"], size=Nb, bounds=(0.0, np.pi/2))
    P = dist.Field(name="P", bases=(sb, bb))
    tsf = [dist.Field(name=f"ts{i}", bases=bb) for i in range(ns_lift)]
    tbf = [dist.Field(name=f"tb{i}", bases=sb) for i in range(nb_lift)]
    ds = lambda F: d3.Differentiate(F, c["s"]); db = lambda F: d3.Differentiate(F, c["b"])
    ns = dict(P=P, ds=ds, db=db, mu=MU, s=c["s"], b=c["b"])
    terms = []
    for i, f in enumerate(tsf):
        ns[f.name] = f
        ns[f"Ls{i}"] = (lambda F, n=-(i+1), o=ns_lift: d3.Lift(F, sb.derivative_basis(o), n))
        terms.append(f"Ls{i}({f.name})")
    for j, f in enumerate(tbf):
        ns[f.name] = f
        ns[f"Lb{j}"] = (lambda F, n=-(j+1), o=nb_lift: d3.Lift(F, bb.derivative_basis(o), n))
        terms.append(f"Lb{j}({f.name})")
    sop = "ds(ds(P))" if ns_lift == 2 else "ds(P)"
    bop = "db(db(P))" if nb_lift == 2 else "db(P)"
    prob = d3.LBVP([P] + tsf + tbf, namespace=ns)
    prob.add_equation(f"{sop} + {bop} + " + " + ".join(terms) + " = 0")
    prob.add_equation("P(b='left')  = 0")
    if nb_lift >= 2: prob.add_equation("P(b='right') = 0")
    prob.add_equation("ds(P)(s='left')  - mu*P(s='left')  = 0")
    if ns_lift >= 2: prob.add_equation("ds(P)(s='right') - mu*P(s='right') = 0")
    solver = prob.build_solver()
    solver.build_matrices(solver.subproblems, ['L'])
    M = solver.subproblems[0].L_min.toarray()
    S = np.linalg.svd(M, compute_uv=False)
    return M.shape[0], int((S/S[0] < 1e-12).sum()), S[0]/max(S[-1], 1e-300)

print("=== nullity vs (#s-lifts x #b-lifts), all WELL POSED ===")
for (a, b) in [(2,2),(1,2),(2,1),(1,1)]:
    dim, n, cond = nullity(24, 16, a, b)
    print(f" order_s={a} order_b={b}  dim={dim} nullity={n} predicted={a*b} "
          f"cond={cond:.2e}  {'MATCH' if n==a*b else 'MISMATCH'}")

import probe4
print("\n=== stress sweep: does the sparse LU ever fail? (inhomogeneous data) ===")
bad = 0
for a in [16,20,24,28,32,40,48,56,64,72,80,96,112,128]:
    for b in [12,16,24,32,48,64]:
        if b > a: continue
        try:
            t0=time.time(); g,nm,em,tn = probe4.run(a,b); dt=time.time()-t0
            flag = "" if g < 1e-9 else "  <-- POOR"
            if g >= 1e-9: bad += 1
            print(f" Ns={a:4d} Nb={b:4d} rel={g:.2e} maxtau={tn:9.2e} {dt:5.2f}s{flag}")
        except Exception as ex:
            bad += 1
            print(f" Ns={a:4d} Nb={b:4d} EXC {type(ex).__name__}: {str(ex)[:70]}")
print(f"failures/poor: {bad}")
