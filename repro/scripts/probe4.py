"""Decisive test: INHOMOGENEOUS BC data with the naive 2+2 double-Chebyshev tau.
If the 4 redundant BC rows are only redundant on the zero RHS, this breaks."""
import numpy as np, dedalus.public as d3, logging
logging.getLogger('dedalus').setLevel(logging.ERROR)
MU = 1.6576; S0, S1 = 10.0, 25.0
SM = 0.5*(S0+S1)

def exact_fields(sg, bg):
    """Generic smooth P with NONZERO data in all four conditions."""
    t = sg - SM
    P   = (np.sin(2*bg) + 0.3*np.sin(4*bg) + 0.13*t*np.cos(3*bg)
           + 0.07*t**2*(1.0 + 0.2*np.cos(bg)))
    Ps  = 0.13*np.cos(3*bg) + 0.14*t*(1.0 + 0.2*np.cos(bg))
    Pss = 0.14*(1.0 + 0.2*np.cos(bg)) * np.ones_like(t)
    Pbb = (-4*np.sin(2*bg) - 0.3*16*np.sin(4*bg) - 0.13*9*t*np.cos(3*bg)
           - 0.07*t**2*0.2*np.cos(bg))
    return P, Ps, Pss, Pbb

def run(Ns, Nb, variant="A", inhomog=True):
    c = d3.CartesianCoordinates("s", "b")
    dist = d3.Distributor(c, dtype=np.float64)
    sb = d3.ChebyshevT(c["s"], size=Ns, bounds=(S0, S1))
    bb = d3.ChebyshevT(c["b"], size=Nb, bounds=(0.0, np.pi/2))
    sg = dist.local_grid(sb); bg = dist.local_grid(bb)
    P = dist.Field(name="P", bases=(sb, bb))
    ts1 = dist.Field(name="ts1", bases=bb); ts2 = dist.Field(name="ts2", bases=bb)
    tb1 = dist.Field(name="tb1", bases=sb); tb2 = dist.Field(name="tb2", bases=sb)
    R = dist.Field(name="R", bases=(sb, bb))
    D0 = dist.Field(name="D0", bases=sb); D1 = dist.Field(name="D1", bases=sb)
    RL = dist.Field(name="RL", bases=bb); RR = dist.Field(name="RR", bases=bb)
    ds = lambda F: d3.Differentiate(F, c["s"]); db = lambda F: d3.Differentiate(F, c["b"])
    Ls = lambda F, n: d3.Lift(F, sb.derivative_basis(2), n)
    Lb = lambda F, n: d3.Lift(F, bb.derivative_basis(2), n)

    Pe, Pse, Psse, Pbbe = exact_fields(sg, bg)
    R["g"] = Psse + 2*MU*Pse + MU**2*Pe + Pbbe
    if inhomog:
        b0 = np.array([[0.0]]); b1 = np.array([[np.pi/2]])
        D0["g"] = exact_fields(sg, b0)[0]
        D1["g"] = exact_fields(sg, b1)[0]
        RL["g"] = exact_fields(np.array([[S0]]), bg)[1]
        RR["g"] = exact_fields(np.array([[S1]]), bg)[1]
    ns = dict(P=P, ts1=ts1, ts2=ts2, tb1=tb1, tb2=tb2, R=R, D0=D0, D1=D1, RL=RL, RR=RR,
              ds=ds, db=db, Ls=Ls, Lb=Lb, mu=MU, s=c["s"], b=c["b"])
    taus = " + Ls(ts1,-1) + Ls(ts2,-2) + Lb(tb1,-1) + Lb(tb2,-2)"
    prob = d3.LBVP([P, ts1, ts2, tb1, tb2], namespace=ns)
    prob.add_equation("ds(ds(P)) + 2*mu*ds(P) + mu**2*P + db(db(P))" + taus + " = R")
    prob.add_equation("P(b='left')  = D0")
    prob.add_equation("P(b='right') = D1")
    prob.add_equation("ds(P)(s='left')  = RL")
    prob.add_equation("ds(P)(s='right') = RR")
    solver = prob.build_solver()
    solver.solve(); P.change_scales(1)
    num = np.asarray(P["g"]).copy()
    e = np.abs(num - Pe)
    return (e.max()/np.abs(Pe).max(), np.abs(num).max(), np.abs(Pe).max(),
            max(float(np.abs(t['c']).max()) for t in (ts1,ts2,tb1,tb2)))

print("=== INHOMOGENEOUS BC data, naive 2+2 tau ===")
for (a,b) in [(24,16),(32,24),(48,32),(64,48),(96,64),(128,96),(48,48),(64,64),(160,120)]:
    try:
        g,nm,em,tn = run(a,b)
        print(f" Ns={a:4d} Nb={b:4d}  relerr={g:.3e}  |num|max={nm:.6f} |exact|max={em:.6f} maxtau={tn:.3e}")
    except Exception as ex:
        print(f" Ns={a:4d} Nb={b:4d}  EXC {type(ex).__name__}: {str(ex)[:90]}")
