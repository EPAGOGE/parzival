import numpy as np, dedalus.public as d3
import logging
logging.getLogger('dedalus').setLevel(logging.ERROR)

MU = 1.6576
S0, S1 = 10.0, 25.0

def run(Ns, Nb, mode="A", subst=False):
    c = d3.CartesianCoordinates("s", "b")
    dist = d3.Distributor(c, dtype=np.float64)
    sb = d3.ChebyshevT(c["s"], size=Ns, bounds=(S0, S1))
    bb = d3.ChebyshevT(c["b"], size=Nb, bounds=(0.0, np.pi/2))
    sg = dist.local_grid(sb); bg = dist.local_grid(bb)
    P = dist.Field(name="P", bases=(sb, bb))
    ts1 = dist.Field(name="ts1", bases=bb)
    ts2 = dist.Field(name="ts2", bases=bb)
    tb1 = dist.Field(name="tb1", bases=sb)
    tb2 = dist.Field(name="tb2", bases=sb)
    R = dist.Field(name="R", bases=(sb, bb))
    ds = lambda F: d3.Differentiate(F, c["s"])
    db = lambda F: d3.Differentiate(F, c["b"])
    Ls = lambda F, n: d3.Lift(F, sb.derivative_basis(2), n)
    Lb = lambda F, n: d3.Lift(F, bb.derivative_basis(2), n)
    ang = np.sin(2*bg) + 0.3*np.sin(4*bg)
    if subst:
        exact = np.broadcast_to(ang, (Ns, Nb)).copy() * np.ones((Ns,1))
        R["g"] = (MU**2-4)*np.sin(2*bg) + 0.3*(MU**2-16)*np.sin(4*bg)
    else:
        exact = np.exp(MU*sg) * ang
        R["g"] = np.exp(MU*sg) * ((MU**2-4)*np.sin(2*bg) + 0.3*(MU**2-16)*np.sin(4*bg))
    ns = dict(P=P, ts1=ts1, ts2=ts2, tb1=tb1, tb2=tb2, R=R,
              ds=ds, db=db, Ls=Ls, Lb=Lb, mu=MU, s=c["s"], b=c["b"])
    if subst:
        main = "ds(ds(P)) + 2*mu*ds(P) + mu**2*P + db(db(P))"
        rob = ("ds(P)(s='left')  = 0", "ds(P)(s='right') = 0")
    else:
        main = "ds(ds(P)) + db(db(P))"
        rob = ("ds(P)(s='left')  - mu*P(s='left')  = 0",
               "ds(P)(s='right') - mu*P(s='right') = 0")
    taus = " + Ls(ts1,-1) + Ls(ts2,-2) + Lb(tb1,-1) + Lb(tb2,-2)"
    prob = d3.LBVP([P, ts1, ts2, tb1, tb2], namespace=ns)
    prob.add_equation(main + taus + " = R")
    prob.add_equation("P(b='left')  = 0")
    prob.add_equation("P(b='right') = 0")
    prob.add_equation(rob[0]); prob.add_equation(rob[1])
    solver = prob.build_solver()
    solver.solve()
    P.change_scales(1)
    num = np.asarray(P["g"]).copy()
    err = np.abs(num - exact)
    glob = err.max()/np.abs(exact).max()
    per_s = (err.max(axis=1)/np.maximum(np.abs(exact).max(axis=1), 1e-300)).max()
    return glob, per_s, np.abs(num).max(), np.abs(exact).max(), solver

for subst in (False, True):
    print(f"--- subst={subst} ---")
    for (Ns,Nb) in [(32,24),(48,32),(64,48),(96,64)]:
        try:
            g,ps,nm,em,_ = run(Ns,Nb,subst=subst)
            print(f"Ns={Ns:3d} Nb={Nb:3d}  globrel={g:.3e}  worst_per_s_rel={ps:.3e}  |num|max={nm:.6e} |exact|max={em:.6e}")
        except Exception as e:
            print(f"Ns={Ns:3d} Nb={Nb:3d}  EXC {type(e).__name__}: {e}")
