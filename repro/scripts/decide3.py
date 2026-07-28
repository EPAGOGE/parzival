"""(1) Does A's 'Newton trap' actually fire?  (2) Does B's mask rescue A's broken matsolvers?"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import logging
import numpy as np
import dedalus.public as d3
from dedalus.tools.config import config

sys.path.insert(0, "/Users/epagogellc/parzival/boussinesq")
from polar_tau2d_gate_b import build, TAU_TERMS, apply_corner_mask, MU, S0, S1

for nm in list(logging.root.manager.loggerDict):
    logging.getLogger(nm).setLevel(logging.ERROR)

Ns, Nb = 32, 24

# ---------------------------------------------------------------- (1) Newton trap
print("(1) NLBVP step norm summed over ALL perturbations (taus included).")
print("    A predicts this NEVER reaches zero when unmasked.  B measures 2.3e-15.\n")
for mask in (False, True):
    B = build(Ns, Nb)
    ns, U, F = B["ns"], B["U"], B["F"]
    b, s = B["b"], B["s"]
    shape = np.sin(2 * b) + 0.3 * np.sin(4 * b)
    lap_shape = -4.0 * np.sin(2 * b) - 4.8 * np.sin(4 * b)
    exact = np.broadcast_to(shape, (Ns, Nb)).copy() * np.ones_like(s)
    F["g"] = MU**2 * shape + lap_shape + 0.1 * shape**3
    p = d3.NLBVP([U, B["ts1"], B["ts2"], B["tb1"], B["tb2"]], namespace=ns)
    p.add_equation("lap(U) + 2*MU*ds(U) + MU**2*U + " + TAU_TERMS + " = F - 0.1*U**3 + 0.1*U**3 - 0.1*U**3 + 0.1*U**3")
    e0 = p.add_equation("U(beta=0) = 0")
    e1 = p.add_equation("U(beta=BE) = 0")
    p.add_equation("ds(U)(s=S0) = 0")
    p.add_equation("ds(U)(s=S1) = 0")
    if mask:
        apply_corner_mask(B["tb1"], B["tb2"], e0, e1)
    sv = p.build_solver()
    U["g"] = 0.9 * shape
    hist = []
    for _ in range(8):
        sv.newton_iteration()
        pn = float(np.sqrt(sum(np.sum(np.abs(q["c"])**2) for q in sv.perturbations)))
        taun = max(np.abs(t["g"]).max() for t in B["taus"])
        hist.append((pn, taun))
        if pn < 1e-14:
            break
    err = float(np.abs(U["g"] - exact).max() / np.abs(exact).max())
    print(f"  mask={mask!s:5}  final err={err:.2e}")
    for i, (pn, tn) in enumerate(hist):
        print(f"      it{i}  |dx|(ALL perts, taus incl)={pn:.3e}   |tau| value={tn:.3e}")
    print()

# ------------------------------------------------------- (2) matsolvers under mask
print("(2) Matsolvers A reported BROKEN -- do they work once masked?\n")
names = ["SuperluColamdFactorizedTranspose", "SuperluColamdSpsolve",
         "SuperluColamdFactorized", "SuperluNaturalFactorized",
         "SparseInverse", "ScipyDenseLU", "DenseInverse"]
for mask in (False, True):
    print(f"  --- mask={mask} ---")
    for name in names:
        try:
            B = build(Ns, Nb)
            ns, U, F = B["ns"], B["U"], B["F"]
            b, s = B["b"], B["s"]
            shape = np.sin(2 * b) + 0.3 * np.sin(4 * b)
            lap_shape = -4.0 * np.sin(2 * b) - 4.8 * np.sin(4 * b)
            exact = np.broadcast_to(shape, (Ns, Nb)).copy() * np.ones_like(s)
            F["g"] = MU**2 * shape + lap_shape
            p = d3.LBVP([U, B["ts1"], B["ts2"], B["tb1"], B["tb2"]], namespace=ns)
            p.add_equation("lap(U) + 2*MU*ds(U) + MU**2*U + " + TAU_TERMS + " = F")
            e0 = p.add_equation("U(beta=0) = 0")
            e1 = p.add_equation("U(beta=BE) = 0")
            p.add_equation("ds(U)(s=S0) = 0")
            p.add_equation("ds(U)(s=S1) = 0")
            if mask:
                apply_corner_mask(B["tb1"], B["tb2"], e0, e1)
            sv = p.build_solver(matsolver=name)
            sv.solve()
            e = np.abs(U["g"] - exact)
            sRel = np.max(e.max(axis=1) / np.abs(exact).max(axis=1))
            tag = "ok " if np.isfinite(sRel) and sRel < 1e-10 else "BAD"
            print(f"    {name:36s} {tag} sRel={sRel:.2e}")
        except Exception as ex:
            print(f"    {name:36s} {type(ex).__name__}: {str(ex)[:50]}")
    print()
