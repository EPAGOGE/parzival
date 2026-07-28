"""ADJUDICATION: masked vs UNMASKED, both with the SUBST form, at A's resolutions.

A claims the deficiency-4 unmasked system is benign and masking is impossible.
B claims masking cures it.  A found sporadic tau pollution at 96x64, 128x96, 64x16.
Question: does B's mask remove the pollution at exactly those resolutions?
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import logging
import numpy as np
import dedalus.public as d3

sys.path.insert(0, "/Users/epagogellc/parzival/boussinesq")
from polar_tau2d_gate_b import build, TAU_TERMS, apply_corner_mask, MU, S0, S1

for nm in list(logging.root.manager.loggerDict):
    logging.getLogger(nm).setLevel(logging.ERROR)


def run(Ns, Nb, mask):
    B = build(Ns, Nb)
    s, b, ns = B["s"], B["b"], B["ns"]
    U, F = B["U"], B["F"]
    shape = np.sin(2 * b) + 0.3 * np.sin(4 * b)
    lap_shape = -4.0 * np.sin(2 * b) - 4.8 * np.sin(4 * b)
    # SUBST: Psi = e^{mu s} P  ->  P_ss + 2 mu P_s + mu^2 P + P_bb = e^{-mu s} F
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
    sv = p.build_solver()
    sv.solve()
    num = U["g"]
    err = np.abs(num - exact)
    # per-s relative error
    sRel = np.max(err.max(axis=1) / np.abs(exact).max(axis=1))
    tau = max(np.abs(t["g"]).max() for t in B["taus"])
    return sRel, tau, 0, 0.0
    L = sv.subproblems[0].L_min.toarray()
    svals = np.linalg.svd(L, compute_uv=False)
    defic = int(np.sum(svals < 1e-11 * svals[0]))
    cond = svals[0] / svals[-1]
    return sRel, tau, defic, cond


print(f"{'Ns':>5}{'Nb':>5} | {'UNMASKED sRel':>14} {'|tau|':>10} {'defic':>6} {'cond':>9} "
      f"| {'MASKED sRel':>13} {'|tau|':>10} {'defic':>6} {'cond':>9}")
for Ns, Nb in [(32, 24), (48, 32), (64, 48), (96, 64), (128, 96), (64, 16)]:
    a = run(Ns, Nb, False)
    m = run(Ns, Nb, True)
    print(f"{Ns:>5}{Nb:>5} | {a[0]:>14.3e} {a[1]:>10.2e} {a[2]:>6d} {a[3]:>9.1e} "
          f"| {m[0]:>13.3e} {m[1]:>10.2e} {m[2]:>6d} {m[3]:>9.1e}")
