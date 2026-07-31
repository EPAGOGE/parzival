"""
GATE the RADIAL (s = ln r) machinery of the log-polar formulation, and settle the one
formulation choice the solver hinges on.

WHY RADIAL, WHY NOW
-------------------
`angular_gate.py` already gates the beta direction (ODE residual 1.65e-13 against
Chen-Hou's profile). `polar_ops_gate.py` gates the operator identities symbolically.
What is still untested is the RADIAL direction -- and that is precisely the direction
that killed all six Cartesian attempts: the power law does not begin until r ~ 1e8,
i.e. s ~ 18.4, and a box with `Ybox = 8` reaches only s = 2.08. In log-polar the same
reach costs a domain of length ~20 instead of ~1e8. This gate checks that Dedalus can
actually solve on that domain with the Robin far-field condition.

Because Psi = 0 at BOTH beta edges, expanding in `sin(2 k beta)` satisfies both
angular conditions identically (consistent with angular_gate's measured Dirichlet
eigenvalues (2k)^2), and the 2D Poisson problem decouples into independent 1D problems
in s -- one per k. That isolates exactly the new content.

    Psi = sum_k A_k(s) sin(2 k beta),   g(beta) = sum_k g_k sin(2 k beta)
    Psi_ss + Psi_bb = -e^{(2+a)s} g     =>   A_k'' - (2k)^2 A_k = -e^{(2+a)s} g_k

with the exact solution A_k = c_k e^{(2+a)s}, c_k = g_k / ((2k)^2 - (2+a)^2), which is
the same modal formula the angular ODE gives -- so the two gates must agree.

THE FORMULATION CHOICE THIS SETTLES
-----------------------------------
Over s in [10, 25] the true solution spans e^{(2+a)*15} ~ 6e10 -- ten decades. Asking
Chebyshev polynomials to represent that is a conditioning question, not a style
question. So both formulations are run head to head:

  RAW:   solve for A_k directly.                 exact answer: c_k e^{(2+a)s}
  SUBST: Psi = e^{(2+a)s} P, solve for P_k.      exact answer: c_k, a CONSTANT
         P_k'' + 2(2+a) P_k' + ((2+a)^2 - (2k)^2) P_k = -g_k,  Robin becomes P_k' = 0

SUBST is POLAR_SPEC's own line `Pt_ss + 2(2+alpha) Pt_s + (2+alpha)^2 Pt + Pt_bb = -Ot`.
If SUBST wins decisively, the solver must carry the substitution -- and the Cartesian
note "I imposed dOt = 0 (flat) and that FAILED" is explained: the tilde tends to an
ANGULAR function, not a constant, which in log-polar is automatic.
"""
import numpy as np
import dedalus.public as d3
from scipy.io import loadmat
from pathlib import Path

MAT = Path.home() / ("parzival/refs/chen_hou/Perturbed_eqn/Computed profile/"
                     "Steady_state_pertb_oneMesh62036.mat")
S0, S1 = 10.0, 25.0          # inside the asymptotic window (law starts at s ~ 18.4...
                             # and r=1e10..1e12 is s=23..27; [10,25] straddles it)
NS = 128
KMAX = 6


def alpha_and_gk(nk=KMAX, nbeta=400):
    """alpha from Chen-Hou's stored constants, and the sin(2k beta) coefficients of the
    measured angular source g(beta). Reuses angular_gate's extraction so the two gates
    are demonstrably about the same g."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "ag", str(Path(__file__).with_name("angular_gate.py")))
    ag = importlib.util.module_from_spec(spec); sys.modules["ag"] = ag
    spec.loader.exec_module(ag)
    P = ag.load_profile()
    beta, g1 = ag.extract_angular(P, P["w"], nbeta=nbeta)
    alpha = -P["al"]
    # least-squares projection onto sin(2k beta), k = 1..nk
    M = np.stack([np.sin(2 * k * beta) for k in range(1, nk + 1)], axis=1)
    gk, *_ = np.linalg.lstsq(M, g1, rcond=None)
    return alpha, gk, beta, g1, M


def solve_raw(k, gk, alpha, N=NS):
    """A'' - (2k)^2 A = -e^{(2+a)s} g_k ,  A' = (2+a) A at both ends."""
    mu = 2.0 + alpha
    c = d3.Coordinate("s")
    dist = d3.Distributor(c, dtype=np.float64)
    sb = d3.ChebyshevT(c, size=N, bounds=(S0, S1))
    A = dist.Field(name="A", bases=sb)
    t1 = dist.Field(name="t1"); t2 = dist.Field(name="t2")
    R = dist.Field(bases=sb)
    sg = dist.local_grid(sb)
    R["g"] = -np.exp(mu * sg) * gk
    ds = lambda F: d3.Differentiate(F, c)
    lift = lambda F, n: d3.Lift(F, sb.derivative_basis(2), n)
    ns = dict(A=A, t1=t1, t2=t2, R=R, ds=ds, lift=lift, k2=(2.0 * k) ** 2, mu=mu, s=c)
    p = d3.LBVP([A, t1, t2], namespace=ns)
    p.add_equation("ds(ds(A)) - k2*A + lift(t1,-1) + lift(t2,-2) = R")
    p.add_equation("ds(A)(s='left')  - mu*A(s='left')  = 0")
    p.add_equation("ds(A)(s='right') - mu*A(s='right') = 0")
    p.build_solver().solve()
    A.change_scales(1)
    return sg, np.asarray(A["g"]).copy(), mu


def solve_subst(k, gk, alpha, N=NS):
    """P'' + 2(2+a) P' + ((2+a)^2 - (2k)^2) P = -g_k ,  P' = 0 at both ends."""
    mu = 2.0 + alpha
    c = d3.Coordinate("s")
    dist = d3.Distributor(c, dtype=np.float64)
    sb = d3.ChebyshevT(c, size=N, bounds=(S0, S1))
    P = dist.Field(name="P", bases=sb)
    t1 = dist.Field(name="t1"); t2 = dist.Field(name="t2")
    R = dist.Field(bases=sb)
    sg = dist.local_grid(sb)
    R["g"] = -gk * np.ones_like(sg)
    ds = lambda F: d3.Differentiate(F, c)
    lift = lambda F, n: d3.Lift(F, sb.derivative_basis(2), n)
    ns = dict(P=P, t1=t1, t2=t2, R=R, ds=ds, lift=lift, mu=mu,
              lam=mu ** 2 - (2.0 * k) ** 2, s=c)
    p = d3.LBVP([P, t1, t2], namespace=ns)
    p.add_equation("ds(ds(P)) + 2*mu*ds(P) + lam*P + lift(t1,-1) + lift(t2,-2) = R")
    p.add_equation("ds(P)(s='left')  = 0")
    p.add_equation("ds(P)(s='right') = 0")
    p.build_solver().solve()
    P.change_scales(1)
    return sg, np.asarray(P["g"]).copy(), mu


def main():
    alpha, gk, beta, g1, M = alpha_and_gk()
    mu = 2.0 + alpha
    print(f"alpha = {alpha:+.8f}   2+alpha = {mu:.8f}")
    print(f"s domain [{S0}, {S1}] (r = {np.exp(S0):.2e} .. {np.exp(S1):.2e}), NS={NS}")
    print(f"true solution spans e^(mu*ds) = {np.exp(mu*(S1-S0)):.3e}  "
          f"({mu*(S1-S0)/np.log(10):.1f} decades)")
    rec = M @ gk
    print(f"g(beta) reconstruction from {KMAX} sine modes: rel L2 err = "
          f"{np.linalg.norm(rec-g1)/np.linalg.norm(g1):.3e}")
    print(f"\n{'k':>3s} {'g_k':>12s} {'c_k exact':>13s} "
          f"{'RAW rel err':>13s} {'SUBST rel err':>15s}  verdict")
    raw_w, sub_w = [], []
    for i, k in enumerate(range(1, KMAX + 1)):
        ck = gk[i] / ((2.0 * k) ** 2 - mu ** 2)
        sg, A, _ = solve_raw(k, gk[i], alpha)
        exact_A = ck * np.exp(mu * sg)
        # PER-POINT, not global L2. A global norm over 10.8 decades is dominated by
        # the large-s end and STRUCTURALLY CANNOT SEE small-s error -- it scored RAW at
        # 1.1e-14 while RAW's actual worst per-point error is 5.1e-05, all of it at
        # small s. Never score a multi-decade problem with a global norm.
        eR_glob = np.linalg.norm(A - exact_A) / max(np.linalg.norm(exact_A), 1e-300)
        eR = float(np.max(np.abs(A - exact_A) / np.maximum(np.abs(exact_A), 1e-300)))
        sg2, P, _ = solve_subst(k, gk[i], alpha)
        exact_P = ck * np.ones_like(sg2)
        eS_glob = np.linalg.norm(P - exact_P) / max(np.linalg.norm(exact_P), 1e-300)
        eS = float(np.max(np.abs(P - exact_P) / np.maximum(np.abs(exact_P), 1e-300)))
        raw_w.append(eR); sub_w.append(eS)
        v = "SUBST" if eS < eR / 10 else ("RAW" if eR < eS / 10 else "tie")
        print(f"{k:3d} {gk[i]:12.5g} {ck:13.6g} {eR:13.3e} {eS:15.3e}  {v}")
    print(f"\nworst RAW   rel err: {max(raw_w):.3e}")
    print(f"worst SUBST rel err: {max(sub_w):.3e}")
    win = max(raw_w) / max(max(sub_w), 1e-300)
    print(f"\nSUBST is {win:.3g}x more accurate at worst case (PER-POINT metric).")
    print("  For contrast, a GLOBAL L2 metric reports only ~13x and would let RAW pass.")
    ok = max(sub_w) < 1e-10
    print(f"GATE: substituted formulation {'PASS' if ok else 'FAIL'} (<1e-10 required)")
    if ok and win > 10:
        print("=> the solver MUST carry Psi = e^{(2+alpha)s} P. NOT optional: RAW's error\n"
              "   is ~1e-5 at the small-s end, which is the end that matters, and a global\n"
              "   norm hides it completely. An earlier version of this gate used a global\n"
              "   norm, concluded '13x, RAW not disqualified', and that conclusion was\n"
              "   WRONG -- see POLAR_SPEC section 3.")


if __name__ == "__main__":
    main()
