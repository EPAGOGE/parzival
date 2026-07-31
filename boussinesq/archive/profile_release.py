"""
CONTINUATION for the 2D Boussinesq blowup profile: converge with the exponents PINNED,
then RELEASE them so the final values are independently determined.

Stage A pins c_l, c_w at the Chen-Hou values and drops both integral gauges. That
converges cleanly (residual falls by a factor of 2 per Newton step, 2000x total), but
c_l and c_w are INPUTS there, so gamma is not a measurement.

Stage B rebuilds the problem over THE SAME FIELDS -- which now hold the stage-A
solution -- with the two pin equations swapped back for the two weighted integral
gauges, so c_l and c_w become unknowns again. From a 1e-4 residual instead of the
seed's 150%, Newton should stay on the blowup branch rather than sliding to the
c_l -> 0 non-concentrating root (which it reaches robustly from a poor seed: residuals
fell 10x while c_l walked 2.10 -> -0.0006).

**Only the stage-B output is an independent value of gamma.**
Scoring targets (NOT fed to stage B): c_l=3.00649898, c_w=-1.02942516,
gamma=2.9205600 [Chen-Hou arXiv:2210.07191 eq 2.23]; gamma=2.91 [Luo-Hou PNAS 2014].
"""
import argparse, json, pathlib
import numpy as np
import dedalus.public as d3

C_L_T, C_W_T = 3.00649898, -1.02942516
GAMMA_T = 2.9205600
ALPHA = C_W_T / C_L_T


def main(N=32, Ybox=8.0, itersA=14, itersB=20, dampA=0.5, dampB=0.3,
         seed_sign=-1.0, out="../runs/profile_release.json"):
    co = d3.CartesianCoordinates("y1", "y2")
    dist = d3.Distributor(co, dtype=np.float64)
    b1 = d3.ChebyshevT(co["y1"], size=N, bounds=(0, Ybox), dealias=3 / 2)
    b2 = d3.ChebyshevT(co["y2"], size=N, bounds=(0, Ybox), dealias=3 / 2)
    y1 = dist.local_grid(b1); y2 = dist.local_grid(b2)
    ey1, ey2 = co.unit_vector_fields(dist)

    Om = dist.Field(name="Om", bases=(b1, b2))
    B = dist.Field(name="B", bases=(b1, b2))
    Psi = dist.Field(name="Psi", bases=(b1, b2))
    c_l = dist.Field(name="c_l"); c_w = dist.Field(name="c_w")
    t1 = dist.Field(name="t1", bases=b2); t2 = dist.Field(name="t2", bases=b2)
    t3 = dist.Field(name="t3", bases=b1); t4 = dist.Field(name="t4", bases=b1)
    s1 = dist.Field(name="s1", bases=b2); q1 = dist.Field(name="q1", bases=b2)

    lift1 = lambda F, n: d3.Lift(F, b1.derivative_basis(2), n)
    lift2 = lambda F, n: d3.Lift(F, b2.derivative_basis(2), n)
    L1 = lambda F, n: d3.Lift(F, b1.derivative_basis(1), n)
    d1 = lambda F: d3.Differentiate(F, co["y1"])
    d2 = lambda F: d3.Differentiate(F, co["y2"])
    U = d3.skew(d3.grad(Psi))                      # clean gradient, no tau contamination

    y1f = dist.Field(bases=b1); y1f["g"] = y1
    y2f = dist.Field(bases=b2); y2f["g"] = y2
    Y1, Y2 = np.meshgrid(np.ravel(y1), np.ravel(y2), indexing="ij")
    R2 = Y1 ** 2 + Y2 ** 2
    r2f = dist.Field(bases=(b1, b2)); r2f["g"] = R2.reshape(r2f["g"].shape)
    wt = dist.Field(bases=(b1, b2)); wt["g"] = ((1 + R2) ** -2.0).reshape(wt["g"].shape)

    p = (1 + 2 * ALPHA) / 2.0
    A0 = 1.0; C0 = C_L_T * A0 / 4.0
    B["g"] = (seed_sign * C0 * Y1 ** 2 * (1 + R2) ** (p - 1)).reshape(B["g"].shape)
    Om["g"] = (seed_sign * A0 * Y1 * (1 + R2) ** ((ALPHA - 1) / 2)).reshape(Om["g"].shape)
    c_l["g"] = C_L_T; c_w["g"] = C_W_T

    integ = lambda ex: d3.Integrate(d3.Integrate(ex, co["y1"]), co["y2"])
    sval = lambda ex: float(np.ravel(ex.evaluate()["g"])[0])
    CLT, CWT = C_L_T, C_W_T

    RES_OM = c_w * Om + d1(B) - (c_l * y1f + U @ ey1) * d1(Om) - (c_l * y2f + U @ ey2) * d2(Om)
    RES_B = (c_l + 2 * c_w) * B - (c_l * y1f + U @ ey1) * d1(B) - (c_l * y2f + U @ ey2) * d2(B)
    resid = lambda: (float(np.abs(RES_OM.evaluate()["g"]).max()),
                     float(np.abs(RES_B.evaluate()["g"]).max()))

    def core(extra_eqs, ns):
        prob = d3.NLBVP([Psi, Om, B, c_l, c_w, t1, t2, t3, t4, s1, q1], namespace=ns)
        prob.add_equation("lap(Psi) + lift1(t1,-1) + lift1(t2,-2)"
                          " + lift2(t3,-1) + lift2(t4,-2) + Om = 0")
        for bc in ["Psi(y1=0) = 0", "Psi(y1=Ybox) = 0", "Psi(y2=0) = 0", "Psi(y2=Ybox) = 0"]:
            prob.add_equation(bc)
        prob.add_equation("c_w*Om + d1(B) - (c_l*y1f + U@ey1)*d1(Om)"
                          " - (c_l*y2f + U@ey2)*d2(Om) + L1(s1,-1) = 0")
        prob.add_equation("(c_l + 2*c_w)*B - (c_l*y1f + U@ey1)*d1(B)"
                          " - (c_l*y2f + U@ey2)*d2(B) + L1(q1,-1) = 0")
        prob.add_equation("Om(y1=0) = 0")
        prob.add_equation("d1(B)(y1=0) = 0")
        for e in extra_eqs:
            prob.add_equation(e)
        return prob.build_solver()

    hist = {"A": [], "B": []}

    def newton(solver, iters, damping, tag):
        r0 = resid()
        print(f"  [{tag}] start residual |R_Om|={r0[0]:.4e} |R_B|={r0[1]:.4e}", flush=True)
        for it in range(1, iters + 1):
            solver.newton_iteration(damping=damping)
            ro, rb = resid()
            cl = float(np.ravel(c_l["g"])[0]); cw = float(np.ravel(c_w["g"])[0])
            g = -cl / cw if abs(cw) > 1e-300 else float("nan")
            nOm = float(np.abs(Om["g"]).max()); nB = float(np.abs(B["g"]).max())
            hist[tag].append(dict(it=it, res_Om=ro, res_B=rb, c_l=cl, c_w=cw, gamma=g,
                                  amp_Om=nOm, amp_B=nB))
            print(f"  [{tag}] {it:>3} |R_Om|={ro:.4e} |R_B|={rb:.4e} "
                  f"||Om||={nOm:.4e} ||B||={nB:.4e} c_l={cl:+.5f} gamma={g:+.5f}", flush=True)
            if not all(np.isfinite(v) for v in (ro, rb, cl, cw)):
                print(f"  [{tag}] non-finite -- stop"); break
            if max(ro, rb) < 1e-10:
                print(f"  [{tag}] converged"); break

    # ---------- STAGE A: exponents PINNED (branch selection) ----------
    print("STAGE A -- c_l, c_w PINNED at Chen-Hou (inputs, not measurements)")
    newton(core(["c_l = CLT", "c_w = CWT"], dict(locals(), **globals())), itersA, dampA, "A")

    # ---------- STAGE B: RELEASE -- gauges from the converged stage-A field ----------
    E1 = sval(integ(wt * Om ** 2))
    E2 = sval(integ(wt * r2f * Om ** 2))
    print(f"\nSTAGE B -- RELEASED. gauges taken from the stage-A solution: "
          f"E1={E1:.6e} E2={E2:.6e}")
    print("  c_l, c_w are now UNKNOWNS; whatever they land on is independently ours.")
    ns = dict(locals(), **globals())
    newton(core(["integ(wt*Om**2) = E1", "integ(wt*r2f*Om**2) = E2"], ns), itersB, dampB, "B")

    fin = hist["B"][-1] if hist["B"] else None
    res = dict(N=N, Ybox=Ybox, history=hist, target=dict(c_l=C_L_T, c_w=C_W_T, gamma=GAMMA_T),
               released=fin)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(res, indent=2))
    if fin:
        print(f"\n[RELEASED]  c_l={fin['c_l']:.6f} (target {C_L_T})")
        print(f"            c_w={fin['c_w']:.6f} (target {C_W_T})")
        print(f"            gamma={fin['gamma']:.6f} (target {GAMMA_T}; Luo-Hou 2.91)")
        print(f"            residual |R_Om|={fin['res_Om']:.3e} |R_B|={fin['res_B']:.3e}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--Ybox", type=float, default=8.0)
    ap.add_argument("--itersA", type=int, default=14)
    ap.add_argument("--itersB", type=int, default=20)
    ap.add_argument("--dampA", type=float, default=0.5)
    ap.add_argument("--dampB", type=float, default=0.3)
    ap.add_argument("--out", default="../runs/profile_release.json")
    a = ap.parse_args()
    main(N=a.N, Ybox=a.Ybox, itersA=a.itersA, itersB=a.itersB,
         dampA=a.dampA, dampB=a.dampB, out=a.out)
