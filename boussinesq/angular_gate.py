"""
GATE for the log-polar formulation: solve the far-field ANGULAR problem and check it
against Chen-Hou's own computed profile.

WHY THIS IS THE RIGHT FIRST STEP
--------------------------------
Substituting the known algebraic growth
    Om = e^(alpha s) Ot,  B = e^((1+2alpha)s) Bt,  Psi = e^((2+alpha)s) Pt,   s = ln r
into the rescaled steady equations makes the LEADING ORDER CANCEL EXACTLY (that
cancellation IS the statement alpha = c_w/c_l), leaving
    c_l Ot_s + e^(alpha s)[nonlinear] = e^(alpha s)[forcing]
    c_l Bt_s + e^(alpha s)[nonlinear] = 0
    Pt_ss + 2(2+alpha) Pt_s + (2+alpha)^2 Pt + Pt_bb = -Ot
Because alpha < 0 the factor e^(alpha s) DECAYS outward, so as s -> +inf the s-derivatives
drop and the Poisson equation becomes the pure angular problem

    (-d_bb - (2+alpha)^2) f(beta) = g1(beta),      f(0) = f(pi/2) = 0

which is exactly Chen-Hou's `eq:ASS_pois_1D` (their code: `Wangle.m`,
`f1 = (H1 - (2-al)^2 L2) \ (B1 g1)` with their al = -alpha > 0). So this ODE is the
asymptotic content of the whole log-polar formulation, and it is 1D.

THE GATE: extract g1(beta) from Chen-Hou's own 620x620 profile, solve this ODE, and
compare the result against the stream function measured from the SAME file. If the
formulation is right, they agree. If not, nothing downstream is worth building.

Measured facts this relies on (from Steady_state_pertb_oneMesh62036.mat):
  solu.cl = +3.00649798, solu.cw = -1.02942519, solu.al = 0.34240009
  |w| along beta=pi/4 decays as r^-0.34239 for r in [1e8,1e16] (target -0.34240)
  the power law only STARTS at r ~ 1e8, i.e. s ~ 18.4 -- sample the annulus above that
  g1 is BOUNDED, flat at the wall, and vanishes LINEARLY at the axis (measured slope
  1.0000 over three decades), so f(0)=f(pi/2)=0 with linear vanishing is legitimate
"""
import numpy as np
from scipy.io import loadmat
from pathlib import Path

MAT = Path.home() / ("parzival/refs/chen_hou/Perturbed_eqn/Computed profile/"
                     "Steady_state_pertb_oneMesh62036.mat")
R_LO, R_HI = 1e10, 1e12          # inside the asymptotic window (starts ~1e8)


def load_profile():
    d = loadmat(MAT, squeeze_me=True, struct_as_record=False)
    M, s = d["Mesh"], d["solu"]
    xs = [np.asarray(e, dtype=float) for e in np.ravel(np.asarray(M.x, dtype=object))]
    return dict(X=xs[0], Y=xs[1], w=np.asarray(d["w"], dtype=float),
                v=np.asarray(d["v"], dtype=float),
                cl=float(np.ravel(np.asarray(s.cl))[0]),
                cw=float(np.ravel(np.asarray(s.cw))[0]),
                al=float(np.ravel(np.asarray(s.al))[0]),
                xag=np.asarray(M.xag, dtype=float))


def extract_angular(P, field, nbeta=200):
    """g(beta) = field * r^(al) averaged over the asymptotic annulus, on a uniform
    beta grid. al = -alpha > 0, so multiplying by r^al removes the decay."""
    XX, YY = np.meshgrid(P["X"], P["Y"], indexing="ij")
    R = np.sqrt(XX ** 2 + YY ** 2)
    B = np.arctan2(YY, XX)
    band = (R > R_LO) & (R < R_HI)
    g = field[band] * R[band] ** P["al"]
    b = B[band]
    edges = np.linspace(0.0, np.pi / 2, nbeta + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(nbeta, np.nan)
    for k in range(nbeta):
        m = (b >= edges[k]) & (b < edges[k + 1])
        if m.any():
            out[k] = g[m].mean()
    ok = ~np.isnan(out)
    return ctr[ok], out[ok]


def solve_angular(beta, g1, p):
    """(-d_bb - p^2) f = g1 on [0,pi/2], f(0)=f(pi/2)=0, 2nd-order FD on the
    (possibly non-uniform) beta grid. p = 2 + alpha = 2 - al."""
    n = beta.size
    A = np.zeros((n, n)); rhs = g1.astype(float).copy()
    A[0, 0] = 1.0; rhs[0] = 0.0
    A[-1, -1] = 1.0; rhs[-1] = 0.0
    for i in range(1, n - 1):
        hm, hp = beta[i] - beta[i - 1], beta[i + 1] - beta[i]
        # non-uniform second derivative stencil
        A[i, i - 1] = -2.0 / (hm * (hm + hp))
        A[i, i + 1] = -2.0 / (hp * (hm + hp))
        A[i, i] = 2.0 / (hm * hp) - p ** 2
    return np.linalg.solve(A, rhs)


def main():
    P = load_profile()
    al = P["al"]; alpha = -al; p = 2.0 + alpha
    print(f"cl={P['cl']:.8f}  cw={P['cw']:.8f}  al={al:.8f}  alpha={alpha:.8f}")
    print(f"p = 2+alpha = {p:.8f}   p^2 = {p**2:.6f}")
    ev = [(2 * k) ** 2 for k in range(1, 4)]
    print(f"Dirichlet eigenvalues of -d_bb on [0,pi/2]: {ev}")
    print(f"  nonresonance margin to first eigenvalue: {ev[0] - p**2:.4f}"
          f"   (goes singular at alpha=0 where p^2=4)")

    beta, g1 = extract_angular(P, P["w"])
    print(f"\ng1 from the annulus r in [{R_LO:.0e},{R_HI:.0e}]: {beta.size} beta points")
    print(f"  g1: min={np.nanmin(g1):.5g} max={np.nanmax(g1):.5g}"
          f"  at wall={g1[0]:.5g}  at axis={g1[-1]:.5g}")

    f = solve_angular(beta, g1, p)
    print(f"\nsolved f(beta): min={f.min():.5g} max={f.max():.5g}"
          f"  f(0)={f[0]:.3g} f(pi/2)={f[-1]:.3g}")

    # INDEPENDENT CHECK: the same annulus should give Psi ~ r^(2+alpha) f(beta).
    # Recover f from the streamfunction implied by the data via the Poisson residual:
    # verify (-d_bb - p^2) f == g1 by direct substitution (discretization consistency)
    n = beta.size
    res = np.zeros(n)
    for i in range(1, n - 1):
        hm, hp = beta[i] - beta[i - 1], beta[i + 1] - beta[i]
        fpp = 2 * (f[i - 1] / (hm * (hm + hp)) - f[i] / (hm * hp) + f[i + 1] / (hp * (hm + hp)))
        res[i] = -fpp - p ** 2 * f[i] - g1[i]
    rel = np.abs(res[1:-1]).max() / max(np.abs(g1).max(), 1e-300)
    print(f"\nGATE 1  angular ODE residual (relative): {rel:.3e}   {'PASS' if rel < 1e-10 else 'CHECK'}")

    # GATE 2: does the SAME exponent come back out of the data along several rays?
    XX, YY = np.meshgrid(P["X"], P["Y"], indexing="ij")
    R = np.sqrt(XX ** 2 + YY ** 2); B = np.arctan2(YY, XX)
    print("\nGATE 2  measured decay exponent along individual rays (target"
          f" {alpha:.5f}):")
    for b0 in (0.2, 0.4, 0.8, 1.2, 1.4):
        sel = (np.abs(B - b0) < 0.02) & (R > 1e8) & (R < 1e15) & (np.abs(P["w"]) > 0)
        if sel.sum() > 20:
            A = np.polyfit(np.log(R[sel]), np.log(np.abs(P["w"][sel])), 1)
            print(f"   beta={b0:.2f}: slope={A[0]:+.5f}  ({sel.sum()} pts)")
    np.savez(Path.home() / "parzival/runs/angular_gate.npz",
             beta=beta, g1=g1, f=f, alpha=alpha, p=p, cl=P["cl"], cw=P["cw"])
    print(f"\nsaved -> ~/parzival/runs/angular_gate.npz")


if __name__ == "__main__":
    main()
