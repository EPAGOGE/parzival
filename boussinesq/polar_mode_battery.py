"""
DECISIVE BATTERY on the converged unstable mode (+0.20 +- 0.64i, period ~9.8).

THE QUESTION
------------
`polar_spectrum.py` finds a resolution-converged unstable oscillatory pair. The
references report stability. Is that mode a property of the PROFILE, of my GAUGE, or of
the INNER TRUNCATION? Three tests, one pass.

TEST A -- WHERE DOES THE EIGENFUNCTION LIVE?
  Cheapest and most informative, and it was already computed and never examined. If the
  mode is localised at the inner edge `s = S0`, it is boundary-induced and an artefact of
  truncating a domain that should reach the corner. If it lives in the transition band
  `s in [2,15]` (where the march's residual concentrates) or spreads over the domain, it
  is not a boundary artefact.

TEST B -- IS IT THE GAUGE?
  The sharpest version of this is not to swap gauges, it is to REMOVE the gauge: freeze
  `c_l, c_w` at the reference constants so the operator is the plain linearisation with
  no projection at all. If the mode survives with NO gauge, the gauge cannot be its cause
  and the hypothesis recorded in section 14 is dead. Also run two genuine alternatives:
    - point conditions just inside the inner edge (a log-polar stand-in for Chen-Hou's
      corner point functionals, which is what section 14 proposed)
    - projection weighted toward the inner region, where the gauge signal is strongest

  NOTE ON THEORY: a gauge only selects the representative within a symmetry orbit, so it
  can move only the eigenvalues ASSOCIATED WITH THE ORBIT, not the transverse ones --
  PROVIDED the symmetry is exact. Here it is NOT: s-translation maps [S0,S1] to
  [S0-sig, S1-sig], so on a TRUNCATED domain with fixed boundaries the dilation symmetry
  is broken. That is precisely why the gauge could matter here and why this has to be
  measured rather than argued.

TEST C -- IS IT THE INNER TRUNCATION?
  Sweep `S0`. The march-rate version of this test came back flat (~0.35 across
  S0 = -4..-1) but a march rate mixes modes and transient amplification. The EIGENVALUE
  is the clean measurement.
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(HERE / fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def spectrum(M, gauge_mode="l2", target=0.20 + 0.64j):
    """Dense spectrum of the gauged (or ungauged) linearisation about the seed.
    Returns (eigenvalues sorted by -Re, eigenvectors, index of the mode nearest target,
    the free-index mask)."""
    ns, nb = M.ns, M.nb
    n2 = ns * nb
    Ot0, Bt0 = M.Ot0.copy(), M.Bt0.copy()

    # --- install the requested gauge ----------------------------------------
    if gauge_mode == "frozen":
        cl_ref, cw_ref = M.P["cl"], M.P["cw"]
        M.gauge = lambda Ot, Bt, pO, pB: (cl_ref, cw_ref, 1.0)
    elif gauge_mode == "point":
        # two POINT conditions just inside the inner edge -- the log-polar stand-in for
        # Chen-Hou's corner point functionals (their gauge lives at r=0, which log-polar
        # deletes). Row 1 is the first free s-node.
        j1, j2 = nb // 4, nb // 2

        def gp(Ot, Bt, pO, pB):
            KO, LO, MO = pO
            KB, LB, MB = pB
            A = np.array([[LO[1, j1], MO[1, j1]], [LB[1, j2], MB[1, j2]]])
            rhs = -np.array([KO[1, j1], KB[1, j2]])
            try:
                cl, cw = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                return M.P["cl"], M.P["cw"], np.inf
            return float(cl), float(cw), float(np.linalg.cond(A))
        M.gauge = gp
    elif gauge_mode == "inner":
        # L2 projection WEIGHTED toward the inner region, where section 11 measured the
        # gauge signal is strongest (48.7 deg at s=-2, collapsing to 4.9 deg by s=+5)
        w = np.exp(-(M.s - M.s[0])[:, None] / 3.0) * np.ones((1, nb))
        w[:2] = 0.0
        w[:, -2:] = 0.0

        def gi(Ot, Bt, pO, pB):
            KO, LO, MO = pO
            KB, LB, MB = pB
            vA = (Ot, 2.0 * Bt)
            vT = (M.ds(Ot) + M.a0 * Ot, M.ds(Bt) + 2.0 * M.a0 * Bt)
            dot = lambda X, Y: float(np.sum(w * X[0] * Y[0]) + np.sum(w * X[1] * Y[1]))
            A = np.array([[dot((LO, LB), vA), dot((MO, MB), vA)],
                          [dot((LO, LB), vT), dot((MO, MB), vT)]])
            rhs = -np.array([dot((KO, KB), vA), dot((KO, KB), vT)])
            cl, cw = np.linalg.solve(A, rhs)
            return float(cl), float(cw), float(np.linalg.cond(A))
        M.gauge = gi

    mask = np.ones((ns, nb), dtype=bool)
    mask[0, :] = False
    mask[:, -1] = False
    mask = np.concatenate([mask.ravel(), mask.ravel()])
    idx = np.where(mask)[0]
    n = idx.size
    scale = max(np.abs(Ot0).max(), np.abs(Bt0).max())

    J = np.empty((n, n))
    e = np.zeros(n)
    full = np.zeros(2 * n2)
    for j in range(n):
        e[j] = 1.0
        full[:] = 0.0
        full[idx] = e
        dO = full[:n2].reshape(ns, nb)
        dB = full[n2:].reshape(ns, nb)
        eps = 1e-6 * scale
        rp = M.rhs(Ot0 + eps * dO, Bt0 + eps * dB)
        rm = M.rhs(Ot0 - eps * dO, Bt0 - eps * dB)
        col = np.concatenate([((rp[0] - rm[0]) / (2 * eps)).ravel(),
                              ((rp[1] - rm[1]) / (2 * eps)).ravel()])
        J[:, j] = col[idx]
        e[j] = 0.0
    vals, vecs = np.linalg.eig(J)
    order = np.argsort(-vals.real)
    vals, vecs = vals[order], vecs[:, order]
    # the mode of interest: LOW |Im| (high-|Im| ones are grid-scale, |Im| ~ N) and
    # unstable
    cand = np.where((vals.real > 0.02) & (np.abs(vals.imag) < 5.0))[0]
    k = int(cand[np.argmin(np.abs(vals[cand] - target))]) if cand.size else 0
    return vals, vecs, k, idx


def localise(M, vec, idx):
    """Where does the eigenfunction live?"""
    ns, nb = M.ns, M.nb
    full = np.zeros(2 * ns * nb, dtype=complex)
    full[idx] = vec
    O = np.abs(full[:ns * nb].reshape(ns, nb))
    B = np.abs(full[ns * nb:].reshape(ns, nb))
    A = np.maximum(O / max(O.max(), 1e-300), B / max(B.max(), 1e-300))
    ps = A.max(axis=1)
    pb = A.max(axis=0)
    # energy fraction by s band
    bands = [(-99, 0), (0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 99)]
    tot = float((A ** 2).sum())
    frac = []
    for lo, hi in bands:
        m = (M.s >= lo) & (M.s < hi)
        frac.append(float((A[m] ** 2).sum()) / max(tot, 1e-300))
    return ps, pb, bands, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32)
    a = ap.parse_args()
    pm = _mod("pm", "polar_march.py")

    print("=" * 78)
    print("TEST A -- WHERE THE UNSTABLE EIGENFUNCTION LIVES")
    print("=" * 78)
    M = pm.March(a.N, a.N, -2.0, 25.0, filter_on=False)
    vals, vecs, k, idx = spectrum(M)
    print(f"  grid {a.N}x{a.N}   selected mode: {vals[k].real:+.6f} "
          f"{vals[k].imag:+.6f}i   (period {2*np.pi/max(abs(vals[k].imag),1e-9):.2f})")
    ps, pb, bands, frac = localise(M, vecs[:, k], idx)
    print("\n  energy fraction by s band:")
    for (lo, hi), f in zip(bands, frac):
        lo_s = "S0" if lo == -99 else f"{lo}"
        hi_s = "S1" if hi == 99 else f"{hi}"
        bar = "#" * int(round(60 * f))
        print(f"    s in [{lo_s:>2s},{hi_s:>2s}) : {f:7.4f}  {bar}")
    jmax = int(np.argmax(ps))
    print(f"\n  peak of |eigenfunction| at s = {M.s[jmax]:+.3f}"
          f"   (domain [{M.s[0]:.1f}, {M.s[-1]:.1f}])")
    print(f"  value at the inner edge / peak = {ps[0]/max(ps.max(),1e-300):.4f}"
          f"   (1.0 would mean boundary-localised)")
    print(f"  beta profile peak at beta/(pi/2) = {M.b[int(np.argmax(pb))]/(np.pi/2):.4f}")

    print("\n" + "=" * 78)
    print("TEST B -- IS IT THE GAUGE?  (frozen c = NO gauge at all)")
    print("=" * 78)
    print(f"  {'gauge':>10s} {'leading low-|Im| unstable':>30s} {'period':>9s}")
    for gm in ("l2", "frozen", "point", "inner"):
        Mg = pm.March(a.N, a.N, -2.0, 25.0, filter_on=False)
        try:
            v, _, kk, _ = spectrum(Mg, gauge_mode=gm)
            z = v[kk]
            print(f"  {gm:>10s} {z.real:+15.6f} {z.imag:+12.6f}i "
                  f"{2*np.pi/max(abs(z.imag),1e-9):9.2f}", flush=True)
        except Exception as ex:
            print(f"  {gm:>10s}   FAILED: {str(ex)[:44]}", flush=True)

    print("\n" + "=" * 78)
    print("TEST C -- IS IT THE INNER TRUNCATION?  (eigenvalue, not march rate)")
    print("=" * 78)
    print(f"  {'S0':>6s} {'r_inner':>9s} {'leading low-|Im| unstable':>30s} {'period':>9s}")
    for S0 in (-4.0, -3.0, -2.0, -1.0, 0.0):
        Ms = pm.March(a.N, a.N, S0, 25.0, filter_on=False)
        try:
            v, _, kk, _ = spectrum(Ms)
            z = v[kk]
            print(f"  {S0:6.1f} {np.exp(S0):9.4g} {z.real:+15.6f} {z.imag:+12.6f}i "
                  f"{2*np.pi/max(abs(z.imag),1e-9):9.2f}", flush=True)
        except Exception as ex:
            print(f"  {S0:6.1f}   FAILED: {str(ex)[:44]}", flush=True)


if __name__ == "__main__":
    main()
