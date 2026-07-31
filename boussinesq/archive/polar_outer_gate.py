"""GATE THE OUTER BOUNDARY CONDITION.  Three checks, none of which mentions alpha.

CLAIM 1 (algebra).  The annihilator acts as advertised on the three far-field modes.
  Build the outer row's symbol P(lam; k) for a single beta mode k = 2j and evaluate it on
    const           lam = 0
    decaying        lam = -(k + mu)
    growing         lam = +(k - mu)
    forced tail     lam = a0
  `dtn`  must kill {const, decaying} and NOT the growing one.
  `dtn3` must kill {const, decaying, forced} and NOT the growing one.
  `neumann` kills only the constant -- which is the defect.

CLAIM 2 (conditioning).  cond(Poisson) ~ e^(2.3475 * XMAX) under `neumann`, because the
  BVP's solution space contains the growing branch.  Under `dtn` the growing branch is out
  of the solution space, so the growth with XMAX must FLATTEN.  This is the whole point:
  XMAX=32 currently fails Newton and XMAX=40 overflows.

CLAIM 3 (MMS).  Manufactured exact solution built from the DISCRETE beta eigenvector, so
  Pt_bb is exact and the only discretisation error is radial:

      Pt* = phi_j(b) [ 1 - e^(-(k_j + mu) xi) ]          (corner value 0, as required)

  Every variant should recover it; a variant that cannot is broken. Then repeat with the
  forced tail added,

      Pt* = phi_j(b) [ 1 - e^(-(k_j+mu) xi) + c ( e^(a0 xi) - 1 ) ]

  which `dtn3` admits exactly and `dtn`/`neumann` do not -- so the error ordering
  dtn3 << dtn, neumann is a PREDICTION, not a fit.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(HERE / fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def poisson_matrix(C):
    """Rebuild the assembled Poisson matrix (splu keeps no copy of it)."""
    nx, nb, g, mu = C.nx, C.nb, C.g, C.mu
    G2, Gm = np.diag(g ** 2), np.diag(g * (1.0 - g))
    As = G2 @ (C.Dx2 + 2 * mu * C.Dx + mu ** 2 * np.eye(nx)) + Gm @ (C.Dx + mu * np.eye(nx))
    A = sp.kron(sp.csr_matrix(As), sp.identity(nb, format="csr")) \
        + sp.kron(sp.identity(nx, format="csr"), sp.csr_matrix(C.Db2))
    A = sp.lil_matrix(A)
    rid = lambda i, j: i * nb + j
    for i in range(nx):
        for j in (0, nb - 1):
            r = rid(i, j)
            A.rows[r], A.data[r] = [r], [1.0]
    S = C._beta_root() if C.outer.startswith("dtn") else None
    a0, Dx, Dx2 = C.a0, C.Dx, C.Dx2
    if C.outer == "neumann":
        od, oc = Dx[nx - 1, :], None
    elif C.outer == "dtn":
        od, oc = Dx2[nx - 1, :] + mu * Dx[nx - 1, :], Dx[nx - 1, :]
    else:
        Dx3 = (Dx @ Dx2)[nx - 1, :]
        od = Dx3 + (mu - a0) * Dx2[nx - 1, :] - mu * a0 * Dx[nx - 1, :]
        oc = Dx2[nx - 1, :] - a0 * Dx[nx - 1, :]
    for j in range(1, nb - 1):
        r = rid(0, j)
        A.rows[r], A.data[r] = [r], [1.0]
        r = rid(nx - 1, j)
        acc = {rid(k, j): float(od[k]) for k in range(nx)}
        if oc is not None:
            for jj in range(1, nb - 1):
                s = float(S[j, jj])
                if s == 0.0:
                    continue
                for k in range(nx):
                    c = rid(k, jj)
                    acc[c] = acc.get(c, 0.0) + s * float(oc[k])
        cols = sorted(acc)
        A.rows[r], A.data[r] = cols, [acc[c] for c in cols]
    return sp.csr_matrix(A)


def symbol(variant, lam, k, mu, a0):
    """The outer row applied to e^(lam xi) phi_k, in closed form."""
    if variant == "neumann":
        return lam
    if variant == "dtn":
        return lam * (lam + k + mu)
    return lam * (lam + k + mu) * (lam - a0)


def check_algebra(mu, a0, k=2.0):
    print(f"CLAIM 1  algebra of the outer row   (k = 2j = {k}, mu = {mu:.6f}, a0 = {a0:.6f})")
    modes = {"const   lam=0": 0.0,
             f"decay   lam={-(k+mu):+.4f}": -(k + mu),
             f"grow    lam={+(k-mu):+.4f}": +(k - mu),
             f"forced  lam={a0:+.4f}": a0}
    print(f"  {'mode':>22} " + "".join(f"{v:>14}" for v in ("neumann", "dtn", "dtn3")))
    for nm, lam in modes.items():
        vals = [symbol(v, lam, k, mu, a0) for v in ("neumann", "dtn", "dtn3")]
        print(f"  {nm:>22} " + "".join(
            f"{('KILL' if abs(v) < 1e-12 else f'{v:+.4f}'):>14}" for v in vals))
    print()


def check_conditioning(Ns=(36,), XMAXs=(10.0, 15.0, 20.0, 25.0, 32.0, 40.0)):
    pc = _mod("pc", "polar_corner.py")
    print("CLAIM 2  cond(Poisson) vs XMAX.  predicted neumann growth ~ e^(2.3475 XMAX)")
    print(f"  {'N':>3} {'XMAX':>6} " + "".join(f"{v:>13}" for v in ("neumann", "dtn", "dtn3"))
          + f" {'e^(2.3475 L)':>14}")
    prev = {}
    for N in Ns:
        for X in XMAXs:
            row, out = [], []
            for v in ("neumann", "dtn", "dtn3"):
                C = pc.Corner(N, N, X, outer=v)
                A = poisson_matrix(C).toarray()
                c = float(np.linalg.cond(A))
                row.append(c)
                r = np.log(c / prev[v]) / (X - prev["_X"]) if v in prev else np.nan
                out.append(f"{c:13.3e}")
            print(f"  {N:3d} {X:6.1f} " + "".join(out) + f" {np.exp(2.3475 * X):14.3e}")
            for v, c in zip(("neumann", "dtn", "dtn3"), row):
                prev[v] = c
            prev["_X"] = X
    print("\n  growth rate d(ln cond)/d(XMAX) over the whole range:")
    for v in ("neumann", "dtn", "dtn3"):
        cs = []
        for X in XMAXs:
            C = pc.Corner(Ns[0], Ns[0], X, outer=v)
            cs.append(np.log(float(np.linalg.cond(poisson_matrix(C).toarray()))))
        p = np.polyfit(XMAXs, cs, 1)[0]
        print(f"    {v:>8}: {p:+.4f} per unit XMAX     (growing branch = +2.3475)")
    print()


def check_mms(N=36, XMAX=25.0, cforced=0.0):
    pc = _mod("pc", "polar_corner.py")
    tag = "decaying only" if cforced == 0.0 else f"decaying + {cforced} * forced tail"
    print(f"CLAIM 3  MMS at N={N} XMAX={XMAX}:  {tag}")
    print(f"  {'variant':>8} {'rel L-inf err':>15} {'rel L2 err':>13} {'err at outer node':>19}")
    for v in ("neumann", "dtn", "dtn3"):
        C = pc.Corner(N, N, XMAX, outer=v)
        nx, nb, mu, a0, g = C.nx, C.nb, C.mu, C.a0, C.g
        # discrete beta eigenpair, slowest interior mode
        M = -C.Db2[1:-1, 1:-1]
        w, V = np.linalg.eig(M)
        i0 = np.argmin(w.real)
        k = float(np.sqrt(w.real[i0]))
        phi = np.zeros(nb)
        phi[1:-1] = V[:, i0].real
        phi /= np.abs(phi).max()
        c1 = k + mu
        xi = C.x

        def f(lam):
            return np.exp(lam * xi) - 1.0                    # vanishes at xi = 0

        Pex = np.outer(-f(-c1) + cforced * f(a0), phi)       # 1 - e^{-c1 xi} + c(e^{a0 xi}-1)
        # RHS from the CONTINUOUS operator, using Db2 phi = -k^2 phi exactly
        def op(lam):
            return (g ** 2) * (lam ** 2 + 2 * mu * lam + mu ** 2) \
                + g * (1.0 - g) * (lam + mu) - k ** 2
        rhs = np.outer(-(op(-c1) - op(0.0) * 0.0) * 0.0, phi)  # placeholder, built below
        rhs = (np.outer(-(op(-c1)) * np.exp(-c1 * xi), phi)
               + np.outer((op(0.0)) * np.ones_like(xi), phi)
               + cforced * (np.outer(op(a0) * np.exp(a0 * xi), phi)
                            - np.outer(op(0.0) * np.ones_like(xi), phi)))
        # boundary rows carry 0 for this manufactured solution by construction
        b = rhs.ravel().copy()
        b[C.brows] = 0.0
        Pn = C.lu.solve(b).reshape(nx, nb)
        d = Pn - Pex
        sc = np.abs(Pex).max()
        print(f"  {v:>8} {np.abs(d).max()/sc:15.3e} {la.norm(d)/la.norm(Pex):13.3e} "
              f"{np.abs(d[-1]).max()/sc:19.3e}")
    print()


if __name__ == "__main__":
    pc = _mod("pc", "polar_corner.py")
    C0 = pc.Corner(24, 24, 25.0)
    check_algebra(C0.mu, C0.a0)
    check_conditioning()
    check_mms(cforced=0.0)
    check_mms(cforced=0.3)
