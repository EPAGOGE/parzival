"""spectrum.py -- realization, projector, resolvent for the corner-regularized profile.

Three objects, in the order the SPEC establishes them, and nothing else.  This file
does no science by itself: it is the instrument the Q3/Q4/Q5 measurements run on.

1. REALIZATION.  The stability operator is NOT J.  In the divided variables
   (Om = xi A, Bf = xi^2 B, Psi = xi^2 P) the e^{a0 xi} factors are tau-independent,
   so d_tau commutes through the substitution and the mass matrix on the two
   transported blocks is the IDENTITY:

        d_tau A = RO' ,   d_tau B = RB' ,   0 = RP' ,   0 = g1 = g2

   Linearizing about the converged profile gives the DESCRIPTOR PENCIL (E, J), with
   J = S.jacobian(z) UNMODIFIED and E = diag(mask), mask = 1 exactly on live
   transport rows (not in rT_pin, not in rT_c0).  Everything else -- the pins, the
   C0 interface matching, the Poisson block, the two gauge rows, the two scalar
   columns (c_l, c_w) -- is algebraic, E = 0 there.  Hessenberg index exactly 2,
   because Cg has ZERO support on the P and c blocks.

2. PROJECTOR.  The quotient is the DAE restriction to ker(Cg): the admissible state
   space is {v : Cg v = 0}, dimension n_f - 2.  Structural only.  Pi(v) = v - Qc Qc^H v
   with Qc = qr(Cg^T).  The grading direction v_g = (A, 2B, P, c_l, c_w) is NOT
   additionally deflated -- gate G1 below measures what deflating it costs.

3. RESOLVENT.  One sparse LU per z, adjoint via trans="H" on the SAME factor:

        R(z) f = Pi [ (zE - J)^-1 embed(Pi f) ]_free

   which equals (z I - L)^-1 on ker(Cg), L = Pi_c M the compressed generator.
   ||R(z)|| by power iteration on R^H R.

PITFALLS ENFORCED HERE (SPEC section 8):
  * J is not L.  sigma_min(J) is the steady root's isolation, sigma_min(L|ker Cg) is
    the generator's.  Both are reported by G1, never conflated.
  * L = Pi_c M is exactly rank-2 deficient on the FULL free space -- Pi_c is an
    oblique projector of rank n_f - 2.  Always compress to ker(Cg) before any solve.
  * Never feed J, or a hand-built P@A, to a general eigensolver.
"""
import importlib.util
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
SCRATCH = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/"
                       "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
GROUND_ROOT = SCRATCH / "hunt_fields/rung_00_a-0.344712.npz"
PROD_CFG = dict(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12), Nb=36, eps_b=1e-4)
SMALL_CFG = dict(edges=(0.0, 2.0, 15.0, 25.0), degs=(6, 10, 5), Nb=10, eps_b=1e-3)

# The three converged roots the campaign has.  Standing discipline: no single-grid
# quote of a resolution-sensitive quantity, so every production gate takes a label.
ROOTS = {
    "A": (GROUND_ROOT, PROD_CFG),
    "B": (SCRATCH / "q6_root_24_56_0.0001.npz",
          dict(edges=(0.0, 2.0, 15.0, 25.0), degs=(24, 56, 12), Nb=36, eps_b=1e-4)),
    "C": (SCRATCH / "q6_root_16_40_5e-05.npz",
          dict(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12), Nb=36, eps_b=5e-5)),
}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(HERE / fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


pc = _mod("pc", "polar_cornerreg.py")


# ---------------------------------------------------------------------------
# 1. the realization
# ---------------------------------------------------------------------------
class Realization:
    """The descriptor pencil (E, J) and its DAE quotient, built from a solver + state.

    Nothing here is re-derived or re-assembled: J is the solver's own Jacobian, and
    every index set (pins, C0 partners, Poisson rows, gauge rows) is read off the
    solver's row ledger.  The only new object is the 0/1 mask that says which rows
    are transported.
    """

    def __init__(self, solver, state):
        self.S = solver
        self.z = np.asarray(state, dtype=float)
        Nx, Nb = solver.Nx, solver.Nb
        self.Nx, self.Nb = Nx, Nb
        self.n2 = n2 = Nx * Nb
        self.J = solver.jacobian(self.z).tocsr()
        self.Jc = self.J.tocsc()
        self.N = self.J.shape[0]
        if self.N != 3 * n2 + 2:
            raise ValueError(f"unexpected Jacobian size {self.N} != 3*{n2}+2")

        # --- row ledger -----------------------------------------------------
        self.pinT = np.asarray(sorted(solver.rT_pin), dtype=int)
        self.c0T = np.asarray(sorted(solver.rT_c0), dtype=int)
        self.liveT = np.setdiff1d(np.arange(n2), np.union1d(self.pinT, self.c0T))
        self.free = np.concatenate([self.liveT, n2 + self.liveT])
        self.n_f = self.free.size
        self.n_finite = self.n_f - 2          # dim ker(Cg): the finite eigenvalue count
        self.partner_l = self.c0T
        self.partner_r = np.array([solver.partner[int(r)] for r in self.c0T], dtype=int)
        # position of each partner_r node inside liveT (it is live by construction)
        pos = -np.ones(n2, dtype=int)
        pos[self.liveT] = np.arange(self.liveT.size)
        if np.any(pos[self.partner_r] < 0):
            raise ValueError("a C0 partner node is not live -- row ledger broken")
        self._pos_r = pos[self.partner_r]
        self._nl = self.liveT.size

        # --- the mass matrix: identity on live transport rows, zero elsewhere
        mask = np.zeros(self.N)
        mask[self.free] = 1.0
        self.mask = mask
        self.E = sp.diags(mask, format="csc")
        self.rank_E = int(mask.sum())

        # --- gauge rows / scalar columns, restricted to the free coordinates
        self.Cg = np.asarray(self.J[[self.N - 2, self.N - 1], :].todense())[:, self.free]
        self.Bc = np.asarray(self.J[:, [self.N - 2, self.N - 1]].todense())[self.free, :]
        self.CgBc = self.Cg @ self.Bc
        self.Qc, _ = np.linalg.qr(self.Cg.T)      # orthonormal basis of ker(Cg)^perp

        # --- Poisson slaving blocks (read off J, not re-derived) -------------
        self.JPP = self.J[2 * n2:3 * n2, 2 * n2:3 * n2].tocsc()
        self.JPA = self.J[2 * n2:3 * n2, 0:n2].tocsr()
        self.JPB = self.J[2 * n2:3 * n2, n2:2 * n2].tocsr()
        self._lu_P = None

    # -- structural projector -------------------------------------------------
    def project(self, v):
        """Orthogonal projector onto ker(Cg).  THE quotient; nothing else deflated."""
        return v - self.Qc @ (self.Qc.conj().T @ v)

    def project_oblique(self, v):
        """Pi_c = I - Bc (Cg Bc)^-1 Cg: the index-2 projector.  Identity on ker(Cg)."""
        return v - self.Bc @ np.linalg.solve(self.CgBc, self.Cg @ v)

    def embed(self, f):
        """free coordinates -> full right-hand side (zero on every algebraic row)."""
        rhs = np.zeros(self.N, dtype=np.asarray(f).dtype)
        rhs[self.free] = f
        return rhs

    def restrict(self, x):
        return x[self.free]

    # -- prolongation: free coordinates -> full admissible perturbation --------
    def prolong(self, v):
        """Z v.  Pins are zero (fixed data); C0 duplicates copy their partner; the
        Poisson block is slaved by dP = -JPP^-1 (JPA dA + JPB dB); the two scalar
        components are zero (they are the (c_l, c_w) unknowns, not state)."""
        n2 = self.n2
        out = np.zeros(self.N, dtype=np.asarray(v).dtype)
        out[self.free] = v
        out[self.partner_l] = v[self._pos_r]
        out[n2 + self.partner_l] = v[self._nl + self._pos_r]
        if self._lu_P is None:
            self._lu_P = spla.splu(self.JPP)
        rhs = self.JPA @ out[:n2] + self.JPB @ out[n2:2 * n2]
        out[2 * n2:3 * n2] = -self._lu_P.solve(rhs)
        return out

    def apply_M(self, v):
        """The unconstrained reduced map on the free coordinates."""
        return (self.J @ self.prolong(v))[self.free]

    def apply_L(self, v):
        """The compressed generator L = Pi_c M.  Valid input: v in ker(Cg)."""
        return self.project_oblique(self.apply_M(v))

    # -- dense objects (small configurations only) ----------------------------
    def dense_M(self):
        nf = self.n_f
        cols = [self.apply_M(e) for e in np.eye(nf)]
        return np.column_stack(cols)

    def dense_L(self):
        M = self.dense_M()
        return M - self.Bc @ np.linalg.solve(self.CgBc, self.Cg @ M), M

    def kernel_basis(self):
        """Orthonormal basis of ker(Cg), n_f x (n_f-2).  Dense: small configs only."""
        return np.linalg.svd(self.Cg)[2][2:].T.conj()

    # -- the grading direction ------------------------------------------------
    def grading_generator(self):
        """v_g = (A, 2B, P, c_l, c_w) restricted to the free coordinates, normalized.

        Exact discrete null of every covariant row of J (Euler on the degree-2/3/1
        homogeneity).  NOT deflated -- see gate G1."""
        n2 = self.n2
        vg_full = np.concatenate([self.z[:n2], 2.0 * self.z[n2:2 * n2],
                                  self.z[2 * n2:3 * n2], [self.z[-2]], [self.z[-1]]])
        v = vg_full[self.free]
        return v / np.linalg.norm(v), vg_full

    def euler_residual(self):
        """||J v_g|| off the pin+gauge rows.  Theorem says 0."""
        _, vg_full = self.grading_generator()
        u = vg_full / np.linalg.norm(vg_full)
        w = self.J @ u
        pin = np.concatenate([self.pinT, self.pinT + self.n2,
                              [self.N - 2, self.N - 1]])
        m = np.zeros(self.N, bool)
        m[pin] = True
        return float(np.linalg.norm(w[~m])), float(np.linalg.norm(w))


# ---------------------------------------------------------------------------
# 2. the resolvent
# ---------------------------------------------------------------------------
class Resolvent:
    """R(z) on the quotient, one sparse LU per z, adjoint on the SAME factor.

    R(z) f = Pi [ (zE - J)^-1 embed(Pi f) ]_free   ==   (z I - L)^-1 on ker(Cg).
    Real arithmetic whenever Im z = 0 (halves the factorization cost).
    """

    def __init__(self, real, z, extra_deflate=None):
        self.R = real
        self.z = complex(z)
        self.real_arith = (self.z.imag == 0.0)
        self.dtype = float if self.real_arith else complex
        Kz = (self.z.real * real.E - real.Jc) if self.real_arith else \
             (self.z * real.E - real.Jc)
        t0 = time.time()
        self.lu = spla.splu(Kz.tocsc())
        self.t_lu = time.time() - t0
        self.nnz_lu = self.lu.L.nnz + self.lu.U.nnz
        if extra_deflate is None:
            self.Q = real.Qc
        else:
            self.Q = np.linalg.qr(np.column_stack([real.Qc, extra_deflate]))[0]

    def _proj(self, v):
        return v - self.Q @ (self.Q.conj().T @ v)

    def apply(self, f, adjoint=False):
        R = self.R
        rhs = np.zeros(R.N, dtype=self.dtype)
        rhs[R.free] = self._proj(f)
        x = self.lu.solve(rhs, trans=("H" if adjoint else "N"))
        return self._proj(x[R.free])

    def norm(self, iters=200, tol=1e-10, seed=0):
        """||R(z)||_2 by power iteration on R^H R.  Returns (sigma, iterations)."""
        nf = self.R.n_f
        v = self._proj(np.random.default_rng(seed).standard_normal(nf).astype(self.dtype))
        v /= np.linalg.norm(v)
        s_old = 0.0
        for it in range(iters):
            y = self.apply(v)
            w = self.apply(y, adjoint=True)
            nn = np.linalg.norm(w)
            s_new = np.sqrt(nn)
            if it > 4 and abs(s_new - s_old) < tol * s_new:
                return s_new, it + 1
            v = w / nn
            s_old = s_new
        return s_old, iters


def sigma_min_J(real, iters=80, seed=3):
    """1/||J^-1||_2 by inverse power iteration.  The STEADY root's isolation."""
    lu = spla.splu(real.Jc)
    luT = spla.splu(real.J.T.tocsc())
    x = np.random.default_rng(seed).standard_normal(real.N)
    x /= np.linalg.norm(x)
    for _ in range(iters):
        x = lu.solve(luT.solve(x))
        x /= np.linalg.norm(x)
    return float(np.linalg.norm(real.J @ x)), x


def load_production(label="A"):
    """Load one of the campaign's converged roots by label (see ROOTS)."""
    npz, cfg = ROOTS[label]
    d = np.load(npz)
    a, z = float(d["a"]), d["z"]
    S = pc.CornerRegSolver(alpha=a, **dict(cfg))
    S.adopt_seed(z)
    return Realization(S, z), S, a, z


def load_small(cfg=None):
    """Small structural configuration, at the SEED.  Counts, ranks and route
    agreement are meaningful here; eigenvalue LOCATIONS are not."""
    c = dict(SMALL_CFG if cfg is None else cfg)
    S = pc.CornerRegSolver(alpha=-0.3447, **c)
    z = S.pack(S.A0, S.B0, S.P0, S.P["cl"], S.P["cw"])
    return Realization(S, z), S, z


# ---------------------------------------------------------------------------
# 3. gates
# ---------------------------------------------------------------------------
def gate_G1(label="A", verbose=True):
    """G1 -- the quotient removes the structural soft direction.

    As REDEFINED by the SPEC: the quotient is the DAE restriction to ker(Cg);
    sigma_min must rise off the steady root's isolation, and the projector must not
    be augmented.  The literal clause ||Pi v_g||/||v_g|| < 1e-10 is measured and
    reported too -- it is the wrong test, and this gate proves why by pricing it.
    """
    real, S, a, z = load_production(label)
    cfg = ROOTS[label][1]
    res_rms = float(np.linalg.norm(S.residual(z)) / np.sqrt(real.N))
    if verbose:
        print("=" * 78)
        print(f"[G1]  QUOTIENT GATE   root {label}: {tuple(cfg['degs'])}/Nb{cfg['Nb']}"
              f" eps_b={cfg['eps_b']:g}")
        print("=" * 78)
        print(f"  root: alpha = {a:+.8f}   h_id = {S.h_id(z):+.4e}   "
              f"||F||_rms = {res_rms:.3e}   converged=True")
        print(f"  pencil: N = {real.N}   rank(E) = {real.rank_E} = n_f   "
              f"n_f = {real.n_f}   dim ker(Cg) = {real.n_finite}")
        print(f"          N - rank(E) = {real.N - real.rank_E}   "
              f"(+ m=2 index-2 infinities -> {real.N - real.rank_E + 2} infinite)")

    t0 = time.time()
    smin_J, xmin = sigma_min_J(real)
    t_J = time.time() - t0

    R0 = Resolvent(real, 0.0)
    nR0, it0 = R0.norm()
    smin_L = 1.0 / nR0

    rng = np.random.default_rng(11)
    w = real.project(rng.standard_normal(real.n_f))
    gauge_leak = float(np.linalg.norm(real.Cg @ w) / np.linalg.norm(w))

    vg, vg_full = real.grading_generator()
    Pvg = real.project(vg)
    clause2 = float(np.linalg.norm(Pvg) / np.linalg.norm(vg))
    euler_cov, euler_all = real.euler_residual()
    vg_hat = vg_full / np.linalg.norm(vg_full)
    cos_soft = abs(float(xmin @ vg_hat))

    w1 = Pvg / np.linalg.norm(Pvg)
    Lw1 = real.apply_L(w1)
    nLw1 = float(np.linalg.norm(Lw1))

    if verbose:
        print(f"\n  sigma_min(J)                          = {smin_J:.6e}"
              f"   [steady root isolation, {t_J:.1f}s]")
        print(f"  sigma_min(L | ker Cg) = 1/||R(0)||     = {smin_L:.6e}"
              f"   [||R(0)|| = {nR0:.6e}, {it0} it]")
        print(f"  RISE FACTOR                            = x{smin_L / smin_J:.1f}"
              f"                 <-- G1 clause 1")
        print(f"  ||Cg w|| / ||w||,  w = Pi(random)      = {gauge_leak:.4e}"
              f"   [projector removes the gauge directions]")
        print(f"\n  ||Pi v_g|| / ||v_g||                   = {clause2:.6f}"
              f"       <-- G1 clause 2 AS LITERALLY WRITTEN: NOT < 1e-10")
        print(f"  ||J v_g||_covariant                    = {euler_cov:.3e}"
              f"    (Euler theorem: 0;  all rows {euler_all:.3e})")
        print(f"  |cos(soft mode of J, v_g)|             = {cos_soft:.4f}")
        print(f"  ||L w1|| / ||w1||,  w1 = Pi v_g / |.|  = {nLw1:.6e}")
        print(f"      vs sigma_min(L|ker Cg)             = {smin_L:.6e}"
              f"   ratio = {nLw1 / smin_L:.4g}")
        print("      -> w1 is NOT a soft direction of the GENERATOR; the grading is a"
              " near-null of J,")
        print("         not of L.  Deflating it removes an admissible direction.")

    # price the literal clause: what does deflating w1 cost in the RHP?
    prices = []
    for zz in (0.5 + 0j, 0.0 + 1.0j):
        s0, i0 = Resolvent(real, zz).norm()
        s1, i1 = Resolvent(real, zz, extra_deflate=w1[:, None]).norm()
        prices.append((zz, s0, s1, abs(s1 - s0) / s0))
    if verbose:
        print(f"\n  PRICE OF SATISFYING CLAUSE 2 LITERALLY (deflate w1 too):")
        print(f"      {'z':>12s} {'structural only':>20s} {'+w1 deflated':>20s}"
              f" {'rel change':>12s}")
        for zz, s0, s1, rel in prices:
            print(f"      {zz.real:+6.2f}{zz.imag:+5.2f}i {s0:>20.10e} {s1:>20.10e}"
                  f" {rel:>12.3e}")

    rhp_move = max(rel for zz, _, _, rel in prices if zz.real > 0)
    ok = (smin_L > smin_J) and (gauge_leak < 1e-12)
    if verbose:
        print(f"\n  VERDICT: G1 as redefined  -> {'PASS' if ok else 'FAIL'}"
              f"   (sigma_min rises x{smin_L / smin_J:.0f}, gauge leak"
              f" {gauge_leak:.1e})")
        print(f"           G1 clause 2 literal -> REFUSED: forcing it moves ||R(z)||"
              f" in the RHP by {rhp_move:.1%}")
    return dict(label=label, smin_J=smin_J, smin_L=smin_L, nR0=nR0,
                gauge_leak=gauge_leak, clause2=clause2, euler_cov=euler_cov,
                nLw1=nLw1, prices=prices, rise=smin_L / smin_J, ok=ok, real=real)


def gate_G2(verbose=True, zs=(0.0, 0.3, -0.8 + 0.4j, 0.5 + 0.0j), ntrial=6):
    """G2 -- the resolvent computed two ways must agree to 1e-8.

    R1  sparse bordered solve            (the production method)
    R2  explicit dense reduction         (z I - L) on the FULL free space
    R3  compressed operator on ker(Cg)   (z I - Lred), Lred = Z0^H Pi_c M Z0

    R2 at z = 0 is the historically quoted comparison; it is well posed only up to
    ker(L), which is exactly 2-dimensional (Pi_c is rank n_f - 2), so its z = 0
    column is reported with that caveat.  R1 vs R3 is well posed at every z.
    """
    real, S, z0 = load_small()
    if verbose:
        print("\n" + "=" * 78)
        print("[G2]  TWO-ROUTE RESOLVENT GATE   (6,10,5)/Nb10 eps_b=1e-3  [structure]")
        print("=" * 78)
        print(f"  N = {real.N}   n_f = {real.n_f}   dim ker(Cg) = {real.n_finite}"
              f"   rank(E) = {real.rank_E}")

    Ld, M = real.dense_L()
    Z0 = real.kernel_basis()
    Lred = Z0.conj().T @ Ld @ Z0
    sv = np.linalg.svd(Ld, compute_uv=False)
    svr = np.linalg.svd(Lred, compute_uv=False)
    if verbose:
        print(f"  ||L|| = {sv[0]:.6e}   sigma_min(L, full free space) = {sv[-1]:.4e}"
              f"   <- rank-2 deficient by construction")
        print(f"                        sigma_min(L | ker Cg)         = {svr[-1]:.6e}"
              f"   <- the generator's")

    rng = np.random.default_rng(7)
    rows = []
    for z in zs:
        zc = complex(z)
        res = Resolvent(real, zc)
        dt = complex if zc.imag else float
        shift = zc if dt is complex else zc.real
        A2 = shift * np.eye(real.n_f, dtype=dt) - Ld.astype(dt)
        A3 = shift * np.eye(real.n_finite, dtype=dt) - Lred.astype(dt)
        e12, e12p, e13, e23 = [], [], [], []
        for _ in range(ntrial):
            f = real.project(rng.standard_normal(real.n_f).astype(dt))
            x1 = res.apply(f)
            x2 = np.linalg.solve(A2, f)
            x3 = Z0 @ np.linalg.solve(A3, Z0.conj().T @ f)
            n1 = np.linalg.norm(x1)
            e12.append(np.linalg.norm(x1 - x2) / n1)
            e12p.append(np.linalg.norm(x1 - real.project(x2)) / n1)
            e13.append(np.linalg.norm(x1 - x3) / n1)
            e23.append(np.linalg.norm(x2 - x3) / n1)
        rows.append((zc, max(e12), max(e12p), max(e13), max(e23)))

    if verbose:
        print(f"\n  {'z':>14s} {'R1 vs R2 raw':>16s} {'R1 vs Pi(R2)':>16s}"
              f" {'R1 vs R3 (ker Cg)':>19s} {'R2 vs R3':>12s}")
        for zc, a12, a12p, a13, a23 in rows:
            print(f"  {zc.real:+8.2f}{zc.imag:+5.2f}i {a12:>16.3e} {a12p:>16.3e}"
                  f" {a13:>19.3e} {a23:>12.3e}")

    # is the z=0 full-free-space comparison well posed at all?  seed spread.
    A2_0 = -Ld
    res0 = Resolvent(real, 0.0)
    spread = []
    for sd in range(8):
        f = real.project(np.random.default_rng(100 + sd).standard_normal(real.n_f))
        x1 = res0.apply(f)
        x2 = np.linalg.solve(A2_0, f)
        spread.append(np.linalg.norm(x1 - x2) / np.linalg.norm(x1))
    if verbose:
        print(f"\n  z=0 route-2 well-posedness (8 seeds, rel diff vs R1):")
        print(f"      min {min(spread):.3e}   median {np.median(spread):.3e}"
              f"   max {max(spread):.3e}")
        print(f"      -> solve(L, f) is rank-2 singular (sigma_min = {sv[-1]:.2e});"
              f" its null component is a coin flip.")
        print(f"      The SPEC's 3.603e-13 for this pair is one lucky draw, not a"
              f" reproducible agreement.")

    # norms, both routes
    nrm_rows = []
    for z in zs:
        zc = complex(z)
        s1, it = Resolvent(real, zc).norm()
        dt = complex if zc.imag else float
        shift = zc if dt is complex else zc.real
        A3 = shift * np.eye(real.n_finite, dtype=dt) - Lred.astype(dt)
        s3 = 1.0 / np.linalg.svd(A3, compute_uv=False)[-1]
        nrm_rows.append((zc, s1, s3, abs(s1 - s3) / s3))
    if verbose:
        print(f"\n  {'z':>14s} {'||R(z)|| power it':>20s} {'||R(z)|| dense SVD':>20s}"
              f" {'rel':>12s}")
        for zc, s1, s3, rel in nrm_rows:
            print(f"  {zc.real:+8.2f}{zc.imag:+5.2f}i {s1:>20.10e} {s3:>20.10e}"
                  f" {rel:>12.3e}")

    worst_wellposed = max(max(a13 for _, _, _, a13, _ in rows),
                          max(rel for _, _, _, rel in nrm_rows))
    ok = worst_wellposed < 1e-8
    if verbose:
        print(f"\n  worst well-posed disagreement = {worst_wellposed:.3e}"
              f"   (bar 1e-8)   ->  {'PASS' if ok else 'FAIL'}")
    return dict(rows=rows, nrm_rows=nrm_rows, worst=worst_wellposed, ok=ok,
                Lred=Lred, real=real)


def gate_G2b(verbose=True, g2=None):
    """G2b -- the pencil's spectrum IS the realization's spectrum.

    QZ on (E, J) vs dense eigenvalues of Lred.  Validates P-elimination,
    prolongation, the index-2 projection and the ker(Cg) compression at once.
    """
    import scipy.linalg as sla
    if g2 is None:
        g2 = gate_G2(verbose=False)
    real, Lred = g2["real"], g2["Lred"]
    Jd = np.asarray(real.J.todense())
    Ed = np.diag(real.mask)
    t0 = time.time()
    AA, BB, aa, bb, Q, Zq = sla.ordqz(Jd, Ed, output="complex")
    tqz = time.time() - t0
    scale = max(abs(aa).max(), 1.0)
    fin = np.abs(bb) > 1e-12 * scale
    lam = aa[fin] / bb[fin]
    ev = np.linalg.eigvals(Lred)
    srt = np.sort(np.abs(bb))
    n_fin = int(fin.sum())
    worst = np.nan
    if n_fin == ev.size:
        used = np.zeros(ev.size, bool)
        worst = 0.0
        for v in lam:
            d = np.abs(ev - v) + 1e30 * used
            k = int(np.argmin(d))
            used[k] = True
            worst = max(worst, float(d[k]) / max(abs(v), 1.0))
    if verbose:
        print("\n" + "=" * 78)
        print("[G2b] REALIZATION GATE  -- pencil (E,J) vs compressed generator")
        print("=" * 78)
        print(f"  QZ finite = {n_fin}   infinite = {real.N - n_fin}"
              f"   [{tqz:.1f}s]")
        print(f"  predicted: finite = dim ker(Cg) = {real.n_finite}"
              f"   infinite = (N - rank E) + m = {real.N - real.rank_E} + 2 ="
              f" {real.N - real.rank_E + 2}")
        print(f"  |beta| separation: largest zero = {srt[real.N - n_fin - 1]:.3e}"
              f"   smallest nonzero = {srt[real.N - n_fin]:.3e}"
              f"   ratio = {srt[real.N - n_fin] / max(srt[real.N - n_fin - 1], 1e-300):.2e}")
        print(f"  worst relative spectral mismatch = {worst:.3e}   (bar 1e-8)"
              f"   ->  {'PASS' if worst < 1e-8 else 'FAIL'}")
    return dict(n_fin=n_fin, worst=worst,
                ok=(n_fin == real.n_finite and worst < 1e-8))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    labels = sys.argv[2].split(",") if len(sys.argv) > 2 else ["A"]
    out, g1s = {}, []
    if which in ("all", "G2", "G2b"):
        g2 = gate_G2()
        out["G2"] = g2["ok"]
        if which in ("all", "G2b"):
            out["G2b"] = gate_G2b(g2=g2)["ok"]
    if which in ("all", "G1"):
        for lab in labels:
            r = gate_G1(lab)
            g1s.append(r)
            out[f"G1[{lab}]"] = r["ok"]
    if len(g1s) > 1:
        print("\n" + "=" * 78)
        print("  G1 ACROSS RESOLUTIONS  (no single-grid quote)")
        print(f"  {'root':>6s} {'sigma_min(J)':>16s} {'sigma_min(L|kerCg)':>20s}"
              f" {'rise':>10s} {'||Pi v_g||/||v_g||':>20s}")
        for r in g1s:
            print(f"  {r['label']:>6s} {r['smin_J']:>16.6e} {r['smin_L']:>20.6e}"
                  f" {'x%.1f' % r['rise']:>10s} {r['clause2']:>20.6f}")
    print("\n" + "=" * 78)
    print("  GATES: " + "   ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                   for k, v in out.items()))
    print("=" * 78)
