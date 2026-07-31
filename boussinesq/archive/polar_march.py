"""
LOG-POLAR DYNAMIC-RESCALING MARCHER for the Luo-Hou corner profile.

METHOD: MARCH, NOT NEWTON -- settled by reading Chen-Hou's source. `run_pertb.m:1` is
"Construct the approximate steady state by solving the dynamic rescaling equations"; the
loop is RK2/Heun and stops at `max(|Fv|,|Fw|) < 2e-10`; a recursive grep for
newton|jacobian over their whole `Perturbed_eqn/` folder matches only a mesh-map
Jacobian. The profile is a saddle ONLY with c_l, c_w free. They are not free: Chen-Hou
recompute them algebraically at EVERY RK stage (`F_pertb_2lev.m:135-137`), which REMOVES
the two unstable directions from the phase space rather than stabilising them. What
remains is an attractor -- the literal content of their title, "STABLE nearly
self-similar blowup".

FORMULATION
-----------
alpha0 is FIXED in the change of variables (NOT slaved to c_w/c_l, which moves in time
and would make the substitution itself time-dependent):

    Om = e^(a0 s) Ot ,   B = e^((1+2a0) s) Bt ,   Psi = e^((2+a0) s) Pt

Then c_l, c_w stay free scalars and every equation picks up ONE deviation factor
(c_w - c_l a0), which vanishes exactly at the fixed point:

  d_t Ot = (c_w - c_l a0) Ot - c_l Ot_s
           - e^(a0 s)[(Pt_s + (2+a0)Pt) Ot_b - Pt_b (Ot_s + a0 Ot)]
           + e^(a0 s)[cos b (Bt_s + (1+2a0)Bt) - sin b Bt_b]
  d_t Bt = 2(c_w - c_l a0) Bt - c_l Bt_s
           - e^(a0 s)[(Pt_s + (2+a0)Pt) Bt_b - Pt_b (Bt_s + (1+2a0)Bt)]
  Pt_ss + 2(2+a0) Pt_s + (2+a0)^2 Pt + Pt_bb = -Ot        [elliptic, solved each stage]

All three verified on Chen-Hou's converged profile in `polar_residual_gate.py`
(inner window R1 6.6e-3 / R2 8.9e-3; far window 1.6e-2 / 9.9e-3 / 2.5e-2).

THE GAUGE -- it writes itself
-----------------------------
c_l and c_w enter the RHS AFFINELY, so for any linear functional F, dF/dt is affine in
(c_l, c_w) and pinning two functionals is a 2x2 linear solve. The right two are not
arbitrary: the system has exactly two scaling symmetries, with tangent directions (in
substituted variables)

    v_amp   = ( Ot,                2 Bt              )
    v_trans = ( Ot_s + a0 Ot,      Bt_s + 2 a0 Bt    )

Requiring the evolution to have NO component along either is the log-polar analogue of
Chen-Hou's corner slaving -- and `polar_gauge_sweep.py` measured that this 2x2 is
well-conditioned (48.7 deg, cond 4.9) exactly when the inner edge sits at s = -2, which
is why the domain is what it is. Their gauge is evaluated at r = 0, which log-polar
deletes; this one is evaluated on the domain and is basis-free.

DOMAIN  s in [-2, 25], beta in [0, pi/2]
  inner  s=-2 : the gauge signal saturates here (48.7 deg) and the seed still has 34.6
                cells of real data. Further out the signal collapses (4.9 deg at s=+5)
                and the gauge becomes indeterminate -- a well-posedness failure, not a
                resolution compromise.
  outer  s=25 : Chen-Hou's own PRODUCTION mesh reaches only s=30.7 and fits the far
                field on s in [22.3,29.0]; s_max ~ 37 oversizes by ~12 units.

BOUNDARY CONDITIONS (all measured, see POLAR_SPEC sections 8 and 11)
  beta=0    wall     : Om, B FREE (Ot -> 1.12496, Bt -> 3.88927). Psi = 0.
  beta=pi/2 axis     : Om = 0 (LINEAR zero), B = 0 (DOUBLE zero, Bt ~ eps^1.9992),
                       Psi = 0.
  s = S1    outer    : ONLY Psi. d_s Pt = 0. Om and B get NOTHING here -- c_l > 0 makes
                       this the OUTFLOW edge for first-order transport, and a condition
                       there is the singular-Jacobian trap already logged in this lab.
  s = S0    inner    : inflow. Corner Taylor expansion gives the asymptotics
                       Ot ~ r^(1-a0), Bt ~ r^(1-2a0), Pt ~ r^(-a0); measured log-slopes
                       converge to these going inward (1.306/1.638/0.322 against
                       1.342/1.685/0.342). Stage 1 uses seed DIRICHLET (exact, makes the
                       first run a clean consistency test); stage 2 swaps in the Robin
                       form to measure the inner-truncation error.

FIRST TEST: the seed is (approximately) a steady state, so marching FROM it must not
move. That single run exercises the RHS, the elliptic solve, the gauge and every BC
against a known answer.
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(HERE / fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def cheb(N):
    """Chebyshev-Gauss-Lobatto nodes on [-1,1] (descending) and the differentiation
    matrix. Trefethen's construction; verified below by GATE 0."""
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def grid(N, lo, hi, kte=0.0):
    """Nodes ASCENDING on [lo,hi] plus first/second derivative matrices.

    kte > 0 applies the KOSLOFF-TAL-EZER map

        xi -> arcsin(kte * xi) / arcsin(kte)

    which pulls nodes OFF the endpoints toward the interior; kte -> 1 approaches uniform
    spacing, kte = 0 is plain Chebyshev.

    WHY: a single Chebyshev grid on [-2,25] puts its FEWEST points exactly where this
    problem is hardest. Measured at N=64: mid-domain spacing ds ~ 0.67 against ~0.007 at
    the edges, a factor ~100, giving 12 nodes in the first 2 units of s but only 8 across
    s in [5,10) -- and s in [2,15] is the TRANSITION band where neither the corner
    asymptotics nor the far-field power law holds, and where the residual concentrates.
    Uniform refinement cannot fix that RATIO, and was measured not to (drift is
    resolution-converged across N = 48/64/96). Bonus: the map also relaxes the Chebyshev
    O(N^-2) CFL toward O(N^-1), so dt can be larger."""
    D, x = cheb(N)
    idx = np.argsort(x)
    x = x[idx]
    D = D[np.ix_(idx, idx)]
    if kte > 0:
        a = np.arcsin(kte)
        xm = np.arcsin(kte * x) / a                 # mapped node positions in [-1,1]
        dxm = (kte / a) / np.sqrt(1.0 - (kte * x) ** 2)   # d(xm)/d(x)
        xp = lo + (hi - lo) * (xm + 1.0) / 2.0
        Dp = (D / dxm[:, None]) * (2.0 / (hi - lo))
    else:
        xp = lo + (hi - lo) * (x + 1.0) / 2.0
        Dp = D * (2.0 / (hi - lo))
    return xp, Dp, Dp @ Dp


class Poisson:
    """Pt_ss + 2 mu Pt_s + mu^2 Pt + Pt_bb = -Ot, with Pt=0 on both beta edges,
    d_s Pt = 0 at s=S1 and d_s Pt + a0 Pt = 0 at s=S0.

    Built ONCE as a sparse Kronecker system and PREFACTORED -- the operator is constant,
    so every stage costs only a triangular solve. Boundary rows are REPLACED (not lifted
    with tau terms), which sidesteps the double-Chebyshev tau nullity entirely: there are
    no tau unknowns to be undetermined."""

    def __init__(self, s, Ds, Ds2, b, Db, Db2, mu, a0, inner_robin=True):
        ns, nb = s.size, b.size
        Is, Ib = sp.identity(ns, format="csr"), sp.identity(nb, format="csr")
        A = sp.kron(sp.csr_matrix(Ds2 + 2 * mu * Ds + mu ** 2 * np.eye(ns)), Ib) \
            + sp.kron(Is, sp.csr_matrix(Db2))
        A = sp.lil_matrix(A)
        rid = lambda i, j: i * nb + j
        self.brows = []
        for i in range(ns):                       # beta edges: Pt = 0
            for j in (0, nb - 1):
                r = rid(i, j)
                A.rows[r], A.data[r] = [r], [1.0]
                self.brows.append(r)
        for j in range(1, nb - 1):                # s edges: Robin / Neumann
            r = rid(ns - 1, j)                    # outer: d_s Pt = 0
            A.rows[r] = [rid(k, j) for k in range(ns)]
            A.data[r] = list(Ds[ns - 1, :])
            self.brows.append(r)
            r = rid(0, j)                         # inner
            A.rows[r] = [rid(k, j) for k in range(ns)]
            row = Ds[0, :].astype(float).copy()
            if inner_robin:
                row[0] += a0                      # d_s Pt + a0 Pt = 0  (Pt ~ r^-a0)
            A.data[r] = list(row)
            self.brows.append(r)
        self.lu = spla.splu(sp.csc_matrix(A))
        self.shape = (ns, nb)
        self.brows = np.array(self.brows)

    def solve(self, Ot):
        rhs = (-Ot).ravel().copy()
        rhs[self.brows] = 0.0
        return self.lu.solve(rhs).reshape(self.shape)


def cheb_fwd(F, axis):
    """Grid -> Chebyshev coefficients on Gauss-Lobatto nodes, via the cosine FFT.
    Nodes here are ASCENDING, so flip before/after."""
    G = np.flip(F, axis=axis)
    n = G.shape[axis]
    sl = tuple(slice(1, -1) if k == axis else slice(None) for k in range(G.ndim))
    Ge = np.concatenate([G, np.flip(G, axis=axis)[sl]], axis=axis)
    C = np.real(np.fft.fft(Ge, axis=axis))
    keep = tuple(slice(0, n) if k == axis else slice(None) for k in range(G.ndim))
    return C[keep] / (n - 1)


def cheb_bwd(C, axis):
    """Chebyshev coefficients -> grid (inverse of cheb_fwd)."""
    n = C.shape[axis]
    sl = tuple(slice(-2, 0, -1) if k == axis else slice(None) for k in range(C.ndim))
    Ce = np.concatenate([C, C[sl]], axis=axis)
    G = np.real(np.fft.ifft(Ce, axis=axis))
    keep = tuple(slice(0, n) if k == axis else slice(None) for k in range(C.ndim))
    return np.flip(G[keep] * (n - 1), axis=axis)


def houli(n, alpha=36.0, order=36, cutoff=0.65):
    """Hou-Li exponential filter (J.Comput.Phys 2007): ~1 up to cutoff*k_max, then a
    very high-order rolloff that annihilates the aliasing-contaminated top modes while
    leaving the physical spectrum untouched to roundoff. Same form as the validated
    `dedalus_axisym.py:make_filter`.

    WHY THIS IS NEEDED HERE, and why it is not a fudge: Chebyshev collocation of pure
    ADVECTION carries no dissipation, and this system's beta-transport term
    (Pt_s + mu Pt) Ot_b is exactly that. Measured without a filter: the top-25% beta
    coefficient energy of Ot grows 7.1e-05 -> 3.2e-02, a factor 449 over tau = 6, while
    the s-direction tail SATURATES at 2.3e-04 -- i.e. the growth is grid-scale and
    beta-directional, not physical. Both reference codes carry dissipation of their own
    (Chen-Hou B-splines plus 30-step frozen-velocity substepping; Liu FIRST-ORDER
    UPWIND); a bare collocation scheme has none, so one must be supplied."""
    k = np.arange(n) / max(n - 1, 1)
    return np.exp(-alpha * (np.maximum(k - cutoff, 0.0) / (1 - cutoff)) ** order)


class March:
    def __init__(self, Ns=96, Nb=96, S0=-2.0, S1=25.0, inner="dirichlet",
                 filter_on=True, cutoff=0.65, kte=0.0):
        ps = _mod("ps", "polar_seed.py")
        gs = _mod("gs", "polar_gauge_sweep.py")
        self.P = ps.load()
        self.a0 = self.P["alpha"]
        self.mu = 2.0 + self.a0
        self.s, self.Ds, self.Ds2 = grid(Ns - 1, S0, S1, kte=kte)
        eps = 1e-3
        self.b, self.Db, self.Db2 = grid(Nb - 1, eps, np.pi / 2 - eps)
        self.ns, self.nb = self.s.size, self.b.size
        self.inner = inner
        # seed
        Om, B, Psi = gs.fields_on(ps, self.P, self.s, self.b)
        self.Ot0 = Om * np.exp(-self.a0 * self.s)[:, None]
        self.Bt0 = B * np.exp(-(1.0 + 2.0 * self.a0) * self.s)[:, None]
        self.Pt0 = Psi * np.exp(-self.mu * self.s)[:, None]
        self.Ot, self.Bt = self.Ot0.copy(), self.Bt0.copy()
        self.poisson = Poisson(self.s, self.Ds, self.Ds2, self.b, self.Db, self.Db2,
                               self.mu, self.a0)
        self.filter_on = filter_on
        self.fs = houli(self.ns, cutoff=cutoff)[:, None]
        self.fb = houli(self.nb, cutoff=cutoff)[None, :]
        self.E = np.exp(self.a0 * self.s)[:, None]
        self.cosb, self.sinb = np.cos(self.b)[None, :], np.sin(self.b)[None, :]

    # --- spatial helpers -----------------------------------------------------
    def ds(self, F):
        return self.Ds @ F

    def db(self, F):
        return F @ self.Db.T

    def parts(self, Ot, Bt):
        """RHS split into the c-INDEPENDENT part and the coefficients of c_l and c_w.
        d_t X = K + c_l * L + c_w * M.  Exploiting that the RHS is AFFINE in (c_l,c_w)
        is what makes the gauge a 2x2 linear solve rather than an iteration."""
        Pt = self.poisson.solve(Ot)
        Ot_s, Ot_b = self.ds(Ot), self.db(Ot)
        Bt_s, Bt_b = self.ds(Bt), self.db(Bt)
        Pt_s, Pt_b = self.ds(Pt), self.db(Pt)
        a0, mu, E = self.a0, self.mu, self.E
        advO = (Pt_s + mu * Pt) * Ot_b - Pt_b * (Ot_s + a0 * Ot)
        srcO = self.cosb * (Bt_s + (1.0 + 2.0 * a0) * Bt) - self.sinb * Bt_b
        advB = (Pt_s + mu * Pt) * Bt_b - Pt_b * (Bt_s + (1.0 + 2.0 * a0) * Bt)
        KO = -E * advO + E * srcO
        LO = -(Ot_s + a0 * Ot)             # coefficient of c_l
        MO = Ot                            # coefficient of c_w
        KB = -E * advB
        LB = -(Bt_s + 2.0 * a0 * Bt)
        MB = 2.0 * Bt
        return Pt, (KO, LO, MO), (KB, LB, MB)

    def gauge(self, Ot, Bt, pO, pB):
        """Choose (c_l, c_w) so the evolution has NO component along either scaling
        direction. 2x2 linear solve; its conditioning is the angle measured by
        polar_gauge_sweep.py (48.7 deg at S0=-2)."""
        KO, LO, MO = pO
        KB, LB, MB = pB
        vA = (Ot, 2.0 * Bt)
        vT = (self.ds(Ot) + self.a0 * Ot, self.ds(Bt) + 2.0 * self.a0 * Bt)
        # interior weights: exclude boundary rows/cols where one-sided stencils dominate
        W = np.zeros_like(Ot)
        W[2:-2, 2:-2] = 1.0
        dot = lambda X, Y: float(np.sum(W * X[0] * Y[0]) + np.sum(W * X[1] * Y[1]))
        A = np.array([[dot((LO, LB), vA), dot((MO, MB), vA)],
                      [dot((LO, LB), vT), dot((MO, MB), vT)]])
        rhs = -np.array([dot((KO, KB), vA), dot((KO, KB), vT)])
        cond = np.linalg.cond(A)
        cl, cw = np.linalg.solve(A, rhs)
        return float(cl), float(cw), float(cond)

    def rhs(self, Ot, Bt):
        Pt, pO, pB = self.parts(Ot, Bt)
        cl, cw, cond = self.gauge(Ot, Bt, pO, pB)
        dOt = pO[0] + cl * pO[1] + cw * pO[2]
        dBt = pB[0] + cl * pB[1] + cw * pB[2]
        # BOUNDARY: inflow at s=S0 only; nothing at s=S1 (outflow). beta edges: Om and B
        # are FREE at the wall, and pinned at the axis (Om linear zero, B double zero).
        if self.inner == "dirichlet":
            dOt[0, :] = 0.0                   # frozen at the seed value
            dBt[0, :] = 0.0
        else:
            dOt[0, :] = 0.0                   # value set by robin_project() after the step
            dBt[0, :] = 0.0
        dOt[:, -1] = 0.0                      # axis: Om = 0
        dBt[:, -1] = 0.0                      # axis: B = 0
        return dOt, dBt, Pt, cl, cw, cond

    def filt(self, F, bc_rows):
        """Hou-Li filter, then RESTORE the pinned boundary values.

        A filter applied to the whole field overwrites the Dirichlet nodes that the RHS
        pins every step. The resulting mismatch at the inflow edge is itself a large
        perturbation and it feeds back: measured, an unrestored filter DIVERGED by
        tau = 2 (max|dOt| 0.073 -> 18.6) -- strictly worse than no filter at all. Filter
        the field, then put the boundary back."""
        if not self.filter_on:
            return F
        keep = {k: F[v].copy() for k, v in bc_rows.items()}
        C = cheb_fwd(cheb_fwd(F, 0), 1) * self.fs * self.fb
        G = cheb_bwd(cheb_bwd(C, 1), 0)
        for k, v in bc_rows.items():
            G[v] = keep[k]
        return G

    def tails(self, F, frac=0.25):
        out = []
        for ax, n in ((0, self.ns), (1, self.nb)):
            C = np.abs(cheb_fwd(F, ax))
            k0 = int(n * (1 - frac))
            hi = np.take(C, range(k0, n), axis=ax)
            out.append(float(np.sqrt((hi ** 2).sum())
                             / max(np.sqrt((C ** 2).sum()), 1e-300)))
        return out

    def robin_project(self):
        """Impose the ASYMPTOTIC inner condition instead of freezing the seed value.

        The corner Taylor expansion gives Ot ~ r^(1-a0), Bt ~ r^(1-2a0) (measured
        log-slopes converge to 1.342 / 1.685 going inward). Written as
        d_s F = q F at s = S0 and solved for the edge value in terms of the interior:

            F[0] = ( sum_{k>=1} Ds[0,k] F[k] ) / ( q - Ds[0,0] )

        Frozen Dirichlet pins the inflow edge to an INTERPOLATED value forever; this lets
        it move with the solution, which is the difference between a persistent forcing
        and a boundary condition."""
        D0 = self.Ds[0]
        for F, q in ((self.Ot, 1.0 - self.a0), (self.Bt, 1.0 - 2.0 * self.a0)):
            F[0, :] = (D0[1:] @ F[1:, :]) / (q - D0[0])

    def step(self, dt):
        """RK2 / Heun, matching Chen-Hou's RK2_pertb_F.m. The gauge is recomputed at
        EVERY stage, exactly as they do."""
        dO1, dB1, _, cl, cw, cond = self.rhs(self.Ot, self.Bt)
        O1, B1 = self.Ot + dt * dO1, self.Bt + dt * dB1
        dO2, dB2, _, _, _, _ = self.rhs(O1, B1)
        bc = {"inner": (0, slice(None)), "axis": (slice(None), -1)}
        self.Ot = self.filt(0.5 * (O1 + self.Ot + dt * dO2), bc)
        self.Bt = self.filt(0.5 * (B1 + self.Bt + dt * dB2), bc)
        if self.inner != "dirichlet":
            self.robin_project()
        return cl, cw, cond, dO1, dB1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, default=96)
    ap.add_argument("--Nb", type=int, default=96)
    ap.add_argument("--S0", type=float, default=-2.0)
    ap.add_argument("--S1", type=float, default=25.0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--dt", type=float, default=1e-3)
    a = ap.parse_args()

    M = March(a.Ns, a.Nb, a.S0, a.S1)
    print(f"grid {M.ns} x {M.nb}   s in [{a.S0}, {a.S1}]   alpha0 = {M.a0:+.8f}")
    print(f"Chen-Hou reference:  c_l = {M.P['cl']:.8f}   c_w = {M.P['cw']:.8f}")

    # GATE 0 -- differentiation matrices on a known function
    f = np.exp(0.3 * M.s)
    e1 = np.abs(M.ds(f[:, None])[:, 0] - 0.3 * f).max() / np.abs(0.3 * f).max()
    g = np.sin(3 * M.b)
    e2 = np.abs(M.db(g[None, :])[0] - 3 * np.cos(3 * M.b)).max() / 3.0
    print(f"\nGATE 0  spectral derivatives: d_s {e1:.3e}   d_b {e2:.3e}"
          f"   {'PASS' if max(e1, e2) < 1e-9 else 'FAIL'}")

    # GATE 1 -- elliptic solve against the seed's own Pt
    Pt = M.poisson.solve(M.Ot0)
    I = (slice(2, -2), slice(2, -2))
    rel = np.abs(Pt[I] - M.Pt0[I]).max() / max(np.abs(M.Pt0[I]).max(), 1e-300)
    print(f"GATE 1  elliptic solve vs the seed's Psi (from their velocity): "
          f"max rel {rel:.3e}   {'PASS' if rel < 0.05 else 'CHECK'}")

    # GATE 2 -- the gauge must RECOVER Chen-Hou's constants from the seed
    _, pO, pB = M.parts(M.Ot0, M.Bt0)
    cl, cw, cond = M.gauge(M.Ot0, M.Bt0, pO, pB)
    ecl = abs(cl - M.P["cl"]) / abs(M.P["cl"])
    ecw = abs(cw - M.P["cw"]) / abs(M.P["cw"])
    print(f"GATE 2  gauge on the seed:  c_l = {cl:+.6f} (err {ecl:.3e})   "
          f"c_w = {cw:+.6f} (err {ecw:.3e})   cond {cond:.3g}")
    print(f"        alpha implied = {cw/cl:+.6f}  vs  {M.a0:+.6f}"
          f"   {'PASS' if max(ecl, ecw) < 0.15 else 'CHECK'}")

    # GATE 3 -- the seed must be a near-steady state
    dOt, dBt, _, cl, cw, cond = M.rhs(M.Ot0, M.Bt0)
    sO = np.abs(dOt[I]).max() / max(np.abs(M.Ot0[I]).max(), 1e-300)
    sB = np.abs(dBt[I]).max() / max(np.abs(M.Bt0[I]).max(), 1e-300)
    print(f"GATE 3  seed steadiness: max|dOt|/|Ot| = {sO:.3e}   "
          f"max|dBt|/|Bt| = {sB:.3e}")
    print("        (Chen-Hou stop their own march at max|F| < 2e-10 on THEIR mesh; on a")
    print("         different grid the interpolated seed cannot be that steady, but it")
    print("         must be SMALL and must not grow when marched.)")

    if a.steps:
        print(f"\nmarching {a.steps} steps at dt={a.dt}")
        print(f"  {'step':>5s} {'c_l':>11s} {'c_w':>11s} {'alpha':>10s} "
              f"{'cond':>9s} {'max|dOt|':>11s} {'|Ot|max':>10s}")
        for k in range(a.steps):
            cl, cw, cond, dO, dB = M.step(a.dt)
            if k % max(1, a.steps // 20) == 0 or k == a.steps - 1:
                print(f"  {k:5d} {cl:11.6f} {cw:11.6f} {cw/cl:10.6f} {cond:9.3g} "
                      f"{np.abs(dO[I]).max():11.3e} {np.abs(M.Ot[I]).max():10.4f}")
        print(f"\n  drift from seed: |Ot-Ot0|/|Ot0| = "
              f"{np.abs(M.Ot[I]-M.Ot0[I]).max()/np.abs(M.Ot0[I]).max():.3e}")


if __name__ == "__main__":
    main()
