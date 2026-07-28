"""
CORNER-INCLUSIVE frame: xi = ln(1 + r).  Fixes what section 15 identified.

WHY
---
Section 15 concluded that log-polar (`s = ln r`) is the right frame for the FAR FIELD and
the WRONG frame for STABILITY, because stability here is decided at the corner --
Chen-Hou's gauge is evaluated at `r = 0`, their conserved functionals are corner
derivatives, and their stability is stated for perturbations vanishing QUADRATICALLY AT
THE ORIGIN. `s = ln r` sends `r = 0` to `s = -infinity`, so no truncation `S0` can
express that space.

THE FIX IS ONE MAP, NOT A MATCHED HYBRID

    xi = ln(1 + r)      <=>     r = e^xi - 1

  * near r = 0:      xi ~ r          -- the CORNER IS AT xi = 0, a finite INCLUDED point
  * at large r:      xi ~ ln r       -- identical to log-polar, so every far-field result
                                        gated in POLAR_SPEC sections 1-9 carries over

and the substitution becomes `(1+r)^p = e^(p xi)`, i.e. EXACTLY the old `e^(p s)` in the
far field:

    Om = e^(a xi) Ot ,   B = e^((1+2a) xi) Bt ,   Psi = e^((2+a) xi) Pt

OPERATORS.  With g(xi) = r/(1+r) = 1 - e^(-xi)  (g -> 0 at the corner, g -> 1 far out):

    r d_r = g d_xi ,   d_1 = (1/r)( g cos b d_xi - sin b d_b )
    Lap   = (1/r^2)( g^2 d_xixi + g(1-g) d_xi + d_bb )

THE EQUATIONS (all exponential prefactors cancel exactly as before; at g = 1 each one
reduces term-by-term to the log-polar form already verified in polar_residual_gate.py):

  d_t Ot = -c_l g (Ot_xi + a Ot) + c_w Ot
           - (e^(a xi)/g)[ (Pt_xi + mu Pt) Ot_b - Pt_b (Ot_xi + a Ot) ]
           + (e^(a xi)/g)[ g cos b (Bt_xi + (1+2a) Bt) - sin b Bt_b ]

  d_t Bt = -c_l g (Bt_xi + (1+2a) Bt) + (c_l + 2 c_w) Bt
           - (e^(a xi)/g)[ (Pt_xi + mu Pt) Bt_b - Pt_b (Bt_xi + (1+2a) Bt) ]

  g^2 (Pt_xixi + 2 mu Pt_xi + mu^2 Pt) + g(1-g)(Pt_xi + mu Pt) + Pt_bb = -g^2 Ot

The `1/g` factors are apparent, not real: at the corner `Pt ~ xi^2`, `Ot ~ xi`, so each
bracket is `O(xi^2)` and the quotient vanishes like `xi`. Chebyshev clusters at `xi = 0`
so the first interior node still has `g ~ 0.06` at N = 32; the corner node itself is
pinned by a boundary condition, so `1/g` is never evaluated at `g = 0`.

THE PAYOFF.  The Poisson equation carries NO singular coefficients, and at `g -> 0` it
degenerates to `Pt_bb = 0`, which with `Pt = 0` on both beta edges forces `Pt = 0` at the
corner. **The regularity condition at the origin appears on its own rather than being
imposed** -- which is exactly the structure log-polar could not express.

CONTROLLED COMPARISON.  This keeps the SAME gauge (L2 projection of the two scaling
tangents) as polar_march.py. Only the FRAME changes. If the unstable mode of section 14
disappears, the frame was the cause -- which is the hypothesis section 15 states.
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


class Corner:
    def __init__(self, Nx=48, Nb=48, XMAX=25.0, filter_on=False, eps_b=1e-3,
                 outer="neumann"):
        pm = _mod("pm", "polar_march.py")
        ps = _mod("ps", "polar_seed.py")
        gs = _mod("gs", "polar_gauge_sweep.py")
        self.pm = pm
        self.P = ps.load()
        self.a0 = self.P["alpha"]
        self.mu = 2.0 + self.a0
        # xi grid INCLUDES the corner xi = 0
        self.x, self.Dx, self.Dx2 = pm.grid(Nx - 1, 0.0, XMAX)
        # beta-endpoint offset. Introduced early purely for interpolation safety at the
        # axis; it was a hard-coded 1e-3 until the mathlit triangulation pointed out that
        # the Nb sweep (section 34) holds it FIXED and therefore structurally cannot see
        # any error it causes -- refining Nb does not refine the offset away. Now a knob.
        eps = float(eps_b)
        self.eps_b = eps
        self.b, self.Db, self.Db2 = pm.grid(Nb - 1, eps, np.pi / 2 - eps)
        self.nx, self.nb = self.x.size, self.b.size
        self.outer = str(outer)
        self.g = 1.0 - np.exp(-self.x)                     # r/(1+r)
        self.r = np.exp(self.x) - 1.0
        self.E = np.exp(self.a0 * self.x)[:, None]
        self.G = self.g[:, None]
        self.cosb, self.sinb = np.cos(self.b)[None, :], np.sin(self.b)[None, :]
        self.filter_on = filter_on
        self.fx = pm.houli(self.nx)[:, None]
        self.fs = self.fx            # March.filt() looks for `fs`
        self.fb = pm.houli(self.nb)[None, :]

        # --- seed, evaluated at r = e^xi - 1 --------------------------------
        s_eq = np.log(np.maximum(self.r, 1e-12))           # log-polar s for the seed
        Om, B, Psi = gs.fields_on(ps, self.P, s_eq, self.b)
        # --- ANALYTIC CORNER SEED, blended in ------------------------------
        # The reference profile cannot be interpolated near r = 0: its mesh spacing there
        # is 0.00390625, so r = 0.016 is only FOUR cells and anything below that is
        # extrapolation. The xi frame reaches exactly into that region (unlike log-polar,
        # which cut at r = 0.135), and 1/g amplifies it.
        #
        # The leading corner behaviour is known analytically from parity plus two
        # constants measured to 8-9 digits in polar_gauge_gate.py:
        #     Om = w_x(0) y1          = 1.19620314 * r cos b     (odd in y1)
        #     B  = th_xx(0)/2 * y1^2  = 0.89909566 * r^2 cos^2 b (even, double zero)
        # Both reproduce the measured edge behaviour of section 8 (Om a LINEAR zero at
        # the axis and nonzero at the wall; B a DOUBLE zero at the axis), and they agree
        # with the interpolated seed to 0.17-0.5% over r in [0.01, 0.05] -- an
        # independent confirmation of both constants. Blend in log r across [0.02, 0.1]:
        # analytic inside, interpolated outside, smoothstep between.
        #
        # Psi needs NO analytic seed: it is SOLVED from the Poisson equation given Ot.
        # (Its leading r^2 coefficient is not fixed by corner constants anyway -- it is
        # set by matching to the outer solution.)
        WX, THXX = 1.19620314, 1.79819132
        RB0, RB1 = 0.02, 0.10
        rr = self.r[:, None]
        cb = np.cos(self.b)[None, :]
        Om_a = WX * rr * cb
        B_a = 0.5 * THXX * rr ** 2 * cb ** 2
        t = np.clip((np.log(np.maximum(rr, 1e-300)) - np.log(RB0))
                    / (np.log(RB1) - np.log(RB0)), 0.0, 1.0)
        wgt = t * t * (3.0 - 2.0 * t)                  # smoothstep: 0=analytic, 1=interp
        Om = wgt * Om + (1.0 - wgt) * Om_a
        B = wgt * B + (1.0 - wgt) * B_a
        self.blend = (RB0, RB1)

        # substituted seeds, formed AFTER the blend
        self.Ot0 = Om * np.exp(-self.a0 * self.x)[:, None]
        self.Bt0 = B * np.exp(-(1.0 + 2.0 * self.a0) * self.x)[:, None]
        self.Pt0 = Psi * np.exp(-self.mu * self.x)[:, None]

        # CORNER ROW ONLY. At xi = 0 we have r = 0, so y1 = r cos b = 0 and the exact
        # conditions are Ot = Bt = Pt = 0 (Om ~ r, B ~ r^2, Psi ~ r^2). The reference
        # data is also below its own resolution there, so this replaces a bad
        # extrapolation with the exact value.
        #
        # DO NOT ALSO ZERO THE AXIS COLUMN. The beta grid stops at pi/2 - 1e-3, NOT at
        # pi/2, so the last column is NOT zero (measured Ot = 0.326 there, since
        # Ot ~ 32.7 * eps). Zeroing it puts a spurious jump in the seed and corrupts the
        # beta derivative: it drove Ot_b to 0.489 against the correct 0.702, a 30% error
        # that propagated straight into the advection bracket and wrecked the gauge
        # (c_l = 4.65, c_w = +0.49 with the WRONG SIGN). polar_march.py zeroes the TIME
        # DERIVATIVE on that column, which freezes the value; it never zeroes the value.
        for A in (self.Ot0, self.Bt0, self.Pt0):
            A[0, :] = 0.0
        self.Ot, self.Bt = self.Ot0.copy(), self.Bt0.copy()
        self._build_poisson()

    def _beta_root(self):
        """S = (-Db2 restricted to the interior, Dirichlet)^(1/2), embedded back into
        nb x nb with zero edge rows/columns.

        The far-field separation needs 2j, not (2j)^2. Db2 supplies (2j)^2 -- but on the
        TRUNCATED wedge, and that is not a discretisation error.

        Measured eigenvalues 4.0102054 / 16.041 / 36.092 / 64.163 look like the exact
        4 / 16 / 36 / 64 to 0.25%, and the natural reading is discretisation error. It is
        not: they match the truncated-wedge exact values (j*pi/(pi/2 - 2*eps_b))^2 to
        3.4e-13, and lam1 = 4.0102054 is reproduced IDENTICALLY at Nb = 24/36/52/96/160.
        The 0.25% is exactly 8*eps_b/pi = 2.5465e-3 (measured 2.5514e-3). So the quantity is
        accurate to 1e-13 ON THE WRONG DOMAIN, and no Nb refinement can see it -- which is
        also why POLAR_SPEC section 34's Nb sweep concluded "beta is converged".

        Consequence for the far-field exponents: the TRUE ones are lam_+ = 2j - mu =
        0.3424 / 2.3424 / 4.3424 / ..., and the measured 0.3449 / 2.3475 / 4.3500 / ...
        are each high by 2j*4*eps_b/pi (6 of 6). The identity the offset hides is worth
        stating: mu = 2 + a0, so the slowest growing branch is lam_1 = 2 - mu = -a0
        EXACTLY -- the far field's slowest growth rate is the reciprocal of the substitution
        itself. The dtn/dtn3 rows built from this S therefore annihilate exponents that are
        off by 4*eps_b/pi per mode. Harmless in practice (both reproduce `neumann` to 7-8
        digits, d(alpha) = 8e-10 to 2.5e-8), but do not read these numbers as exact.

        The square root is the Dirichlet-to-Neumann operator of the half-strip; it is
        nonlocal in beta, which is why the outer row couples all beta nodes."""
        M = -self.Db2[1:-1, 1:-1]
        w, V = np.linalg.eig(M)
        if np.abs(w.imag).max() > 1e-8 * np.abs(w.real).max():
            raise RuntimeError("beta operator has complex spectrum; sqrt branch ambiguous")
        Si = (V * np.sqrt(w.real)) @ np.linalg.inv(V)
        S = np.zeros((self.nb, self.nb))
        S[1:-1, 1:-1] = Si.real
        return S

    def _build_poisson(self):
        nx, nb = self.nx, self.nb
        g, mu = self.g, self.mu
        G2 = np.diag(g ** 2)
        Gm = np.diag(g * (1.0 - g))
        As = G2 @ (self.Dx2 + 2 * mu * self.Dx + mu ** 2 * np.eye(nx)) \
            + Gm @ (self.Dx + mu * np.eye(nx))
        A = sp.kron(sp.csr_matrix(As), sp.identity(nb, format="csr")) \
            + sp.kron(sp.identity(nx, format="csr"), sp.csr_matrix(self.Db2))
        A = sp.lil_matrix(A)
        rid = lambda i, j: i * nb + j
        self.brows = []
        for i in range(nx):                       # beta edges: Pt = 0
            for j in (0, nb - 1):
                r = rid(i, j)
                A.rows[r], A.data[r] = [r], [1.0]
                self.brows.append(r)
        # OUTER ROW.  `d_xi Pt = 0` is NOT the far-field condition, and this is the one
        # place the two ends of the domain were treated differently.  At g -> 1 the
        # homogeneous far field separates as Pt ~ e^(lam xi) phi_j(b) with
        #
        #     (lam + mu)^2 = (2j)^2      =>     lam = -mu +- 2j
        #
        # and since mu = 2 + a0 = 1.6576 < 2, EVERY '+' branch has lam = 2j - mu > 0.
        # Measured: +0.3449 +2.3475 +4.3500 +6.3526 +8.3551 +10.3577 -- eight of eight
        # GROW.  A two-point BVP whose solution space contains e^(2.3475 xi) has condition
        # number ~ e^(2.3475 L): 1.6e15 at L=15, 2.6e20 at L=20, 3.1e25 at L=25, 3.7e32 at
        # L=32.  That is the measured wall -- XMAX=32 fails Newton at ||F||=3.9e-3 and
        # XMAX=40 overflows and collapses onto Om = B = 0 -- and it is why alpha OSCILLATES
        # with XMAX (-0.084%, +3.13%, +0.085%, +4.99%) instead of converging.
        #
        # `neumann` (the original) imposes d_xi Pt = 0, which does not exclude the growing
        # branch; it picks whatever admixture of it makes the derivative vanish at the last
        # node.  `dtn` applies the exact annihilator of {const, decaying branch},
        #
        #     Pt_xixi + (S + mu) Pt_xi = 0 ,      S = (-Db2|Dirichlet)^(1/2)
        #
        # verified term by term: on a constant it gives 0; on e^(-(2j+mu) xi) it gives
        # (2j+mu)^2 - (2j+mu)^2 = 0; on the growing e^((2j-mu) xi) it gives
        # (2j-mu)(2j) != 0.  So the growing modes leave the discrete solution space
        # instead of being fought by the linear algebra.  This is the same move that made
        # the corner the best-behaved part of the domain (1.1e-4 against 4.3e-2 out here),
        # applied to the end where we never made it.
        # Written as  outer_diag(d_xi) acting in xi alone  +  S @ outer_cpl(d_xi), so the
        # beta coupling is one matrix product and the algebra is checkable by expansion:
        #   dtn   d(d + S + mu)          = [d^2 + mu d]          + S[d]
        #   dtn3  d(d + S + mu)(d - a0)  = [d^3 + (mu-a0)d^2 - mu a0 d] + S[d^2 - a0 d]
        S = self._beta_root() if self.outer.startswith("dtn") else None
        a0 = self.a0
        Dx, Dx2 = self.Dx, self.Dx2
        if self.outer == "neumann":
            outer_diag, outer_cpl = Dx[nx - 1, :], None
        elif self.outer == "dtn":
            outer_diag, outer_cpl = Dx2[nx - 1, :] + mu * Dx[nx - 1, :], Dx[nx - 1, :]
        elif self.outer == "dtn3":
            # also admits the FORCED tail e^(a0 xi) -- which is what the substituted fields
            # actually decay by (measured tail slope -0.45..-0.49 for Ot/Bt/Pt against
            # a0 = -0.3424), and which neither `neumann` nor `dtn` annihilates.
            Dx3 = (Dx @ Dx2)[nx - 1, :]
            outer_diag = Dx3 + (mu - a0) * Dx2[nx - 1, :] - mu * a0 * Dx[nx - 1, :]
            outer_cpl = Dx2[nx - 1, :] - a0 * Dx[nx - 1, :]
        else:
            raise ValueError(f"unknown outer condition {self.outer!r}")
        for j in range(1, nb - 1):
            r = rid(0, j)                         # CORNER: Pt = 0 (Psi ~ r^2)
            A.rows[r], A.data[r] = [r], [1.0]
            self.brows.append(r)
            r = rid(nx - 1, j)
            acc = {}
            for k in range(nx):                   # the beta-diagonal part
                acc[rid(k, j)] = float(outer_diag[k])
            if outer_cpl is not None:             # + (S @ outer_cpl), nonlocal in beta
                for jj in range(1, nb - 1):
                    s = float(S[j, jj])
                    if s == 0.0:
                        continue
                    for k in range(nx):
                        c = rid(k, jj)
                        acc[c] = acc.get(c, 0.0) + s * float(outer_cpl[k])
            cols = sorted(acc)
            A.rows[r], A.data[r] = cols, [acc[c] for c in cols]
            self.brows.append(r)
        self.lu = spla.splu(sp.csc_matrix(A))
        self.brows = np.array(self.brows)

    def poisson(self, Ot):
        rhs = (-(self.G ** 2) * Ot).ravel().copy()
        rhs[self.brows] = 0.0
        return self.lu.solve(rhs).reshape(self.nx, self.nb)

    def dx(self, F):
        return self.Dx @ F

    def db(self, F):
        return F @ self.Db.T

    def parts(self, Ot, Bt, Pt_frozen=None):
        Pt = self.poisson(Ot) if Pt_frozen is None else Pt_frozen
        Ot_x, Ot_b = self.dx(Ot), self.db(Ot)
        Bt_x, Bt_b = self.dx(Bt), self.db(Bt)
        Pt_x, Pt_b = self.dx(Pt), self.db(Pt)
        a0, mu, G, E = self.a0, self.mu, self.G, self.E
        Ginv = np.zeros_like(G)
        np.divide(1.0, G, out=Ginv, where=(G > 1e-13))     # corner row is pinned anyway
        advO = (Pt_x + mu * Pt) * Ot_b - Pt_b * (Ot_x + a0 * Ot)
        srcO = G * self.cosb * (Bt_x + (1.0 + 2.0 * a0) * Bt) - self.sinb * Bt_b
        advB = (Pt_x + mu * Pt) * Bt_b - Pt_b * (Bt_x + (1.0 + 2.0 * a0) * Bt)
        KO = (E * Ginv) * (-advO + srcO)
        LO = -G * (Ot_x + a0 * Ot)
        MO = Ot
        KB = (E * Ginv) * (-advB)
        # + Bt: the Bt equation's RHS is (c_l + 2 c_w) B, so c_l appears THERE too, and
        # that contribution survives the substitution. Dropping it made LB differ from the
        # log-polar value by exactly -Bt (measured -1.0049 against +2.1800 at r~1e9,
        # Bt = 3.1847), which flipped the sign of c_w and gave alpha = +0.105 instead of
        # -0.342. At g = 1 this reduces to -(Bt_xi + 2 a0 Bt), i.e. polar_march.py's LB.
        LB = -G * (Bt_x + (1.0 + 2.0 * a0) * Bt) + Bt
        MB = 2.0 * Bt
        return Pt, (KO, LO, MO), (KB, LB, MB)

    def gauge_conserving(self, Ot, Bt, pO, pB):
        """CHEN-HOU'S ACTUAL GAUGE: freeze the two CORNER FUNCTIONALS.

        Their march conserves w_x(0) and (theta/x)_x(0) to 1e-11 over ~18787 steps, and
        the recon identified that conservation as "the structural reason their march
        cannot collapse to the zero field". This is only expressible now that the corner
        is IN the domain -- log-polar deletes r = 0, which is what section 15 concluded.

        In xi variables at the corner (xi ~ r, e^(p xi) -> 1):
            w_x(0)   = Ot_xi(0, b=0)        th_xx(0) = Bt_xixi(0, b=0)

        Choose (c_l, c_w) so BOTH are stationary. The RHS is affine in them, so this is a
        2x2 LINEAR solve, not an iteration:

            d/dt [Ot_xi(0,0)]   = 0     ->  (Dx  @ [KO + cl LO + cw MO])[0,0] = 0
            d/dt [Bt_xixi(0,0)] = 0     ->  (Dx2 @ [KB + cl LB + cw MB])[0,0] = 0
        """
        KO, LO, MO = pO
        KB, LB, MB = pB
        j = 0                                     # beta node nearest the wall
        row1 = [float((self.Dx @ LO)[0, j]), float((self.Dx @ MO)[0, j])]
        row2 = [float((self.Dx2 @ LB)[0, j]), float((self.Dx2 @ MB)[0, j])]
        A = np.array([row1, row2])
        rhs = -np.array([float((self.Dx @ KO)[0, j]),
                         float((self.Dx2 @ KB)[0, j])])
        try:
            cl, cw = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return self.P["cl"], self.P["cw"], np.inf
        return float(cl), float(cw), float(np.linalg.cond(A))

    def gauge_l2(self, Ot, Bt, pO, pB):
        """L2 projection of the two scaling tangents -- the polar_march.py gauge, kept so
        the frame comparison stays controlled."""
        KO, LO, MO = pO
        KB, LB, MB = pB
        vA = (Ot, 2.0 * Bt)
        # DILATION IS NOT A TRANSLATION IN xi. With xi = ln(1+r), r -> lam r gives
        # d xi / d sigma = r/(1+r) = g, so the dilation generator is g d_xi, NOT d_xi.
        # Carrying the log-polar tangent (the g = 1 limit) here is correct far out and
        # WRONG at the corner, and it wrecks the gauge: it returned c_l = 4.10 and
        # c_w = +0.42 (WRONG SIGN, against -1.029), which in turn produced a far-field
        # residual of 8.9 because the far field needs c_w = c_l * alpha.
        #   Om -> Om,  B -> e^-sigma B,  Psi -> e^-2sigma Psi   under dilation, so
        #   d Ot / d sigma = g (Ot_xi + a Ot)
        #   d Bt / d sigma = g (Bt_xi + (1+2a) Bt) - Bt
        # Both reduce to the log-polar forms at g = 1.
        G = self.G
        vT = (G * (self.dx(Ot) + self.a0 * Ot),
              G * (self.dx(Bt) + (1.0 + 2.0 * self.a0) * Bt) - Bt)
        W = np.zeros_like(Ot)
        W[2:-2, 2:-2] = 1.0
        dot = lambda X, Y: float(np.sum(W * X[0] * Y[0]) + np.sum(W * X[1] * Y[1]))
        A = np.array([[dot((LO, LB), vA), dot((MO, MB), vA)],
                      [dot((LO, LB), vT), dot((MO, MB), vT)]])
        rhs = -np.array([dot((KO, KB), vA), dot((KO, KB), vT)])
        return (*np.linalg.solve(A, rhs), float(np.linalg.cond(A)))

    def gauge(self, Ot, Bt, pO, pB):
        if getattr(self, "gauge_mode", "l2") == "conserve":
            return self.gauge_conserving(Ot, Bt, pO, pB)
        return self.gauge_l2(Ot, Bt, pO, pB)

    def rhs(self, Ot, Bt, Pt_frozen=None):
        Pt, pO, pB = self.parts(Ot, Bt, Pt_frozen)
        cl, cw, cond = self.gauge(Ot, Bt, pO, pB)
        dOt = pO[0] + cl * pO[1] + cw * pO[2]
        dBt = pB[0] + cl * pB[1] + cw * pB[2]
        dOt[0, :] = 0.0                      # CORNER: Om ~ r  => Ot(0) = 0
        dBt[0, :] = 0.0                      # CORNER: B ~ r^2 => Bt(0) = 0
        dOt[:, -1] = 0.0                     # axis
        dBt[:, -1] = 0.0
        return dOt, dBt, Pt, cl, cw, cond

    def step_two_tier(self, k, k2, n_inner=30):
        """Chen-Hou's TWO-TIER substepping (`run_pertb.m:56-59`).

        One outer RK2 step WITH the Poisson solve, then `n_inner` RK2 steps with the
        VELOCITY FROZEN -- no Poisson solve at all. Their comment calls it "much faster
        (more than 10 times)", but speed is not why it matters here.

        The system is two TRANSPORT equations (local: information moves along
        characteristics at finite speed) coupled to one ELLIPTIC equation (global: every
        point depends on every other point instantaneously). Treating both with the same
        timestep forces the local relaxation to proceed at the pace of the global
        coupling, and every local adjustment is immediately redistributed over the whole
        domain by the next Poisson solve. Freezing Psi lets the transport relax ~31x more
        often than the elliptic coupling is updated, which is a genuinely different
        effective dynamics -- not an optimisation.

        The gauge IS still recomputed at every stage, as they do (`opt.tran = 1` still
        calls RK_df, which recomputes cl, cw from the current field)."""
        cl, cw, cond, dO, dB = self.step(k)          # outer: full, with Poisson
        Pt_frozen = self.poisson(self.Ot)            # freeze the velocity here
        for _ in range(n_inner):
            dO1, dB1, _, cl, cw, cond = self.rhs(self.Ot, self.Bt, Pt_frozen)
            O1, B1 = self.Ot + k2 * dO1, self.Bt + k2 * dB1
            dO2, dB2, _, _, _, _ = self.rhs(O1, B1, Pt_frozen)
            self.Ot = 0.5 * (O1 + self.Ot + k2 * dO2)
            self.Bt = 0.5 * (B1 + self.Bt + k2 * dB2)
            if self.filter_on:
                bc = {"c": (0, slice(None)), "a": (slice(None), -1)}
                self.Ot = self.pm.March.filt(self, self.Ot, bc)
                self.Bt = self.pm.March.filt(self, self.Bt, bc)
        return cl, cw, cond, dO, dB

    def step(self, dt):
        dO1, dB1, _, cl, cw, cond = self.rhs(self.Ot, self.Bt)
        O1, B1 = self.Ot + dt * dO1, self.Bt + dt * dB1
        dO2, dB2, _, _, _, _ = self.rhs(O1, B1)
        self.Ot = 0.5 * (O1 + self.Ot + dt * dO2)
        self.Bt = 0.5 * (B1 + self.Bt + dt * dB2)
        if self.filter_on:
            bc = {"c": (0, slice(None)), "a": (slice(None), -1)}
            self.Ot = self.pm.March.filt(self, self.Ot, bc)
            self.Bt = self.pm.March.filt(self, self.Bt, bc)
        return cl, cw, cond, dO1, dB1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Nx", type=int, default=48)
    ap.add_argument("--Nb", type=int, default=48)
    ap.add_argument("--XMAX", type=float, default=25.0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--dt", type=float, default=1e-3)
    a = ap.parse_args()

    C = Corner(a.Nx, a.Nb, a.XMAX)
    print(f"CORNER-INCLUSIVE frame  xi = ln(1+r),  xi in [0, {a.XMAX}]  "
          f"grid {C.nx}x{C.nb}")
    print(f"  r spans {C.r[0]:.3g} .. {C.r[-1]:.3g}   (the CORNER r=0 IS in the domain)")
    print(f"  g = r/(1+r): first interior node g={C.g[1]:.4f}, last g={C.g[-1]:.8f}")
    print(f"  reference c_l={C.P['cl']:.8f} c_w={C.P['cw']:.8f} alpha={C.a0:+.8f}")

    f = np.exp(0.3 * C.x)
    e1 = np.abs(C.dx(f[:, None])[:, 0] - 0.3 * f).max() / np.abs(0.3 * f).max()
    print(f"\nGATE 0  spectral d_xi: {e1:.3e}  {'PASS' if e1 < 1e-9 else 'FAIL'}")

    Pt = C.poisson(C.Ot0)
    I = (slice(2, -2), slice(2, -2))
    rel = np.abs(Pt[I] - C.Pt0[I]).max() / max(np.abs(C.Pt0[I]).max(), 1e-300)
    print(f"GATE 1  Poisson vs the seed's Psi: max rel {rel:.3e}  "
          f"{'PASS' if rel < 0.10 else 'CHECK'}")

    _, pO, pB = C.parts(C.Ot0, C.Bt0)
    cl, cw, cond = C.gauge(C.Ot0, C.Bt0, pO, pB)
    print(f"GATE 2  gauge: c_l={cl:+.6f} (err {abs(cl-C.P['cl'])/abs(C.P['cl']):.3e})  "
          f"c_w={cw:+.6f} (err {abs(cw-C.P['cw'])/abs(C.P['cw']):.3e})  cond {cond:.4g}")
    print(f"        alpha implied {cw/cl:+.6f} vs {C.a0:+.6f}")

    dOt, dBt, _, _, _, _ = C.rhs(C.Ot0, C.Bt0)
    print(f"GATE 3  seed steadiness: max|dOt|/|Ot| "
          f"{np.abs(dOt[I]).max()/np.abs(C.Ot0[I]).max():.3e}   max|dBt|/|Bt| "
          f"{np.abs(dBt[I]).max()/np.abs(C.Bt0[I]).max():.3e}")

    if a.steps:
        print(f"\nmarching {a.steps} steps at dt={a.dt}")
        O0 = C.Ot0.copy(); sc = np.abs(O0[I]).max()
        print(f"  {'step':>6s} {'tau':>7s} {'c_l':>11s} {'alpha':>10s} "
              f"{'max|dOt|':>11s} {'drift':>11s}")
        for k in range(a.steps):
            cl, cw, cond, dO, dB = C.step(a.dt)
            if k % max(1, a.steps // 12) == 0 or k == a.steps - 1:
                print(f"  {k:6d} {k*a.dt:7.2f} {cl:11.6f} {cw/cl:10.6f} "
                      f"{np.abs(dO[I]).max():11.4e} "
                      f"{np.abs(C.Ot[I]-O0[I]).max()/sc:11.4e}", flush=True)


if __name__ == "__main__":
    main()
