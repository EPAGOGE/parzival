"""RADIAL-PANEL SOLVER: piecewise-Chebyshev in xi, Pt kept as an UNKNOWN, sparse Newton.

WHY THIS EXISTS (premortem wf_703cc27b-9dc + collapse test, 2026-07-26/27). The (N, XMAX)
surface is genuinely 2-D, every knob on the single-grid solver is exhausted at ~1-2%
against a target known to 3e-7, and the floor is STRUCTURAL: one global Chebyshev grid
cannot put resolution where the field varies (43% of nodes sat on xi>15 carrying 1.1% of
the variation at L=25), and the dense direct Newton dies near N~52.

TWO DECISIONS, EACH LOAD-BEARING:

1. Pt IS AN UNKNOWN, NOT ELIMINATED. The old solver eliminates Pt through a prefactored
   Poisson LU; that inverse is GLOBAL, so the reduced Jacobian is dense no matter how the
   grid is structured -- panelizing the grid under that formulation buys nothing.  Keeping
   (Ot, Bt, Pt) all as unknowns makes every operator LOCAL (per-panel derivative blocks,
   pointwise coefficients), the full Jacobian sparse, and -- the free bonus -- removes the
   n-Poisson-solves-per-iteration cost of A_exact, which dominated wall clock at N>=44.

2. DUPLICATED INTERFACE NODES + EXPLICIT MATCHING EQUATIONS (classical patching).  Each
   panel owns both its endpoints; the redundant rows carry continuity equations instead of
   the PDE.  Counting per field (K panels, panel k of degree p_k):
     transport (Ot, Bt -- 1st order in xi, characteristics run OUTWARD since c_l > 0):
       PDE at interior nodes AND at panel right-endpoints (outflow: PDE is valid there,
       and this formulation imposes NO outer radial condition on Ot/Bt -- POLAR_SPEC
       sections 8/10: a condition at outflow is the trap that once gave 1e12 step norms);
       C0 continuity at each panel-left-endpoint (the inflow side);
       the global corner row (xi=0) stays PINNED to the exact values, as before.
     Poisson (Pt -- 2nd order in xi):
       PDE at interior nodes; Pt=0 at the corner (identity row, consistent -- see
       POLAR_SPEC section 16 caveat); d_xi Pt = 0 at the outer endpoint ('neumann', the
       form the DtN experiments exonerated to 8e-10..2.5e-8 in alpha);
       C0 at each interface left-endpoint, C1 (one-sided derivative match) at each
       interface right-endpoint.
   Beta treatment is IDENTICAL to the single-grid solver (same Chebyshev grid on
   [eps_b, pi/2 - eps_b], Pt=0 on both beta edges, Ot/Bt axis column pinned to seed),
   so single-panel runs are the SAME discrete system as polar_stability and must
   reproduce its alpha to solver tolerance -- that is gate G1, and it is not optional.

CONSTRAINTS AND GAUGE: unchanged. c_l, c_w are unknowns; g1 pins Ot_xi(0,0) = WX_REF;
g2 is the 'd1' form (Dx @ (Bt/g))[0,0] = THXX_REF/2 (Vt[0]=0 exact).  The damped outer
alpha loop, the accepted-step flag, and the open-residual reporting all carry over --
those harness rules were bought with real blood this week.

PINNED ROWS AS IDENTITY EQUATIONS: instead of deleting pinned unknowns (the old mask),
pinned rows carry z - z_seed = 0.  Same solution, much simpler sparse indexing, and the
residual on those rows is exactly 0 at every Newton iterate so norms are comparable.
"""
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


class PanelSolver:
    WX_REF, THXX_REF = 1.19620314, 1.79819132     # polar_gauge_gate.py
    # CORNER-DUST DE-COLLOCATION RADIUS.  Measured root cause of the K=2 stall
    # (2026-07-27): the transport PDE collocated at wall-line (j=0) nodes with
    # x_i below ~0.02 is nearly vacuous -- its advective coefficient
    # EG*Pt_b - cl*G is ~ -0.01 there -- and once the first interior node crosses
    # x1 ~ 0.02 those rows go numerically dependent on the corner Bt / near-wall
    # Poisson / beta-edge rows (row-normalized left-null energy 0.37 on B(1..5,0)
    # + 0.63 on P(1..5,1..); sigma_min 7.5e-7 -> 2e-9 at the seed, 3.5e-10 at the
    # stall, and the stalled system is INCONSISTENT: truncated-SVD Newton floors at
    # ||F|| ~ 1.3e-8).  Interface location / outer row / panel-1 degree are all
    # measured non-causes; K=1 crosses the same threshold at deg >= 40 (x1 <=
    # 0.023, sigma_min 4e-8) -- this degeneracy, not LU cost, is likely what
    # killed the dense solver near N~52.  Cure: do not collocate the PDE at dust
    # nodes; carry an INTERPOLATION (de-collocation) row instead -- the value at
    # a dust node is the panel's wall-line polynomial through the NON-dust nodes.
    # Exactly satisfiable (consistency restored), zero rows touched for K=1
    # deg35 (x1=0.030: G1 binary-identical).  Measured on the (0,2,15)/(16,18)
    # reproducer: newton 5 steps to 3e-14, solution sigma_min 1.4e-8.  OPEN: at
    # deeper corner clustering ((24,18): x1=0.009, (32,18): x1=0.005) margins
    # are still thin and Newton limps -- the fixed threshold is a first cut, not
    # yet a uniform rule; revisit before pushing panel-0 degree past ~20.
    CORNER_DUST = 0.025

    def __init__(self, edges=(0.0, 2.0, 15.0), degs=(16, 48), Nb=36, eps_b=1e-3,
                 alpha=None):
        pm = _mod("pm", "polar_march.py")
        ps = _mod("ps", "polar_seed.py")
        gs = _mod("gs", "polar_gauge_sweep.py")
        self.edges, self.degs = list(map(float, edges)), list(map(int, degs))
        self.K = len(self.degs)
        assert len(self.edges) == self.K + 1
        # --- radial composite grid -------------------------------------------
        xs, Ds, D2s = [], [], []
        for a, b, p in zip(self.edges[:-1], self.edges[1:], self.degs):
            x, D, D2 = pm.grid(p, a, b)
            xs.append(x); Ds.append(D); D2s.append(D2)
        self.sizes = [len(x) for x in xs]
        self.offs = np.concatenate([[0], np.cumsum(self.sizes)])
        self.x = np.concatenate(xs)
        self.Nx = len(self.x)
        self.Dx = sp.block_diag([sp.csr_matrix(D) for D in Ds], format="csr")
        self.Dx2 = sp.block_diag([sp.csr_matrix(D) for D in D2s], format="csr")
        self.lefts = [int(self.offs[k]) for k in range(self.K)]
        self.rights = [int(self.offs[k] + self.sizes[k] - 1) for k in range(self.K)]
        # --- beta grid (identical to Corner) ---------------------------------
        self.eps_b = float(eps_b)
        self.b, Db, Db2 = pm.grid(Nb - 1, self.eps_b, np.pi / 2 - self.eps_b)
        self.Nb = len(self.b)
        self.Db, self.Db2 = np.asarray(Db), np.asarray(Db2)
        # --- coefficients -----------------------------------------------------
        self.P = ps.load()
        self.a0 = float(self.P["alpha"] if alpha is None else alpha)
        self._coef()
        # --- seed, replicating polar_corner exactly on the panel nodes -------
        r = np.exp(self.x) - 1.0
        s_eq = np.log(np.maximum(r, 1e-12))
        Om, B, Psi = gs.fields_on(ps, self.P, s_eq, self.b)
        WX, THXX = self.WX_REF, self.THXX_REF
        RB0, RB1 = 0.02, 0.10
        rr = r[:, None]; cb = np.cos(self.b)[None, :]
        Om_a = WX * rr * cb
        B_a = 0.5 * THXX * rr ** 2 * cb ** 2
        t = np.clip((np.log(np.maximum(rr, 1e-300)) - np.log(RB0))
                    / (np.log(RB1) - np.log(RB0)), 0.0, 1.0)
        w = t * t * (3.0 - 2.0 * t)
        Om = w * Om + (1.0 - w) * Om_a
        B = w * B + (1.0 - w) * B_a
        self.Ot0 = Om * np.exp(-self.a0 * self.x)[:, None]
        self.Bt0 = B * np.exp(-(1.0 + 2.0 * self.a0) * self.x)[:, None]
        for A in (self.Ot0, self.Bt0):
            A[0, :] = 0.0
        # duplicated interface nodes: seed both copies identically (same xi)
        for k in range(1, self.K):
            self.Ot0[self.lefts[k], :] = self.Ot0[self.rights[k - 1], :]
            self.Bt0[self.lefts[k], :] = self.Bt0[self.rights[k - 1], :]
        self._rows()                       # row masks BEFORE the Pt seed solve needs them
        self.Pt0 = self._pt_seed(self.Ot0)

    # -------------------------------------------------------------------------
    def _coef(self):
        self.mu = 2.0 + self.a0
        self.g = 1.0 - np.exp(-self.x)
        self.G = self.g[:, None]
        self.E = np.exp(self.a0 * self.x)[:, None]
        Ginv = np.zeros_like(self.G)
        np.divide(1.0, self.G, out=Ginv, where=(self.G > 1e-13))
        self.EG = self.E * Ginv
        self.cosb = np.cos(self.b)[None, :]
        self.sinb = np.sin(self.b)[None, :]

    def set_alpha(self, a):
        self.a0 = float(a)
        self._coef()

    def _rows(self):
        """Row-role masks over one field's (Nx, Nb) grid, flattened i*Nb+j."""
        Nx, Nb = self.Nx, self.Nb
        rid = lambda i, j: i * Nb + j
        corner = [rid(0, j) for j in range(Nb)]
        axis = [rid(i, Nb - 1) for i in range(Nx)]
        il = [rid(i, j) for k in range(1, self.K) for i in [self.lefts[k]]
              for j in range(Nb)]
        ir = [rid(i, j) for k in range(self.K - 1) for i in [self.rights[k]]
              for j in range(Nb)]
        bedge = [rid(i, j) for i in range(Nx) for j in (0, Nb - 1)]
        outer = [rid(self.rights[-1], j) for j in range(1, Nb - 1)]
        self.r_corner = np.array(sorted(set(corner)), dtype=int)
        self.r_axis = np.array(sorted(set(axis)), dtype=int)
        # transport special rows: corner + axis pin, interface-left C0
        self.rT_pin = np.array(sorted(set(corner) | set(axis)), dtype=int)
        self.rT_c0 = np.array(sorted(set(il) - set(axis)), dtype=int)
        # corner-dust de-collocation rows (transport, wall line j=0, first panel):
        # interior nodes of panel 0 with x_i < CORNER_DUST lose their PDE row and
        # carry  F(i,0) - sum_m c_m F(m,0) = 0  over the non-dust panel-0 nodes.
        s0 = self.sizes[0]
        dust = [i for i in range(1, s0 - 1) if self.x[i] < self.CORNER_DUST]
        self.rT_interp = np.array([rid(i, 0) for i in dust], dtype=int)
        keep = [m for m in range(s0) if m not in dust]
        self.interp_coef = {}
        for i in dust:
            cs = []
            for m in keep:
                c = 1.0
                for l in keep:
                    if l != m:
                        c *= (self.x[i] - self.x[l]) / (self.x[m] - self.x[l])
                cs.append((m, c))
            self.interp_coef[i] = cs
        # Poisson special rows: beta edges + corner + outer + C0 + C1
        self.rP_bedge = np.array(sorted(set(bedge)), dtype=int)
        self.rP_corner = np.array(sorted(set(corner) - set(bedge)), dtype=int)
        self.rP_outer = np.array(sorted(set(outer) - set(bedge)), dtype=int)
        self.rP_c0 = np.array(sorted(set(il) - set(bedge)), dtype=int)
        self.rP_c1 = np.array(sorted(set(ir) - set(bedge)), dtype=int)
        # partner map for C0/C1 rows: duplicate node on the neighbour panel
        self.partner = {}
        for k in range(1, self.K):
            for j in range(self.Nb):
                self.partner[rid(self.lefts[k], j)] = rid(self.rights[k - 1], j)
                self.partner[rid(self.rights[k - 1], j)] = rid(self.lefts[k], j)

    # -------------------------------------------------------------------------
    def _ops(self):
        """Field-space sparse operators, built once per alpha."""
        n2 = self.Nx * self.Nb
        I2 = sp.identity(n2, format="csr")
        DX = sp.kron(self.Dx, sp.identity(self.Nb), format="csr")
        DX2 = sp.kron(self.Dx2, sp.identity(self.Nb), format="csr")
        DB = sp.kron(sp.identity(self.Nx), sp.csr_matrix(self.Db), format="csr")
        DBB = sp.kron(sp.identity(self.Nx), sp.csr_matrix(self.Db2), format="csr")
        return I2, DX, DX2, DB, DBB

    def _pt_block(self, I2, DX, DX2, DBB):
        d = lambda v: sp.diags(np.broadcast_to(v, (self.Nx, self.Nb)).ravel())
        g2 = self.G ** 2
        gg = self.G * (1.0 - self.G)
        mu = self.mu
        Lp = (d(g2) @ (DX2 + 2 * mu * DX + mu * mu * I2)
              + d(gg) @ (DX + mu * I2) + DBB)
        return self._overwrite_pt_rows(Lp.tolil(), DX)

    def _overwrite_pt_rows(self, L, DX):
        """Replace Poisson special rows: identity (corner/beta edges), one-sided
        Neumann (outer), C0 and C1 (interfaces)."""
        DXl = DX.tolil()
        for r in np.concatenate([self.rP_bedge, self.rP_corner]):
            L.rows[r] = [int(r)]; L.data[r] = [1.0]
        for r in self.rP_outer:
            L.rows[r] = list(DXl.rows[r]); L.data[r] = list(DXl.data[r])
        for r in self.rP_c0:
            L.rows[r] = sorted([int(r), self.partner[int(r)]])
            L.data[r] = [1.0, -1.0] if L.rows[r][0] == int(r) else [-1.0, 1.0]
        for r in self.rP_c1:
            p = self.partner[int(r)]
            cols = list(DXl.rows[r]) + list(DXl.rows[p])
            vals = list(DXl.data[r]) + [-v for v in DXl.data[p]]
            L.rows[r] = cols; L.data[r] = vals
        return sp.csr_matrix(L)

    def _pt_seed(self, Ot):
        I2, DX, DX2, DB, DBB = self._ops()
        Lp = self._pt_block(I2, DX, DX2, DBB)
        rhs = (-(self.G ** 2) * Ot).ravel().copy()
        for r in np.concatenate([self.rP_bedge, self.rP_corner, self.rP_outer,
                                 self.rP_c0, self.rP_c1]):
            rhs[r] = 0.0
        return spla.spsolve(sp.csc_matrix(Lp), rhs).reshape(self.Nx, self.Nb)

    # -------------------------------------------------------------------------
    def pack(self, Ot, Bt, Pt, cl, cw):
        return np.concatenate([Ot.ravel(), Bt.ravel(), Pt.ravel(), [cl, cw]])

    def unpack(self, z):
        n2 = self.Nx * self.Nb
        Ot = z[:n2].reshape(self.Nx, self.Nb)
        Bt = z[n2:2 * n2].reshape(self.Nx, self.Nb)
        Pt = z[2 * n2:3 * n2].reshape(self.Nx, self.Nb)
        return Ot, Bt, Pt, float(z[-2]), float(z[-1])

    def residual(self, z):
        Ot, Bt, Pt, cl, cw = self.unpack(z)
        dx = lambda F: (self.Dx @ F)
        db = lambda F: F @ self.Db.T
        a0, mu = self.a0, self.mu
        Ot_x, Ot_b = dx(Ot), db(Ot)
        Bt_x, Bt_b = dx(Bt), db(Bt)
        Pt_x, Pt_b = dx(Pt), db(Pt)
        PmuP = Pt_x + mu * Pt
        OxaO = Ot_x + a0 * Ot
        BxbB = Bt_x + (1.0 + 2.0 * a0) * Bt
        advO = PmuP * Ot_b - Pt_b * OxaO
        srcO = self.G * self.cosb * BxbB - self.sinb * Bt_b
        advB = PmuP * Bt_b - Pt_b * BxbB
        RO = self.EG * (-advO + srcO) + cl * (-self.G * OxaO) + cw * Ot
        RB = self.EG * (-advB) + cl * (-self.G * BxbB + Bt) + cw * 2.0 * Bt
        RP = ((self.G ** 2) * ((self.Dx2 @ Pt) + 2 * mu * Pt_x + mu * mu * Pt)
              + self.G * (1.0 - self.G) * PmuP + (Pt @ self.Db2.T)
              + (self.G ** 2) * Ot)
        ro, rb, rp = RO.ravel(), RB.ravel(), RP.ravel()
        # transport special rows
        for r in self.rT_pin:
            ro[r] = Ot.ravel()[r] - self.Ot0.ravel()[r]
            rb[r] = Bt.ravel()[r] - self.Bt0.ravel()[r]
        for r in self.rT_c0:
            p = self.partner[int(r)]
            ro[r] = Ot.ravel()[r] - Ot.ravel()[p]
            rb[r] = Bt.ravel()[r] - Bt.ravel()[p]
        for r in self.rT_interp:
            cs = self.interp_coef[int(r) // self.Nb]
            ro[r] = Ot.ravel()[r] - sum(c * Ot[m, 0] for m, c in cs)
            rb[r] = Bt.ravel()[r] - sum(c * Bt[m, 0] for m, c in cs)
        # Poisson special rows
        PtF, PxF = Pt.ravel(), Pt_x.ravel()
        for r in np.concatenate([self.rP_bedge, self.rP_corner]):
            rp[r] = PtF[r]
        for r in self.rP_outer:
            rp[r] = PxF[r]
        for r in self.rP_c0:
            rp[r] = PtF[r] - PtF[self.partner[int(r)]]
        for r in self.rP_c1:
            rp[r] = PxF[r] - PxF[self.partner[int(r)]]
        # constraints (d1 form)
        g1 = float((self.Dx @ Ot)[0, 0]) - self.WX_REF
        Vt = np.zeros_like(Bt)
        Vt[1:, :] = Bt[1:, :] / self.g[1:, None]
        g2 = float((self.Dx @ Vt)[0, 0]) - 0.5 * self.THXX_REF
        return np.concatenate([ro, rb, rp, [g1, g2]])

    # -------------------------------------------------------------------------
    def jacobian(self, z):
        Ot, Bt, Pt, cl, cw = self.unpack(z)
        n2 = self.Nx * self.Nb
        I2, DX, DX2, DB, DBB = self._ops()
        d = lambda v: sp.diags(np.asarray(v).ravel())
        bc = lambda v: sp.diags(np.broadcast_to(v, (self.Nx, self.Nb)).ravel())
        a0, mu = self.a0, self.mu
        Ot_b = Ot @ self.Db.T
        Bt_b = Bt @ self.Db.T
        Pt_b = Pt @ self.Db.T
        PmuP = (self.Dx @ Pt) + mu * Pt
        OxaO = (self.Dx @ Ot) + a0 * Ot
        BxbB = (self.Dx @ Bt) + (1.0 + 2.0 * a0) * Bt
        EG, G = bc(self.EG), bc(self.G)
        DXa = DX + a0 * I2
        DXb = DX + (1.0 + 2.0 * a0) * I2
        DXm = DX + mu * I2
        # field blocks
        J_OO = EG @ (-(d(PmuP) @ DB) + d(Pt_b) @ DXa) + cl * (-(G @ DXa)) + cw * I2
        J_OB = EG @ (bc(self.G * self.cosb) @ DXb - bc(self.sinb) @ DB)
        J_OP = EG @ (-(d(Ot_b) @ DXm) + d(OxaO) @ DB)
        J_BB = EG @ (-(d(PmuP) @ DB) + d(Pt_b) @ DXb) + cl * (-(G @ DXb) + I2) \
            + 2.0 * cw * I2
        J_BP = EG @ (-(d(Bt_b) @ DXm) + d(BxbB) @ DB)
        J_PO = bc(self.G ** 2)
        J_PP = self._pt_block(I2, DX, DX2, DBB)
        # c-columns from the affine structure
        # MO must be a COPY: Ot.ravel() is a VIEW into the caller's z, and fix_T zeroes
        # entries of these arrays -- without the copy, building the Jacobian silently
        # zeroed z's pinned Ot entries (caught by the G0 state-checksum: |z| changed and
        # the residual at "the same" z moved by 24). The other three are fresh arrays
        # from arithmetic, but copy them too so nobody has to re-derive which is which.
        LO = (-self.G * OxaO).ravel().copy(); MO = Ot.ravel().copy()
        LB = (-self.G * BxbB + Bt).ravel().copy(); MB = (2.0 * Bt).ravel().copy()
        # overwrite transport special rows (identity / C0), zero their c-cols
        def fix_T(Jself, Jother1, Jother2, Lc, Mc):
            Jself = Jself.tolil()
            Z1, Z2 = Jother1.tolil(), Jother2.tolil()
            for r in self.rT_pin:
                Jself.rows[r] = [int(r)]; Jself.data[r] = [1.0]
                Z1.rows[r] = []; Z1.data[r] = []
                Z2.rows[r] = []; Z2.data[r] = []
                Lc[r] = 0.0; Mc[r] = 0.0
            for r in self.rT_c0:
                p = self.partner[int(r)]
                cols = sorted([int(r), p])
                Jself.rows[r] = cols
                Jself.data[r] = [1.0, -1.0] if cols[0] == int(r) else [-1.0, 1.0]
                Z1.rows[r] = []; Z1.data[r] = []
                Z2.rows[r] = []; Z2.data[r] = []
                Lc[r] = 0.0; Mc[r] = 0.0
            for r in self.rT_interp:
                cs = self.interp_coef[int(r) // self.Nb]
                ent = sorted([(int(r), 1.0)] + [(m * self.Nb, -c) for m, c in cs])
                Jself.rows[r] = [e[0] for e in ent]
                Jself.data[r] = [e[1] for e in ent]
                Z1.rows[r] = []; Z1.data[r] = []
                Z2.rows[r] = []; Z2.data[r] = []
                Lc[r] = 0.0; Mc[r] = 0.0
            return sp.csr_matrix(Jself), sp.csr_matrix(Z1), sp.csr_matrix(Z2)

        J_OO, J_OB, J_OP = fix_T(J_OO, J_OB, J_OP, LO, MO)
        J_BB, J_BO_z, J_BP = fix_T(J_BB, sp.csr_matrix((n2, n2)), J_BP, LB, MB)
        # Poisson rows: J_PP already carries special rows; zero J_PO there
        J_PO = J_PO.tolil()
        for r in np.concatenate([self.rP_bedge, self.rP_corner, self.rP_outer,
                                 self.rP_c0, self.rP_c1]):
            J_PO.rows[r] = []; J_PO.data[r] = []
        J_PO = sp.csr_matrix(J_PO)
        # constraint rows
        rid0 = 0
        Cg = sp.lil_matrix((2, 3 * n2))
        Dx0 = self.Dx[0, :].toarray().ravel()
        for k in range(self.Nx):
            if Dx0[k] != 0.0:
                Cg[0, k * self.Nb + 0] = Dx0[k]
                if k >= 1:
                    Cg[1, n2 + k * self.Nb + 0] = Dx0[k] / self.g[k]
        Z = sp.csr_matrix((n2, n2))
        top = sp.bmat([[J_OO, J_OB, J_OP],
                       [J_BO_z, J_BB, J_BP],
                       [J_PO, Z, J_PP]], format="csr")
        Bcol = np.zeros((3 * n2, 2))
        Bcol[:n2, 0] = LO; Bcol[:n2, 1] = MO
        Bcol[n2:2 * n2, 0] = LB; Bcol[n2:2 * n2, 1] = MB
        J = sp.bmat([[top, sp.csr_matrix(Bcol)],
                     [sp.csr_matrix(Cg), sp.csr_matrix((2, 2))]], format="csc")
        return J

    # -------------------------------------------------------------------------
    def newton(self, z0=None, steps=40, tol=1e-11, verbose=False):
        z = self.pack(self.Ot0, self.Bt0, self.Pt0,
                      self.P["cl"], self.P["cw"]) if z0 is None else z0.copy()
        f = self.residual(z)
        r = np.linalg.norm(f) / np.sqrt(f.size)
        prev, taken = r, 0
        for it in range(steps):
            J = self.jacobian(z)
            try:
                dz = spla.splu(J).solve(-f)
            except RuntimeError as ex:
                print(f"    [newton] sparse LU FAILED: {ex} -- Jacobian singular, "
                      f"not a linesearch stall", flush=True)
                return z, f, r, taken
            lam, best = 1.0, None
            for _ in range(12):
                ft = self.residual(z + lam * dz)
                rt = np.linalg.norm(ft) / np.sqrt(ft.size)
                if rt < prev:
                    best = (z + lam * dz, ft, rt)
                    break
                lam *= 0.5
            if best is None:
                break
            z, f, r = best
            prev = r
            taken += 1
            if verbose:
                print(f"    it{it:02d} ||F||={r:.4e} c_l={z[-2]:.6f} "
                      f"alpha={z[-1]/z[-2]:+.8f}", flush=True)
            if r < tol:
                break
        return z, f, r, taken


def converge(edges, degs, Nb=36, eps_b=1e-3, theta=0.5, outer=80, tol=1e-11,
             verbose=False):
    """Damped outer alpha loop over the panel solver. Returns (S, z, r, info).

    The solver is built ONCE and only its coefficients (a0, mu, E, EG) are updated per
    pass -- the SEED and therefore the pinned-row targets stay frozen at the initial
    alpha, exactly as the single-grid solver behaves (its alpha override never rebuilt
    C.Ot0; measured benign for alpha at ~4e-9).  Rebuilding the seed each pass moves the
    identity-row targets under Newton's feet mid-continuation, which is what stalled the
    first K=2 attempt at pass 2."""
    a, z0 = None, None
    hist = []
    S = PanelSolver(edges=edges, degs=degs, Nb=Nb, eps_b=eps_b, alpha=None)
    for k in range(outer):
        if a is not None:
            S.set_alpha(a)
        z, f, r, taken = S.newton(z0=z0, tol=tol, verbose=verbose)
        if taken == 0:
            return S, z, r, dict(converged=False, reason="zero_steps", passes=k + 1)
        cl, cw = float(z[-2]), float(z[-1])
        an = cw / cl
        z0 = z
        hist.append(an)
        if a is not None and abs(an - a) < 1e-9 and r < tol:
            return S, z, r, dict(converged=True, alpha=an, cl=cl, passes=k + 1,
                                 hist=hist[-4:])
        a = an if a is None else a + theta * (an - a)
    return S, z, r, dict(converged=False, reason="outer_cap", passes=outer,
                         hist=hist[-4:])
