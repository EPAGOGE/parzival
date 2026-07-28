"""FREED-PIN SOLVER (adjudicated spec 2026-07-28) -- prototype subclass.

FreePinSolver frees the corner data THXX from its pinned REF value and closes
the system with the corner identity itself:

    z = [A(n2), B(n2), P(n2), THXX', cl, cw]          (3*n2 + 3 unknowns)
    F = [ro(n2), rb(n2), rp(n2), g1, g2, gI]          (3*n2 + 3 equations)

with exactly ONE new unknown THXX' (= z[3*n2]) and ONE new equation

    gI = cl - 2*THXX'/WX_REF = 0                       (the corner identity)

WX' is eliminated (== WX_REF) as the mandatory degree-breaking normalization of
the exact scaling symmetry T_s: (A,B,P,cl,cw) -> (sA, s^2 B, sP, s cl, s cw),
under which RO'->s^2 RO', RB'->s^3 RB', RP'->s RP' and alpha=cw/cl is invariant
(verified on the discrete residual to 1e-13; g2 functional carries weight 2).

Three row sites change relative to the base CornerRegSolver (nothing else):
  1. B-corner pin rows (global n2+j, j=0..Nb-1, INCLUDING the corner/axis node
     j=Nb-1): target becomes (THXX'/2) cos^2(b_j)  -- LIVE via _refresh().
  2. g2 (global 3n2+1): target becomes THXX'/2 (live; retargeted by adding
     0.5*(THXX_REF - THXX') to the base row).
  3. NEW row gI (global 3n2+2): cl - 2*THXX'/WX_REF.
A-corner pins and g1 keep static WX_REF targets: they ARE the normalization.

New Jacobian column (index 3n2): -cos^2(b_j)/2 at rows n2+j (all j incl. Nb-1),
-1/2 at the g2 row, -2/WX_REF at the gI row; zero elsewhere (in particular on
every P row).  New Jacobian row (gI): +1 at cl, -2/WX_REF at THXX', 0 at cw.

Implementation pattern: refresh-then-super.  _refresh(TH) writes the live pin
values into self.B0[0,:]; the base residual/_slave/jacobian read self.B0 for the
corner rows, so the pins are live at EVERY linesearch trial and slave projection
(kills the silent revert-to-pinned failure mode at all three sites at once).
The layout keeps z[-2]=cl, z[-1]=cw, so the inherited newton()/newton_plain()
iterations run verbatim (only the z0=None default seed is filled in here).

Ramp/pin fallback mode (spec section 6, Fallback 1): set S.TH_target = t to
swap the gI row for the pin row THXX' - t = 0 (one-row mode flag); set it back
to None to restore the identity closure.  g2 and the B-corner pins stay live at
THXX' in BOTH modes -- this is what fixes the derivation-2 hazard of hacking
THXX_REF (which desyncs g2 from B0).

The base solver file /Users/epagogellc/parzival/boussinesq/polar_cornerreg.py
is imported, never edited.  set_alpha only refreshes coefficients, so there is
NO rebuild-per-alpha: pins stay live functions of the unknown vector.

__main__ runs the mandatory gates:
  G0  FD-vs-analytic Jacobian directional derivative, 4 random directions
      (3 with a THXX' component, 1 with zero THXX' component as a base-block
      regression), on the full (3n2+3) system at a perturbed ground-grid state;
      PASS < 1e-5 worst relative deviation.  The corner/axis node row
      n2 + rid(0, Nb-1) is reported individually.  Also run in pin mode.
  G0b slave-liveness guard: perturb THXX' in z, call _slave, assert the
      returned B(0,j) equals 0.5*(THXX'+d)*cos^2 b_j to machine precision.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import scipy.sparse as sp

BOUS = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
_spec = importlib.util.spec_from_file_location("pc", str(BOUS / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(_spec)
sys.modules["pc"] = pc
_spec.loader.exec_module(pc)


class FreePinSolver(pc.CornerRegSolver):
    """Corner-regularized solver with the B-corner data THXX promoted to an
    unknown and the corner identity cl = 2*THXX'/WX_REF as its closure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TH_target = None          # None -> identity closure; float -> pin row
        n2 = self.Nx * self.Nb
        self._n2 = n2
        self._n3 = 3 * n2
        # static THXX' column over the base (3n2+2) rows: B-corner pins + g2.
        rows = np.concatenate([n2 + np.arange(self.Nb), [self._n3 + 1]])
        vals = np.concatenate([-0.5 * np.cos(self.b) ** 2, [-0.5]])
        self._tcol = sp.csc_matrix(
            (vals, (rows, np.zeros_like(rows))), shape=(self._n3 + 2, 1))

    # -- freed-layout helpers -------------------------------------------------
    def _refresh(self, TH):
        """Make the B-corner pin values a live function of the unknown."""
        self.B0[0, :] = 0.5 * TH * np.cos(self.b) ** 2

    def _strip(self, z):
        """Freed-layout (3n2+3) -> base-layout (3n2+2) vector."""
        return np.concatenate([z[:self._n3], z[-2:]])

    def _th(self, z):
        assert z.size == self._n3 + 3, \
            f"freed-layout vector expected (len {self._n3 + 3}), got {z.size}"
        return float(z[self._n3])

    def pack_free(self, A, B, Pf, TH, cl, cw):
        return np.concatenate([A.ravel(), B.ravel(), Pf.ravel(), [TH, cl, cw]])

    def unpack_free(self, z):
        A, B, Pf, cl, cw = self.unpack(self._strip(z))
        return A, B, Pf, self._th(z), cl, cw

    def seed_z(self, TH0=None):
        TH0 = self.THXX_REF if TH0 is None else float(TH0)
        return self.pack_free(self.A0, self.B0, self.P0, TH0,
                              self.P["cl"], self.P["cw"])

    @classmethod
    def from_seed(cls, npz_path, edges=(0.0, 2.0, 15.0, 25.0),
                  degs=(16, 40, 12), Nb=36, eps_b=1e-4, TH0=None):
        """Build a solver on the seed's grid, alpha from its 'a' key, with the
        axis/corner pin data A0/B0 seeded WHOLESALE from the npz fields (spec
        section 4: constructed axis data differs from the branch axis by ~4.5x
        its own amplitude; the npz corner circle equals the analytic pins to
        0.0, so the static A-corner pins are unchanged and B0[0,:] is
        overwritten live anyway).  Returns (solver, z0_freed)."""
        d = np.load(npz_path)
        zb = np.asarray(d["z"], dtype=float)
        S = cls(edges=edges, degs=degs, Nb=Nb, eps_b=eps_b,
                alpha=float(d["a"]))
        n2 = S.Nx * S.Nb
        assert zb.size == 3 * n2 + 2, \
            f"seed len {zb.size} != 3*n2+2 = {3 * n2 + 2}: wrong grid"
        S.A0 = zb[:n2].reshape(S.Nx, S.Nb).copy()
        S.B0 = zb[n2:2 * n2].reshape(S.Nx, S.Nb).copy()
        S.P0 = zb[2 * n2:3 * n2].reshape(S.Nx, S.Nb).copy()
        TH0 = S.THXX_REF if TH0 is None else float(TH0)
        z0 = np.concatenate([zb[:3 * n2], [TH0], zb[-2:]])
        return S, z0

    # -- the three changed row sites -----------------------------------------
    def residual(self, z):
        TH = self._th(z)
        self._refresh(TH)
        Fb = super().residual(self._strip(z))
        Fb[-1] += 0.5 * (self.THXX_REF - TH)         # g2 target -> THXX'/2
        if self.TH_target is None:
            gI = float(z[-2]) - 2.0 * TH / self.WX_REF
        else:
            gI = TH - float(self.TH_target)
        return np.concatenate([Fb, [gI]])

    def jacobian(self, z):
        TH = self._th(z)
        self._refresh(TH)                            # base J has no A0/B0 dep;
        Jb = super().jacobian(self._strip(z)).tocsc()  # refresh for consistency
        n3 = self._n3
        if self.TH_target is None:
            g_t, g_cl = -2.0 / self.WX_REF, 1.0
        else:
            g_t, g_cl = 1.0, 0.0
        bot_l = sp.csr_matrix((1, n3))
        bot_t = sp.csr_matrix(np.array([[g_t]]))
        bot_r = sp.csr_matrix(np.array([[g_cl, 0.0]]))
        return sp.bmat([[Jb[:, :n3], self._tcol, Jb[:, n3:]],
                        [bot_l, bot_t, bot_r]], format="csc")

    def _slave(self, z):
        TH = self._th(z)
        self._refresh(TH)
        zb = super()._slave(self._strip(z))
        return np.concatenate([zb[:self._n3], [TH], zb[-2:]])

    # -- inherited iterations, freed default seed -----------------------------
    def newton(self, z0=None, **kw):
        return super().newton(z0=self.seed_z() if z0 is None else z0, **kw)

    def newton_plain(self, z0=None, **kw):
        return super().newton_plain(z0=self.seed_z() if z0 is None else z0, **kw)


def converge_free(npz_path, edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12),
                  Nb=36, eps_b=1e-4, TH0=None, theta=0.5, outer=80, tol=1e-11,
                  verbose=False):
    """Damped outer alpha loop for the freed solver (the module-level converge
    reimplemented for the subclass; solver built once, seed from the npz,
    coefficients updated via set_alpha -- pins stay live, no rebuild)."""
    S, z0 = FreePinSolver.from_seed(npz_path, edges=edges, degs=degs, Nb=Nb,
                                    eps_b=eps_b, TH0=TH0)
    a, hist = None, []
    for k in range(outer):
        if a is not None:
            S.set_alpha(a)
        z, f, r, taken = S.newton(z0=z0, tol=tol, verbose=verbose)
        if taken == 0:
            return S, z, r, dict(converged=False, reason="zero_steps",
                                 passes=k + 1, hist=hist[-4:])
        cl, cw = float(z[-2]), float(z[-1])
        an = cw / cl
        TH = float(z[3 * S.Nx * S.Nb])
        z0 = z
        hist.append((an, TH))
        if a is not None and abs(an - a) < 1e-9 and r < tol:
            return S, z, r, dict(converged=True, alpha=an, cl=cl, THXX=TH,
                                 passes=k + 1, hist=hist[-4:])
        a = an if a is None else a + theta * (an - a)
    return S, z, r, dict(converged=False, reason="outer_cap", passes=outer,
                         hist=hist[-4:])


# =============================================================================
if __name__ == "__main__":
    HF = pathlib.Path(__file__).parent / "hunt_fields"
    SEED = HF / "rung_00_a-0.344712.npz"
    print("=== build: FreePinSolver on the ground grid, seeded from npz ===")
    S, z_seed = FreePinSolver.from_seed(SEED, degs=(16, 40, 12), Nb=36,
                                        eps_b=1e-4)
    n2, n3 = S._n2, S._n3
    print(f"grid Nx={S.Nx} Nb={S.Nb} n2={n2}  system {n3 + 3} x {n3 + 3}")
    F0 = S.residual(z_seed)
    print(f"seed residual: ||F||_rms={np.linalg.norm(F0) / np.sqrt(F0.size):.3e}"
          f"  gI(seed)={F0[-1]:+.6e}  (spec cross-check: -1.0605e-3)")

    # --- G0: FD directional derivative vs jacobian @ v ----------------------
    print("\n=== G0: FD vs analytic Jacobian (PASS < 1e-5) ===")
    rng = np.random.default_rng(20260728)
    # perturbed generic state: exercise the nonlinear terms, TH away from REF
    zg = z_seed.copy()
    zg[:n3] *= (1.0 + 0.05 * rng.standard_normal(n3))
    zg[n3] = 1.93                      # THXX' well off REF
    zg[-2] += 0.11
    zg[-1] -= 0.07
    h = 1e-6
    corner_axis_row = n2 + (S.Nb - 1)  # B pin row of the corner/axis node
    worst = 0.0
    for k in range(4):
        v = rng.standard_normal(zg.size)
        if k == 3:
            v[n3] = 0.0                # base-block regression direction
        v /= np.linalg.norm(v)
        Jv = np.asarray(S.jacobian(zg) @ v).ravel()
        fd = (S.residual(zg + h * v) - S.residual(zg - h * v)) / (2.0 * h)
        rel = np.linalg.norm(fd - Jv) / max(np.linalg.norm(Jv), 1e-300)
        ca = abs(fd[corner_axis_row] - Jv[corner_axis_row])
        gi = abs(fd[-1] - Jv[-1])
        tag = "TH-bearing" if k < 3 else "TH-zero   "
        print(f"  dir {k} [{tag}]  rel dev = {rel:.3e}   "
              f"corner/axis row abs dev = {ca:.3e}   gI row abs dev = {gi:.3e}")
        worst = max(worst, rel)
    print(f"  G0 worst relative deviation = {worst:.3e}  "
          f"-> {'PASS' if worst < 1e-5 else 'FAIL'} (gate 1e-5)")

    # --- G0 in pin (ramp fallback) mode --------------------------------------
    S.TH_target = 2.1
    v = rng.standard_normal(zg.size)
    v /= np.linalg.norm(v)
    Jv = np.asarray(S.jacobian(zg) @ v).ravel()
    fd = (S.residual(zg + h * v) - S.residual(zg - h * v)) / (2.0 * h)
    rel_pin = np.linalg.norm(fd - Jv) / max(np.linalg.norm(Jv), 1e-300)
    S.TH_target = None
    print(f"  pin-mode (TH_target=2.1) rel dev = {rel_pin:.3e}  "
          f"-> {'PASS' if rel_pin < 1e-5 else 'FAIL'}")

    # --- G0b: slave-liveness guard -------------------------------------------
    print("\n=== G0b: slave-liveness (revert-to-pinned tripwire) ===")
    dTH = 0.017
    zp = z_seed.copy()
    zp[n3] += dTH
    zs = S._slave(zp)
    Bs = zs[n2:2 * n2].reshape(S.Nx, S.Nb)
    want = 0.5 * (z_seed[n3] + dTH) * np.cos(S.b) ** 2
    dev = np.max(np.abs(Bs[0, :] - want))
    print(f"  max |B(0,:) - 0.5*(TH+d)*cos^2 b| = {dev:.3e}  "
          f"-> {'PASS' if dev < 1e-14 else 'FAIL'} (machine precision)")

    ok = worst < 1e-5 and rel_pin < 1e-5 and dev < 1e-14
    print(f"\nALL GATES: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
