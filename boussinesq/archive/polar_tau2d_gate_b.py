"""
polar_tau2d_gate_b.py -- DOUBLE-CHEBYSHEV TAU GATE for the log-polar profile solve.
(Independent "B" answer.  Written to a *_b.py path because polar_tau2d_gate.py
already existed on disk when this reconnaissance started; it was neither read
nor overwritten.)

RUN:  ~/parzival/.venv-dedalus/bin/python3 ~/parzival/boussinesq/polar_tau2d_gate_b.py

WHAT IS BEING SETTLED
---------------------
Every engine in this lab (dedalus_axisym.py, dedalus_bsq.py) is Fourier x Chebyshev:
ONE bounded direction, taus lifted from ONE basis.  The log-polar profile solve needs
ChebyshevT x ChebyshevT on (s, beta) in [S0,S1] x [0, pi/2], with TWO boundary
conditions per direction.  That double-direction tau construction is untested here.

TEST PROBLEM (all four BC kinds the real solver needs):
    Psi_ss + Psi_bb = F(s,beta)
    Psi        = 0        at beta = 0        (Dirichlet, wall)
    Psi        = 0        at beta = pi/2     (Dirichlet, symmetry line)
    d_s Psi    = mu Psi   at s = S0          (ROBIN, far-field power law)
    d_s Psi    = mu Psi   at s = S1          (ROBIN)

MANUFACTURED SOLUTIONS (both satisfy ALL FOUR conditions exactly):
    (1)  Psi = e^{mu s} sin(2 beta)                    F = (mu^2 -  4) Psi
    (2)  Psi = e^{mu s} (sin 2b + 0.3 sin 4b)          F = e^{mu s} [ (mu^2-4) sin2b
                                                            + 0.3 (mu^2-16) sin4b ]
    A sum of beta modes only satisfies ONE COMMON Robin condition because the
    s-dependence e^{mu s} is SHARED; two modes with different s-exponents would not.
    That is why (2) is built with a common e^{mu s} factor.
    S0=10, S1=25, mu=1.6576  ->  the exact field spans e^{mu(S1-S0)} = 6.3e10, i.e.
    the same ~10 decades the real far-field tail spans (power law starts at r~1e8).

HEADLINE RESULTS (measured; reproduced verbatim by running this file)
---------------------------------------------------------------------
1. The NAIVE 4-tau construction is SINGULAR.  Rank deficiency is EXACTLY 4, at every
   resolution, on every domain (so it is structural, not roundoff).  cond ~ 1e19-1e20.
   It still "solves" (SuperLU returns something) and the globally-normalised error
   still looks like 1.1e-14 -- but the tau fields come back at 1.7e18 = O(|Psi|)
   instead of roundoff, which is the tell.  THE TRAP THE TASK WARNED ABOUT: judged by
   gRel alone the naive build looks fine; judged by |tau| and by sRel it is garbage.
2. The degeneracy is the CORNER OVERLAP, and it obeys an exact law:
        rank deficiency  =  SUM over equations of  (# s-lifts) x (# beta-lifts)
   Verified 4/4 on (n_s,n_b) in {1,2}x{1,2}, and on a coupled 2-field system
   (Psi 2nd order + Om 1st order: 2*2 + 1*1 = 5 predicted, 5 measured).
3. THE FIX: mask 4 tau coefficients and 4 (redundant) BC rows via `valid_modes`.
   deficiency 4 -> 0;  cond 1.6e5 (was 3.9e19);  |tau| 7.6e-16 (was 6.0e17).
4. Tau surgery is NECESSARY BUT NOT SUFFICIENT.  Solving for the RAW Psi over 10
   decades has an irreducible relative-accuracy floor at the small-s end of
   eps * (dynamic range) = 2.22e-16 * 6.28e10 = 1.40e-5 (measured 1.74e-5 at 64x40,
   NOT improving with resolution).  Solving the SUBSTITUTED variable Psi = e^{mu s} P
   removes it entirely.

FINAL NUMBER (recommended construction: masked taus + exponential substitution):
    max relative error vs the exact two-mode manufactured solution,
    measured PER-s (so the small-amplitude end of the 10-decade range counts equally):
        Ns x Nb = 16x12 -> 3.88e-08
                  24x16 -> 4.26e-12
                  32x24 -> 1.77e-15     <- machine precision, spectrally converged
                  48x32 -> 1.66e-15
                  64x40 -> 1.76e-15
    Single-mode solution e^{mu s} sin(2 beta): 2.34e-15 at 32x24, 2.78e-15 at 48x32.
    Newton (NLBVP) on a nonlinear variant: Jacobian rank 876/876, cond 1.6e5,
    quadratic convergence to |dx|=2.5e-15 in 5 iterations, final error 1.97e-15.

NOTE ON THE REFERENCE SOURCES
-----------------------------
refs/selfSimilarEulerBoussinesq/README.md:118-119 says of the Boussinesq case:
"This is not finished."  The only double-Chebyshev 2D code in that repo is
boussinesq/backup/mwe.m (chebop2 with lbc/rbc/ubc/dbc) and
boussinesq/backup/periodicDoesSomething.m, which carries its own "FB ERROR" note at
line 9.  The working Newton machinery there (boussinesq/+fb/
calcDampedStepSizeAndDirection.m:40-43) is 1D `chebop`, not 2D.  So that reference
does NOT supply a validated double-Chebyshev construction to copy; this gate does.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
import numpy as np
import dedalus.public as d3

for _nm in ("dedalus", "subsystems", "solvers", "dedalus.core.subsystems"):
    logging.getLogger(_nm).setLevel(logging.ERROR)

S0, S1 = 10.0, 25.0
MU = 1.6576
BE = np.pi / 2
SING_TOL = 1e-11          # singular-value cutoff, relative to sigma_max


# =============================================================================
# THE TAU CONSTRUCTION  (this is the deliverable -- everything else is testing)
# =============================================================================
def build(Ns, Nb, s0=S0, s1=S1, dealias=1.0):
    """ChebyshevT(s) x ChebyshevT(beta), fields + the FOUR tau fields + lifts.

    TAU CONSTRUCTION, exactly:
      * 4 tau fields for a 2nd-order operator in 2 bounded directions.
      * ts1, ts2  live on the BETA basis alone -> shape (1, Nb).  They carry the two
        s-boundary conditions.  Lifted with  d3.Lift(t, sb.derivative_basis(2), -1)
        and (..., -2): the s output basis is ChebyshevT with Jacobi parameter k=2,
        and indices -1 / -2 are its last two polynomials, i.e. s-modes Ns-1, Ns-2 --
        exactly the two rows that d_ss(Psi) cannot reach.
      * tb1, tb2  live on the S basis alone -> shape (Ns, 1).  They carry the two
        beta-boundary conditions.  Lifted with d3.Lift(t, bb.derivative_basis(2), -1)
        and (..., -2), i.e. beta-modes Nb-1, Nb-2.
      * The lift index is per-direction and NEGATIVE only (Dedalus enforces n<0,
        operators.py:4285).
    """
    coords = d3.CartesianCoordinates("s", "beta")
    dist = d3.Distributor(coords, dtype=np.float64)
    sb = d3.ChebyshevT(coords["s"], size=Ns, bounds=(s0, s1), dealias=dealias)
    bb = d3.ChebyshevT(coords["beta"], size=Nb, bounds=(0.0, BE), dealias=dealias)
    s = dist.local_grid(sb)
    b = dist.local_grid(bb)

    U = dist.Field(name="U", bases=(sb, bb))     # the unknown (Psi, or P if rescaled)
    F = dist.Field(name="F", bases=(sb, bb))     # source
    ts1 = dist.Field(name="ts1", bases=bb)       # (1, Nb)  -> s-BC taus
    ts2 = dist.Field(name="ts2", bases=bb)
    tb1 = dist.Field(name="tb1", bases=sb)       # (Ns, 1)  -> beta-BC taus
    tb2 = dist.Field(name="tb2", bases=sb)

    sb2 = sb.derivative_basis(2)                 # k=2 output basis in s
    bb2 = bb.derivative_basis(2)                 # k=2 output basis in beta

    ns = dict(
        U=U, F=F, ts1=ts1, ts2=ts2, tb1=tb1, tb2=tb2,
        Ls=lambda A, n: d3.Lift(A, sb2, n),      # lift into the s direction
        Lb=lambda A, n: d3.Lift(A, bb2, n),      # lift into the beta direction
        ds=lambda A: d3.Differentiate(A, coords["s"]),
        db=lambda A: d3.Differentiate(A, coords["beta"]),
        lap=lambda A: d3.Laplacian(A, coords),
        MU=MU, S0=s0, S1=s1, BE=BE,
    )
    return dict(coords=coords, dist=dist, sb=sb, bb=bb, s=s, b=b,
                U=U, F=F, ts1=ts1, ts2=ts2, tb1=tb1, tb2=tb2, ns=ns,
                taus=(ts1, ts2, tb1, tb2))


TAU_TERMS = "Ls(ts1,-1) + Ls(ts2,-2) + Lb(tb1,-1) + Lb(tb2,-2)"


def apply_corner_mask(tb1, tb2, eqD0, eqD1):
    """KILL THE 4-DIMENSIONAL CORNER DEGENERACY.

    WHY: the equation row block is (Ns x Nb) in the (k=2, k=2) output basis.
    d_ss(Psi) reaches s-modes 0..Ns-3 (all beta); d_bb(Psi) reaches beta-modes
    0..Nb-3 (all s).  The 2x2 CORNER  {s in Ns-2,Ns-1} x {beta in Nb-2,Nb-1}
    is reachable by NO derivative of Psi -- but by BOTH the ts pair (via their
    top 2 beta coefficients) and the tb pair (via their top 2 s coefficients).
    Four rows, eight tau coefficients feeding them -> a 4-dim redundancy.

    MEASURED CONFIRMATION (gate G3 below):
      * the 4-dim RIGHT null space has ZERO energy in Psi -- it is 100% tau.
        So Psi is unique in exact arithmetic; the damage is purely conditioning.
      * the 4-dim LEFT null space is 100% inside the two beta-DIRICHLET BC rows.

    THE FIX: remove 4 tau columns and the 4 redundant rows, via `valid_modes`
    boolean masks (dedalus/core/subsystems.py:540-552 filters both, then checks
    squareness).  Masks MUST be applied AFTER add_equation and BEFORE build_solver.
      - tb1, tb2 are (Ns, 1): [-2:] drops their top 2 s-modes          -> -4 columns
      - the two Dirichlet equations are (Ns, 1): [-2:] drops the top   -> -4 rows
        2 s-modes of each.  (The BC is then enforced Galerkin-wise on the first
        Ns-2 s-modes; measured residual max|Psi(beta=0)|/|Psi| stays at 1e-16,
        so nothing is actually lost -- see the bcres column.)

    You must remove EXACTLY as many rows as columns.  If you mask only one side,
    Dedalus raises `ValueError: Non-square system: group=..., I=..., J=...`
    (subsystems.py:552) -- a loud failure, not a silent one.  Gate G6 exercises it.
    """
    tb1.valid_modes[-2:] = False
    tb2.valid_modes[-2:] = False
    eqD0["valid_modes"][-2:] = False
    eqD1["valid_modes"][-2:] = False


def make_lbvp(Ns, Nb, rescale, mask, dealias=1.0, maskmode="tb+dirichlet"):
    """Assemble the LBVP.  rescale=False solves for Psi directly;
    rescale=True solves for P with Psi = e^{MU s} P."""
    B = build(Ns, Nb, dealias=dealias)
    s, b, ns = B["s"], B["b"], B["ns"]
    U, F = B["U"], B["F"]

    shape = np.sin(2 * b) + 0.3 * np.sin(4 * b)          # two-mode angular profile
    lap_shape = -4.0 * np.sin(2 * b) - 4.8 * np.sin(4 * b)
    F_raw = np.exp(MU * s) * (MU ** 2 * shape + lap_shape)   # = Psi_ss + Psi_bb

    if not rescale:
        F["g"] = F_raw
        exact = np.exp(MU * s) * shape
        eqn = f"lap(U) + {TAU_TERMS} = F"
        rob = ("ds(U)(s=S0) - MU*U(s=S0) = 0", "ds(U)(s=S1) - MU*U(s=S1) = 0")
    else:
        # Psi = e^{MU s} P  =>  P_ss + 2 MU P_s + MU^2 P + P_bb = e^{-MU s} F
        # Robin d_s Psi = MU Psi  =>  P_s = 0 at both ends (the MU cancels exactly)
        F["g"] = F_raw * np.exp(-MU * s)
        exact = shape * np.ones_like(s)
        eqn = f"lap(U) + 2*MU*ds(U) + MU**2*U + {TAU_TERMS} = F"
        rob = ("ds(U)(s=S0) = 0", "ds(U)(s=S1) = 0")

    prob = d3.LBVP([U, B["ts1"], B["ts2"], B["tb1"], B["tb2"]], namespace=ns)
    prob.add_equation(eqn)
    eqD0 = prob.add_equation("U(beta=0) = 0")
    eqD1 = prob.add_equation("U(beta=BE) = 0")
    eqR0 = prob.add_equation(rob[0])
    eqR1 = prob.add_equation(rob[1])

    if mask:
        if maskmode == "tb+dirichlet":
            apply_corner_mask(B["tb1"], B["tb2"], eqD0, eqD1)
        elif maskmode == "ts+robin":             # symmetric alternative
            B["ts1"].valid_modes[:, -2:] = False
            B["ts2"].valid_modes[:, -2:] = False
            eqR0["valid_modes"][:, -2:] = False
            eqR1["valid_modes"][:, -2:] = False
        elif maskmode == "cols_only":            # deliberately unbalanced
            B["tb1"].valid_modes[-2:] = False
            B["tb2"].valid_modes[-2:] = False
        else:
            raise ValueError(maskmode)
    return B, prob, exact


# =============================================================================
# measurement helpers
# =============================================================================
def _diagnose(prob, B, exact, want_svd=True):
    solver = prob.build_solver()
    sp = solver.subproblems[0]
    solver.build_matrices([sp], ["L"])
    A = sp.L_min.toarray()
    n = A.shape[0]
    if want_svd:
        sv = np.linalg.svd(A, compute_uv=False)
        defic = int((sv <= sv.max() * SING_TOL).sum())
        cond = float(sv.max() / max(sv.min(), 1e-300))
    else:
        sv, defic, cond = None, -1, float("nan")
    solver.solve()
    U = B["U"]
    num = U["g"]
    amax = float(np.abs(num).max())
    gmax = float(np.abs(exact).max())
    # (i) globally-normalised error -- the FLATTERING metric
    gRel = float(np.abs(num - exact).max() / gmax)
    # (ii) per-s relative error -- max over s of (max_b |err|) / (max_b |exact|).
    #      This is the metric that matters for a 10-decade far-field tail: it asks
    #      whether the SMALL-amplitude end of the range is resolved at all.
    sRel = float((np.abs(num - exact).max(axis=1) / np.abs(exact).max(axis=1)).max())
    coords = B["coords"]
    d0 = float(np.abs(d3.Interpolate(U, coords["beta"], 0.0).evaluate()["g"]).max())
    d1 = float(np.abs(d3.Interpolate(U, coords["beta"], BE).evaluate()["g"]).max())
    bcres = max(d0, d1) / max(amax, 1e-300)
    taumax = max(float(np.abs(t["g"]).max()) for t in B["taus"])
    return dict(n=n, defic=defic, cond=cond, gRel=gRel, sRel=sRel,
                bcres=bcres, tau=taumax, fieldnorm=amax)


_HDR = (f"{'case':>26s} {'n':>5s} {'defic':>5s} {'cond':>9s} {'|field|':>10s} "
        f"{'gRel':>9s} {'sRel':>9s} {'bcRes':>8s} {'|tau|':>9s}")


def _row(tag, r):
    print(f"{tag:>26s} {r['n']:5d} {r['defic']:5d} {r['cond']:9.2e} "
          f"{r['fieldnorm']:10.3e} {r['gRel']:9.2e} {r['sRel']:9.2e} "
          f"{r['bcres']:8.1e} {r['tau']:9.2e}")


# =============================================================================
# GATES
# =============================================================================
def g1_naive_is_singular():
    """G1: the naive 4-tau construction is rank deficient by EXACTLY 4, and its
    tau fields come back at O(|Psi|) -- the signature of a populated null space."""
    print("\nG1  NAIVE 4-tau construction (no mask) -- expected: defic = 4")
    print(_HDR)
    ok = True
    for Ns, Nb in [(24, 16), (32, 24), (48, 32)]:
        B, prob, ex = make_lbvp(Ns, Nb, rescale=False, mask=False)
        r = _diagnose(prob, B, ex)
        _row(f"raw   naive {Ns}x{Nb}", r)
        ok &= (r["defic"] == 4)
    # same structural deficiency on an O(1) domain -> it is NOT a roundoff artifact
    for Ns, Nb in [(16, 12), (24, 16)]:
        B = build(Ns, Nb, s0=0.0, s1=1.0)
        s, b = B["s"], B["b"]
        shape = np.sin(2 * b) + 0.3 * np.sin(4 * b)
        B["F"]["g"] = np.exp(MU * s) * (MU**2 * shape - 4*np.sin(2*b) - 4.8*np.sin(4*b))
        prob = d3.LBVP([B["U"], B["ts1"], B["ts2"], B["tb1"], B["tb2"]], namespace=B["ns"])
        B["ns"]["S0"], B["ns"]["S1"] = 0.0, 1.0
        prob.add_equation(f"lap(U) + {TAU_TERMS} = F")
        prob.add_equation("U(beta=0) = 0")
        prob.add_equation("U(beta=BE) = 0")
        prob.add_equation("ds(U)(s=S0) - MU*U(s=S0) = 0")
        prob.add_equation("ds(U)(s=S1) - MU*U(s=S1) = 0")
        r = _diagnose(prob, B, np.exp(MU * s) * shape)
        _row(f"O(1)dom naive {Ns}x{Nb}", r)
        ok &= (r["defic"] == 4)
    print(f"  -> deficiency is exactly 4 everywhere: {ok}")
    return ok


def g2_mask_regularises():
    """G2: the corner mask takes deficiency 4 -> 0 and cond 1e20 -> ~1e5,
    with tau fields collapsing to roundoff."""
    print("\nG2  MASKED construction -- expected: defic = 0, |tau| ~ eps")
    print(_HDR)
    ok = True
    rows = []
    for Ns, Nb in [(24, 16), (32, 24), (48, 32), (64, 40)]:
        B, prob, ex = make_lbvp(Ns, Nb, rescale=False, mask=True)
        r = _diagnose(prob, B, ex)
        _row(f"raw   MASK {Ns}x{Nb}", r)
        rows.append(r)
        ok &= (r["defic"] == 0 and r["cond"] < 1e8)
    print("  -> the RAW-Psi sRel column stalls near %.1e." % rows[-1]["sRel"])
    print("     PREDICTED floor = eps * dynamic range = %.2e * %.2e = %.2e"
          % (np.finfo(float).eps, np.exp(MU * (S1 - S0)),
             np.finfo(float).eps * np.exp(MU * (S1 - S0))))
    print("     Tau surgery cannot fix this; only the exponential substitution can.")
    return ok


def g3_null_space_anatomy(Ns=16, Nb=12):
    """G3: locate the null space.  RIGHT null space must be 100% tau (so Psi is
    mathematically unique); LEFT null space must be 100% in the Dirichlet rows
    (so those are the rows it is legitimate to drop)."""
    print("\nG3  NULL-SPACE ANATOMY of the naive system (Ns=%d, Nb=%d)" % (Ns, Nb))
    B, prob, ex = make_lbvp(Ns, Nb, rescale=False, mask=False)
    solver = prob.build_solver()
    sp = solver.subproblems[0]
    solver.build_matrices([sp], ["L"])
    A = sp.L_min.toarray()
    n = A.shape[0]
    Um, sv, Vt = np.linalg.svd(A)
    varlist = [B["U"], B["ts1"], B["ts2"], B["tb1"], B["tb2"]]
    sizes = [sp.field_size(v) for v in varlist]
    offs = np.cumsum([0] + sizes)
    P = sp.pre_right_pinv.toarray()
    colvar = np.array([np.searchsorted(offs, np.nonzero(P[j])[0][0], side="right") - 1
                       for j in range(P.shape[0])])
    eqs = prob.equations
    esz = [sp.field_size(e["eqn"]) for e in eqs]
    eoffs = np.cumsum([0] + esz)
    PL = sp.pre_left.toarray()
    roweq = np.array([np.searchsorted(eoffs, np.nonzero(PL[i])[0][0], side="right") - 1
                      for i in range(PL.shape[0])])
    eqnames = ["PDE", "Dirichlet b=0", "Dirichlet b=pi/2", "Robin s=S0", "Robin s=S1"]
    print("  singular values (smallest 6):",
          np.array2string(sv[-6:], precision=2, max_line_width=200))
    psi_leak = 0.0
    bc_frac = 0.0
    for m in range(4):
        v = Vt[n - 1 - m]
        ev = np.array([np.sum(v[colvar == i] ** 2) for i in range(len(varlist))])
        ev /= ev.sum()
        u = Um[:, n - 1 - m]
        eu = np.array([np.sum(u[roweq == i] ** 2) for i in range(len(eqs))])
        eu /= eu.sum()
        psi_leak = max(psi_leak, ev[0])
        bc_frac = max(bc_frac, eu[1] + eu[2])
        print("   sv=%.1e | RIGHT: " % sv[n - 1 - m]
              + " ".join(f"{v_.name}={e:.3f}" for v_, e in zip(varlist, ev))
              + " | LEFT: " + " ".join(f"{nm}={e:.3f}" for nm, e in zip(eqnames, eu)))
    # The RIGHT null space is exactly zero in Psi (measured ~1e-28: this is a hard
    # structural statement).  The LEFT null space is ~99.7% inside the two Dirichlet
    # BC blocks with a ~0.3% leak into the Robin blocks -- the null space is not
    # perfectly block-aligned, so 0.99 is the honest bar, not 1-1e-6.
    ok = (psi_leak < 1e-20) and (bc_frac > 0.99)
    print(f"  -> max Psi energy in right null space = {psi_leak:.2e} (want ~0; bar 1e-20)")
    print(f"  -> max Dirichlet share of left null space = {bc_frac:.6f} (bar 0.99)")
    return ok


def g4_deficiency_law(Ns=16, Nb=12):
    """G4: deficiency = sum over equations of (# s-lifts)*(# beta-lifts)."""
    print("\nG4  DEFICIENCY LAW:  defic = SUM_eqns (n_s_lifts * n_beta_lifts)")

    def single(ns_tau, nb_tau):
        coords = d3.CartesianCoordinates("s", "beta")
        dist = d3.Distributor(coords, dtype=np.float64)
        sb = d3.ChebyshevT(coords["s"], size=Ns, bounds=(0.0, 1.0))
        bb = d3.ChebyshevT(coords["beta"], size=Nb, bounds=(0.0, BE))
        U = dist.Field(name="U", bases=(sb, bb))
        F = dist.Field(name="F", bases=(sb, bb))
        sbk, bbk = sb.derivative_basis(ns_tau), bb.derivative_basis(nb_tau)
        tsl = [dist.Field(name=f"ts{i}", bases=bb) for i in range(ns_tau)]
        tbl = [dist.Field(name=f"tb{i}", bases=sb) for i in range(nb_tau)]
        nsp = dict(U=U, F=F, BE=BE,
                   ds=lambda A: d3.Differentiate(A, coords["s"]),
                   db=lambda A: d3.Differentiate(A, coords["beta"]),
                   Ls=lambda A, n: d3.Lift(A, sbk, n),
                   Lb=lambda A, n: d3.Lift(A, bbk, n))
        nsp.update({f"ts{i}": t for i, t in enumerate(tsl)})
        nsp.update({f"tb{i}": t for i, t in enumerate(tbl)})
        sop = "ds(ds(U))" if ns_tau == 2 else "ds(U)"
        bop = "db(db(U))" if nb_tau == 2 else "db(U)"
        tt = " + ".join([f"Ls(ts{i},{-(i+1)})" for i in range(ns_tau)]
                        + [f"Lb(tb{i},{-(i+1)})" for i in range(nb_tau)])
        prob = d3.LBVP([U] + tsl + tbl, namespace=nsp)
        prob.add_equation(f"{sop} + {bop} + {tt} = F")
        for bc in (["U(beta=0) = 0", "U(beta=BE) = 0"][:nb_tau]
                   + ["U(s=0) = 0", "U(s=1) = 0"][:ns_tau]):
            prob.add_equation(bc)
        solver = prob.build_solver()
        sp = solver.subproblems[0]
        solver.build_matrices([sp], ["L"])
        A = sp.L_min.toarray()
        sv = np.linalg.svd(A, compute_uv=False)
        return int((sv <= sv.max() * SING_TOL).sum()), A.shape[0]

    ok = True
    for nst in (1, 2):
        for nbt in (1, 2):
            d, n = single(nst, nbt)
            good = (d == nst * nbt)
            ok &= good
            print(f"   n_s_lifts={nst} n_b_lifts={nbt}: n={n:4d} "
                  f"defic={d} predicted={nst*nbt}  {'OK' if good else 'MISMATCH'}")
    d, n, predicted = _coupled_case(24, 16, mask=False)
    ok &= (d == predicted)
    print(f"   coupled 2-field (Psi 2nd order + Om 1st order): n={n} defic={d} "
          f"predicted={predicted}  {'OK' if d == predicted else 'MISMATCH'}")
    dm, nm, _ = _coupled_case(24, 16, mask=True)
    ok &= (dm == 0)
    print(f"   coupled 2-field WITH masks:                     n={nm} defic={dm}")
    return ok


def _coupled_case(Ns, Nb, mask):
    """Psi: 2nd order (2 s-lifts, 2 b-lifts).  Om: 1st order (1 s-lift, 1 b-lift).
    Predicted deficiency = 2*2 + 1*1 = 5."""
    coords = d3.CartesianCoordinates("s", "beta")
    dist = d3.Distributor(coords, dtype=np.float64)
    sb = d3.ChebyshevT(coords["s"], size=Ns, bounds=(S0, S1))
    bb = d3.ChebyshevT(coords["beta"], size=Nb, bounds=(0.0, BE))
    Psi = dist.Field(name="Psi", bases=(sb, bb))
    Om = dist.Field(name="Om", bases=(sb, bb))
    F = dist.Field(name="F", bases=(sb, bb))
    G = dist.Field(name="G", bases=(sb, bb))
    ps1 = dist.Field(name="ps1", bases=bb); ps2 = dist.Field(name="ps2", bases=bb)
    pb1 = dist.Field(name="pb1", bases=sb); pb2 = dist.Field(name="pb2", bases=sb)
    os1 = dist.Field(name="os1", bases=bb); ob1 = dist.Field(name="ob1", bases=sb)
    sb1, bb1 = sb.derivative_basis(1), bb.derivative_basis(1)
    sb2, bb2 = sb.derivative_basis(2), bb.derivative_basis(2)
    ns = dict(Psi=Psi, Om=Om, F=F, G=G, ps1=ps1, ps2=ps2, pb1=pb1, pb2=pb2,
              os1=os1, ob1=ob1, BE=BE, S0=S0, S1=S1,
              Ls2=lambda A, n: d3.Lift(A, sb2, n), Lb2=lambda A, n: d3.Lift(A, bb2, n),
              Ls1=lambda A: d3.Lift(A, sb1, -1), Lb1=lambda A: d3.Lift(A, bb1, -1),
              ds=lambda A: d3.Differentiate(A, coords["s"]),
              db=lambda A: d3.Differentiate(A, coords["beta"]),
              lap=lambda A: d3.Laplacian(A, coords))
    prob = d3.LBVP([Psi, Om, ps1, ps2, pb1, pb2, os1, ob1], namespace=ns)
    prob.add_equation("lap(Psi) + Ls2(ps1,-1) + Ls2(ps2,-2)"
                      " + Lb2(pb1,-1) + Lb2(pb2,-2) + Om = F")
    prob.add_equation("ds(Om) + db(Om) + Ls1(os1) + Lb1(ob1) + Psi = G")
    eD0 = prob.add_equation("Psi(beta=0) = 0")
    eD1 = prob.add_equation("Psi(beta=BE) = 0")
    prob.add_equation("ds(Psi)(s=S0) = 0")
    prob.add_equation("ds(Psi)(s=S1) = 0")
    eO0 = prob.add_equation("Om(beta=0) = 0")
    prob.add_equation("Om(s=S0) = 0")
    if mask:
        pb1.valid_modes[-2:] = False
        pb2.valid_modes[-2:] = False
        eD0["valid_modes"][-2:] = False
        eD1["valid_modes"][-2:] = False
        ob1.valid_modes[-1:] = False          # 1 s-lift x 1 b-lift -> 1 extra
        eO0["valid_modes"][-1:] = False
    solver = prob.build_solver()
    sp = solver.subproblems[0]
    solver.build_matrices([sp], ["L"])
    A = sp.L_min.toarray()
    sv = np.linalg.svd(A, compute_uv=False)
    return int((sv <= sv.max() * SING_TOL).sum()), A.shape[0], 5


def g5_rescaled_spectral_convergence():
    """G5: the RECOMMENDED construction -- masked taus + Psi = e^{mu s} P.
    Machine precision uniformly across the whole 10-decade range."""
    print("\nG5  RECOMMENDED: masked taus + substitution Psi = e^{mu s} P")
    print(_HDR)
    ok = True
    best = 1.0
    for Ns, Nb in [(16, 12), (24, 16), (32, 24), (48, 32), (64, 40)]:
        B, prob, ex = make_lbvp(Ns, Nb, rescale=True, mask=True)
        r = _diagnose(prob, B, ex)
        _row(f"resc  MASK {Ns}x{Nb}", r)
        if Ns >= 32:
            ok &= (r["defic"] == 0 and r["sRel"] < 1e-13)
            best = min(best, r["sRel"])
    print(f"  -> best per-s relative error = {best:.3e}")
    return ok, best


def g6_single_mode_and_failure_modes():
    """G6: (a) single-mode manufactured solution; (b) unbalanced mask -> loud error;
    (c) symmetric mask location also works; (d) dealias=3/2 unaffected."""
    print("\nG6  SINGLE MODE, UNBALANCED MASK, ALTERNATIVE MASK, DEALIAS")
    ok = True
    # (a) single mode Psi = e^{mu s} sin(2 beta),  F = (mu^2-4) Psi
    for rescale in (False, True):
        for Ns, Nb in [(32, 24), (48, 32)]:
            B = build(Ns, Nb)
            s, b, ns = B["s"], B["b"], B["ns"]
            F_raw = (MU ** 2 - 4.0) * np.exp(MU * s) * np.sin(2 * b)
            if rescale:
                B["F"]["g"] = F_raw * np.exp(-MU * s)
                ex = np.sin(2 * b) * np.ones_like(s)
                eqn = f"lap(U) + 2*MU*ds(U) + MU**2*U + {TAU_TERMS} = F"
                rob = ("ds(U)(s=S0) = 0", "ds(U)(s=S1) = 0")
            else:
                B["F"]["g"] = F_raw
                ex = np.exp(MU * s) * np.sin(2 * b)
                eqn = f"lap(U) + {TAU_TERMS} = F"
                rob = ("ds(U)(s=S0) - MU*U(s=S0) = 0", "ds(U)(s=S1) - MU*U(s=S1) = 0")
            prob = d3.LBVP([B["U"], B["ts1"], B["ts2"], B["tb1"], B["tb2"]], namespace=ns)
            prob.add_equation(eqn)
            e0 = prob.add_equation("U(beta=0) = 0")
            e1 = prob.add_equation("U(beta=BE) = 0")
            prob.add_equation(rob[0]); prob.add_equation(rob[1])
            apply_corner_mask(B["tb1"], B["tb2"], e0, e1)
            r = _diagnose(prob, B, ex)
            _row(f"{'resc' if rescale else 'raw '} 1mode {Ns}x{Nb}", r)
            if rescale and Ns >= 32:
                ok &= (r["defic"] == 0 and r["sRel"] < 1e-13)
    # (b) unbalanced mask must raise Non-square
    try:
        B, prob, ex = make_lbvp(32, 24, rescale=True, mask=True, maskmode="cols_only")
        _diagnose(prob, B, ex, want_svd=False)
        print("   (b) unbalanced mask: NO ERROR RAISED  <-- unexpected")
        ok = False
    except ValueError as e:
        print(f"   (b) unbalanced mask correctly raises: {str(e)[:70]}")
    # (c) symmetric mask location
    B, prob, ex = make_lbvp(32, 24, rescale=True, mask=True, maskmode="ts+robin")
    r = _diagnose(prob, B, ex)
    _row("resc MASK(ts+robin) 32x24", r)
    ok &= (r["defic"] == 0)
    # (d) dealias 3/2 (what the real nonlinear solver will use)
    B, prob, ex = make_lbvp(32, 24, rescale=True, mask=True, dealias=1.5)
    r = _diagnose(prob, B, ex)
    _row("resc MASK dealias3/2 32x24", r)
    ok &= (r["defic"] == 0 and r["sRel"] < 1e-13)
    return ok


def g7_newton_jacobian():
    """G7: masks propagate into an NLBVP Jacobian (problems.py:397 copies
    var.valid_modes onto the perturbation fields), so Newton sees a NONSINGULAR
    Jacobian.  Also a live check that ||field|| is reported beside the residual."""
    print("\nG7  NLBVP / NEWTON -- Jacobian rank and quadratic convergence")
    ok = True
    for mask in (False, True):
        coords = d3.CartesianCoordinates("s", "beta")
        dist = d3.Distributor(coords, dtype=np.float64)
        Ns, Nb = 32, 24
        sb = d3.ChebyshevT(coords["s"], size=Ns, bounds=(S0, S1))
        bb = d3.ChebyshevT(coords["beta"], size=Nb, bounds=(0.0, BE))
        s = dist.local_grid(sb); b = dist.local_grid(bb)
        P = dist.Field(name="P", bases=(sb, bb))
        F = dist.Field(name="F", bases=(sb, bb))
        ts1 = dist.Field(name="ts1", bases=bb); ts2 = dist.Field(name="ts2", bases=bb)
        tb1 = dist.Field(name="tb1", bases=sb); tb2 = dist.Field(name="tb2", bases=sb)
        sb2, bb2 = sb.derivative_basis(2), bb.derivative_basis(2)
        ns = dict(P=P, F=F, ts1=ts1, ts2=ts2, tb1=tb1, tb2=tb2, MU=MU, BE=BE,
                  S0=S0, S1=S1,
                  Ls=lambda A, n: d3.Lift(A, sb2, n),
                  Lb=lambda A, n: d3.Lift(A, bb2, n),
                  ds=lambda A: d3.Differentiate(A, coords["s"]),
                  lap=lambda A: d3.Laplacian(A, coords))
        shape = np.sin(2 * b) + 0.3 * np.sin(4 * b)
        exact = shape * np.ones_like(s)
        # target root is P = shape for:  lap P + 2mu P_s + mu^2 P + 0.1 P^3 = F
        F["g"] = MU**2*shape + (-4*np.sin(2*b) - 4.8*np.sin(4*b)) + 0.1*shape**3
        tt = TAU_TERMS.replace("U", "P")
        prob = d3.NLBVP([P, ts1, ts2, tb1, tb2], namespace=ns)
        prob.add_equation(f"lap(P) + 2*MU*ds(P) + MU**2*P + {tt} - F = -0.1*P**3")
        e0 = prob.add_equation("P(beta=0) = 0")
        e1 = prob.add_equation("P(beta=BE) = 0")
        prob.add_equation("ds(P)(s=S0) = 0")
        prob.add_equation("ds(P)(s=S1) = 0")
        if mask:
            apply_corner_mask(tb1, tb2, e0, e1)
        solver = prob.build_solver()
        P["g"] = 0.9 * shape                      # seed near the target root
        hist = []
        for _ in range(10):
            solver.newton_iteration()
            pn = float(np.sqrt(sum(np.sum(np.abs(p["c"])**2)
                                   for p in solver.perturbations)))
            hist.append((pn, float(np.abs(P["g"]).max())))
            if pn < 1e-13:
                break
        sp = solver.subproblems[0]
        dF = sp.dF_min.toarray()
        rank = np.linalg.matrix_rank(dF)
        cond = np.linalg.cond(dF)
        err = float(np.abs(P["g"] - exact).max() / np.abs(exact).max())
        print(f"   mask={str(mask):5s} Jacobian rank {rank}/{dF.shape[0]} "
              f"cond={cond:.1e} iters={len(hist)} finalErr={err:.2e}")
        print("      " + "  ".join(f"|dx|={h[0]:.1e} |P|={h[1]:.4f}" for h in hist[:5]))
        if mask:
            ok &= (rank == dF.shape[0] and cond < 1e8 and err < 1e-12)
    return ok


# =============================================================================
def main():
    print("=" * 108)
    print("polar_tau2d_gate_b.py -- ChebyshevT x ChebyshevT tau construction")
    print(f"  domain s in [{S0}, {S1}], beta in [0, pi/2];  mu = {MU}")
    print(f"  exact-solution dynamic range e^(mu*(S1-S0)) = {np.exp(MU*(S1-S0)):.3e}")
    print("  gRel = err / max|exact| (global)   sRel = max_s [ max_b|err| / max_b|exact| ]")
    print("=" * 108)
    results = []
    results.append(("G1 naive construction is singular (defic==4)", g1_naive_is_singular()))
    results.append(("G3 null space: 100% tau / 99.7% Dirichlet rows", g3_null_space_anatomy()))
    results.append(("G4 deficiency law defic=sum n_s*n_b", g4_deficiency_law()))
    results.append(("G2 mask regularises (defic==0, cond<1e8)", g2_mask_regularises()))
    ok5, best = g5_rescaled_spectral_convergence()
    results.append(("G5 masked+rescaled reaches sRel<1e-13", ok5))
    results.append(("G6 single mode / failure modes / dealias", g6_single_mode_and_failure_modes()))
    results.append(("G7 Newton Jacobian nonsingular", g7_newton_jacobian()))
    print("\n" + "=" * 108)
    allok = True
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
        allok &= bool(ok)
    print(f"\n  BEST MEASURED per-s relative error (recommended construction): {best:.3e}")
    print(f"  OVERALL: {'PASS' if allok else 'FAIL'}")
    print("=" * 108)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
