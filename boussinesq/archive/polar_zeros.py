"""SPECTRUM, DONE THE WAY THE STRUCTURE ASKS FOR.

Everything here replaces `la.eig(P @ A)` in polar_stability.spectrum, which has now
manufactured the same artifact three separate times (R(0) ~ 1.6e15, cond(L) ~ 1e16, an
inflated median kappa). The mathlit triangulation (refs/mathlit/, run wf_1a9bc154-3b6)
identified what that object actually is, and none of the three identifications was ours:

  1. L = [I - B (Cg B)^-1 Cg] A is the generator of a Hessenberg INDEX-2 DAE, and its
     spectrum is the set of INVARIANT (transmission) ZEROS of the triple (A, B, Cg),
     together with 0 at multiplicity m = 2.  Zeros are the finite generalized eigenvalues
     of the ROSENBROCK PENCIL

         [[A, B], [Cg, 0]]  -  lambda * diag(I, 0)

     and control theory computes them with a structure-respecting pencil algorithm, never
     with a general eigensolver on an explicitly-formed projection, precisely because
     zeros are far worse conditioned than poles.  (arXiv:2509.24105.)

  2. In ambient coordinates the pencil pseudospectrum is NOT A PROPERTY OF THE DYNAMICS:
     premultiplying by any invertible T leaves the DAE and its solutions untouched but
     changes {z : ||(zE-A)^-1|| > 1/eps} completely (Embree-Keeler 1601.00044 Fig 2).
     The invariant object lives in a UNITARY basis of the solution subspace.  So we
     compress onto Z = orthonormal basis of ker(Cg) and work there.  The two null
     directions are then gone BY CONSTRUCTION rather than by remembering to drop
     sigma[-1] and sigma[-2].

  3. The obliqueness, not the constraint, is what makes L non-normal.  Split

         M = Z^T A Z  -  (Z^T B) (Cg B)^-1 (Cg A Z)

     the first term is the plain ORTHOGONAL compression (which typically REDUCES
     departure from normality) and the second is a rank-2 oblique correction vanishing
     iff range(B) = range(Cg^T).  The number that governs it is not cond(Cg B) -- which
     is what we have been reporting -- but

         ||P|| = 1 / sin(theta_min(ker Cg, range B)) = ||I - P||   exactly

     (Kato; Xu-Zikatanov 1307.4393 Cor 4.2), from one SVD.

WHAT MAY BE BELIEVED, AND WHERE.  Boegli-Marletta-Tretter (1907.09599) Thm 7.1: spectral
pollution under domain truncation is CONFINED to the essential numerical range W_e, and
every isolated eigenvalue outside W_e is correctly approximated.  Thm 6.3 is the converse
and it is brutal: EVERY point of W_e is realisable as a spurious eigenvalue of a perfectly
legitimate Galerkin method.  Prop 7.2 makes W_e computable when the coefficients are
asymptotically constant -- and ours are, because g -> 1 and E = e^(a0 xi) -> 0 with
a0 < 0.  `essential_numerical_range` below builds the frozen far-field symbol from the
SAME closed-form linearisation A_exact uses and computes the numerical range of S(k) over
k, by supporting hyperplanes.  Anything inside the resulting region is unfalsifiable at
this domain size; anything outside is guaranteed to converge.

BY HAND, the limit is worth stating because it is a sharp prediction the code should
reproduce.  As xi -> infinity, E -> 0 kills every advection and source bracket, leaving

    rO -> -c_l dOt_xi + (c_w - c_l a0) dOt        rB -> -c_l dBt_xi + 2 (c_w - c_l a0) dBt

and at alpha self-consistency a0 = c_w/c_l EXACTLY, so both zeroth-order terms vanish and
the symbol is p_inf(k) = -i c_l k on both components.  W_e should therefore come out as
(a thin neighbourhood of) the IMAGINARY AXIS.  If it does, then none of the eigenvalues we
have been arguing about -- +6.87, +0.76, -0.35, +1.05, +0.43 +- 0.60i -- is pollution in
the Boegli-Marletta-Tretter sense, and that whole explanation is off the table.  The width
of the computed region measures how far from asymptotically-constant XMAX = 25 really is.
"""
import numpy as np
import scipy.linalg as la


# ---------------------------------------------------------------------------
# 1. slice basis and the compressed operator
# ---------------------------------------------------------------------------
def slice_basis(Cg):
    """Z : orthonormal basis of ker(Cg), shape (n, n-m). Via SVD, not QR of a null-space
    guess -- we want the numerically-orthonormal complement, and m is tiny."""
    U, s, Vt = la.svd(Cg, full_matrices=True)
    m = Cg.shape[0]
    return Vt[m:].T.conj()


def compress(A, B, Cg, Z=None):
    """M = Z^T A Z - (Z^T B)(Cg B)^-1 (Cg A Z), plus the orthogonal part on its own.

    Returns (M, M_orth, Z). M_orth = Z^T A Z is the constraint WITHOUT the obliqueness;
    comparing departure-from-normality of the two separates the two effects."""
    if Z is None:
        Z = slice_basis(Cg)
    AZ = A @ Z
    M_orth = Z.T.conj() @ AZ
    corr = (Z.T.conj() @ B) @ la.solve(Cg @ B, Cg @ AZ)
    return M_orth - corr, M_orth, Z


def proj_norm(B, Cg, Z=None):
    """||P|| = ||I - P|| = 1/sin(theta_min(ker Cg, range B)) for the oblique projector
    P = I - B(CgB)^-1 Cg.  This is the quantity that governs the obliqueness; cond(Cg B)
    is not.

    COMPUTED FROM THE SINES DIRECTLY, never as sqrt(1 - cos^2).  The earlier version took
    the largest singular value of Z^H orth(B) -- a cosine near 1 -- and formed
    sqrt(1 - c*c), which is catastrophic cancellation in exactly the regime this function
    exists to measure.  It returned 7.000057e-06 at N=28 where the stable route gives
    7.000002e-06, and its absolute error floor is ~1e-8 in the sine, so ANY ||P|| above
    ~1e8 from the old routine was noise.  The sines are the singular values of
    orth(Cg^H)^H orth(B) -- the projection of range(B) onto the ORTHOGONAL COMPLEMENT of
    ker(Cg) -- so no subtraction is involved and no clip is needed."""
    Qb = la.orth(B)
    Qc = la.orth(Cg.T.conj())                  # orthonormal basis of range(Cg^H) = ker(Cg)^perp
    sin_min = float(la.svdvals(Qc.T.conj() @ Qb).min())
    return (np.inf if sin_min == 0.0 else 1.0 / sin_min), sin_min


def departure_from_normality(M):
    """Henrici: sqrt(||M||_F^2 - sum |lambda_i|^2). Zero iff M is normal."""
    w = la.eigvals(M)
    v = la.norm(M, "fro") ** 2 - float(np.sum(np.abs(w) ** 2))
    return float(np.sqrt(max(v, 0.0)))


# ---------------------------------------------------------------------------
# 2. the zeros, by QZ on the Rosenbrock pencil
# ---------------------------------------------------------------------------
def rosenbrock_zeros(A, B, Cg, tol_inf=1e10):
    """Invariant zeros of (A, B, Cg) = finite generalized eigenvalues of

        [[A, B], [Cg, 0]] - lambda * diag(I, 0)

    QZ never forms (Cg B)^-1 and puts the m construction directions at INFINITY, where
    they belong, instead of at zero where they contaminate every statistic. Returns
    (finite_zeros_sorted, n_infinite, alpha, beta) with the raw QZ output kept so the
    infinite/finite split is auditable rather than asserted."""
    n, m = A.shape[0], Cg.shape[0]
    K = np.zeros((n + m, n + m))
    K[:n, :n], K[:n, n:], K[n:, :n] = A, B, Cg
    E = np.zeros((n + m, n + m))
    E[:n, :n] = np.eye(n)
    aa, bb, q, z = la.qz(K, E, output="complex")
    al, be = np.diag(aa), np.diag(bb)
    fin = np.abs(be) > np.abs(al) / tol_inf
    w = al[fin] / be[fin]
    return w[np.argsort(-w.real)], int((~fin).sum()), al, be


def nearest(w, target, exclude_zero=1e-8):
    """The eigenvalue closest to `target`, ignoring the construction zeros."""
    ww = w[np.abs(w) > exclude_zero]
    if ww.size == 0:
        return np.nan
    return ww[np.argmin(np.abs(ww - target))]


# ---------------------------------------------------------------------------
# 3. essential numerical range from the frozen far-field symbol
# ---------------------------------------------------------------------------
def far_field_symbol(St, x, k, i_star=None):
    """S(k): the 2 nb x 2 nb symbol of the linearised operator with coefficients frozen
    at radial node i_star (default: the outermost) and d_xi -> i k.

    Mirrors A_exact term for term. The only nonlocal piece is the Poisson solve, which at
    frozen xi becomes the nb x nb matrix

        dP = -[ (ik + mu)^2 g^2 I + g(1-g)(ik + mu) I + Db2 ]^-1 ( g^2 dOt )

    i.e. the same operator polar_corner._build_poisson assembles, with Dx -> ik I. The
    beta edges keep their Dirichlet rows, so the block is invertible."""
    C = St.C
    S = St.S
    nb = C.nb
    i = (C.nx - 1) if i_star is None else int(i_star)
    Ot, Bt = S.unpack(x[:-2])
    cl, cw = float(x[-2]), float(x[-1])
    a0, mu = C.a0, C.mu
    g = float(C.g[i])
    E = float(np.exp(a0 * C.x[i]))
    EG = E / g

    Pt = C.poisson(Ot)
    row = lambda F: np.asarray(F)[i, :]
    Pt_b = row(C.db(Pt))
    PmuP = row(C.dx(Pt) + mu * Pt)
    Ot_b, Bt_b = row(C.db(Ot)), row(C.db(Bt))
    OxaO = row(C.dx(Ot) + a0 * Ot)
    BxbB = row(C.dx(Bt) + (1.0 + 2.0 * a0) * Bt)

    I = np.eye(nb, dtype=complex)
    Db, Db2 = C.Db.astype(complex), C.Db2.astype(complex)
    dg = lambda v: np.diag(np.asarray(v, dtype=complex))
    ik = 1j * float(k)

    # Poisson block with Dirichlet beta edges (same rows _build_poisson pins)
    Lp = (g ** 2) * ((ik + mu) ** 2) * I + g * (1.0 - g) * (ik + mu) * I + Db2
    rhs = -(g ** 2) * I
    Lp, rhs = Lp.copy(), rhs.copy()
    for j in (0, nb - 1):
        Lp[j, :] = 0.0
        Lp[j, j] = 1.0
        rhs[j, :] = 0.0
    Pmap = la.solve(Lp, rhs)                      # dP = Pmap @ dOt
    dP_x = ik * Pmap
    dP_b = Db @ Pmap
    dP_mu = (ik + mu) * Pmap

    cb, sb = np.cos(C.b), np.sin(C.b)
    # dOt block
    dadvO_O = dg(Ot_b) @ dP_mu + dg(PmuP) @ Db - dg(OxaO) @ dP_b - dg(Pt_b) * (ik + a0)
    rO_O = EG * (-dadvO_O) + cl * (-g * (ik + a0)) * I + cw * I
    dsrcO_B = g * dg(cb) * (ik + 1.0 + 2.0 * a0) - dg(sb) @ Db
    rO_B = EG * dsrcO_B
    # dBt block
    dadvB_O = dg(Bt_b) @ dP_mu - dg(BxbB) @ dP_b
    rB_O = EG * (-dadvB_O)
    dadvB_B = dg(PmuP) @ Db - dg(Pt_b) * (ik + 1.0 + 2.0 * a0)
    rB_B = EG * (-dadvB_B) + cl * (-g * (ik + 1.0 + 2.0 * a0) + 1.0) * I + 2.0 * cw * I

    return np.block([[rO_O, rO_B], [rB_O, rB_B]])


def essential_numerical_range(St, x, ks=None, thetas=64, i_star=None):
    """W_e = conv{ numerical range of S(k) : k in R }, by supporting hyperplanes:

        h(theta) = max_k lambda_max( Herm( e^{i theta} S(k) ) )

    Returns (thetas, h, extent) where extent is the max real part of the region -- the
    single number that decides whether an observed eigenvalue with Re > 0 can be
    pollution. A region hugging the imaginary axis means it cannot."""
    if ks is None:
        ks = np.concatenate([-np.logspace(2, -2, 40), [0.0], np.logspace(-2, 2, 40)])
    th = np.linspace(0.0, 2.0 * np.pi, thetas, endpoint=False)
    h = np.full(th.size, -np.inf)
    for k in ks:
        Sk = far_field_symbol(St, x, k, i_star=i_star)
        for t, ang in enumerate(th):
            R = np.exp(1j * ang) * Sk
            H = 0.5 * (R + R.conj().T)
            h[t] = max(h[t], float(la.eigvalsh(H)[-1]))
    return th, h, float(h[0])                     # h[0] = max Re over the whole region


def support_contains(th, h, z, slack=0.0):
    """Is z inside conv{...} described by the support function h(theta)?"""
    return bool(np.all(np.real(np.exp(1j * th) * z) <= h + slack))
