"""Essential numerical range of the CORNER-REGULARIZED generator, measured.

The campaign's standing statement 'max Re W_e = 0' is an ASSERTION, not a
measurement: it reads off polar_zeros.py's by-hand far-field limit, which says the
undivided symbol tends to p_inf(k) = -i c_l k once c_w - a0 c_l = 0 (measured
-2.47e-12).  Two things are wrong with using that as the pollution bound here.

  (a) It is the symbol of the UNDIVIDED operator.  We discretize in the divided
      variables A = Ot/xi, B = Bt/xi^2, P = Pt/xi^2.  The substitution is a
      SIMILARITY (S_r = S_u^-1 exactly), so it moves no spectrum -- but the
      numerical range is not similarity invariant, and the essential numerical
      range is the object the pollution theorem is stated in.  By hand the
      divided diagonal symbol is -i c_l k - m c_l / xi (m = 1 on the A row, 2 on
      the B row): a strictly NEGATIVE real part of order c_l/XMAX ~ 0.12, absent
      from the undivided limit.
  (b) polar_zeros.essential_numerical_range needs asymptotically constant
      coefficients on an infinite domain (Boegli-Marletta-Tretter Prop 7.2).  Our
      domain is FINITE, xi in [0, 25].  On a finite domain with smooth
      coefficients the singular sequences are the high-frequency ones, so the
      right object is

          W_e = conv  U_{xi*}  lim_{k -> inf} W( S(xi*, k) )

      -- freeze at EVERY radial node, not only the outermost, and take k large.

This module builds S(xi*, k) for the corner-regularized system term by term from
residual()/jacobian(), eliminates the slaved Poisson block at frozen xi (Dirichlet
beta edges, as _fix_p_rows installs them), and reports max Re W(S) = lambda_max of
the Hermitian part.  A region with max Re < 0 means NO point of the open right
half plane is reachable as spectral pollution, which is what makes any RHP
eigenvalue believable.
"""
import numpy as np
import scipy.linalg as sla


def symbol(S, z, i, k):
    """2 Nb x 2 Nb symbol of the generator, coefficients frozen at radial node i,
    d_xi -> i k, Poisson block eliminated at frozen xi."""
    A, B, Pf, cl, cw = S.unpack(z)
    Nb = S.Nb
    a0, mu = S.a0, S.mu
    xi = float(S.x[i])
    G1 = float(S.G1c[i])
    E1 = float(np.exp(a0 * xi) / G1)
    Db = np.asarray(S.Db, dtype=complex)
    Db2 = np.asarray(S.Db2, dtype=complex)
    I = np.eye(Nb, dtype=complex)
    dg = lambda v: np.diag(np.asarray(v, dtype=complex))
    ik = 1j * float(k)

    # first-order bundles with d_xi -> ik  (scalars, they multiply the identity)
    la = 1.0 + xi * (ik + a0)
    lb = 2.0 + xi * (ik + 1.0 + 2.0 * a0)
    lp = 2.0 + xi * (ik + mu)

    # base-state row data, exactly the quantities residual()/jacobian() form
    dx = lambda F: (S.Dx @ F)
    db = lambda F: F @ S.Db.T
    A_b, B_b, P_b = db(A)[i], db(B)[i], db(Pf)[i]
    LAa = (A + S.XI * (dx(A) + a0 * A))[i]
    LB2b = (2.0 * B + S.XI * (dx(B) + (1.0 + 2.0 * a0) * B))[i]
    LPp = (2.0 * Pf + S.XI * (dx(Pf) + mu * Pf))[i]
    cosb, sinb = np.asarray(S.cosb).ravel(), np.asarray(S.sinb).ravel()

    J_AA = E1 * (-(dg(LPp) @ Db) + dg(P_b) * la) + cl * (-(G1 * la)) * I + cw * I
    J_AB = E1 * (G1 * dg(cosb) * lb - dg(sinb) @ Db)
    J_AP = E1 * (-(dg(A_b)) * lp + dg(LAa) @ Db)
    J_BB = -E1 * (dg(LPp) @ Db - dg(P_b) * lb) + cl * (I - G1 * lb * I) + 2.0 * cw * I
    J_BP = -E1 * (dg(B_b) * lp - dg(LB2b) @ Db)
    J_PA = (xi * G1 ** 2) * I
    J_PP = (G1 ** 2 * (xi ** 2 * ik ** 2 + (4 * xi + 2 * mu * xi ** 2) * ik
                       + (2 + 4 * mu * xi + mu * mu * xi ** 2)) * I
            + G1 * (1.0 - xi * G1) * lp * I + Db2)
    for j in (0, Nb - 1):                       # Dirichlet beta edges
        J_PP[j, :] = 0.0
        J_PP[j, j] = 1.0
        J_PA[j, :] = 0.0
    Pmap = -sla.solve(J_PP, J_PA)               # dP = Pmap dA
    return np.block([[J_AA + J_AP @ Pmap, J_AB],
                     [J_BP @ Pmap, J_BB]])


def max_re(S, z, i, k):
    Sk = symbol(S, z, i, k)
    H = 0.5 * (Sk + Sk.conj().T)
    w = sla.eigvalsh(H)
    return float(w[-1]), float(w[0])


def support(S, z, nodes, ks, thetas=48):
    """h(theta) = max over (node, k) of lambda_max Herm(e^{i theta} S).  Returns
    (theta, h); h[0] is max Re W_e."""
    th = np.linspace(0.0, 2.0 * np.pi, thetas, endpoint=False)
    h = np.full(th.size, -np.inf)
    for i in nodes:
        for k in ks:
            Sk = symbol(S, z, i, k)
            for t, ang in enumerate(th):
                R = np.exp(1j * ang) * Sk
                H = 0.5 * (R + R.conj().T)
                h[t] = max(h[t], float(sla.eigvalsh(H)[-1]))
    return th, h
