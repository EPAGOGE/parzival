"""ADJUDICATION A -- the pencil's spectral bookkeeping, decided by dense QZ.

D1 says: E-nullity = 2910 = P 2556 + c 2 + pins 212 + C0 140.
D2 says: infinite eigenvalues = 2912 = 212 + 140 + 2556 + 4, the extra 4 being two
         size-2 Jordan blocks at infinity from the index-2 gauge constraints, so the
         finite count is dim ker(Cg) = n_f - 2.

These are different claims about deg det(zE - J).  Decide by QZ on a config small
enough to diagonalize whole, and cross-check three ways:
  (i)  finite eigenvalue count of the pencil (E, J)          [ the pencil ]
  (ii) eigenvalues of the dense reduced generator on ker(Cg) [ the realization ]
  (iii) argument-principle winding count on a contour        [ the production method ]
(iii) is the method that must scale to production size; validating it against (i) here
is the only place it can be validated.

STRUCTURE ONLY.  This config is at the SEED, not a converged root: eigenvalue
LOCATIONS here are meaningless.  Counts, ranks and method agreement are not.
"""
import importlib.util, pathlib, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla

H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
spec = importlib.util.spec_from_file_location("pc", str(H / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(spec); sys.modules["pc"] = pc; spec.loader.exec_module(pc)


def build(S, z):
    Nx, Nb = S.Nx, S.Nb; n2 = Nx * Nb
    J = S.jacobian(z).tocsr()
    allr = np.arange(n2)
    liveT = np.setdiff1d(allr, np.union1d(S.rT_pin, S.rT_c0))
    free_rows = np.concatenate([liveT, n2 + liveT])
    part_l = np.array(sorted(S.rT_c0), dtype=int)
    part_r = np.array([S.partner[int(r)] for r in part_l], dtype=int)
    spec_p = np.concatenate([S.rP_bedge, S.rP_outer, S.rP_c0, S.rP_c1, S.rP_cornerI])
    coefP = np.broadcast_to(-(S.XI * S.G1 ** 2), (Nx, Nb)).ravel().copy()
    coefP[spec_p] = 0.0
    return dict(J=J, n2=n2, liveT=liveT, free_rows=free_rows, part_l=part_l,
                part_r=part_r, coefP=coefP, Nx=Nx, Nb=Nb)


def dense_L(S, K):
    J, n2, fr = K["J"], K["n2"], K["free_rows"]
    Jd = np.asarray(J.todense()); nf = fr.size
    Lp = Jd[2 * n2:3 * n2, 2 * n2:3 * n2]
    Pmap = np.linalg.solve(Lp, np.diag(K["coefP"]))
    Z = np.zeros((J.shape[1], nf)); Z[fr, np.arange(nf)] = 1.0
    Z[K["part_l"], :] = Z[K["part_r"], :]
    Z[n2 + K["part_l"], :] = Z[n2 + K["part_r"], :]
    Z[2 * n2:3 * n2, :] = Pmap @ Z[:n2, :]
    M = Jd[np.ix_(fr, np.arange(J.shape[1]))] @ Z
    Bc = Jd[np.ix_(fr, [J.shape[1] - 2, J.shape[1] - 1])]
    Cg = Jd[np.ix_([J.shape[0] - 2, J.shape[0] - 1], fr)]
    L = M - Bc @ np.linalg.solve(Cg @ Bc, Cg @ M)
    return L, Bc, Cg, M


S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(6, 10, 5), Nb=10,
                       eps_b=1e-3, alpha=-0.3447)
z0 = S.pack(S.A0, S.B0, S.P0, S.P["cl"], S.P["cw"])
K = build(S, z0)
J = K["J"]; n2 = K["n2"]; fr = K["free_rows"]; nf = fr.size; N = J.shape[0]
mask = np.zeros(N); mask[K["liveT"]] = 1.0; mask[n2 + K["liveT"]] = 1.0
E = np.diag(mask)
Jd = np.asarray(J.todense())

print(f"CONFIG (6,10,5)/Nb10   N={N}  n2={n2}  n_f={nf}  rank(E)={int(mask.sum())}")
print(f"  row ledger: pins 2x{S.rT_pin.size}={2*S.rT_pin.size}   "
      f"C0 2x{S.rT_c0.size}={2*S.rT_c0.size}   P {n2}   gauge 2   c-cols 2")
print(f"  N - rank(E) = {N - int(mask.sum())}   (D1's E-nullity reading)")

# ---------------------------------------------------------------- (i) QZ
t0 = time.time()
AA, BB, aa, bb, Q, Zq = sla.ordqz(Jd, E, output="complex")
tqz = time.time() - t0
scale = max(abs(aa).max(), 1.0)
finite = np.abs(bb) > 1e-12 * scale          # beta != 0  <=>  finite eigenvalue
nfin = int(finite.sum())
lam = aa[finite] / bb[finite]
print(f"\n(i)  QZ on (E,J):  finite eigenvalues = {nfin}   infinite = {N - nfin}"
      f"    [{tqz:.1f}s]")
print(f"     D1 predicts finite = {int(mask.sum())} (= rank E);  "
      f"D2 predicts finite = {nf - 2} (= dim ker Cg)")
print(f"     ==> {'D2' if nfin == nf - 2 else ('D1' if nfin == int(mask.sum()) else 'NEITHER')}")
# how cleanly separated are the two groups?
srt = np.sort(np.abs(bb))
print(f"     |beta| separation: largest 'zero' = {srt[N-nfin-1]:.3e}   "
      f"smallest 'nonzero' = {srt[N-nfin]:.3e}   ratio = {srt[N-nfin]/max(srt[N-nfin-1],1e-300):.2e}")

# ---------------------------------------------------------------- (ii) realization
Ld, Bc, Cg, M = dense_L(S, K)
Z0 = np.linalg.svd(Cg)[2][2:].T.conj()          # orthonormal basis of ker(Cg)
Lred = Z0.T.conj() @ Ld @ Z0
ev_red = np.linalg.eigvals(Lred)
print(f"\n(ii) dense realization: L on ker(Cg) is {Lred.shape[0]}x{Lred.shape[1]}, "
      f"{Lred.shape[0]} eigenvalues")
# match the two spectra
a1 = np.sort_complex(np.round(lam, 8)); a2 = np.sort_complex(np.round(ev_red, 8))
if len(a1) == len(a2):
    # greedy nearest match
    used = np.zeros(len(a2), bool); worst = 0.0
    for v in lam:
        d = np.abs(ev_red - v) + 1e30 * used
        k = int(np.argmin(d)); used[k] = True
        worst = max(worst, float(d[k]) / max(abs(v), 1.0))
    print(f"     pencil-vs-realization spectra: worst relative mismatch = {worst:.3e}")
else:
    print(f"     COUNT MISMATCH: pencil {len(a1)} vs realization {len(a2)}")

npos = int((lam.real > 0).sum())
print(f"\n     eigenvalues with Re > 0: {npos} of {nfin}   "
      f"(max Re = {lam.real.max():+.4e}, min Re = {lam.real.min():+.4e})")
print(f"     |lam| range: {np.abs(lam).min():.3e} .. {np.abs(lam).max():.3e}")

# ---------------------------------------------------------------- (iii) argument principle
# N_enclosed = (1/2pi i) oint tr((zE-J)^-1 E) dz
def count_contour(R=None, ctr=None, npts=180, exact=True, probes=40, seed=1):
    """Winding count on a circle; 'exact' uses a dense trace, else Hutchinson."""
    rng = np.random.default_rng(seed)
    Om = rng.choice([-1.0, 1.0], size=(N, probes)) if not exact else None
    tot = 0.0 + 0j
    for k in range(npts):
        th = 2 * np.pi * (k + 0.5) / npts
        z = ctr + R * np.exp(1j * th)
        dz = 1j * R * np.exp(1j * th) * (2 * np.pi / npts)
        Kz = z * E - Jd
        if exact:
            tr = np.trace(np.linalg.solve(Kz, E))
        else:
            Y = np.linalg.solve(Kz, E @ Om)
            tr = np.einsum("ij,ij->", Om, Y) / probes
        tot += tr * dz
    return tot / (2j * np.pi)

for (ctr, R) in ((0.0, 1e6), (0.0, 1.0), (5.0, 4.99)):
    c_ex = count_contour(R=R, ctr=ctr, exact=True)
    c_hu = count_contour(R=R, ctr=ctr, exact=False)
    inside = int(((lam.real - ctr) ** 2 + lam.imag ** 2 < R ** 2).sum())
    print(f"\n(iii) contour |z-{ctr}|={R:g}:  exact trace = {c_ex.real:+.4f}"
          f"{c_ex.imag:+.2e}i   Hutchinson(40) = {c_hu.real:+.3f}"
          f"   direct count = {inside}")
