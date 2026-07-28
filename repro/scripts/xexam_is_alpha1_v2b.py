"""Cross-exam part 2: (a) is the converged deg24_56 root the SAME BRANCH as the
deg16 candidate (fingerprints: c3/c1 sign at xi=1, near-wall dip below pinned
corner profile, outward-displaced A peak, B amplitude)? (b) re-verify the
advocate's 203x corner-panel spectral-tail claim. (c) mint EJA objects.
Measurement only."""
import importlib.util, sys
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad"
sys.path.insert(0, SCRATCH)
spec = importlib.util.spec_from_file_location(
    "pc", "/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py")
pc = importlib.util.module_from_spec(spec)
sys.modules["pc"] = pc
spec.loader.exec_module(pc)
HF = SCRATCH + "/hunt_fields/"
EDGES = (0.0, 2.0, 15.0, 25.0)

def load(name):
    d = np.load(HF + name)
    return {k: d[k] for k in d.files}

def fingerprints(z, S, label):
    n2 = S.Nx * S.Nb
    A = z[:n2].reshape(S.Nx, S.Nb)
    B = z[n2:2*n2].reshape(S.Nx, S.Nb)
    cl, cw = float(z[-2]), float(z[-1])
    x, b = S.x, S.b
    # c3/c1 of A at xi=1.0 (node at panel center for even corner deg)
    i1 = int(np.argmin(np.abs(x - 1.0)))
    M = np.stack([np.cos((2*k+1)*b) for k in range(5)], axis=1)
    c, *_ = np.linalg.lstsq(M, A[i1], rcond=None)
    # near-wall radial profile at j=0
    prof = A[:, 0]
    s0 = S.sizes[0]
    corner = slice(0, s0)
    dipk = int(np.argmin(prof[corner][1:]) + 1)
    pinned0 = A[0, 0]   # = WX cos(eps_b) ~ 1.196
    peakk = int(np.argmax(prof))
    print(f"  {label}:")
    print(f"    xi(node)={x[i1]:.4f}  c3/c1={c[1]/c[0]:+.6f}  (ground ref -0.1805, cand deg16 +0.0940)")
    print(f"    near-wall A: pinned A(0,0)={pinned0:.4f}  dip={prof[corner][dipk]:.4f}@xi={x[corner][dipk]:.4f}"
          f"  (below pin: {prof[corner][dipk] < pinned0})")
    print(f"    A peak {prof[peakk]:.4f}@xi={x[peakk]:.4f}   max|B|={np.max(np.abs(B)):.4f}"
          f"   cl={cl:+.5f} cw={cw:+.5f} corner c=(cl-2cw)/4={0.25*(cl-2*cw):+.5f}")
    return c[1]/c[0], x[peakk], prof[corner][dipk]

print("== same-branch fingerprint check: deg16 candidate vs deg24 re-hunt root ==")
d16 = load("branch1_deg16_40.npz")
S16 = pc.CornerRegSolver(edges=EDGES, degs=(16,40,12), Nb=36, eps_b=1e-5, alpha=float(d16["a"]))
f16 = fingerprints(d16["z"], S16, "deg16_40 (eps1e-5, a=-0.42172919)")
d24 = load("branch1_deg24_56.npz")
S24 = pc.CornerRegSolver(edges=EDGES, degs=(24,56,12), Nb=36, eps_b=1e-5, alpha=float(d24["a"]))
f24 = fingerprints(d24["z"], S24, "deg24_56 (eps1e-5, a=-0.42554621)")
g = load("rung_00_a-0.344712.npz")
Sg = pc.CornerRegSolver(edges=EDGES, degs=(16,40,12), Nb=36, eps_b=1e-4, alpha=float(g["a"]))
fg = fingerprints(g["z"], Sg, "GROUND rung_00 (eps1e-4, a=-0.34471229)")

print("\n== re-verify corner-panel spectral tails (advocate's 203x claim) ==")
def tailA(z, S):
    n2 = S.Nx * S.Nb
    A = z[:n2].reshape(S.Nx, S.Nb)
    ncp = S.sizes[0]
    tt = S.x[:ncp] - 1.0     # [0,2] -> [-1,1]
    fr = []
    for j in range(S.Nb):
        c = np.polynomial.chebyshev.chebfit(tt, A[:ncp, j], ncp - 1)
        fr.append(np.max(np.abs(c[-3:])) / np.max(np.abs(c)))
    fr = np.array(fr)
    return float(np.sqrt(np.mean(fr**2)))
d1e4 = load("branch1_eps1e-4.npz")
S164 = pc.CornerRegSolver(edges=EDGES, degs=(16,40,12), Nb=36, eps_b=1e-4, alpha=float(d1e4["a"]))
tg = tailA(g["z"], Sg)
tc = tailA(d1e4["z"], S164)
t24 = tailA(d24["z"], S24)
print(f"  A corner-panel tail rms: ground {tg:.3e}  candidate deg16 {tc:.3e} ({tc/tg:.0f}x)"
      f"  candidate deg24 {t24:.3e}")

print("\n== EJA ledger ==")
from eja_bridge import mk_witness, refuse
w1 = mk_witness("advocate_fresh_eval_L2_deg24", 3.006e-4,
                "operator_matched_replica_L2_deg24", 6.079e-10,
                scenario={"file": "branch1_deg24_56.npz",
                          "resolution": "stale-pin systematic: set_alpha does not refresh A0/B0 pins;"
                                        " max pin shift 9.52e-5 == fresh ||F||_inf; interior PDE rows <=2.7e-10"},
                tags=("forensics", "residual"))
print(f"  W1 fresh vs replica residual: 3.006e-4 vs 6.079e-10, divergence={w1.divergence:.4f}")
r1 = refuse("deg24_56 is an unconverged mid-secant intermediate (advocate's voiding of the direction datum)",
            distance=3.006e-4, scale=6.079e-10,
            why="run-replica residual 6.0e-12 rms matches the log's 5.97e-12 to 3 digits; "
                "|a-cw/cl|=5.7e-10 shows the secant closed; the 3.0e-4 fresh-eval excess "
                "localizes 100% in RO pin rows and equals the 9.52e-5 stale-pin shift -- "
                "the file IS converged, the advocate's instrument was mismatched")
print(f"  REFUSAL r1: refused={r1['refused']} ratio={r1['ratio']:.2e}x (claim rested on an instrument artifact)")
w2 = mk_witness("branch_alpha_deg16", -0.42172919, "branch_alpha_deg24_converged", -0.42554621,
                scenario={"axis": "deg (16,40,12)->(24,56,12), both converged (rms 3.3e-12 / 6.0e-12)",
                          "direction": "AWAY from alpha_1=-0.4168236 by 3.82e-3",
                          "caveat": "deg24 pins stale at -0.42172919 (pin shift 9.5e-5, plausible alpha bias O(1e-4))"},
                tags=("resolution", "direction"))
print(f"  W2 converged deg-ladder step: -0.42172919 -> -0.42554621 (AWAY from alpha_1), divergence={w2.divergence:.4f}")
r2 = refuse("features-sit-on-grid-nodes as evidence of grid-scale representation",
            distance=0.0, scale=1.0,
            why="reported dip/peak coordinates (0.0761, 0.4444, 1.3827, 1.5556) ARE the corner-panel "
                "collocation nodes; nodal-argmax lands on a node by construction -- circular, zero information")
print(f"  REFUSAL r2: refused={r2['refused']} (vacuous evidence item)")
