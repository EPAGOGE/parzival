#!/usr/bin/env python3
"""Ghost-stance measurements: spectral-tail under-resolution diagnostic +
refinement-motion audit + EJA minting. Seconds of compute, no solves."""
import importlib.util, sys, json
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad"
sys.path.insert(0, SCRATCH)

spec = importlib.util.spec_from_file_location(
    'pc', '/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py')
pc = importlib.util.module_from_spec(spec); sys.modules['pc'] = pc
spec.loader.exec_module(pc)

EDGES = (0.0, 2.0, 15.0, 25.0)
NB = 36

def load(fname, degs):
    d = np.load(f"{SCRATCH}/hunt_fields/{fname}")
    z = d['z']; a = float(d['a'])
    nx = sum(dg + 1 for dg in degs)
    n = nx * NB
    A = z[0:n].reshape(nx, NB)
    B = z[n:2*n].reshape(nx, NB)
    P = z[2*n:3*n].reshape(nx, NB)
    cl, cw = float(z[3*n]), float(z[3*n+1])
    return dict(A=A, B=B, P=P, cl=cl, cw=cw, a=a, nx=nx, degs=degs)

def panels(degs):
    """index slices of the duplicated-interface panel layout"""
    out, i0 = [], 0
    for dg in degs:
        out.append(slice(i0, i0 + dg + 1))
        i0 += dg + 1
    return out

def cheb_tail(field, sl, frac=0.2):
    """RMS-over-beta Chebyshev mode energies on one radial panel; tail fraction."""
    n = sl.stop - sl.start
    xj = np.cos(np.pi * np.arange(n) / (n - 1))       # CGL nodes on [-1,1]
    C = np.polynomial.chebyshev.chebfit(xj, field[sl, :], n - 1)  # (n, NB)
    e = np.sqrt(np.mean(C**2, axis=1))                # per-mode energy
    ktail = max(2, int(round(frac * n)))
    return float(np.sum(e[-ktail:]) / np.sum(e)), e

def beta_tail(field, rows, frac=0.2):
    """Chebyshev tail in the beta direction, averaged over given radial rows."""
    nb = field.shape[1]
    xj = np.cos(np.pi * np.arange(nb) / (nb - 1))
    C = np.polynomial.chebyshev.chebfit(xj, field[rows, :].T, nb - 1)  # (nb, len(rows))
    e = np.sqrt(np.mean(C**2, axis=1))
    ktail = max(2, int(round(frac * nb)))
    return float(np.sum(e[-ktail:]) / np.sum(e))

D16 = (16, 40, 12)
D24 = (24, 56, 12)

ground   = load('rung_00_a-0.344712.npz', D16)   # ground at its own alpha
walked   = load('rung_10_a-0.422247.npz', D16)   # ground branch walked to a near candidate
cand     = load('find_half.npz', D16)            # THE candidate at frozen a=alpha_1
cand_pol = load('branch1_eps1e-4.npz', D16)      # secant-polished branch root
cand16   = load('branch1_deg16_40.npz', D16)
cand24   = load('branch1_deg24_56.npz', D24)

print("=== cw/cl and refinement motion ===")
for tag, r in [('ground', ground), ('walked_rung10', walked), ('find_half', cand),
               ('branch1_eps1e-4', cand_pol), ('branch1_deg16_40', cand16),
               ('branch1_deg24_56', cand24)]:
    print(f"{tag:18s} a={r['a']:+.7f} cl={r['cl']:+.6f} cw={r['cw']:+.6f} "
          f"cw/cl={r['cw']/r['cl']:+.9f} c=(cl-2cw)/4={(r['cl']-2*r['cw'])/4:+.6f}")

m_branch = abs(cand24['cw']/cand24['cl'] - cand16['cw']/cand16['cl'])
print(f"\nbranch deg-refinement motion |d(cw/cl)| 16->24: {m_branch:.6e}"
      f"  (ground deg0 16->24 sensitivity: 7e-4; ratio {m_branch/7e-4:.2f}x)")
c16 = (cand16['cl']-2*cand16['cw'])/4; c24 = (cand24['cl']-2*cand24['cw'])/4
print(f"branch corner amplitude c under refinement: {c16:+.6f} -> {c24:+.6f} "
      f"({100*(c24-c16)/c16:+.1f}%)")

print("\n=== radial Chebyshev tail fractions (last 20% of modes, RMS over beta) ===")
names = ['corner(0-2)', 'mid(2-15)', 'outer(15-25)']
tails = {}
for tag, r in [('ground', ground), ('walked_rung10', walked), ('find_half', cand),
               ('branch1_deg16_40', cand16), ('branch1_deg24_56', cand24)]:
    sls = panels(r['degs'])
    row = {}
    for fld in ('A', 'B', 'P'):
        row[fld] = [cheb_tail(r[fld], sl)[0] for sl in sls]
    tails[tag] = row
    print(tag)
    for fld in ('A', 'B', 'P'):
        print("   " + fld + "  " + "  ".join(f"{nm}={v:.3e}" for nm, v in zip(names, row[fld])))

print("\n=== beta-direction Chebyshev tail (corner-panel rows 1..16, last 20% modes) ===")
for tag, r in [('ground', ground), ('find_half', cand), ('branch1_deg16_40', cand16),
               ('branch1_deg24_56', cand24)]:
    rows = np.arange(1, min(17, r['nx']))
    bt = {f: beta_tail(r[f], rows) for f in ('A', 'B', 'P')}
    print(f"{tag:18s} " + "  ".join(f"{f}={v:.3e}" for f, v in bt.items()))

# ---- ratios that carry the stance ----
gA = tails['ground']['A'][0]; cA = tails['find_half']['A'][0]
gB = tails['ground']['B'][0]; cB = tails['find_half']['B'][0]
print(f"\ncorner-panel tail ratio candidate/ground: A {cA/gA:.1f}x  B {cB/gB:.1f}x")
cA24 = tails['branch1_deg24_56']['A'][0]; cB24 = tails['branch1_deg24_56']['B'][0]
print(f"candidate corner tail under refinement: A {tails['branch1_deg16_40']['A'][0]:.3e} -> {cA24:.3e}"
      f"   B {tails['branch1_deg16_40']['B'][0]:.3e} -> {cB24:.3e}")

# ================= EJA =================
from eja_bridge import (mk_witness, refuse, conditional_axiom, shared_constant_audit)

print("\n=== EJA objects ===")
# W1: refinement-motion witness. Prediction A (real converged root, ground-transfer
# sensitivity): |d(cw/cl)| under deg 16->24 ~ 7e-4. Prediction B: measured branch motion.
w1 = mk_witness("real_root_ground_transfer_pred_deg_motion", 7e-4,
                "measured_branch_deg_motion", m_branch,
                {"axis": "deg0/degmid 16/40 -> 24/56", "eps": "1e-4(assumed)"},
                ("ghost", "refinement"))
print("W1 refinement-motion witness divergence:", w1.divergence)

# W2: spectral-tail witness at the SAME grid (corner panel, field B).
w2 = mk_witness("ground_corner_B_tail", gB, "candidate_corner_B_tail", cB,
                {"grid": "(16,40,12)/Nb36/eps1e-4", "panel": "xi<2", "modes": "last 20%"},
                ("ghost", "resolution"))
print("W2 corner-B spectral-tail witness divergence:", w2.divergence)

# R1: negative control -- residual smallness has ZERO discriminating power.
r1 = refuse("smallF_implies_continuum_root", 0.0, 1.0,
            "||F||=5.4e-14 is a DISCRETE residual; ghost and real hypotheses both "
            "predict machine-zero -- distance 0 at scale 1, no discriminating power")
print("R1:", json.dumps(r1))

# R2: honesty control -- the dry cold re-hunt alone cannot convict.
r2 = refuse("dry_rehunt_alone_proves_ghost", 1.0, 1.0,
            "(20,48,12) hunt was FROM-SCRATCH; a real root with the measured tiny "
            "basin (1-of-8 starts, structured starts dry at 1e-2..1e-3) also fails "
            "cold hunts -- indistinguishable without interpolation seeding")
print("R2:", json.dumps(r2))

# Shared-constant audit over every run that 'confirms' the candidate root.
audit = shared_constant_audit([
    dict(run="find_half", deg0=16, degmid=40, degc=12, Nb=36, edges="0,2,15,25",
         seed="chen-hou-interp-lineage", start="half-amplitude-deflation", eps=1e-4),
    dict(run="branch1_eps1e-4", deg0=16, degmid=40, degc=12, Nb=36, edges="0,2,15,25",
         seed="chen-hou-interp-lineage", start="half-amplitude-deflation", eps=1e-4),
    dict(run="branch1_eps5e-05", deg0=16, degmid=40, degc=12, Nb=36, edges="0,2,15,25",
         seed="chen-hou-interp-lineage", start="half-amplitude-deflation", eps=5e-5),
    dict(run="branch1_eps3e-05", deg0=16, degmid=40, degc=12, Nb=36, edges="0,2,15,25",
         seed="chen-hou-interp-lineage", start="half-amplitude-deflation", eps=3e-5),
    dict(run="branch1_eps1e-05", deg0=16, degmid=40, degc=12, Nb=36, edges="0,2,15,25",
         seed="chen-hou-interp-lineage", start="half-amplitude-deflation", eps=1e-5),
    dict(run="branch1_deg16_40", deg0=16, degmid=40, degc=12, Nb=36, edges="0,2,15,25",
         seed="chen-hou-interp-lineage", start="half-amplitude-deflation", eps=1e-4),
])
print("shared_constant_audit:", json.dumps(audit, default=str))

# Conditional axiom: the ghost claim, with its falsifier named at mint.
ax = conditional_axiom(
    "candidate_root_is_discretization_ghost",
    "the a=-0.4168236 deflated-multistart root is an artifact root of the "
    "(16,40,12)/Nb36 collocation system with no continuum limit: its cw/cl moved "
    f"{m_branch:.2e} (5.5x ground's full deg sensitivity) and its corner amplitude "
    f"c moved {100*(c24-c16)/c16:+.1f}% under the ONE refinement measured, its corner "
    "layer grows unboundedly (dA/dxi(0) rms 25.0->33.1), and it fits no continuum "
    "reference (18.1% into the a1->a2 gap, no family slot)",
    domain="grids (16,40,12) eps 1e-5..1e-4 + the single (24,56,12) refinement file; "
           "all runs Chen-Hou seed lineage, half-amplitude deflation start",
    residual="structured morphology unexplained by generic ghost-hood: corner algebra "
             "P(0,b)=c sin2b holds to 1e-3, eigenmode-like 2-node difference field, "
             "eps-flat cw/cl via clean scale-mode cancellation",
    falsifier="interpolation-seeded warm-start Newton at (20,48,12) and (24,56,12) "
              "converging with cw/cl motion CONTRACTING toward a limit (successive "
              "deg-ladder deltas shrinking below ground's 7e-4 scale) -- Cauchy "
              "behavior kills ghost-hood",
    evidence={"deg_motion": m_branch, "corner_c_change_frac": (c24-c16)/c16,
              "tail_ratio_B_corner": cB/gB, "dry_rehunt_residual": 3.5e-3})
print("AXIOM minted:", ax.name, "| falsifier:", ax.residual if not hasattr(ax, 'falsifier') else ax.falsifier)
