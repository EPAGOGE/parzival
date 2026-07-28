#!/usr/bin/env python3
"""Cross-exam part 2: (a) alpha-scan the deg24_56 file -- is the FIELD a
converged root at (24,56,12) with the stored 'a' merely the next secant
abscissa?  (b) scale-mode decomposition of the deg16->deg24 (cl,cw) motion.
(c) EJA minting with the measured numbers."""
import importlib.util, sys, json
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad"
sys.path.insert(0, SCRATCH)
spec = importlib.util.spec_from_file_location(
    'pc', '/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py')
pc = importlib.util.module_from_spec(spec); sys.modules['pc'] = pc
spec.loader.exec_module(pc)

d24 = np.load(f"{SCRATCH}/hunt_fields/branch1_deg24_56.npz")
z24, a24 = d24['z'], float(d24['a'])
cl24, cw24 = float(z24[-2]), float(z24[-1])
print(f"deg24 stored a = {a24:+.9f}")
print(f"deg24 own cw/cl = {cw24/cl24:+.9f}  (diff from stored a: {cw24/cl24-a24:+.3e})")

S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(24, 56, 12),
                       Nb=36, eps_b=1e-5, alpha=a24)

def rms_at(a):
    S.set_alpha(a)
    f = S.residual(z24)
    return np.linalg.norm(f) / np.sqrt(f.size)

print("\nalpha scan (RMS residual of the SAVED field vs frozen alpha):")
offs = [0.0, 1e-6, -1e-6, 3e-6, -3e-6, 1e-5, -1e-5, 3e-5, -3e-5, 1e-4, -1e-4]
vals = [(a24 + o, rms_at(a24 + o)) for o in offs]
for a, r in sorted(vals):
    print(f"  a={a:+.9f}  RMS={r:.3e}")
# refine around the min with a parabola on log-RMS then golden-ish bisection
best_a, best_r = min(vals, key=lambda t: t[1])
lo, hi = best_a - 5e-6, best_a + 5e-6
for _ in range(24):
    m1, m2 = lo + 0.382*(hi-lo), lo + 0.618*(hi-lo)
    if rms_at(m1) < rms_at(m2): hi = m2
    else: lo = m1
a_star = 0.5*(lo+hi); r_star = rms_at(a_star)
print(f"  minimizing alpha a* = {a_star:+.9f}, RMS(a*) = {r_star:.3e}")
print(f"  a* - stored_a = {a_star-a24:+.3e};  a* - own cw/cl = {a_star-cw24/cl24:+.3e}")
print(f"  self-consistency defect h = cw/cl - a* = {cw24/cl24-a_star:+.3e}")

d16 = np.load(f"{SCRATCH}/hunt_fields/branch1_deg16_40.npz")
z16 = d16['z']; cl16, cw16 = float(z16[-2]), float(z16[-1])
print("\n=== scale-mode decomposition of the deg16->deg24 (cl,cw) motion ===")
s = cl24 / cl16
print(f"  cl {cl16:+.6f} -> {cl24:+.6f}  scale s = {s:.6f}")
print(f"  cw {cw16:+.6f} -> {cw24:+.6f}; pure-scale prediction s*cw16 = {s*cw16:+.6f}")
print(f"  non-scale part of cw motion: {cw24 - s*cw16:+.6f} "
      f"(= {abs(cw24-s*cw16)/abs(cw24-cw16)*100:.1f}% of total cw motion)")
print(f"  cw/cl motion: {cw24/cl24 - cw16/cl16:+.6e} (scale mode cancels exactly)")
print(f"  c=(cl-2cw)/4: {(cl16-2*cw16)/4:+.6f} -> {(cl24-2*cw24)/4:+.6f} "
      f"({100*((cl24-2*cw24)-(cl16-2*cw16))/(cl16-2*cw16):+.1f}%) -- scales with s "
      f"(pure-scale would give {s*(cl16-2*cw16)/4:+.6f})")

# ============ EJA ============
from eja_bridge import mk_witness, refuse

print("\n=== EJA objects (cross-exam) ===")
w1 = mk_witness("advocate_deg24_maybe_unconverged_intermediate_RMS", 3.5e-3,
                "measured_min_RMS_of_saved_field_at_alpha_star", r_star,
                {"file": "branch1_deg24_56.npz", "grid": "(24,56,12)/Nb36/eps1e-5",
                 "note": "stored a is the NEXT secant abscissa; field converged at a*"},
                ("ghost", "cross-exam"))
print(f"W1 converged-or-not witness: dry-hunt-scale 3.5e-3 vs measured {r_star:.2e}; "
      f"divergence={w1.divergence}")

w2 = mk_witness("ghost_reading_c_collapse_is_drift", 0.255,
                "scale_mode_explained_fraction_of_cw_motion",
                float(1 - abs(cw24 - s*cw16)/abs(cw24 - cw16)),
                {"axis": "deg (16,40,12)->(24,56,12)", "s": float(s)},
                ("ghost", "cross-exam"))
print(f"W2 scale-mode witness: divergence={w2.divergence}")

r1 = refuse("A_tail_growth_proves_corner_nonconvergence", 0.14, 0.34,
            "only A's corner tail grows (+14%) under deg16->deg24; B decays -34% "
            "and P decays -37%, and B carries 62% of the branch-separation energy "
            "-- the majority-of-identity fields ARE spectrally converging; distance "
            "0.14 vs opposing-signal scale 0.34")
print("R1:", json.dumps(r1))

r2 = refuse("5p45x_deg_sensitivity_is_ghost_specific", 3.817e-3, 7e-4,
            "the 3.8e-3 cw/cl motion came from moving TWO axes at once (deg0 16->24 "
            "AND degmid 40->56) at eps=1e-5, vs ground's single-axis 7e-4 prior at "
            "eps=1e-4; an under-resolved REAL root (corner tail 100-800x ground) also "
            "predicts amplified deg sensitivity -- the ratio does not discriminate "
            "ghost from under-resolved-real")
print("R2:", json.dumps(r2))
