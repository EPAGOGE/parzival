"""Cross-examination of the is_alpha1 advocate: residual forensics on the deg24_56
file. The run log (branch1_res.log) printed ||F||_rms=5.97e-12 for (24,56,12); the
advocate's fresh-solver re-eval got ||F||_2=3.0e-4 and called the file 'an
unconverged mid-secant intermediate'. Those cannot both describe the same vector
under the same operator. Resolve by measurement:
  A. forensic: |a - cw/cl| per file (secant break state)
  B. fresh-solver residual (advocate's path) with ROW-BLOCK localization
  C. run-replica residual: construct at the run's construction alpha
     (-0.42172919), set_alpha(stored a)  [replicates stale-pin state]
  D. pin-value sensitivity: ||A0(a1)-A0(a2)|| on pinned rows for da=3.8e-3
  E. grid-node vacuity: corner-panel nodes vs 'features sit on nodes' claim
Measurement only: residual evaluations, no Newton.
"""
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

print("== A. forensic: stored a vs stored cw/cl ==")
files = ["branch1_deg16_40.npz", "branch1_deg24_56.npz", "branch1_eps1e-4.npz",
         "branch1_eps1e-05.npz", "find_half.npz", "rung_00_a-0.344712.npz"]
data = {}
for nm in files:
    d = load(nm)
    data[nm] = d
    z = d["z"]
    cl, cw = float(z[-2]), float(z[-1])
    a = float(d["a"]) if "a" in d else np.nan
    print(f"  {nm:28s} a={a:+.9f} cw/cl={cw/cl:+.9f} |a-cw/cl|={abs(a-cw/cl):.2e} len(z)={len(z)}")

def blocknorms(S, F, label):
    n2 = S.Nx * S.Nb
    ro, rb, rp = F[:n2], F[n2:2*n2], F[2*n2:3*n2]
    g = F[-2:]
    pin = np.zeros(n2, dtype=bool); pin[S.rT_pin] = True
    c0 = np.zeros(n2, dtype=bool); c0[S.rT_c0] = True
    interior = ~(pin | c0)
    rp_special = np.zeros(n2, dtype=bool)
    for arr in (S.rP_bedge, S.rP_outer, S.rP_c0, S.rP_c1, S.rP_cornerI):
        rp_special[arr] = True
    rms = np.linalg.norm(F)/np.sqrt(F.size)
    print(f"  {label}: ||F||_2={np.linalg.norm(F):.3e} rms={rms:.3e} inf={np.max(np.abs(F)):.3e}")
    print(f"     RO: pin-rows inf={np.max(np.abs(ro[pin])):.2e}  c0 inf={np.max(np.abs(ro[c0])):.2e}"
          f"  interior inf={np.max(np.abs(ro[interior])):.2e}")
    print(f"     RB: pin-rows inf={np.max(np.abs(rb[pin])):.2e}  interior inf={np.max(np.abs(rb[interior])):.2e}")
    print(f"     RP: special inf={np.max(np.abs(rp[rp_special])):.2e}  interior inf={np.max(np.abs(rp[~rp_special])):.2e}")
    print(f"     gauge g1={g[0]:+.2e} g2={g[1]:+.2e}")
    return rms

print("\n== B. fresh-solver residual (advocate's path), deg24_56, eps=1e-5 ==")
d = data["branch1_deg24_56.npz"]
z24, a24 = d["z"], float(d["a"])
Sf = pc.CornerRegSolver(edges=EDGES, degs=(24,56,12), Nb=36, eps_b=1e-5, alpha=a24)
Ff = Sf.residual(z24)
rms_fresh = blocknorms(Sf, Ff, "FRESH(alpha=stored a)")

print("\n== C. run-replica residual: construct at -0.42172919, set_alpha(stored a) ==")
Sr = pc.CornerRegSolver(edges=EDGES, degs=(24,56,12), Nb=36, eps_b=1e-5, alpha=-0.42172919)
Sr.set_alpha(a24)
Fr = Sr.residual(z24)
rms_replica = blocknorms(Sr, Fr, "REPLICA(init@-0.42172919, set_alpha)")

print("\n   also replicate the FULL run path: init@-0.42172919 -> set_alpha(A1) -> set_alpha(stored a):")
Sr2 = pc.CornerRegSolver(edges=EDGES, degs=(24,56,12), Nb=36, eps_b=1e-5, alpha=-0.42172919)
Sr2.set_alpha(-0.4168236)
Sr2.set_alpha(a24)
Fr2 = Sr2.residual(z24)
print(f"   full-path replica rms={np.linalg.norm(Fr2)/np.sqrt(Fr2.size):.3e} (should equal C if set_alpha is memoryless)")

print("\n== controls: fresh-eval on the OTHER files at their own configs ==")
d = data["branch1_deg16_40.npz"]
S = pc.CornerRegSolver(edges=EDGES, degs=(16,40,12), Nb=36, eps_b=1e-5, alpha=float(d["a"]))
F = S.residual(d["z"]); print(f"  deg16_40 fresh: rms={np.linalg.norm(F)/np.sqrt(F.size):.3e} L2={np.linalg.norm(F):.3e}")
d = data["branch1_eps1e-05.npz"]
S = pc.CornerRegSolver(edges=EDGES, degs=(16,40,12), Nb=36, eps_b=1e-5, alpha=float(d["a"]))
F = S.residual(d["z"]); print(f"  eps1e-05 fresh: rms={np.linalg.norm(F)/np.sqrt(F.size):.3e} L2={np.linalg.norm(F):.3e}")
# eps1e-4 file was solved with S constructed at alpha=A1 (branch1_polish stage 1)
d = data["branch1_eps1e-4.npz"]
S = pc.CornerRegSolver(edges=EDGES, degs=(16,40,12), Nb=36, eps_b=1e-4, alpha=float(d["a"]))
F = S.residual(d["z"]); print(f"  eps1e-4  fresh(alpha=stored): rms={np.linalg.norm(F)/np.sqrt(F.size):.3e}")
S = pc.CornerRegSolver(edges=EDGES, degs=(16,40,12), Nb=36, eps_b=1e-4, alpha=-0.4168236)
S.set_alpha(float(d["a"]))
F = S.residual(d["z"]); print(f"  eps1e-4  replica(init@A1,set_alpha): rms={np.linalg.norm(F)/np.sqrt(F.size):.3e}")

print("\n== D. pin-value alpha-sensitivity at (24,56,12)/eps1e-5 ==")
Sa = Sf   # constructed at stored a
Sb = pc.CornerRegSolver(edges=EDGES, degs=(24,56,12), Nb=36, eps_b=1e-5, alpha=-0.42172919)
dA0 = np.abs(Sa.A0.ravel()[Sa.rT_pin] - Sb.A0.ravel()[Sb.rT_pin])
dB0 = np.abs(Sa.B0.ravel()[Sa.rT_pin] - Sb.B0.ravel()[Sb.rT_pin])
print(f"  pinned-row seed shift for da=3.82e-3: max|dA0|={dA0.max():.2e}  max|dB0|={dB0.max():.2e}")
print(f"  (if C ~ run's 5.97e-12*sqrt(N)~6e-10 and B >> C, the fresh-vs-run gap is init-state, size above)")

print("\n== E. grid-node vacuity: corner-panel nodes at (16,40,12) ==")
S16 = pc.CornerRegSolver(edges=EDGES, degs=(16,40,12), Nb=36, eps_b=1e-4, alpha=-0.4168236)
nodes = S16.x[:S16.sizes[0]]
print(f"  corner nodes: {np.round(nodes, 4)}")
print(f"  node spacing around xi=1.4: {np.round(np.diff(nodes)[8:12], 4)}")
print(f"  -> any feature READ OFF this grid lies on a node by construction;")
print(f"     'features sit on nodes to <5e-4' is circular unless located by interpolation")
