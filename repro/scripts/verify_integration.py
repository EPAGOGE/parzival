"""Regression gate for the h_id integration: alpha MUST be unchanged; the membership
card must separate the ground root from the adjudicated ghost."""
import importlib.util, pathlib, sys
import numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
spec = importlib.util.spec_from_file_location("pc", str(H / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(spec); sys.modules["pc"] = pc; spec.loader.exec_module(pc)

print("[1] REGRESSION: converge (16,40,12)/Nb36/eps=1e-4 -- alpha must equal -0.34471229", flush=True)
S, z, r, info = pc.converge(edges=(0.0,2.0,15.0,25.0), degs=(16,40,12), Nb=36, eps_b=1e-4)
if not info.get("converged"):
    print(f"  NOT CONVERGED: {info}"); sys.exit(1)
a = info["alpha"]; d = a - (-0.34471229)
print(f"  alpha = {a:+.8f}   delta vs recorded = {d:+.2e}   h_id = {info['h_id']:+.3e}   ||F||={r:.1e}", flush=True)
print(f"  REGRESSION {'PASS' if abs(d) < 1e-7 else 'FAIL'} (gate 1e-7)", flush=True)

print("\n[2] MEMBERSHIP CARD on saved fields (ground vs adjudicated ghost)", flush=True)
cases = [("ground rung_00", "rung_00_a-0.344712.npz", (16,40,12), 1e-4),
         ("ghost deg16",    "branch1_deg16_40.npz",   (16,40,12), 1e-5),
         ("ghost deg24",    "branch1_deg24_56.npz",   (24,56,12), 1e-5),
         ("ghost deg28",    "branch1_deg28_64_18.npz",(28,64,18), 1e-5)]
for name, f, degs, eps in cases:
    dd = np.load(SCR / "hunt_fields" / f)
    Sx = pc.CornerRegSolver(edges=(0.0,2.0,15.0,25.0), degs=degs, Nb=36, eps_b=eps,
                            alpha=float(dd["a"]))
    print(f"  {name:16s} cl={float(dd['z'][-2]):8.4f}  h_id = {Sx.h_id(dd['z']):+.4e}", flush=True)

print("\n[3] adopt_seed + parametrized corner data smoke", flush=True)
S2 = pc.CornerRegSolver(edges=(0.0,2.0,15.0,25.0), degs=(16,40,12), Nb=36, eps_b=1e-4,
                        wx=1.19620314, thxx=1.79819132)
before = np.abs(S2.A0 - z[:S2.Nx*S2.Nb].reshape(S2.Nx, S2.Nb)).max()
S2.adopt_seed(z)
after = np.abs(S2.A0 - z[:S2.Nx*S2.Nb].reshape(S2.Nx, S2.Nb)).max()
print(f"  |A0 - converged A|: before adopt {before:.2e} -> after {after:.2e}", flush=True)
print(f"  explicit wx/thxx round-trip: wx={S2.wx} thxx={S2.thxx}  h_id(z)={S2.h_id(z):+.3e}", flush=True)
print("done", flush=True)
