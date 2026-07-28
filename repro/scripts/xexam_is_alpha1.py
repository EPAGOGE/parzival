"""Cross-examination of the is_alpha1 advocate case.  Measurement only:
residual evaluations (no Newton), interpolant evaluations, harmonic fits.

Targets:
 T1  Is branch1_deg24_56.npz converged or a mid-secant intermediate?
     (advocate: ||F||_2=3.0e-4, unconverged, direction void.
      branch1_res.log: -0.42554621 with ||F||=5.97e-12, secant done.)
 T2  Self-consistency: cw/cl(z) vs stored a.
 T3  Fingerprints of the deg24 root: c3/c1 sign at xi=1, A-peak location,
     corner dip below pinned profile  -> same branch or different object?
 T4  Node-quantization tautology: true (interpolant) peak location off-node?
 T5  Spectral-tail verification (ground vs candidate corner-panel A, deg16;
     candidate deg24).
"""
import importlib.util, sys, pathlib
import numpy as np
from numpy.polynomial import chebyshev as C

SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
HF = SCR / "hunt_fields"

spec = importlib.util.spec_from_file_location("pc", "/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py")
pc = importlib.util.module_from_spec(spec); sys.modules["pc"] = pc; spec.loader.exec_module(pc)

def build(degs, eps_b, alpha):
    return pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=degs, Nb=36,
                              eps_b=eps_b, alpha=alpha)

def resnorms(S, z):
    F = S.residual(z)
    return float(np.max(np.abs(F))), float(np.linalg.norm(F))

def panel_interp(S, F2d, k, jcols, xs_dense):
    """Chebyshev interpolant of field F2d on panel k, per beta column."""
    i0, i1 = int(S.offs[k]), int(S.offs[k] + S.sizes[k])
    xs = S.x[i0:i1]; a, b = xs[0], xs[-1]
    t = 2 * (xs - a) / (b - a) - 1
    td = 2 * (xs_dense - a) / (b - a) - 1
    out = {}
    for j in jcols:
        c = C.chebfit(t, F2d[i0:i1, j], deg=len(xs) - 1)
        out[j] = C.chebval(td, c)
    return out

def tail_rms(S, F2d, k, j):
    i0, i1 = int(S.offs[k]), int(S.offs[k] + S.sizes[k])
    xs = S.x[i0:i1]
    t = 2 * (xs - xs[0]) / (xs[-1] - xs[0]) - 1
    c = C.chebfit(t, F2d[i0:i1, j], deg=len(xs) - 1)
    return float(np.sqrt(np.mean(np.abs(c[-3:]) ** 2)) / np.max(np.abs(c)))

def harmonics(S, Arow, nh=5):
    M = np.stack([np.cos((2 * m - 1) * S.b) for m in range(1, nh + 1)], axis=1)
    c, *_ = np.linalg.lstsq(M, Arow, rcond=None)
    return c

print("=" * 78)
print("T1/T2: residual + self-consistency of branch1_deg24_56.npz")
d24 = np.load(HF / "branch1_deg24_56.npz")
a24, z24 = float(d24["a"]), d24["z"]
S24 = build((24, 56, 12), 1e-5, a24)
assert z24.size == 3 * S24.Nx * S24.Nb + 2, (z24.size, S24.Nx)
rinf, r2 = resnorms(S24, z24)
cl24, cw24 = float(z24[-2]), float(z24[-1])
h24 = cw24 / cl24
print(f"  stored a          = {a24:+.9f}")
print(f"  cw/cl of stored z = {h24:+.9f}   |a - cw/cl| = {abs(a24-h24):.2e}")
print(f"  ||F||_inf = {rinf:.3e}   ||F||_2 = {r2:.3e}   (advocate claimed 9.5e-5 / 3.0e-4)")
# also at eps_b=1e-4 (advocate's 100x claim; run base was eps_b=1e-5)
S24b = build((24, 56, 12), 1e-4, a24)
rinf_b, r2_b = resnorms(S24b, z24)
print(f"  same z at eps_b=1e-4: ||F||_inf = {rinf_b:.3e}  ||F||_2 = {r2_b:.3e}")

print()
print("controls: branch1_eps1e-05.npz and find_half.npz on their own grids")
d16 = np.load(HF / "branch1_eps1e-05.npz")
a16, z16 = float(d16["a"]), d16["z"]
S16 = build((16, 40, 12), 1e-5, a16)
rinf16, r216 = resnorms(S16, z16)
h16 = float(z16[-1]) / float(z16[-2])
print(f"  eps1e-05 control: a={a16:+.9f} cw/cl={h16:+.9f} |diff|={abs(a16-h16):.1e}"
      f"  ||F||_inf={rinf16:.2e} ||F||_2={r216:.2e}")
dfh = np.load(HF / "find_half.npz")
afh, zfh = float(dfh["a"]), dfh["z"]
Sfh = build((16, 40, 12), 1e-4, afh)
rinf_fh, r2_fh = resnorms(Sfh, zfh)
print(f"  find_half control: a={afh:+.9f}  ||F||_inf={rinf_fh:.2e} ||F||_2={r2_fh:.2e}")

print()
print("=" * 78)
print("T3: fingerprints of the deg24 root vs deg16 candidate vs ground")
A24, B24, P24, _, _ = S24.unpack(z24)
A16f, B16f, P16f, _, _ = Sfh.unpack(zfh)
dg = np.load(HF / "rung_00_a-0.344712.npz")
Sg = build((16, 40, 12), 1e-4, float(dg["a"]))
Ag, Bg, Pg, clg, cwg = Sg.unpack(dg["z"])

xd = np.linspace(0.0, 2.0, 4001)
for name, S, A in (("ground(deg16)", Sg, Ag), ("cand(deg16)", Sfh, A16f),
                   ("cand(deg24)", S24, A24)):
    # xi=1.0 harmonic ratio c3/c1 (interpolate each beta column to xi=1)
    i0, i1 = int(S.offs[0]), int(S.offs[0] + S.sizes[0])
    xs = S.x[i0:i1]; t1 = 2 * (1.0 - xs[0]) / (xs[-1] - xs[0]) - 1
    row = np.array([C.chebval(t1, C.chebfit(2*(xs-xs[0])/(xs[-1]-xs[0])-1,
                                            A[i0:i1, j], deg=len(xs)-1))
                    for j in range(S.Nb)])
    c = harmonics(S, row)
    # near-axis beta station (largest cos b) radial profile on corner panel
    j0 = int(np.argmax(np.cos(S.b)))
    prof = panel_interp(S, A, 0, [j0], xd)[j0]
    ipk = int(np.argmax(prof))
    dip = float(np.min(prof[xd <= 0.5]) / A[i0, j0])   # vs pinned corner value
    print(f"  {name:14s} c3/c1(xi=1) = {c[1]/c[0]:+.6f}   A-peak(interp) xi = {xd[ipk]:.4f} "
          f"val {prof[ipk]:+.4f}   min(A/A_pin, xi<=0.5) = {dip:+.4f}")
print(f"  deg24 root: cl={cl24:+.6f} cw={cw24:+.6f}  (deg16 cand: +5.22425/-2.20395; "
      f"ground: {clg:+.6f}/{cwg:+.6f})")

print()
print("=" * 78)
print("T4: node-quantization tautology -- true interpolant peak vs nearest node")
for name, S, A in (("cand(deg16)", Sfh, A16f), ("cand(deg24)", S24, A24)):
    j0 = int(np.argmax(np.cos(S.b)))
    prof = panel_interp(S, A, 0, [j0], xd)[j0]
    xpk = xd[int(np.argmax(prof))]
    i0, i1 = int(S.offs[0]), int(S.offs[0] + S.sizes[0])
    xs = S.x[i0:i1]
    dnode = float(np.min(np.abs(xs - xpk)))
    spacing = float(np.min(np.diff(np.sort(np.abs(xs - xpk)))[0])) if False else None
    near = xs[np.argsort(np.abs(xs - xpk))[:2]]
    print(f"  {name:14s} interp peak xi={xpk:.4f}; nearest nodes {near[0]:.4f},{near[1]:.4f}; "
          f"off-node distance = {dnode:.4f} (advocate: features ON nodes to <5e-4)")

print()
print("=" * 78)
print("T5: corner-panel Chebyshev tail (last-3 coeff rms / max coeff), near-axis col")
for name, S, F, lab in (("ground(deg16)", Sg, Ag, "A"), ("cand(deg16)", Sfh, A16f, "A"),
                        ("cand(deg24)", S24, A24, "A"),
                        ("ground(deg16)", Sg, Bg, "B"), ("cand(deg16)", Sfh, B16f, "B"),
                        ("cand(deg24)", S24, B24, "B")):
    j0 = int(np.argmax(np.cos(S.b)))
    print(f"  {name:14s} {lab}: tail = {tail_rms(S, F, 0, j0):.3e}")

print()
print("also: deg16->deg24 alpha ladder recap from branch1_res.log:")
print("  (16,40,12) -0.42172919  ||F||=3.3e-12 ; (24,56,12) -0.42554621  ||F||=5.97e-12")
print(f"  motion = {(-0.42554621) - (-0.42172919):+.5e}  (alpha_1 - deg16 root = "
      f"{(-0.4168236) - (-0.42172919):+.5e})")
