import importlib.util, sys, numpy as np
sys.path.insert(0, ".")
for nm, f in (("ag", "angular_gate.py"), ("ps", "polar_seed.py")):
    sp = importlib.util.spec_from_file_location(nm, f)
    m = importlib.util.module_from_spec(sp); sys.modules[nm] = m; sp.loader.exec_module(m)
import ag, ps
Pg = ag.load_profile()
bg, g1 = ag.extract_angular(Pg, Pg["w"])
print("angular_gate beta range: [%.5f, %.5f], %d bins" % (bg.min(), bg.max(), bg.size))
P = ps.load()
sgrid = np.linspace(19.0, 30.0, 48)
bgrid = np.linspace(0.02, np.pi/2 - 0.02, 64)
print("polar_seed  beta range: [%.5f, %.5f], %d pts" % (bgrid.min(), bgrid.max(), bgrid.size))
Ot, Bt, Om, B = ps.seed_on_grid(P, sgrid, bgrid)
mine_at = Ot[24]
ov = (bg >= bgrid.min()) & (bg <= bgrid.max())
print("\noverlapping beta bins: %d of %d  (%d OUTSIDE my range -> np.interp CLAMPS them)"
      % (ov.sum(), bg.size, (~ov).sum()))
if (~ov).sum():
    print("  outside:", np.array2string(bg[~ov], precision=4))
    print("  g there:", np.array2string(g1[~ov], precision=4))
mine = np.interp(bg, bgrid, mine_at)
print("\n  rel L2 over ALL bins      : %.4e" % (np.linalg.norm(mine-g1)/np.linalg.norm(g1)))
print("  rel L2 over OVERLAP only  : %.4e"
      % (np.linalg.norm(mine[ov]-g1[ov])/np.linalg.norm(g1[ov])))
# where is the error?
loc = np.abs(mine-g1)/np.maximum(np.abs(g1), 1e-30)
k = np.argsort(loc)[::-1][:6]
print("\n  worst 6 bins by RELATIVE error:")
for i in k:
    print("    beta=%.5f  angular_gate=%9.5g  seed=%9.5g  rel=%.3e  %s"
          % (bg[i], g1[i], mine[i], loc[i], "" if ov[i] else "<- CLAMPED (outside range)"))
# now evaluate the seed ON angular_gate's own beta points, no clamping
b2 = bg.copy()
Ot2, _, _, _ = ps.seed_on_grid(P, sgrid, b2)
m2 = Ot2[24]
print("\n  RE-EVALUATED on angular_gate's own beta points (no interpolation, no clamp):")
print("    rel L2 = %.4e" % (np.linalg.norm(m2-g1)/np.linalg.norm(g1)))
loc2 = np.abs(m2-g1)/np.maximum(np.abs(g1),1e-30)
k2 = np.argsort(loc2)[::-1][:5]
for i in k2:
    print("    beta=%.5f  ag=%9.5g  seed=%9.5g  rel=%.3e" % (bg[i], g1[i], m2[i], loc2[i]))
