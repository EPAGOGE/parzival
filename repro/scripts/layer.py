import importlib.util, sys, numpy as np
sys.path.insert(0, ".")
sp = importlib.util.spec_from_file_location("ps", "polar_seed.py")
ps = importlib.util.module_from_spec(sp); sys.modules["ps"] = ps; sp.loader.exec_module(ps)
P = ps.load()
H = np.pi / 2
print("Ot(beta) approaching the SYMMETRY LINE beta=pi/2, where Om must vanish (odd).")
print("Sampling eps = pi/2 - beta geometrically, at three well-separated s.\n")
eps = np.array([3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5,1e-6,1e-7,1e-8])
for s0 in (20.0, 25.0, 30.0):
    Ot,_,_,_ = ps.seed_on_grid(P, np.array([s0]), H - eps)
    v = Ot[0]
    print("  s=%.1f  (r=%.2e)" % (s0, np.exp(s0)))
    print("    eps :", " ".join("%9.1e" % e for e in eps))
    print("    Ot  :", " ".join("%9.4f" % x for x in v))
    # local slope d log|Ot| / d log eps  -> the vanishing order at the axis
    m = v > 0
    if m.sum() > 3:
        sl = np.gradient(np.log(v[m]), np.log(eps[m]))
        print("    dlogOt/dlogeps:", " ".join("%9.3f" % x for x in sl))
print("\nAlso the WALL side, eps = beta, where Om need NOT vanish:")
for s0 in (25.0,):
    Ot,_,_,_ = ps.seed_on_grid(P, np.array([s0]), eps)
    print("  s=%.1f  Ot:" % s0, " ".join("%9.4f" % x for x in Ot[0]))
