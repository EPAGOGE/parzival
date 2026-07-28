import importlib.util, sys, numpy as np
spec = importlib.util.spec_from_file_location("ag", "angular_gate.py")
ag = importlib.util.module_from_spec(spec); sys.modules["ag"] = ag
spec.loader.exec_module(ag)
P = ag.load_profile()
beta, g1 = ag.extract_angular(P, P["w"], nbeta=600)
print("g(beta) from Chen-Hou's own profile, %d angular samples" % beta.size)
print("  g at wall(beta->0)=%.5g   at axis(beta->pi/2)=%.5g   max=%.5g\n"
      % (g1[0], g1[-1], np.abs(g1).max()))
print("SINE-SERIES CONVERGENCE  g = sum_k g_k sin(2k beta)")
print("  smooth g on the closed wedge => EXPONENTIAL decay in K")
print("  g nonzero / singular at an EDGE => ALGEBRAIC decay\n")
print("  %5s %12s %8s" % ("K", "rel L2 err", "ratio"))
prev = None; Ks = [2, 4, 8, 16, 32, 64, 128, 256]; errs = []
for K in Ks:
    M = np.stack([np.sin(2 * k * beta) for k in range(1, K + 1)], axis=1)
    c, *_ = np.linalg.lstsq(M, g1, rcond=None)
    e = np.linalg.norm(M @ c - g1) / np.linalg.norm(g1)
    errs.append(e)
    print("  %5d %12.4e %8s" % (K, e, ("%8.2f" % (prev / e)) if prev else "-"))
    prev = e
K = np.array(Ks, float); E = np.array(errs); m = E > 1e-13
p = np.polyfit(np.log(K[m]), np.log(E[m]), 1)
print("\n  algebraic fit: err ~ K^%+.3f" % p[0])
if p[0] > -3.5:
    print("  => ALGEBRAIC decay. g is NOT smooth on the closed wedge; a sine basis,")
    print("     which forces g(0)=g(pi/2)=0, fights the actual edge behaviour.")
M = np.stack([np.sin(2 * k * beta) for k in range(1, 33)], axis=1)
c, *_ = np.linalg.lstsq(M, g1, rcond=None)
res = np.abs(M @ c - g1)
print("\n  RESIDUAL LOCALISATION at K=32 (mean |residual| by region):")
for lab, sl in (("wall edge  beta<0.05", beta < 0.05),
                ("interior 0.05..1.52", (beta >= 0.05) & (beta <= 1.52)),
                ("axis edge  beta>1.52", beta > 1.52)):
    if sl.sum():
        print("    %-22s %.4e   (%d pts)" % (lab, res[sl].mean(), sl.sum()))
print("\n  g near WALL (first 5): %s" % np.array2string(g1[:5], precision=4))
print("  g near AXIS (last  5): %s" % np.array2string(g1[-5:], precision=4))
