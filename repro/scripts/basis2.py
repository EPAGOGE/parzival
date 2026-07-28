import importlib.util, sys, numpy as np
spec = importlib.util.spec_from_file_location("ag", "angular_gate.py")
ag = importlib.util.module_from_spec(spec); sys.modules["ag"] = ag
spec.loader.exec_module(ag)
P = ag.load_profile()
beta, g1 = ag.extract_angular(P, P["w"], nbeta=600)
n = beta.size
print("g endpoints: g(wall)=%.5g  g(axis)=%.5g   -> g does NOT vanish at the wall\n"
      % (g1[0], g1[-1]))
bases = {
    "sin(2k b)      [0 at BOTH edges]  ": lambda K: [np.sin(2*k*beta) for k in range(1,K+1)],
    "cos((2k+1) b)  [free at 0, 0 at pi/2]": lambda K: [np.cos((2*k+1)*beta) for k in range(0,K)],
    "Chebyshev in b [free at both]     ": lambda K: [np.cos(k*np.arccos(np.clip(4*beta/np.pi-1,-1,1))) for k in range(K)],
}
Ks = [2, 4, 8, 16, 24, 32, 40]
print("  %-38s %s" % ("basis", "".join("%11s" % ("K=%d" % K) for K in Ks)))
for name, gen in bases.items():
    row = []
    for K in Ks:
        if K > n - 2: row.append("    --   "); continue
        M = np.stack(gen(K), axis=1)
        c, *_ = np.linalg.lstsq(M, g1, rcond=None)
        e = np.linalg.norm(M @ c - g1) / np.linalg.norm(g1)
        row.append("%11.3e" % e)
    print("  %-38s %s" % (name, "".join(row)))
print("\n  (n = %d angular samples, so K beyond ~%d is underdetermined -- ignore)" % (n, n-2))
