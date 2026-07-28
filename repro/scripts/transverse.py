"""IS THE SLICE GOING NON-TRANSVERSE, AND DOES IT DO SO WITH N?

theta_min(ker Cg, range B) = 0.0004 deg at N=28. That is the real obliqueness number
(Kato; Xu-Zikatanov 1307.4393 Cor 4.2) and it is FIVE ORDERS OF MAGNITUDE worse than
cond(Cg B) = 29.3 suggests. The witness-set cross-connection warned about exactly this:
our two corner point-functionals are a SLICE of the 2-dimensional component {Om = B = 0},
and a slice chosen for physical convenience rather than genericity can be non-transverse.

MECHANISM (prediction to test). Cg reads DERIVATIVES AT THE CORNER, with Chebyshev row
weights Dx[0,:] ~ N^2 and Dx2[0,:] ~ N^4.  B is the pair of gauge tangents, whose leading
terms carry a factor g = 1 - e^(-xi), which VANISHES at the corner and is O(N^-2) at the
first interior node.  So the constraints look exactly where the gauge directions are
weakest.  If that is the mechanism then

    sin theta_min  ~  N^(-p)  with p > 0,

||P|| grows without bound under refinement, and the N-ceiling is a transversality failure
rather than a conditioning accident (math/0603716 Thm 3.2: at a fold it is the
TRANSVERSALITY that controls the bordered solve, not how small sigma_min is).

Also settles the +1.05 question across N: it is absent from the N=28 spectrum entirely
and the compressed, QZ and ambient routes all agree that it is.
"""
import sys, pathlib, numpy as np, scipy.linalg as la
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
import importlib.util


def mod(name, fn):
    sp = importlib.util.spec_from_file_location(name, str(H / fn))
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m


pst = mod("pst", "polar_stability.py")
pz = mod("pz", "polar_zeros.py")

print(f"{'N':>3} {'dim':>5} {'||F||':>10} {'alpha':>13} {'cond(CgB)':>10} "
      f"{'sin(th)':>10} {'||P||':>10} {'s(CgB)min':>10} {'||B||':>9} {'||Cg||':>9}  real zeros in (-2.5,9)")
prev = None
for N in (28, 36, 44, 52):
    St, x, r, cl, cw = pst.converge_exact(N)
    _, _, cCB, A, B, Cg = St.spectrum(x)
    M, M_orth, Z = pz.compress(A, B, Cg)
    pn_, smin = pz.proj_norm(B, Cg, Z)
    sCB = la.svdvals(Cg @ B)
    w = la.eigvals(M)
    rl = np.sort(w[np.abs(w.imag) < 1e-6].real)[::-1]
    rl = [float(z) for z in rl if -2.5 < z < 9.0]
    print(f"{N:3d} {St.n:5d} {r:10.2e} {cw/cl:+13.8f} {cCB:10.3g} {smin:10.3e} {pn_:10.3e} "
          f"{sCB[-1]:10.3e} {la.norm(B):9.3g} {la.norm(Cg):9.3g}  "
          + " ".join(f"{z:+.4f}" for z in rl[:6]), flush=True)
    if prev is not None:
        Np, sp_ = prev
        p = np.log(sp_ / smin) / np.log(N / Np)
        print(f"      -> sin(theta) scaling exponent between N={Np} and N={N}:  p = {p:+.2f}"
              f"   (sin ~ N^-p)", flush=True)
    prev = (N, smin)
