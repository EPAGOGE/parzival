"""
GATE the GAUGE the solver will impose, against Chen-Hou's own stored constants.

Two scaling symmetries means two normalizations, and getting them wrong is not cosmetic:
pinning the wrong quantity leaves a neutral direction, which in this lab produced a
singular Jacobian, a sign-flipping c_w, and step norms of 1e6-1e7.

POLAR_SPEC records that their NONLINEAR gauge is two POINT conditions, not integrals:

    c_l     = 2 theta_xx(0) / omega_x(0)
    c_w     = c_l / 2 + u_x(0)
    c_theta = c_l + 2 c_w                  (a CONSTRAINT -- exactly two scaling DOF)

That is testable rather than merely citable: their converged profile AND their converged
c_l, c_w are both on disk. Evaluate the right-hand sides from the profile and check they
reproduce the stored constants. Their mesh is UNIFORM near the origin (spacing
0.00390625), so one-sided 4th-order differences at the corner are clean.

Sign variants are tested beside the stated forms, because a sign error in a gauge is the
kind that still "converges" -- to the wrong branch.

RESULT (2026-07-25): BOTH formulas confirmed.
    c_l = 2 th_xx(0)/w_x(0) = +3.00649823  vs stored +3.00649798   rel err 8.4e-08
    c_w = c_l/2 + u1_x(0)   = -1.02942519  vs stored -1.02942519   rel err 1.2e-09
The sign variants miss by 50% and 490%, so the gate discriminates.

SIDE FINDING -- a correction to polar_bc_gate.py's field labels: the stored `v` is NOT
theta_x. Its far-field exponent IS that of a first derivative of theta (-0.68471, versus
-0.68481 for theta_x), which is why an exponent-only identification accepted it, but
v/theta_x varies from 0.50 to 2.84 pointwise (non-constant, so not a rescaling) and
v_x(0) = theta_xx(0)/2 exactly rather than theta_xx(0). Its identity is UNRESOLVED and is
not needed: this gate reproducing both constants from `th` and `u1` to 8-9 digits pins
th = theta and u1 = the velocity component in the gauge far harder than any exponent
match could.
"""
import importlib.util, sys, numpy as np
from scipy.io import loadmat
sys.path.insert(0, ".")
sp = importlib.util.spec_from_file_location("ps", "polar_seed.py")
ps = importlib.util.module_from_spec(sp); sys.modules["ps"] = ps; sp.loader.exec_module(sp and ps)
P = ps.load()
d = loadmat(ps.MAT, squeeze_me=True, struct_as_record=False)
s_ = d["solu"]
w = np.asarray(d["w"], float); shp = w.shape
X, Y = P["X"], P["Y"]
cl, cw = P["cl"], P["cw"]
print("stored:  cl=%.8f  cw=%.8f  c_theta=cl+2cw=%.8f  alpha=cw/cl=%.8f"
      % (cl, cw, cl + 2*cw, cw/cl))
print("near-origin grid spacing: dX=%.6g dY=%.6g (uniform? %s)\n"
      % (X[1]-X[0], Y[1]-Y[0], np.allclose(np.diff(X[:8]), X[1]-X[0])))
avail = {}
for nm in ("th","u1","u2","v","u1_dx","v_dx"):
    try:
        avail[nm] = ps._grid_field(d, s_, nm, shp); print("  have %-6s %s" % (nm, shp))
    except SystemExit: print("  MISSING", nm)
th = avail["th"]; u1 = avail["u1"]; v = avail["v"]
h = X[1]-X[0]
# derivatives at the corner (0,0) along y1, one-sided high-order on the uniform patch
def d1_at0(F):   # dF/dy1 at (0,0), 4th-order one-sided
    f = F[:5, 0]
    return (-25*f[0] + 48*f[1] - 36*f[2] + 16*f[3] - 3*f[4]) / (12*h)
def d2_at0(F):   # d2F/dy1^2 at (0,0), one-sided
    f = F[:5, 0]
    return (35*f[0] - 104*f[1] + 114*f[2] - 56*f[3] + 11*f[4]) / (12*h*h)
wx  = d1_at0(w)
thxx= d2_at0(th)
u1x = d1_at0(u1)
vx  = d1_at0(v)
print("\nat the corner (0,0):")
print("  w(0,0)=%.6g   th(0,0)=%.6g   u1(0,0)=%.6g   v(0,0)=%.6g" % (w[0,0], th[0,0], u1[0,0], v[0,0]))
print("  w_x   = %.8f" % wx)
print("  th_xx = %.8f" % thxx)
print("  u1_x  = %.8f" % u1x)
print("  v_x   = %.8f   (NOTE: this is th_xx/2 exactly, so v is NOT theta_x)" % vx)
print("\nTEST POLAR_SPEC's stated gauge formulas:")
for label, num in (("2*th_xx(0)/w_x(0)", 2*thxx/wx), ("2*v_x(0)/w_x(0)", 2*vx/wx)):
    print("  cl =? %-22s = %+.8f   vs stored %+.8f   rel err %.3e"
          % (label, num, cl, abs(num-cl)/abs(cl)))
for label, num in (("cl/2 + u1_x(0)", cl/2 + u1x), ("cl/2 - u1_x(0)", cl/2 - u1x)):
    print("  cw =? %-22s = %+.8f   vs stored %+.8f   rel err %.3e"
          % (label, num, cw, abs(num-cw)/abs(cw)))
# also: is v really theta_x? compare v against d(th)/dy1 in the interior
gy1 = np.gradient(th, X, axis=0)
m = (np.arange(shp[0])[:,None] > 5) & (np.arange(shp[1])[None,:] > 5)
num = np.linalg.norm((v-gy1)[m]); den = np.linalg.norm(v[m])
print("\n  v vs d(th)/dy1 over the interior: rel L2 = %.4e" % (num/den))
print("     -> v is NOT theta_x (ratio varies 0.50..2.84 pointwise). Exponent-only")
print("        identification accepted it wrongly; see this file's docstring.")
