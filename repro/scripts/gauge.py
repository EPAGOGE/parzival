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
print("  v_x   = %.8f  (v = theta_x, so v_x = theta_xx -> cross-check: %.8f)" % (vx, vx))
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
print("\n  sanity: v vs d(th)/dy1 over the interior: rel L2 = %.4e  (confirms v=theta_x)" % (num/den))
