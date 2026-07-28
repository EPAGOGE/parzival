import numpy as np

def lagrag(x, a, b):
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    F = np.zeros_like(x)
    for i in range(n):
        m = np.full_like(x, b[i])
        for j in range(n):
            if j != i:
                m = m * (x - a[j]) / (a[i] - a[j])
        F = F + m
    return F

def find_first_ge(x, a):
    """MATLAB find(x>=a,1) -> 1-based index"""
    idx = np.nonzero(x >= a)[0]
    return idx[0] + 1  # 1-based

def densera_smooth_exp(x, a1, a2, ref):
    # x is 1-D numpy, MATLAB column
    length_ = 18
    ad2 = 1
    i0 = find_first_ge(x, a2)        # 1-based
    kk = 3
    i1 = find_first_ge(x, a1) - kk - length_   # 1-based
    xg = np.log(x[1:]) - np.log(x[:-1])        # xg(k) MATLAB 1-based k -> xg[k-1]
    h0 = xg[i0-1] / ref
    fitx = np.concatenate([np.arange(i1, i1+kk+1), np.arange(i1+length_+kk, i1+length_+kk+kk+1)])
    fity = np.concatenate([xg[i1-1:i1+kk], h0*np.ones(kk+1)])
    ls = np.arange(1, length_+1)
    st = i1 + kk
    mxg = lagrag(i1+kk+ls, fitx, fity)
    ed0 = x[i0-1] * np.exp(ad2*h0)
    i2 = i0 + 5
    ad1 = int(np.ceil((np.log(ed0 / x[st+1-1]) - np.sum(mxg)) / h0))
    mxg = np.concatenate([mxg, np.ones(ad1)*h0])
    fitx2 = np.concatenate([i2-length_+np.arange(-kk,1), i2+np.arange(0,kk+1)])
    fity2 = np.concatenate([h0*np.ones(kk+1), xg[i2-1:i2+kk]])
    mxg = np.concatenate([mxg, lagrag(i2-length_+ls, fitx2, fity2)])
    mxg = np.concatenate([mxg, xg[i2:]])   # xg(i2+1:end) 1-based -> xg[i2:]
    x = np.concatenate([x[:st+1], x[st+1-1]*np.exp(np.cumsum(mxg))])
    return x

def expmesh_smoother_22():
    n = 355; Nexp1 = 90; r0 = 1.025
    N1 = 58; N2 = 38
    r1 = 1.15; r3 = 2.4; Nexp = 221
    d = 2
    N = n + Nexp
    x = np.zeros(N)
    h = 1.0/(N-1)
    h1 = 1/256; h2 = 2/256
    kk = 4
    gd = np.concatenate([np.arange(-kk,1), np.arange(-kk,1)+N2])
    gh = np.concatenate([h1*np.ones(kk+1), h2*np.ones(kk+1)])
    I1 = np.arange(1, N1+1)
    x[I1-1] = (I1-1)*h1
    I2 = np.arange(N1+1, N1+N2+1)
    dx = lagrag(I2-N1, gd, gh)
    for i in range(1, N2+1):
        x[N1+i-1] = x[N1+i-2] + dx[i-1]
    N3 = n - N1 - N2
    h3 = 1.0/N3
    y = np.arange(1, N3+1)*h3
    M = h2/h3
    r = np.log(r0/(1+h3)) / ((1+h3)**d - 1)
    F = lambda z: M*z*np.exp(r*z**d)
    I3 = np.arange(N1+N2+1, n+1)
    x[I3-1] = F(y) + x[N1+N2-1]
    ad = np.ones(Nexp)
    rat = 1.0
    for i in range(1, Nexp+1):
        lam = min((i-1)/Nexp1, 1.0)
        rat = rat*np.exp((1-lam)*np.log(r0) + lam*np.log(r1))
        ad[i-1] = rat
    x[np.arange(n+1, n+Nexp+1)-1] = x[I3[-1]-1]*ad
    print("  pre-refine: N=%d  x_max=%.6e  h1=%.8f" % (len(x), x[-1], x[1]-x[0]))
    x = densera_smooth_exp(x, 30, 40000, 2.1)
    print("  after refine1: N=%d  x_max=%.6e" % (len(x), x[-1]))
    x = densera_smooth_exp(x, 30, 3000, 2.4)
    print("  after refine2: N=%d  x_max=%.6e" % (len(x), x[-1]))
    return x

x1 = expmesh_smoother_22()
n1b = len(x1)
print()
print("=== PRODUCTION MESH ===")
print("n1b            =", n1b)
print("x1_max         = %.6e   s=ln(x1max)=%.4f" % (x1[-1], np.log(x1[-1])))
print("corner sqrt2*  = %.6e   s=%.4f" % (np.sqrt(2)*x1[-1], np.log(np.sqrt(2)*x1[-1])))
print("x1[57]         = %.6f (58th pt, uniform end)" % x1[57])
print("h near 0       = %.10f  (1/256=%.10f)" % (x1[1]-x1[0], 1/256))
rat = x1[1:]/x1[:-1]
print("last 3 ratios  =", rat[-3:])
print("r_st=x1(end-60)= %.6e  s=%.4f" % (x1[-61], np.log(x1[-61])))
print("r_ed=x1(end-12)= %.6e  s=%.4f" % (x1[-13], np.log(x1[-13])))
print("Farfit window x1(n1-50)..x1(n1-15) = %.4e .. %.4e  (s=%.2f..%.2f)"
      % (x1[n1b-50-1], x1[n1b-15-1], np.log(x1[n1b-50-1]), np.log(x1[n1b-15-1])))
# Meshext exp with ex=28 ratio 1.15
ex = 28; r1 = 1.15
ad = x1[-1]*r1**np.arange(1, ex+1)
gx1 = np.concatenate([x1, ad])
print("gx1_max (ex=28)= %.6e  s=%.4f ; corner %.4e s=%.4f"
      % (gx1[-1], np.log(gx1[-1]), np.sqrt(2)*gx1[-1], np.log(np.sqrt(2)*gx1[-1])))
# log-radial resolution
import collections
for lo, hi in [(1e0,1e2),(1e2,1e4),(1e4,1e6),(1e6,1e13)]:
    m = (x1>=lo)&(x1<hi)
    sel = x1[m]
    if len(sel)>1:
        ds = np.diff(np.log(sel))
        print("  x in [%.0e,%.0e): %3d pts, ds mean %.4f" % (lo,hi,m.sum(),ds.mean()))
print("  pts with x>1e8 :", int((x1>1e8).sum()))
