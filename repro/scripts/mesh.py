import numpy as np

def lagrag(x, a, b):
    x = np.asarray(x, dtype=float); a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    F = np.zeros_like(x)
    for i in range(len(a)):
        m = b[i]*np.ones_like(x)
        for j in range(len(a)):
            if j != i:
                m = m*(x-a[j])/(a[i]-a[j])
        F = F + m
    return F

def densera_smooth_exp(x, a1, a2, ref):
    # x: 1-D numpy, MATLAB 1-based translated
    x = np.asarray(x, dtype=float)
    ln = 18; ad2 = 1
    i0 = int(np.argmax(x >= a2)) + 1          # MATLAB index
    kk = 3
    i1 = int(np.argmax(x >= a1)) + 1 - kk - ln
    xg = np.log(x[1:]) - np.log(x[:-1])       # xg(k) MATLAB = xg[k-1]
    def XG(k): return xg[np.asarray(k, dtype=int)-1]
    h0 = XG(i0)/ref
    fitx = np.concatenate([np.arange(i1, i1+kk+1), i1+ln+kk+np.arange(0, kk+1)])
    fity = np.concatenate([XG(np.arange(i1, i1+kk+1)), h0*np.ones(kk+1)])
    ls = np.arange(1, ln+1)
    st = i1 + kk
    mxg = lagrag(i1+kk+ls, fitx, fity)
    ed0 = x[i0-1]*np.exp(ad2*h0)
    i2 = i0 + 5
    ad1 = int(np.ceil((np.log(ed0/x[st+1-1]) - mxg.sum())/h0))
    mxg = np.concatenate([mxg, np.ones(ad1)*h0])
    fitx2 = np.concatenate([i2-ln+np.arange(-kk, 1), i2+np.arange(0, kk+1)])
    fity2 = np.concatenate([h0*np.ones(kk+1), XG(i2+np.arange(0, kk+1))])
    mxg = np.concatenate([mxg, lagrag(i2-ln+ls, fitx2, fity2)])
    mxg = np.concatenate([mxg, XG(np.arange(i2+1, len(x)))])
    return np.concatenate([x[:st+1], x[st+1-1]*np.exp(np.cumsum(mxg))])

def expmesh():
    n = 355; N1 = 58; N2 = 38; Nexp1 = 90
    r1 = 1.15; r3 = 2.4; Nexp = 221
    r0 = 1.025; d = 2
    N = n + Nexp
    x = np.zeros(N)
    h = 1.0/(N-1)
    h1 = 1/256; h2 = 2/256
    kk = 4
    gd = np.concatenate([np.arange(-kk, 1), np.arange(-kk, 1)+N2])
    gh = np.concatenate([h1*np.ones(kk+1), h2*np.ones(kk+1)])
    I1 = np.arange(1, N1+1)
    x[I1-1] = (I1-1)*h1
    dx = lagrag(np.arange(1, N2+1), gd, gh)
    for i in range(1, N2+1):
        x[N1+i-1] = x[N1+i-2] + dx[i-1]
    N3 = n - N1 - N2
    h3 = 1.0/N3
    y = np.arange(1, N3+1)*h3
    M = h2/h3
    r = np.log(r0/(1+h3))/((1+h3)**d - 1)
    x[N1+N2:n] = M*y*np.exp(r*y**d) + x[N1+N2-1]
    rat = 1.0; ad = np.ones(Nexp)
    for i in range(1, Nexp+1):
        lam = min((i-1)/Nexp1, 1.0)
        rat = rat*np.exp((1-lam)*np.log(r0) + lam*np.log(r1))
        ad[i-1] = rat
    x[n:n+Nexp] = x[n-1]*ad
    x = densera_smooth_exp(x, 30, 40000, 2.1)
    x = densera_smooth_exp(x, 30, 3000, r3)
    return x

x = expmesh()
print('n1b =', len(x))
print('x[0]=%.6g x[1]=%.6g x[-1]=%.6g'%(x[0], x[1], x[-1]))
print('s_max = ln x[-1] = %.5f'%np.log(x[-1]))
print('tail ds = %.6f  (ln1.15=%.6f)'%(np.log(x[-1]/x[-2]), np.log(1.15)))
print('pts > 1e8:', int((x > 1e8).sum()))
print('idx of first x>1e8 (0-based):', int(np.searchsorted(x, 1e8)))
