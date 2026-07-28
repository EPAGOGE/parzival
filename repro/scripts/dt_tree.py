import numpy as np

C, Tst, rho = 1.23, 1.7135, 2.60

def d0(t): return C*(Tst-t)**rho
print("null delta(1.53) =", d0(1.53), " delta(1.45)=",d0(1.45), " delta(0.8)=",d0(0.8))

# resolution floors  delta_min = a/kmax
for N in [256,512,1024,2048,4096]:
    kmax23 = N/3.0          # 2/3 dealias
    for a,lab in [(1.7,'a=1.7'),(9.2,'a=9.2 (1e-4)'),(36.8,'a=36.8 (roundoff)')]:
        dmin = a/kmax23
        s = (dmin/C)**(1/rho)
        print(f"N={N:5d} kmax={kmax23:7.1f} {lab:18s} delta_min={dmin:9.5f}  t_floor={Tst-s:7.4f}")
    print()

# decades gained per doubling of N
for r in [1.0,1.5,2.0,2.6]:
    print(f"rho={r}: decades of (T*-t) per doubling of N = {np.log10(2)/r:.4f};  N factor per decade = {10**r:.1f}")
print()

# Hou-Li cumulative filter ceiling: rho_f(k)=exp(-36 (k/kmax)^36) per application
for Nstage in [4e3,4e4,4e5,4e6]:
    # require cumulative damping > 0.99
    thr = 0.01005/(36*Nstage)
    kfrac = thr**(1/36)
    print(f"N_stage_applications={Nstage:9.0e}: safe k/kmax_filter <= {kfrac:.4f}")
print()

# local-exponent diagnostic  G(t) = -delta/delta' = s/rho_local(s)
def G_null(t): return (Tst-t)/rho
def G_corr(t,rt,Ts,a,sx):
    s = Ts-t
    rl = rt + sx*a*s**sx/(1+a*s**sx)
    return s/rl, rl

ts = np.array([0.80,0.95,1.10,1.25,1.40,1.45,1.50,1.53])
print(" t     G_null   G_A(rho=1.5)  rholoc_A   G_B(rho=1.0)  rholoc_B")
for t in ts:
    ga,ra = G_corr(t,1.5,1.640,5.4,1.25)
    gb,rb = G_corr(t,1.0,1.630,12.0,1.50)
    print(f"{t:5.2f} {G_null(t):8.5f} {ga:12.5f} {ra:9.4f} {gb:13.5f} {rb:9.4f}")
print()
# local slope of G  ->  1/rho_local_effective, over sub-windows
def locslope(f,t1,t2):
    return (f(t2)-f(t1))/(t2-t1)
for (t1,t2) in [(0.80,0.95),(1.10,1.25),(1.45,1.53)]:
    sn = locslope(G_null,t1,t2)
    sa = (G_corr(t2,1.5,1.640,5.4,1.25)[0]-G_corr(t1,1.5,1.640,5.4,1.25)[0])/(t2-t1)
    sb = (G_corr(t2,1.0,1.630,12.0,1.50)[0]-G_corr(t1,1.0,1.630,12.0,1.50)[0])/(t2-t1)
    print(f"window [{t1},{t2}]: rho_from_slope  null={-1/sn:6.3f}  A={-1/sa:6.3f}  B={-1/sb:6.3f}")
