import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
sp=importlib.util.spec_from_file_location("pm","/Users/epagogellc/parzival/boussinesq/polar_march.py")
pm=importlib.util.module_from_spec(sp); sys.modules["pm"]=pm; sp.loader.exec_module(pm)

def cheb_coeffs(F, axis):
    """Chebyshev coefficients via FFT on Gauss-Lobatto nodes (nodes are ASCENDING here,
    so flip first)."""
    G = np.flip(F, axis=axis)
    n = G.shape[axis]
    Ge = np.concatenate([G, np.flip(G, axis=axis)[tuple(
        slice(1,-1) if k==axis else slice(None) for k in range(G.ndim))]], axis=axis)
    C = np.real(np.fft.fft(Ge, axis=axis))[tuple(
        slice(0,n) if k==axis else slice(None) for k in range(G.ndim))]
    return C/(n-1)

def tail(F, axis, frac=0.25):
    C = np.abs(cheb_coeffs(F, axis))
    n = C.shape[axis]; k0 = int(n*(1-frac))
    hi = np.take(C, range(k0,n), axis=axis)
    return float(np.sqrt((hi**2).sum())/max(np.sqrt((C**2).sum()),1e-300))

M = pm.March(64,64,-2.0,25.0)
I=(slice(2,-2),slice(2,-2))
print("GRID-SCALE DIAGNOSTIC: is the growth at HIGH WAVENUMBER?")
print("tail = energy fraction in the top 25%% of Chebyshev modes.\n")
print("%8s %9s %12s %12s %12s %12s" % ("step","tau","tail_s(Ot)","tail_b(Ot)","max|dOt|","c_l"))
for k in range(6001):
    if k%500==0:
        ts, tb = tail(M.Ot,0), tail(M.Ot,1)
        dO,_,_,cl,cw,_ = M.rhs(M.Ot,M.Bt)
        print("%8d %9.2f %12.4e %12.4e %12.4e %12.6f"
              % (k,k*1e-3,ts,tb,np.abs(dO[I]).max(),cl), flush=True)
    M.step(1e-3)
