"""RE-RUN EVERY TABLE THAT THE BROKEN HARNESS PRODUCED.

The four defects fixed are not biases, they manufactured data:
  1. zero-accepted-step returns of the SEED, reported as a converged alpha (the phantom
     -0.34240009311696556 at ||F|| = 1.77e-2)
  2. warm start discarded (x0 = None unconditionally) -> path failures look like
     non-existence
  3. undamped outer fixed-point map with a hard 8-pass cap and no convergence test ->
     period-2 cycles returned as answers
  4. steps=10 against a damped phase that runs 16-22 iterations

So both of today's tables have to be re-taken, and this time every row carries a converged
flag, the accepted-step counts, the open-system residual on the line ||F|| cannot see, and
c_l against its exact free target 3.00649824.

WHAT ALPHA CAN SEE.  d_alpha = (d_w - d_l)/(1 + d_l) to 4.5e-7, so the common mode of
(d_l, d_w) is invisible to it. Print BOTH: `d_cl` is mostly common mode (1.4x-767x larger
than the alpha error) and `diff` = d_w - d_l is what alpha actually reads.
"""
import sys, pathlib, numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
import importlib.util


def mod(name, fn):
    sp = importlib.util.spec_from_file_location(name, str(H / fn))
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m


pst = mod("pst", "polar_stability.py")
REF = -0.34240009
CLS = pst.CL_STAR
CWS = REF * CLS

hdr = (f"{'N':>3} {'XMAX':>5} {'eps_b':>7} {'cnv':>4} {'||F||':>10} {'openF':>10} "
       f"{'axisRMS':>9} {'alpha':>13} {'vs ref':>9} {'d_cl':>9} {'diff':>10} {'steps':>14}")


def run(N, X, e):
    try:
        St, x, r, cl, cw, info = pst.converge_exact(N, XMAX=X, eps_b=e, strict=False)
    except Exception as ex:
        print(f"{N:3d} {X:5.1f} {e:7.0e}  RAISED {type(ex).__name__}: {str(ex)[:88]}", flush=True)
        return
    if not np.isfinite(cl):
        print(f"{N:3d} {X:5.1f} {e:7.0e} {'NO':>4} {r:10.3e}   "
              f"zero Newton steps -> the returned alpha would have been the SEED's", flush=True)
        return
    a = cw / cl
    od = St.S.open_residual(x)
    d_l = (cl - CLS) / CLS
    d_w = (cw - CWS) / CWS
    print(f"{N:3d} {X:5.1f} {e:7.0e} {('yes' if info['converged'] else 'NO'):>4} "
          f"{r:10.3e} {od['open_rms']:10.3e} {od['axis_rms']:9.2e} {a:+13.8f} "
          f"{100*(a-REF)/abs(REF):+8.3f}% {100*d_l:+8.3f}% {100*(d_w-d_l):+9.4f}% "
          f"{str(info['newton_steps'])[:14]:>14}", flush=True)


print("A. THE N LADDER at XMAX=25, eps_b=1e-3   (was: -0.341876 / -0.342108 / -0.337819 / -0.360552)")
print(hdr)
for N in (28, 36, 44, 52):
    run(N, 25.0, 1e-3)

print("\nB. THE XMAX LADDER at N=36  (was: 15 ok / 20 ok / 25 ok / 32 STALL / 40 COLLAPSE)")
print(hdr)
for X in (15.0, 20.0, 25.0, 32.0):
    run(36, X, 1e-3)

print("\nC. THE eps_b LADDER at N=36, XMAX=25  (was: ONLY 1e-3 converged)")
print(hdr)
for e in (1e-2, 3e-3, 1.2e-3, 1e-3, 6e-4, 1e-4, 1e-5):
    run(36, 25.0, e)

print("\nD. THE PHANTOM.  shipped code returned alpha=-0.34240009311696556 at ||F||=1.77e-2 here.")
print(hdr)
run(28, 22.0, 1e-3)
print(f"\nreference alpha = {REF}   c_l* = {CLS:.8f}   c_w* = {CWS:.8f}")
