"""GATE THE SECOND CORNER CONSTRAINT:  'd2' (Bt_xixi) versus 'd1' (Vt_xi, Vt = Bt/g).

Both pin the same continuum quantity.  What differs is how much of the profile's unresolved
radial Chebyshev tail the row amplifies -- |Dx2[0,:]|_1 ~ N^4 against |Dx[0,:]|_1 ~ N^2 --
and alpha reads the pinned pair ONLY through q = THXX/WX^2, so the harmful projection is
dln q = e2 - 2 e1.

Three checks, cheapest first, so a failure is caught before any Newton solve.

  1. CONTINUUM EQUIVALENCE.  On the exact corner form Bt = (THXX/2) xi^2 cos^2 b, both rows
     must return their target to discretisation accuracy.  A variant that fails here is
     simply wrong and nothing else matters.
  2. AMPLIFICATION.  Row 1-norms against N^4 / N^2, and the COHERENCE ratio
     sum|row_k data_k| / |sum(row_k data_k)| -- the quantity that actually decides how much
     signal is lost, as distinct from the row's size.
  3. SOLVE-FREE ERROR ON THE SEED.  e1, e2 and dln q evaluated on Corner's own seed, no
     Newton anywhere.  Predicted: |e2| flattens at ~1% under refinement while |e2'| keeps
     falling.  That CONVERGENCE, not the size at any single N, is the claim.
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(HERE / fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def rows(S):
    """The two constraint rows as vectors over the radial index, at beta node 0."""
    C = S.C
    r1 = C.Dx[0, :].copy()
    if S.constraint == "d2":
        r2 = C.Dx2[0, :].copy()
    else:
        r2 = np.zeros(C.nx)
        r2[1:] = C.Dx[0, 1:] / C.g[1:]
    return r1, r2


def check_continuum(Ns=(28, 36, 44, 52, 64)):
    pn = _mod("pn", "polar_newton.py")
    print("1. CONTINUUM EQUIVALENCE on the exact corner form Bt = (THXX/2) xi^2 cos^2 b")
    print(f"   {'N':>4} {'d2 rel err':>13} {'d1 rel err':>13}")
    for N in Ns:
        out = []
        for v in ("d2", "d1"):
            S = pn.NewtonSolver(N, constraint=v)
            C = S.C
            TH = S.THXX_REF
            Bt = 0.5 * TH * (C.x[:, None] ** 2) * (np.cos(C.b)[None, :] ** 2)
            got = S.g2_of(Bt) + (TH if v == "d2" else 0.5 * TH)
            want = (TH if v == "d2" else 0.5 * TH) * np.cos(C.b[0]) ** 2
            out.append((got - want) / want)
        print(f"   {N:4d} {out[0]:13.3e} {out[1]:13.3e}")
    print()


def check_amplification(Ns=(20, 28, 36, 44, 52, 64, 96)):
    pn = _mod("pn", "polar_newton.py")
    print("2. AMPLIFICATION.  row 1-norm, its N^4 / N^2 reduced form, and the COHERENCE ratio")
    print(f"   {'N':>4} {'|d2|_1':>10} {'/N^4':>10} {'|d1|_1':>10} {'/N^2':>9} "
          f"{'coh(d2)':>10} {'coh(d1)':>10}")
    for N in Ns:
        vals = {}
        for v in ("d2", "d1"):
            S = pn.NewtonSolver(N, constraint=v)
            C = S.C
            _, r2 = rows(S)
            data = C.Bt0[:, 0]
            num = float(np.abs(r2 * data).sum())
            den = abs(float((r2 * data).sum()))
            vals[v] = (float(np.abs(r2).sum()), num / max(den, 1e-300))
        print(f"   {N:4d} {vals['d2'][0]:10.3e} {vals['d2'][0]/N**4:10.3e} "
              f"{vals['d1'][0]:10.3e} {vals['d1'][0]/N**2:9.3e} "
              f"{vals['d2'][1]:10.4g} {vals['d1'][1]:10.4g}")
    print()


def check_seed(Ns=(28, 36, 44, 52, 64, 96), XMAX=25.0):
    """e1, e2, dln q on the seed. No solve. alpha sees only dln q = e2 - 2 e1."""
    pn = _mod("pn", "polar_newton.py")
    print(f"3. SOLVE-FREE ERROR ON THE SEED at XMAX={XMAX}.  alpha reads only dln q = e2 - 2 e1")
    print(f"   {'N':>4} {'e1 %':>10} {'e2(d2) %':>11} {'e2(d1) %':>11} "
          f"{'dlnq(d2)':>11} {'dlnq(d1)':>11} {'gain':>8}")
    first = {}
    for N in Ns:
        e = {}
        for v in ("d2", "d1"):
            S = pn.NewtonSolver(N, XMAX=XMAX, constraint=v)
            C = S.C
            cb = np.cos(C.b[0])
            e1 = (float((C.Dx @ C.Ot0)[0, 0]) - S.WX_REF * cb) / (S.WX_REF * cb)
            tgt = (S.THXX_REF if v == "d2" else 0.5 * S.THXX_REF) * cb ** 2
            got = S.g2_of(C.Bt0) + (S.THXX_REF if v == "d2" else 0.5 * S.THXX_REF)
            e[v] = ((got - tgt) / tgt, e1)
        q2 = e["d2"][0] - 2 * e["d2"][1]
        q1 = e["d1"][0] - 2 * e["d1"][1]
        for k, val in (("d2", abs(q2)), ("d1", abs(q1))):
            first.setdefault(k, val)
        print(f"   {N:4d} {100*e['d2'][1]:10.4f} {100*e['d2'][0]:11.4f} "
              f"{100*e['d1'][0]:11.4f} {q2:11.3e} {q1:11.3e} {abs(q2)/max(abs(q1),1e-300):8.1f}x")
    print(f"\n   CONVERGENCE is the claim, not size at one N:")
    print(f"     |dlnq(d2)| fell by {first['d2']/max(abs(q2),1e-300):.1f}x over this range")
    print(f"     |dlnq(d1)| fell by {first['d1']/max(abs(q1),1e-300):.1f}x over this range")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmax", type=float, default=25.0)
    a = ap.parse_args()
    check_continuum()
    check_amplification()
    check_seed(XMAX=a.xmax)
