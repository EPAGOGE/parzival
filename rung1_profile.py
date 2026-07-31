#!/usr/bin/env python
"""Rung 1: the edge steady-state branch of viscous gCLM by Newton continuation.

Hypothesis under test: the fate boundary A*(a) ~ C/(a_c - a)^0.55 is governed
by an unstable steady state (the "edge state") whose stable manifold separates
decay from blowup; hover = trajectories shadowing it. Mechanism question:
does the branch FOLD at a_c (saddle-node) or DIVERGE (depletion runaway)?

Steady equation on the circle (mean-zero):
    F(w) = u_x*w - a*u*w_x - nu*Lambda(w) = 0,   u_x = H w,  u = -Lambda^{-1} w

All fp64, dense operators from swarm_m1.build_mats. Seed: snapshot of a
near-threshold trajectory at its most-steady moment (min |d ln sup / dt|).
Newton is bordered with a translation phase condition and a mean-zero row.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from swarm_m1 import build_mats, macro_step_np, make_ic, resolve_np

N = 128
NU = 1.0
Hm, Lm, Dm, Gm = build_mats(N)
MATS = {"H": Hm, "L": Lm, "D": Dm, "G": Gm}


def F(w: np.ndarray, a: float) -> np.ndarray:
    u, ux, wx = Gm @ w, Hm @ w, Dm @ w
    return ux * w - a * u * wx - NU * (Lm @ w)


def jac(w: np.ndarray, a: float) -> np.ndarray:
    u, ux, wx = Gm @ w, Hm @ w, Dm @ w
    return (np.diag(w) @ Hm + np.diag(ux)
            - a * (np.diag(wx) @ Gm + np.diag(u) @ Dm) - NU * Lm)


def newton(w0: np.ndarray, a: float, ref: np.ndarray, c0: float = 0.0,
           tol: float = 1e-10, itmax: int = 80
           ) -> tuple[np.ndarray | None, float, float]:
    """Relative-equilibrium Newton: solve F(w) + c*w_x = 0 for (w, c).
    A traveling-wave edge state has c != 0; a true fixed point returns c ~ 0."""
    w, c = w0.copy(), c0
    phase = Dm @ ref
    mean_row = np.ones(N) / N
    res = np.inf
    for _ in range(itmax):
        wx = Dm @ w
        Fv = F(w, a) + c * wx
        res = float(np.max(np.abs(Fv)))
        if res < tol:
            return w, res, c
        top = np.hstack([jac(w, a) + c * Dm, wx[:, None]])
        aug = np.vstack([top,
                         np.append(phase, 0.0)[None, :],
                         np.append(mean_row, 0.0)[None, :]])
        rhs = np.concatenate([-Fv, [-phase @ (w - ref)], [-w.mean()]])
        dz, *_ = np.linalg.lstsq(aug, rhs, rcond=None)
        dw, dc = dz[:-1], float(dz[-1])
        step = 1.0
        for _bt in range(6):
            wn = w + step * dw
            if (np.max(np.abs(F(wn, a) + (c + step * dc) * (Dm @ wn)))
                    < max(res * 1.5, tol)
                    and np.max(np.abs(wn)) > 0.25 * np.max(np.abs(w))):
                break
            step *= 0.5
        w, c = w + step * dw, c + step * dc
    return (w, res, c) if res < 1e-8 else (None, res, c)


def eigs(w: np.ndarray, a: float, c: float = 0.0) -> tuple[float, float, float]:
    """Top real parts of the co-moving linearization spectrum (translation
    mode ~0 kept visible). Returns (lam_u, lam_2, lam_3)."""
    ev = np.linalg.eigvals(jac(w, a) + c * Dm)
    re = np.sort(ev.real)[::-1]
    return float(re[0]), float(re[1]), float(re[2])


def seed_from_hover(a: float, alo: float, ahi: float) -> np.ndarray:
    """Bisect the fate boundary, then snapshot the near-threshold trajectory
    at its most-steady moment."""
    lo, hi = alo, ahi
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        f, _ = resolve_np(np.array([mid]), MATS, NU, "gclm", N,
                          max_steps=150000, ic="cos2", a=a)
        if f[0] == 1:
            hi = mid
        else:
            lo = mid
    amid = 0.5 * (lo + hi)
    print(f"  seed bisection at a={a}: threshold in ({lo:.5f}, {hi:.5f})")
    w = make_ic(np.array([amid]), N, "cos2")
    t = np.zeros(1)
    best, best_rate = None, np.inf
    sup_prev, t_prev = np.max(np.abs(w)), 0.0
    for _ in range(150000):
        w, t = macro_step_np(w, t, MATS, NU, "gclm", a=a)
        sup, tn = float(np.max(np.abs(w))), float(t[0])
        if sup > 1e3 or (sup < 0.1 * amid and tn > 0.5):
            break
        if tn > 1.0 and tn - t_prev > 1e-6:
            rate = abs(np.log(sup) - np.log(sup_prev)) / (tn - t_prev)
            if rate < best_rate:
                best_rate, best = rate, w[0].copy()
        sup_prev, t_prev = sup, tn
    print(f"  seed snapshot: sup={np.max(np.abs(best)):.3f} "
          f"(min |dlnsup/dt|={best_rate:.2e})")
    return best


def main() -> None:
    a0 = 0.93
    seed = seed_from_hover(a0, 15.0, 30.0)
    w, res, c = newton(seed, a0, seed)
    if w is None:
        print(f"NEWTON FAILED at a={a0} (res {res:.2e}, c={c:.4f}) -- edge "
              f"object is neither steady nor traveling; stopping honestly.")
        return
    lu, l2, l3 = eigs(w, a0, c)
    print(f"CONVERGED a={a0}: ||F+cw_x||={res:.2e} c={c:.5f} "
          f"sup={np.max(np.abs(w)):.4f} eig top3 re: {lu:.4f} {l2:.4f} {l3:.4f}")

    # edge verification: nudge along the unstable eigenvector, both signs
    ev, V = np.linalg.eig(jac(w, a0) + c * Dm)
    vu = np.real(V[:, np.argmax(ev.real)])
    vu /= np.max(np.abs(vu))
    fates = []
    for s in (+1e-3, -1e-3):
        wp = (w + s * np.max(np.abs(w)) * vu)[None, :].copy()
        t = np.zeros(1)
        fate = -1
        for _ in range(200000):
            wp, t = macro_step_np(wp, t, MATS, NU, "gclm", a=a0)
            m = float(np.max(np.abs(wp)))
            if m > 1e3:
                fate = 1
                break
            if m < 0.02 * np.max(np.abs(w)):
                fate = 0
                break
        fates.append(fate)
    print(f"edge check (+/- unstable nudge): fates {fates} "
          f"(1=blow 0=decay; edge state <=> one of each)")

    # continuation in a, both directions
    branch = [{"a": a0, "sup": float(np.max(np.abs(w))), "res": res, "c": c,
               "lam_u": lu, "lam_2": l2}]
    for direction, targets in [
        (+1, list(np.arange(0.935, 0.9701, 0.005)) +
             list(np.arange(0.972, 0.99001, 0.002))),
        (-1, [0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]),
    ]:
        wc, cc = w.copy(), c
        for at in targets:
            wn, resn, cn = newton(wc, float(at), wc, c0=cc)
            if wn is None:
                print(f"  branch {'up' if direction>0 else 'down'} stopped at "
                      f"a={at:.4f} (res {resn:.2e})")
                break
            lu, l2, _ = eigs(wn, float(at), cn)
            sup = float(np.max(np.abs(wn)))
            branch.append({"a": float(at), "sup": sup, "res": resn, "c": cn,
                           "lam_u": lu, "lam_2": l2})
            print(f"  a={at:.4f} sup={sup:9.3f} c={cn:8.4f} "
                  f"lam_u={lu:8.4f} lam_2={l2:8.4f}")
            wc, cc = wn, cn
            if sup > 2000:
                print("  amplitude runaway -- stopping this direction")
                break

    branch.sort(key=lambda r: r["a"])
    out = pathlib.Path(__file__).parent / "runs" / "rung1_branch.json"
    out.write_text(json.dumps({"nu": NU, "N": N, "edge_fates": fates,
                               "branch": branch}, indent=2))

    # divergence vs fold diagnosis + fit on the upper branch
    ups = [r for r in branch if r["a"] >= 0.9]
    if len(ups) >= 4:
        aa = np.array([r["a"] for r in ups])
        ss = np.array([r["sup"] for r in ups])
        best = None
        for ac in np.arange(max(aa) + 0.0005, 1.02, 0.0005):
            x, y = np.log(ac - aa), np.log(ss)
            A_ = np.vstack([x, np.ones_like(x)]).T
            coef, resid, *_ = np.linalg.lstsq(A_, y, rcond=None)
            r = float(resid[0]) if len(resid) else 0.0
            if best is None or r < best[0]:
                best = (r, float(ac), float(-coef[0]), float(np.exp(coef[1])))
        _, ac, p, c = best
        print(f"\nbranch fit sup(a) = C/(a_c - a)^p : a_c={ac:.4f} p={p:.3f} "
              f"C={c:.3f}   (fate-boundary fit gave a_c=0.9825, gamma=0.550)")
    print(f"branch saved: {out}")


if __name__ == "__main__":
    main()
