#!/usr/bin/env python3
"""EJA TENSION #75 DISCRIMINATING TEST -- rung-3 config at eps_b=1e-4.

The ladder (M1_GATE_LADDER.out) found Hurwitz-ness CONFIG-dependent, not
size-dependent: rung 3 ((16,48,12)/Nb28, n_f=4104) converges to 2.4e-11 with
an exact Lyapunov solve (residual 5.6e-16) but P is INDEFINITE (4 negative
directions), while the SMALLER-n_f production root (Nb36/(16,40,12),
n_f=4760... larger Nb, comparable n_f) at eps_b=1e-4 is cholesky-certified.
Prime suspect: eps_b. The ladder ran entirely at eps_b=1e-3 (the singular
wedge layer xi^{k-2} regime); production certification lives at 1e-4 where
that layer vanishes.

THIS TEST: the EXACT rung-3 config -- edges (0,2,15,25), degs (16,48,12),
Nb=28, fixed alpha=-0.3447, same Newton recipe and tolerances -- with ONLY
eps_b changed 1e-3 -> 1e-4.
  cholesky FLIPS to PASS  => eps_b is the driver (#75 resolves confirmed).
  cholesky still FAILS    => eps_b is NOT the (sole) driver; suspect survives
                             elimination round, #75 stays open, sharper.
  root fails to converge  => fall back ONCE to the production eps_b=1e-4
                             alpha (-0.3447122873723904) -- labeled as a
                             deviation -- because the question is about
                             cholesky(P), not about fixed-alpha root
                             existence; a no-root outcome at -0.3447 alone
                             would leave #75 unanswered.

Reuses the ladder's own functions unmodified (march_s._converge_ladder_root,
march_s._build_lyapunov_at_root, march_s._save_ladder_certificate).
Definiteness by cholesky ONLY (standing refusal); eigvalsh appears only as
the ladder's own diagnostic COUNT after a cholesky FAIL. Certificate npz
written ONLY on PASS ("skip failed rungs, disk is finite").
"""
import json
import pathlib
import time

import numpy as np

from march_s import (_HERE, _build_lyapunov_at_root, _converge_ladder_root,
                     _save_ladder_certificate)

RUNG3_GRID = dict(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 48, 12), Nb=28,
                  eps_b=1e-4)          # ONLY change vs ladder rung 3: 1e-3 -> 1e-4
ALPHA_LADDER = -0.3447                 # the ladder's fixed alpha (protocol)
ALPHA_PROD_1EM4 = -0.3447122873723904  # production eps_b=1e-4 alpha (fallback only)

OUT = _HERE / "EPSB_DISCRIMINATOR.out"
_lines = []


def log(s=""):
    _lines.append(s)
    print(s, flush=True)


def main():
    t_all = time.time()
    log("=" * 78)
    log("EJA #75 DISCRIMINATING TEST -- rung-3 config, eps_b 1e-3 -> 1e-4")
    log("=" * 78)
    log(f"grid={RUNG3_GRID}")
    log("control (from M1_GATE_LADDER.out, NOT rerun): same grid at "
        "eps_b=1e-3, alpha=-0.3447 -> root 2.425e-11, lyap resid 5.559e-16, "
        "cholesky FAIL (4 negative of 4102).")
    log("")

    alpha_used, deviation = ALPHA_LADDER, None
    log(f"ROOT at fixed alpha={ALPHA_LADDER} (ladder protocol, tolerances "
        f"identical: newton_tol=1e-12, steps=120, root_tol=1e-10)")
    root = _converge_ladder_root(RUNG3_GRID, ALPHA_LADDER)
    log(f"  ||F||_max={root['fmax']:.6e}  ||F||_rms={root['rms']:.6e}  "
        f"newton_taken={root['taken']}  wall={root['wall_s']:.1f}s  "
        f"converged(<1e-10)={root['converged']}")

    if not root["converged"]:
        deviation = (f"fixed-alpha root does not exist at {ALPHA_LADDER} for "
                     f"this grid at eps_b=1e-4 (floor {root['fmax']:.3e}); "
                     f"retried ONCE at the production eps_b=1e-4 alpha "
                     f"{ALPHA_PROD_1EM4} -- a labeled protocol deviation, "
                     f"same everything else.")
        log(f"  NOT CONVERGED -- {deviation}")
        log("")
        log(f"ROOT RETRY at fixed alpha={ALPHA_PROD_1EM4}")
        alpha_used = ALPHA_PROD_1EM4
        root = _converge_ladder_root(RUNG3_GRID, ALPHA_PROD_1EM4)
        log(f"  ||F||_max={root['fmax']:.6e}  ||F||_rms={root['rms']:.6e}  "
            f"newton_taken={root['taken']}  wall={root['wall_s']:.1f}s  "
            f"converged(<1e-10)={root['converged']}")
        if not root["converged"]:
            log("")
            log("VERDICT: INCONCLUSIVE -- no root at EITHER fixed alpha on "
                "this grid at eps_b=1e-4; cholesky question unreachable via "
                "the rung-3 config. (Itself a finding: the eps_b=1e-4 "
                "solvable-grid set excludes rung 3 at both candidate "
                "alphas.) #75 stays open.")
            _finish(t_all)
            return

    log("")
    log("LYAPUNOV at the converged root (ladder recipe, unmodified: dense_L "
        "-> kernel_basis -> Lred -> solve_lyapunov(Lred.T,-I) "
        "[Bartels-Stewart]; definiteness by CHOLESKY ONLY)")
    lyap = _build_lyapunov_at_root(root["S"], root["z0"])
    n = lyap["Lred"].shape[0]
    log(f"  dense_L(): {lyap['t_densel']:.1f}s  kernel_basis(): "
        f"{lyap['t_kb']:.1f}s  solve_lyapunov: {lyap['t_lyap']:.1f}s  (n={n})")
    log(f"  ||Lred||_2={lyap['nrmL']:.6e} (SVD-based norm, not an eigenvalue)")
    log(f"  relative residual ||L^T P + P L + I||/(||P|| ||L||) = "
        f"{lyap['residual']:.6e}")
    if lyap["cholesky_ok"]:
        log("  cholesky(P): PASS")
    else:
        log(f"  cholesky(P): FAIL -- P indefinite ({lyap['n_neg_eig']} "
            f"negative eigenvalue(s) of {n}, diagnostic count via eigvalsh, "
            f"not part of the decision).")

    log("")
    log("-" * 78)
    log("DISCRIMINATION TABLE (same grid/degs/Nb/recipe throughout)")
    log("-" * 78)
    log(f"  eps_b=1e-3  alpha=-0.3447          : cholesky FAIL "
        f"(4 neg of 4102)   [M1_GATE_LADDER.out rung 3]")
    tag = "PASS" if lyap["cholesky_ok"] else f"FAIL ({lyap['n_neg_eig']} neg of {n})"
    log(f"  eps_b=1e-4  alpha={alpha_used:<18}: cholesky {tag}   [this run]")
    log("")
    if lyap["cholesky_ok"]:
        log("VERDICT: CHOLESKY FLIPPED. eps_b IS the driver of "
            "Hurwitz-certifiability at this config: the ladder's universal "
            "FAIL was an artifact of sitting at eps_b=1e-3, inside the "
            "singular wedge layer, not a property of resolution. "
            "Config-dependence localizes to the eps_b axis. #75: resolve "
            "confirmed.")
    else:
        log("VERDICT: NO FLIP. cholesky still FAILS at eps_b=1e-4 on the "
            "rung-3 grid -- eps_b is NOT the (sole) driver; the production "
            "certification must be earning its definiteness elsewhere "
            "(degs ratio / Nb / alpha refinement). #75 stays open, suspect "
            "list shortened by one.")
    if deviation:
        log("")
        log(f"PROTOCOL DEVIATION (recorded): {deviation}")

    if lyap["cholesky_ok"]:
        npz_path = str(_HERE / "pnorm_P_Nb28_epsb1em4.npz")
        _save_ladder_certificate(root, lyap, RUNG3_GRID, alpha_used, npz_path)
        log(f"saved certificate (PASS rung discipline): {npz_path}")

    _finish(t_all)


def _finish(t_all):
    log("")
    log(f"total wall: {time.time() - t_all:.1f}s")
    OUT.write_text("\n".join(_lines) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
