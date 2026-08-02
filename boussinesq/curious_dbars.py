#!/usr/bin/env python3
"""CURIOUS track — W2 kill test (UNGRADED; see CURIOUS.md).

K1-R1 forces |D| = (w-1)/w < 1 for any strict two-scale power-law collapse,
where D = dln(aspect)/dln|omega|.  The 07-30 battery quoted generic |D| ~ 1.01
WITHOUT per-run error bars.  This script puts a moving-block-bootstrap CI on D
for every run whose full-field snapshots survive on disk, and reads each CI
against the admissibility boundary |D| = 1.

Estimator: identical to the 07-30 battery (features.extract HWHM scales,
OLS slope of ln(aspect) vs ln A) so the numbers are commensurable.  The
cross-estimator (integral-width) repeat is the recorded follow-up, not this.

DATA GAP (on the EJA ledger): the generic W-seed snapshot dirs were deleted;
only scalar streams survive.  Tags here are the SURVIVING sets; the decisive
generic verdict needs W-seed re-runs (staged separately).
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, '.')
from features import extract

TAGS_DEFAULT = ["OR_z256r768", "OR_z256r384", "OR_z128r768", "OR_z128r384",
                "G5_11", "NUL1e-3", "NUL1e-4", "N2_1e-3", "N2_1e-4",
                "SYMa", "SYMb", "loc1024", "mpi1024"]
TAGS = sys.argv[1:] or TAGS_DEFAULT

N_BOOT = 2000
RNG = np.random.default_rng(7)


def slope(lnA: np.ndarray, lnasp: np.ndarray) -> float:
    return float(np.polyfit(lnA, lnasp, 1)[0])


def block_bootstrap_ci(lnA: np.ndarray, lnasp: np.ndarray,
                       nboot: int = N_BOOT) -> tuple[float, float]:
    """Moving-block bootstrap (block ~ n^(1/3)) for an autocorrelated series."""
    n = len(lnA)
    blk = max(3, int(round(n ** (1.0 / 3.0))))
    starts = n - blk + 1
    nblocks = int(np.ceil(n / blk))
    sl = np.empty(nboot)
    for i in range(nboot):
        idx = np.concatenate([np.arange(s, s + blk)
                              for s in RNG.integers(0, starts, nblocks)])[:n]
        sl[i] = slope(lnA[idx], lnasp[idx])
    return float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))


def main() -> None:
    out = []
    out.append("W2 KILL TEST: per-run |D| CIs vs the K1-R1 admissibility "
               "boundary |D| < 1")
    out.append("=" * 78)
    out.append(f"{'run':<14}{'n':>4}{'span':>7}{'D':>9}{'CI95':>20}"
               f"{'|D|<1?':>8}  verdict under K1-R1")
    out.append("-" * 78)
    for tag in TAGS:
        try:
            rows = extract(tag)
        except Exception as e:
            out.append(f"{tag:<14}  EXTRACT FAILED: {e}")
            continue
        if len(rows) < 8:
            out.append(f"{tag:<14}{len(rows):>4}   too few gated snapshots")
            continue
        lnA = np.log(np.array([x["A"] for x in rows]))
        lnasp = np.log(np.array([x["aspect"] for x in rows]))
        span = float(lnA[-1] - lnA[0])
        d = slope(lnA, lnasp)
        lo, hi = block_bootstrap_ci(lnA, lnasp)
        if lo >= 1.0 or hi <= -1.0:
            verdict = "FORBIDDEN as strict two-scale"
            adm = "no"
        elif max(abs(lo), abs(hi)) < 1.0:
            verdict = f"admissible; R1 predicts w~{1/(1-min(abs(d),0.999)):.2f}"
            adm = "yes"
        elif lo < -1.0 and hi > 1.0:
            verdict = "UNINFORMATIVE (CI spans both boundaries)"
            adm = "n/a"
        else:
            verdict = "CI straddles the boundary"
            adm = "strad"
        out.append(f"{tag:<14}{len(rows):>4}{span:>7.2f}{d:>9.3f}"
                   f"[{lo:>8.3f},{hi:>7.3f}]{adm:>8}  {verdict}")
    out.append("-" * 78)
    out.append("Reading guide: D is the OLS aspect-drift slope (HWHM estimator,")
    out.append("same as the 07-30 battery). K1-R1: strict two-scale power law")
    out.append("needs |D| < 1 and then forces w = 1/(1-|D|). A CI wholly at or")
    out.append("above 1 refuses the two-exponent picture for that run; a CI")
    out.append("wholly below 1 with w-prediction far from banked w=1 is the W3")
    out.append("tension instead. All UNGRADED; S6 caveat applies (short spans")
    out.append("sit inside the transient; D inherits that limitation).")
    text = "\n".join(out) + "\n"
    with open("CURIOUS_DBARS.out", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
