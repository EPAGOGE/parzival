#!/usr/bin/env python3
"""FRONTIER MAP -- two attention heads, three tiers.

The Clay problem admits TWO solutions, not one. Either construct a finite-time
singularity for 3D Navier-Stokes from smooth finite-energy data, or prove no
such singularity exists. A program that only hunts blowup scores its own
negative results as failures, which is a scoring error: closing a blowup route
is evidence in the regularity direction and must be credited there.

So every finding carries:

  HEAD    which goal it feeds -- BLOWUP, REGULARITY, or NEITHER (machinery)
  TIER    how far it sits from the Clay object itself
  STATUS  what it would take to move it

TIER LADDER. The distance that actually matters, because a result about a model
object is not a result about Navier-Stokes no matter how precise it is:

  T0  internal machinery: solver correctness, gauge constants, free residuals
  T1  the 2D Boussinesq model object
  T2  3D axisymmetric EULER with boundary  (the Hou-Luo object)
  T3  3D axisymmetric NAVIER-STOKES
  T4  full 3D Navier-Stokes, smooth finite-energy data  (the Clay object)
  T5  a proof, either direction

Nothing here reaches T5. Being explicit about the tier is the point: it stops a
T1 number from being narrated as though it were T4 progress, which is the exact
failure this map exists to prevent.

LITERATURE ENTRIES are recorded from search metadata and abstracts only. Where a
paper has not been read in full, it is marked ABSTRACT_ONLY and must not be
leaned on. Preprint claims of NS blowup appear regularly and mostly do not
survive; a recent arXiv id is not a result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BLOWUP, REGULARITY, MACHINERY = "BLOWUP", "REGULARITY", "MACHINERY"

TIERS = {
    0: "internal machinery (solver, gauge, free residuals)",
    1: "2D Boussinesq model object",
    2: "3D axisymmetric Euler with boundary (Hou-Luo)",
    3: "3D axisymmetric Navier-Stokes",
    4: "full 3D Navier-Stokes, smooth finite-energy data (Clay object)",
    5: "a proof, either direction",
}


@dataclass
class Finding:
    fid: str
    head: str
    tier: int
    statement: str
    status: str               # ESTABLISHED | ABDUCED | ABSTRACT_ONLY | REFUTED
    source: str
    conditional_on: list[str] = field(default_factory=list)
    moves_up_if: str = ""


OURS = [
    Finding("O1", MACHINERY, 0,
            "sup|u1| conserved to 0.83% across 7 runs; range-preservation free "
            "residual holds, so the solver is sound even where the regime is not reached",
            "ESTABLISHED", "mythos C5b"),
    Finding("O2", MACHINERY, 0,
            "the solver's c_l = 3.006498 is a GAUGE constant, not the physical "
            "scaling exponent (2.86% from -1/alpha against a 1.8e-6 residual)",
            "ESTABLISHED", "mythos C10"),
    Finding("O3", MACHINERY, 1,
            "the uniform-grid march is not in the self-similar regime and cannot "
            "measure the scaling exponent by any estimator (factor 5.8 from asymptotic)",
            "ESTABLISHED", "mythos C4/C7/C9"),
    Finding("O4", MACHINERY, 1,
            "alpha_0 = -0.34240 +- 3e-5 for the stable profile, two disjoint methods",
            "ESTABLISHED", "corner-regularised spectral Newton + published neural value"),
    Finding("O5", REGULARITY, 2,
            "far-field matching gives the PHYSICAL exponent c_l = -1/alpha = 2.92056, "
            "hence Re_loc ~ (T-t)^{+4.84} -> 0: viscosity dominates the stable "
            "Hou-Luo profile",
            "ABDUCED", "mythos C10",
            ["A1 c_omega=-1", "A8 far-field matching"],
            "a second disjoint derivation of c_omega = alpha * c_l"),
    Finding("O6", REGULARITY, 3,
            "the whole known branch family accumulates at alpha_inf = -0.4722, so NO "
            "member reaches the alpha < -2 needed to escape viscosity; the direct "
            "self-similar Euler -> NS route is closed for this family",
            "ABDUCED", "mythos C12",
            ["A1", "A8", "A9 published alpha_1..alpha_3"],
            "our own non-network computation of alpha_1"),
    Finding("O7", MACHINERY, 1,
            "'lambda ~ 0.33 reciprocal branch, viscosity subdominant, coheres with NS'",
            "REFUTED", "mythos, refuted same session by C10"),
]

# Recorded from search metadata/abstracts on 2026-07-30. NOT read in full.
LIT = [
    Finding("L1", BLOWUP, 2,
            "unstable self-similar singularities discovered for 3D Euler and related "
            "equations by neural search; source of alpha_1..alpha_3",
            "ABSTRACT_ONLY", "alphaxiv 2509.14185 (DeepMind et al, 2025-09)"),
    Finding("L2", BLOWUP, 4,
            "finite-time blowup for an AVERAGED 3D Navier-Stokes equation, same energy "
            "identity and scaling: the supercriticality barrier. Any method using only "
            "energy and scaling cannot decide the real problem",
            "ABSTRACT_ONLY", "alphaxiv 1402.0290 (Tao, 2015)"),
    Finding("L3", BLOWUP, 3,
            "potentially singular behaviour of 3D Navier-Stokes, axisymmetric with "
            "boundary; the NS continuation of the Hou-Luo programme",
            "ABSTRACT_ONLY", "alphaxiv 2107.06509 (Hou, 2022)"),
    Finding("L4", REGULARITY, 4,
            "concentration -> quantitative regularity: the modern regularity-side "
            "toolkit and its current reach",
            "ABSTRACT_ONLY", "alphaxiv 2211.16215 (survey, 2022)"),
    Finding("L5", BLOWUP, 3,
            "numerically constructed UNSTABLE self-similar solutions of NS in R^3, "
            "axisymmetric and homogeneous, used to argue non-uniqueness. Directly "
            "adjacent to our object: NS-side rather than Euler-side self-similarity",
            "ABSTRACT_ONLY", "alphaxiv 2606.07501 (2026-06)"),
    Finding("L6", BLOWUP, 4,
            "claimed STABLE finite-time singularity for 3D NS on T^3 via a 5D-lifted "
            "analytic-profile programme. If it holds this is a Clay solution, which is "
            "itself reason for caution: unread, unrefereed, near-zero engagement",
            "ABSTRACT_ONLY", "alphaxiv 2604.09949 (2026-04)"),
    Finding("L7", BLOWUP, 2,
            "exact C^{1,alpha} self-similar blowup profiles for 3D Euler WITHOUT swirl, "
            "and asymptotically self-similar blowup built on them",
            "ABSTRACT_ONLY", "alphaxiv 2605.15130 (2026-05)"),
    Finding("L8", BLOWUP, 4,
            "finite-time blowup in a shell model of 3D NS with smooth decaying data",
            "ABSTRACT_ONLY", "alphaxiv 2605.13827 (2026-05)"),
]


def report() -> None:
    print("=" * 78)
    print("FRONTIER MAP -- two heads (BLOWUP / REGULARITY), tiers T0..T5")
    print("=" * 78)
    for t in sorted(TIERS):
        print(f"  T{t}  {TIERS[t]}")

    for label, rows in (("OUR FINDINGS", OURS), ("LITERATURE (metadata/abstract only)", LIT)):
        print(f"\n{'-'*78}\n{label}\n{'-'*78}")
        for f in sorted(rows, key=lambda x: (-x.tier, x.fid)):
            tag = {"ESTABLISHED": "EST ", "ABDUCED": "ABD ",
                   "ABSTRACT_ONLY": "ABS ", "REFUTED": "REF "}[f.status]
            print(f"[{tag}] T{f.tier} {f.head:<10} {f.fid}  {f.statement}")
            print(f"           src: {f.source}")
            if f.conditional_on:
                print(f"           conditional on: {', '.join(f.conditional_on)}")
            if f.moves_up_if:
                print(f"           moves up if: {f.moves_up_if}")
            print()

    live = [f for f in OURS if f.status in ("ESTABLISHED", "ABDUCED")]
    print("=" * 78)
    for head in (BLOWUP, REGULARITY):
        ours = [f for f in live if f.head == head]
        best = max((f.tier for f in ours), default=None)
        lit = max((f.tier for f in LIT if f.head == head), default=None)
        print(f"{head:<11} our highest live tier: "
              f"{'T'+str(best) if best is not None else 'none'}"
              f"   |  literature reaches: {'T'+str(lit) if lit is not None else 'n/a'}")
    print()
    print("HONEST POSITION. Everything we have on the BLOWUP head is machinery or")
    print("refuted; we contribute nothing in that direction. Our live contribution is")
    print("on the REGULARITY head at T3, and it is ABDUCED, resting on three")
    print("single-stream anchors. The literature reaches T4 on both heads, so we are")
    print("one full tier below the frontier on our own side and absent on the other.")
    print()
    print("THE BARRIER THAT BINDS BOTH HEADS. L2 (Tao) shows an averaged NS with the")
    print("same energy identity and scaling DOES blow up. So no argument built only")
    print("from energy and scaling can settle either direction. Our viscous criterion")
    print("Re_loc ~ w*l^2 is exactly such an argument. It is therefore structurally")
    print("incapable of reaching T4, however well it is adjudicated -- it can close")
    print("specific routes (which it did) but never the problem.")
    print("=" * 78)


if __name__ == "__main__":
    report()
