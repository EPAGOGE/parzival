#!/usr/bin/env python3
"""FRESH-PASS FAULT REGISTRY -- six independent critiques, grouped by agreement.

METHOD. Six agents were given an identical, conclusion-free description of the
axisymmetric Euler experiment and asked what is WRONG with it. None could read
any file in this repository, search the web, or see any result the campaign had
produced. They shared no context with each other or with the author.

WHY THE INDEPENDENCE IS THE POINT. Sequential reasoning stands on its own
previous step, so its errors are CORRELATED: a wrong premise at step 1 is still
wrong at step 40 and consistency feels like confirmation. Independent samples
cannot make a correlated error, so agreement between them carries information
that self-consistency never does. On 2026-07-30 this pass found in four minutes
several faults a full day of careful sequential work had missed, and invalidated
two results the author had reported within the preceding hour.

READING THE COLUMN. `agree` is how many of six raised the fault unprompted.
6/6 on a technical fault, from samples that cannot have coordinated, is the
strongest evidence available here short of a computation. `ours` marks faults
the campaign had ALSO found independently by its own work -- those are mutual
corroboration, not novelty. `NEW` marks what only this pass caught.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fault:
    fid: str
    agree: int                # of 6
    severity: str
    statement: str
    mechanism: str
    ours: bool = False        # campaign found this independently too
    invalidates: list[str] = field(default_factory=list)
    fix: str = ""


FAULTS = [
    # ---------------- unanimous ----------------
    Fault("F1", 6, "HIGH",
          "the inner boundary at r=0.4 is a fabricated wall",
          "psi1=0 there has no physical counterpart, and psi is a GLOBAL elliptic "
          "object, so the invented condition reshapes the poloidal eddy and "
          "rescales the corner strain by O(1). Two agents added that r=0 is a "
          "REMOVABLE coordinate singularity under standard pole conditions, so the "
          "excision buys nothing and silently changes the problem.",
          fix="solve to r=0 with pole conditions, or demonstrate the r0-sensitivity "
              "is below the reported precision by sweeping r0"),
    Fault("F2", 6, "HIGH",
          "the Casimir trust gate is a volume integral, blind to a vanishing-volume peak",
          "the singular region's volume fraction tends to zero, so its error cannot "
          "move an integrated invariant. Measured on this campaign's own data: "
          "78-88% of records passed the Casimir gate while spectrally unresolved; "
          "loc512 reached a spectral tail of 0.449 at cas_drift 1.0e-3, and on "
          "mpi1024 the gate NEVER tripped while the tail hit 4.3e-2.",
          ours=True,
          invalidates=["every trust window quoted before 2026-07-30"],
          fix="gate on the spectral tail fraction, ALREADY COMPUTED and recorded as "
              "tail_u1/tail_w1 in every stream file and never used as the criterion"),
    Fault("F3", 6, "HIGH",
          "the spectral filter is an uncontrolled free knob",
          "any filter is a dissipation the equations do not contain; it caps peak "
          "growth and can manufacture a clean power law whose slope is the cutoff's. "
          "Filtered and unfiltered runs are different PDEs and cannot be pooled into "
          "one convergence study.",
          fix="filter-strength convergence study, or report filtered and unfiltered "
              "ladders separately"),
    Fault("F4", 6, "HIGH",
          "T* sits inside its own regressor, so the exponents are not identified",
          "T* and the slope are near-perfectly anticorrelated: a family of (T*, "
          "exponent) pairs all give R^2 ~ 1, and serially correlated residuals make "
          "the quoted error bars fiction. The least-resolved final points carry the "
          "largest lever arm.",
          ours=True,
          fix="T*-free derivative estimator: 1/(dln S/dt) is LINEAR in t with slope "
              "1/exponent. The campaign switched to exactly this on the same day."),
    Fault("F5", 6, "HIGH",
          "BKM is never formed and omega1 is not the vorticity",
          "omega1 = omega^theta/r ignores omega^r = -r dz(u1) and omega^z = 2u1 + "
          "r dr(u1). BKM requires the divergence of the time integral of ||omega||_inf "
          "over the FULL vector; growth of a rescaled component cannot distinguish "
          "finite-time blowup from double-exponential but finite growth.",
          fix="accumulate the BKM integral on the full |omega|"),
    Fault("F6", 6, "HIGH",
          "HWHM is grid-quantised with no subgrid interpolation and saturates smoothly",
          "once the peak spans a few points the half-max crossing is a property of "
          "the mesh, and it bends the log-log slope smoothly rather than visibly "
          "breaking it. On a Chebyshev grid the quantisation itself varies by orders "
          "of magnitude as the peak migrates.",
          ours=True,
          fix="subgrid interpolation plus a hard points-per-HWHM gate"),
    Fault("F7", 6, "HIGH",
          "the Navier-Stokes clause is unreachable from what is measured",
          "inviscid, slip wall, no Reynolds number anywhere. The local comparison "
          "that decides whether the collapse survives viscosity is not computable "
          "from the recorded quantities.",
          ours=True,
          fix="the campaign's own C10/C12 derivation answers this analytically "
              "(c_l = -1/alpha, Re_loc ~ (T-t)^4.84 -> 0) rather than by march"),

    # ---------------- new, and damaging ----------------
    Fault("F8", 4, "HIGH",
          "the z and r meshes differ by ~3 orders of magnitude at the collapse point",
          "dz = L/Nz ~ 1.63e-4 uniform, against Chebyshev dr_min ~ 1.6e-7 at the "
          "wall, so l_z hits the mesh floor thousands of times earlier than l_r. "
          "Because the ladder refines at FIXED Nz:Nr = 1:3, every resolution fails "
          "first in the same direction at the same structural scale, so run-to-run "
          "differences stay small AND READ AS GRID CONVERGENCE while all runs carry "
          "the same error.",
          invalidates=["the l_z ladder -1.1637..-1.2017 with steps shrinking to "
                       "0.001, reported as converged",
                       "the aspect-ratio test of the collision hypothesis"],
          fix="refine Nz and Nr independently; a two-dimensional (Nz, Nr) surface, "
              "not a single ray"),
    Fault("F9", 4, "HIGH",
          "single-ray refinement cannot distinguish converged from equally under-resolved",
          "refining both directions together means a grid artefact converges exactly "
          "as smoothly as a physical answer. Agreement across the 128x384 to "
          "1024x3072 ladder is therefore not evidence of convergence.",
          invalidates=["every convergence claim made from this ladder"],
          fix="off-ray runs, e.g. 256x3072 and 1024x768"),
    Fault("F10", 2, "HIGH",
          "Gamma = r^2 u1 is the exactly transported quantity, not u1",
          "sup|Gamma| is EXACTLY constant in time, a free pointwise check that bites "
          "locally where a volume integral cannot. The campaign validated "
          "conservation on sup|u1| to 0.83% -- on the wrong field.",
          invalidates=["the C5b free-residual validation"],
          fix="track sup|r^2 u1|; it is one line and it is exact"),
    Fault("F11", 3, "MEDIUM",
          "the odd-in-z symmetry is never projected out",
          "roundoff seeds the even-in-z modes the exact solution forbids, and those "
          "are precisely the TRANSLATION modes that let the peak drift in z. This is "
          "a mechanism for the argmax wandering that has recurred across this "
          "campaign in two separate observables.",
          ours=True,
          fix="project the odd-in-z component each step"),
    Fault("F12", 1, "HIGH",
          "the CFL bounds the advective timescale, not the vortex-stretching timescale",
          "velocities stay O(10^2) because Gamma is transported, so CFL gives "
          "dt ~ 1.6e-6, while accuracy at sup|omega| ~ 1e6 needs dt <~ 1e-8. The "
          "scheme stays marginally stable and accumulates smoothly wrong phase into "
          "T*. This campaign has NEVER run a temporal refinement study.",
          fix="a dt-refinement ladder at fixed grid"),
    Fault("F13", 1, "HIGH",
          "growth factors are quoted on a field that starts at exactly zero",
          "omega1 == 0 at t=0 by construction in every IC used, while the actual "
          "||omega||_inf ~ 6.2e3 already, since swirl supplies omega^r and omega^z. "
          "Every 'sup|omega1| grew xN' figure is therefore measured on the wrong "
          "field against a null baseline.",
          invalidates=["the first collision-sweep table, x3198 / x1026 / x705"],
          fix="quote ||omega||_inf on the full vector"),
    Fault("F14", 3, "MEDIUM",
          "the module docstring contradicts the code's own initial condition",
          "docstring line 18 states u1 = A exp(-30(1-r^2)) sin(2 pi z/L) while "
          "IC_POWER = 4.0 is the default and its own comment says 'PNAS eq 3a is the "
          "FOURTH power'. Three agents flagged the linear form as non-standard, "
          "reasoning from the stale docstring. The code is right; the documentation "
          "has been wrong for the whole campaign.",
          fix="correct the docstring to exp(-30(1-r^2)^4)"),
]


def report() -> None:
    print("=" * 79)
    print("FRESH-PASS FAULT REGISTRY -- 6 independent critiques, no shared context")
    print("=" * 79)
    print(f"{'id':<5}{'agree':>6}{'sev':>7}  {'src':<5} statement")
    print("-" * 79)
    for f in sorted(FAULTS, key=lambda x: (-x.agree, x.fid)):
        src = "both" if f.ours else "NEW"
        print(f"{f.fid:<5}{str(f.agree)+'/6':>6}{f.severity:>7}  {src:<5} {f.statement}")
    inval = [f for f in FAULTS if f.invalidates]
    print("\n" + "-" * 79)
    print("RESULTS INVALIDATED BY THIS PASS")
    print("-" * 79)
    for f in inval:
        for x in f.invalidates:
            print(f"  [{f.fid}] {x}")
    print("\n" + "-" * 79)
    print(f"unanimous (6/6): {sum(1 for f in FAULTS if f.agree == 6)}   "
          f"already ours: {sum(1 for f in FAULTS if f.ours)}   "
          f"NEW to this pass: {sum(1 for f in FAULTS if not f.ours)}")
    print("\nThe unanimous seven are not a ranking of severity, they are a ranking of")
    print("DETECTABILITY: faults obvious enough that six blind samples all find them.")
    print("The single-agent faults F12 and F13 are rated HIGH and were found ONCE.")
    print("Rarity is not weakness here -- it is the tail the majority vote would")
    print("have discarded, and F13 invalidated a table published minutes earlier.")
    print("=" * 79)


if __name__ == "__main__":
    report()
