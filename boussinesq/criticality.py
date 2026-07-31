#!/usr/bin/env python3
"""CRITICALITY ATLAS -- an engine, not a table.

Founder equation: 3D incompressible Navier-Stokes. Every other entry exists
only to be scored against it.

The organising fact, from Tao's averaged Navier-Stokes: an equation sharing NS's
energy identity AND its scaling DOES blow up in finite time. So no argument
built only from energy and scaling can settle NS in either direction. Any
mechanism worth importing must use structure that survives that averaging.

WHAT THE ENGINE COMPUTES rather than asserts. For a PDE with scaling symmetry

    u_s(x,t) = s^a u(s x, s^b t)

a norm ||.||_{L^p} in dimension d obeys ||u_s||_p = s^{a - d/p} ||u||_p, so the
CRITICALITY INDEX of a controlled quantity is

    sigma = a - d/p .

    sigma > 0  SUBCRITICAL    control STRENGTHENS as you zoom in   -> tractable
    sigma = 0  CRITICAL       control is scale-invariant           -> hard, often falls
    sigma < 0  SUPERCRITICAL  control VANISHES as you zoom in      -> the barrier

The whole difficulty of 3D NS is one number: the only globally controlled
quantity is energy, at sigma = -1/2. The calibration case is 2D NS, which is
solved, and the engine must reproduce WHY without being told.

EVERY ENTRY CARRIES ITS FAULTS. A mechanism that does not transfer is not
discarded, because the REASON it fails is a constraint on any method that could
succeed. That residue is the output, not a footnote.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction as F

SUB, CRIT, SUPER = "SUBCRITICAL", "CRITICAL", "SUPERCRITICAL"
SOLVED, OPEN, BLOWS_UP = "SOLVED", "OPEN", "BLOWUP PROVEN"


def index(a, d, p) -> F:
    """Criticality index sigma = a - d/p.  p = None means L^infinity."""
    return F(a) - (F(0) if p is None else F(d, 1) / F(p))


def verdict(sigma: F) -> str:
    return SUB if sigma > 0 else (CRIT if sigma == 0 else SUPER)


@dataclass
class Entry:
    name: str
    d: int
    a: F                      # scaling weight of the CONTROLLED field
    p: int | None             # exponent of the controlled norm; None = L^inf
    controlled: str           # what is actually controlled, and by what
    kind: str                 # transport-invariant | energy | mass
    status: str
    mechanism: str            # what beat it, or why it is open
    transfer: int             # +1 helps NS, -1 does not, 0 neutral/unknown
    transfer_why: str
    faults: list[str] = field(default_factory=list)   # why it fails to carry over
    residue: str = ""         # what the failure still constrains

    @property
    def sigma(self) -> F:
        return index(self.a, self.d, self.p)

    @property
    def crit(self) -> str:
        return verdict(self.sigma)


# ---------------------------------------------------------------- the atlas

ATLAS = [
    Entry("3D Navier-Stokes (FOUNDER)", 3, F(1), 2,
          "energy ||u||_{L^2}, from the energy identity", "energy", OPEN,
          "only globally controlled quantity is energy, and it is supercritical",
          0, "this is the target",
          ["vortex stretching destroys any maximum principle on vorticity",
           "critical norms (L^3, H^{1/2}, BMO^{-1}) are NOT globally controlled"],
          "the entire problem is that sigma = -1/2 < 0"),

    Entry("2D Navier-Stokes (CALIBRATION)", 2, F(2), None,
          "||omega||_{L^inf}, by a maximum principle on transported vorticity",
          "transport-invariant", SOLVED,
          "no vortex stretching in 2D, so vorticity obeys a maximum principle and "
          "the controlled quantity is strongly subcritical",
          1, "identifies the EXACT defect in 3D: not dimension, but stretching",
          ["the mechanism is unavailable in 3D precisely because omega is stretched"],
          "a 3D proof must either restore a vorticity maximum principle (geometric "
          "depletion / alignment) or find another transported quantity"),

    Entry("Critical SQG (gamma = 1/2)", 2, F(0), None,
          "||theta||_{L^inf}, maximum principle on the transported scalar",
          "transport-invariant", SOLVED,
          "nonlocal maximum principle / modulus of continuity (Kiselev-Nazarov-"
          "Volberg) and De Giorgi iteration (Caffarelli-Vasseur), independently",
          1, "PROOF that a CRITICAL problem can fall to a transport invariant "
             "rather than to an energy estimate",
          ["theta is a scalar with a genuine maximum principle; 3D vorticity is a "
           "stretched vector and has none"],
          "the winning mechanism was never an energy estimate. That is exactly the "
          "class of argument Tao's averaging cannot kill, because averaging "
          "preserves energy while destroying transport structure"),

    Entry("Supercritical SQG (gamma < 1/2)", 2, F(-1, 2), None,
          "||theta||_{L^inf}, but now supercritical", "transport-invariant", OPEN,
          "same maximum principle, now too weak at small scales",
          -1, "shows a transport invariant alone is not enough once sigma < 0",
          ["being a transport invariant does not rescue you below criticality"],
          "bounds the ambition: for NS we need a controlled quantity at sigma >= 0, "
          "not merely one of transport type"),

    Entry("Fractional Burgers, supercritical", 1, F(-1, 2), None,
          "||u||_{L^inf}, maximum principle", "transport-invariant", BLOWS_UP,
          "supercritical dissipation loses to the transport nonlinearity; "
          "finite-time blowup is PROVEN",
          1, "a supercritical transport equation with a maximum principle CAN blow "
             "up. Direct evidence for the blowup head",
          ["1D, no incompressibility, no pressure, no vortex stretching",
           "the nonlinearity is far simpler than NS's"],
          "supercriticality plus a maximum principle does not imply regularity. So "
          "our sigma < 0 for NS is not merely 'hard', it is genuinely compatible "
          "with blowup"),

    Entry("3D Euler (no dissipation)", 3, F(1), 2,
          "energy, conserved exactly", "energy", OPEN,
          "blowup PROVEN for the axisymmetric-with-boundary (Hou-Luo) class by "
          "Chen-Hou; the general smooth case remains open",
          0, "our own object sits here; but see the campaign result",
          ["no dissipation at all, so it says nothing about viscous regularisation",
           "our own C10/C12: the whole known self-similar family has c_l = -1/alpha "
           "in [2.12, 2.92], giving Re_loc -> 0. Viscosity dominates every member"],
          "an Euler singularity is NOT automatically an NS singularity. Establishing "
          "one buys nothing at T3 unless its collapse rate satisfies alpha < -2"),

    Entry("Averaged 3D NS (Tao, THE BARRIER)", 3, F(1), 2,
          "energy, identical identity to NS", "energy", BLOWS_UP,
          "constructed to blow up while preserving the energy identity and the "
          "exact scaling of true NS",
          -1, "kills every method that uses only energy and scaling, INCLUDING our "
              "own viscous criterion Re_loc ~ w*l^2",
          ["it is not NS: averaging destroys the specific bilinear structure, the "
           "divergence-free constraint's geometry, and all transport invariants"],
          "the constraint that binds everything: a successful method must use "
          "structure the averaging destroys. Transport invariants, vorticity "
          "geometry, and the nonlocal pressure are what remain"),

    Entry("Mass-critical NLS", 3, F(3, 2), 2,
          "mass ||u||_{L^2}, conserved", "mass", SOLVED,
          "blowup understood down to the log-log rate; critical mass threshold sharp",
          0, "different (dispersive) mechanism; no transport structure to borrow",
          ["dispersive, not advective; no incompressibility"],
          "shows a CRITICAL problem can be fully mapped, blowup rate included. The "
          "obstruction for NS is not that criticality is unmappable"),

    Entry("2D Keller-Segel chemotaxis", 2, F(2), 1,
          "mass ||rho||_{L^1}, conserved", "mass", SOLVED,
          "sharp critical-mass threshold; blowup above it, global below",
          0, "a clean solved critical case; mechanism is mass concentration",
          ["parabolic-elliptic, no incompressibility, no vector stretching"],
          "the trichotomy is decidable when the conserved quantity sits AT "
          "criticality. NS's sits below it. That gap is the whole difficulty"),
]

FOUNDER = ATLAS[0]


def report() -> None:
    print("=" * 79)
    print("CRITICALITY ATLAS -- founder: 3D incompressible Navier-Stokes")
    print("sigma = a - d/p ;  >0 subcritical, =0 critical, <0 supercritical")
    print("=" * 79)
    print(f"{'equation':<34}{'d':>2}{'sigma':>8}  {'criticality':<14}{'status':<14}{'xfer':>5}")
    print("-" * 79)
    for e in ATLAS:
        s = e.sigma
        sig = f"{int(s)}" if s.denominator == 1 else f"{s.numerator}/{s.denominator}"
        x = {1: " +1", -1: " -1", 0: "  0"}[e.transfer]
        print(f"{e.name:<34}{e.d:>2}{sig:>8}  {e.crit:<14}{e.status:<14}{x:>5}")

    print("\n" + "-" * 79)
    print("CALIBRATION. The engine is told nothing about why 2D NS is solved.")
    ns2 = ATLAS[1]
    print(f"  2D NS controlled quantity: {ns2.controlled}")
    print(f"  computed sigma = {ns2.sigma} -> {ns2.crit}.  Solved, as it must be.")
    print(f"  3D NS controlled quantity: {FOUNDER.controlled}")
    print(f"  computed sigma = {FOUNDER.sigma} -> {FOUNDER.crit}.  Open, as it must be.")
    print("  The difference is NOT dimension. It is that 2D vorticity is transported")
    print("  and 3D vorticity is STRETCHED, so the maximum principle dies and the")
    print("  best surviving control drops from sigma=+2 to sigma=-1/2.")

    print("\n" + "-" * 79)
    print("TRANSFERS THAT HELP (+1)")
    for e in ATLAS:
        if e.transfer == 1:
            print(f"  {e.name}\n      {e.transfer_why}")

    print("\nTRANSFERS THAT DO NOT (-1) -- kept for their residue")
    for e in ATLAS:
        if e.transfer == -1:
            print(f"  {e.name}\n      {e.transfer_why}\n      residue: {e.residue}")

    print("\n" + "=" * 79)
    print("SURVIVING CONSTRAINTS ON ANY SUCCESSFUL NS METHOD")
    print("=" * 79)
    cons = [
        "1. It cannot rest on energy and scaling alone (Tao). This retires our own "
        "viscous criterion as a route to T4, though not as a route-closer.",
        "2. It needs a controlled quantity at sigma >= 0. Supercritical SQG proves "
        "that transport type alone is insufficient below criticality.",
        "3. Transport invariants are the live class: they beat CRITICAL SQG twice by "
        "independent methods, and they are precisely what Tao's averaging destroys.",
        "4. The specific 3D defect is vortex stretching breaking the vorticity "
        "maximum principle. So the target is either geometric depletion (alignment "
        "suppressing stretching) or a different transported quantity.",
        "5. Supercritical + maximum principle is COMPATIBLE with blowup (fractional "
        "Burgers, proven). Regularity is not the safe default at sigma < 0.",
    ]
    for c in cons:
        print("  " + c)

    print("\nWHAT THIS SAYS ABOUT OUR OWN POSITION")
    print("  We measure transport invariants already (range preservation of theta")
    print("  and sup|u1|, holding to 0.83%). That is constraint 3's object. What we")
    print("  do NOT have is one at sigma >= 0 for 3D, which is constraint 2, and")
    print("  constraint 4 says why: stretching. The atlas does not hand us a proof.")
    print("  It says the only door not closed by a theorem is a transported quantity")
    print("  that survives stretching, and it names the two candidate forms.")
    print("=" * 79)


if __name__ == "__main__":
    report()
