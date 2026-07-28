# The degeneracy lattice of 3D axisymmetric Euler with swirl at a wall

> ### !! CORRECTION 2026-07-25 -- the (6,1) DYNAMICS numbers below are VOID
> `IC_POWER` defaulted to **1.0** while PNAS eq 3a is the **FOURTH** power. The
> (2,1)/(6,5) ladders were run with `--ic-power 4`; the (6,1) ladder (`ax_Z3W1_*`)
> and the first discriminator matrix (`disc_D*`) were **not**, so they used a
> different radial initial profile. Every (6,1)-vs-(6,5) dynamical comparison in
> the second half of this document is therefore **confounded and withdrawn**:
> t_s = 0.00506, the "grid-converged 0.78% / 0.03%" ladder, and the
> equal-fractional-distance table. Being re-run as `E_*` with `--ic-power 4`
> explicit.
>
> **UNAFFECTED** (and this is most of the document): the degeneracy forcing law,
> its 7 measurements, and the s = 2 (mod 4) result. `ic_power` multiplies a purely
> RADIAL factor `exp(-30 (1-r^2)^p)` and the law is a statement about z-structure at
> fixed r, so it cannot depend on `ic_power` -- and every zorder measurement used
> one consistent value anyway. The (2,1)-vs-(6,5) comparison is also unaffected:
> both used `--ic-power 4`.
>
> Root cause: the run JSON did not record its own IC. Fixed -- runs now write an
> `ic` block (ic_power, zpow, wamp, wpow, A, r0, lattice orders), and the module
> default is now the paper-faithful 4.0. **An artifact that cannot say what IC
> produced it is not a result, it is a number.**

Measured 2026-07-25 with `dedalus_axisym.py` (Hou-Li variables, annulus r in [0.4,1],
z-periodic, Luo-Hou corner geometry). Everything below is a measurement or a
consequence of one, not a citation.

## Why this question exists

Liu (Caltech thesis 2017, Sec 3.5) and Chen-Huang-Li (arXiv:2604.01868) both explore
*degenerate* initial data for 2D Boussinesq: instead of the Luo-Hou profile they take
data vanishing to higher order at the symmetry point, e.g.

    Liu, Boussinesq:  w     = sin^3(pi x1) (1 - x2)^3            -> vorticity order 3
    CHL, Boussinesq:  Omega = 10 x1^9 / (x1^10 + x2^10 + 16)     -> vorticity order 9

Liu read the s >= 4 family as a distinct blowup class; CHL have since overturned that
reading. The natural transfer question is what the axisymmetric analogue does, since
the Boussinesq-axisym analogue is the reason anyone trusts 2D results as 3D evidence.

## The two order parameters

Write q = ord_z u1 and p = ord_z omega1, orders taken at z = 0 on the wall ring
r -> 1. Both fields are odd about z = 0, so q and p are odd.

theta is the Boussinesq analogue of u1^2, so the "s" of the Boussinesq literature is

    s = ord_z theta = 2q.

Odd parity of u1 forces q odd, hence **s in {2, 6, 10, ...}: s = 2 (mod 4), and
s = 4 is forbidden.** Liu's s = 4 Boussinesq family therefore has no axisymmetric
preimage at all. This was the first sign that the analogue is not faithful at the
level of degeneracy classes.

## The forcing law (measured, then derived)

omega1 obeys (PNAS 2b)

    d_t omega1 + u . grad omega1 = d_z(u1^2).

The source has ord_z d_z(u1^2) = 2q - 1. For the advection term, work the parity out:
u1 odd and omega1 odd make psi1 odd (elliptic equation, odd RHS), so

    u^r = -r d_z psi1        is EVEN in z
    u^z = 2 psi1 + r d_r psi1 is ODD  in z

hence

    u^r d_r omega1 :  even x odd  -> order p
    u^z d_z omega1 :  odd  x even -> order >= 1 + (p-1) = p

Both advection terms carry order >= p, so advection can never *lower* the order. Only
the source can, and only as far as 2q - 1. Therefore

    **ord_z omega1(t > 0) = min( ord_z omega1(0), 2q - 1 ).**

The lattice is two-dimensional but ONE-SIDED: an initial omega1 can be made *less*
degenerate than the forced value and that choice persists; it can never be made
*more* degenerate, because the source instantly regenerates order 2q - 1 underneath
it. Amplitude is irrelevant to this - near z = 0 the lower power wins whatever the
coefficient.

### Verification

Leading z-power of omega1 on the wall ring after 40 steps (t = 8e-5), 256x192,
log-log fit over z/L in [0.004, 0.06]:

| initial data                          | predicted | measured   | R^2     |
|---------------------------------------|-----------|------------|---------|
| q=1, no injection                     | z^1       | z^+0.957   | 0.99946 |
| q=3, no injection                     | z^5       | z^+4.914   | 0.99992 |
| q=5, no injection                     | z^9       | z^+8.871   | 0.99995 |
| q=1, inject omega1 ~ sin^3            | z^1       | z^+1.056   | 0.99940 |
| q=1, inject omega1 ~ sin^9            | z^1       | z^+0.957   | 0.99947 |
| q=3, inject omega1 ~ sin^9  (p0 > 2q-1) | z^5     | z^+4.918   | 0.99993 |
| q=3, inject omega1 ~ sin^1  (p0 < 2q-1) | z^1     | z^+1.001   | 0.99998 |

q = 5 was an out-of-sample prediction: the law was stated from q = 1 and q = 3 only.
G5 z-parity holds to 3-5e-15 on every injected case, so none of this leaks even
parity error.

## What this corrected

I had recorded, after the s = 6 ladder, the caveat *"we only degenerated half the
data - our IC changes u1 (hence theta) but keeps omega1 = psi1 = 0, whereas Liu and
CHL degenerate the vorticity too."* **That caveat was wrong.** With q = 3 the source
d_z(u1^2) ~ sin^5 cos generates omega1 at order 5 within the first few steps
(measured z^4.914). The s = 6 ladder was already the both-fields-degenerate run, and
at vorticity order 5 it was *more* degenerate than Liu's cubic.

So the correct reading of the s = 6 result is stronger than the one I gave: with
theta at order 6 AND omega1 at order 5, the theory-forced law omega1 ~ (t_s - t)^-1
still held at R^2 = 0.9993 with t_s resolution-independent (0.003824 vs 0.003825 at
256 vs 512) and t_s within 0.7% of the s = 2 value.

## Structural consequence

In Boussinesq, theta and omega are independent fields and their degeneracies are
independently choosable - a genuinely 2D family. In axisymmetric Hou-Li variables
theta is tied to u1 and omega1 is *driven by* d_z(u1^2), so of the two order
parameters only q is free downward-unbounded; p is pinned to min(p0, 2q-1). The
axisymmetric system has strictly fewer degeneracy degrees of freedom than its
Boussinesq analogue, and one whole Boussinesq class (s = 4) has no preimage.

**The analogue is not faithful at the level of degeneracy classes.** Conclusions
about degenerate Boussinesq families do not transfer to axisymmetric Euler as such,
in either direction.

## Status of the lattice

| (ord theta, ord omega1) | how reached                | result |
|-------------------------|----------------------------|--------|
| (2, 1)                  | Luo-Hou, zpow=1            | baseline: forced law holds, t_s = 0.003571 |
| (6, 5)                  | zpow=3                     | forced law holds R^2=0.9993, t_s = 0.003547 (0.7% of baseline) |
| (6, 1)                  | zpow=3 + inject wpow=1     | RUNNING (ax_Z3W1_*.json) |
| (10, 9)                 | zpow=5                     | order confirmed, dynamics not run |

The `--wamp/--wpow` knobs exist only to reach the p < 2q-1 column. They cannot make
omega1 more degenerate and the help text now says so; do not reach for them for that.

## Where this points

The IC-family search over degeneracy is close to exhausted and it has been inert:
every lattice point measured so far lands on the same blowup type with t_s moving
under 1%. That is what a *strongly attracting* self-similar fixed point looks like -
which is exactly the CHL Stage-1/Stage-2 picture and exactly what the Chen-Hou
stability spectrum encodes.

So the live question is no longer "which initial data" but "what is the spectrum of
the linearisation at the fixed point". That is Liu's route, and it is the build
specified in POLAR_SPEC.md - not more IC scanning.

---

## The (6,1) run, and a comparison error worth not repeating

Reaching (6,1) needs `--zpow 3 --wamp 100 --wpow 1`: swirl at order 3 (theta order
6) with an injected order-1 vorticity that the source cannot regenerate underneath.
The lattice label is stable under the dynamics - working the orders through
`d_t u1 + u.grad u1 = 2 u1 d_z psi1` with u^r even and u^z odd makes every term
order q, so an order-1 omega1 cannot contaminate u1 downward. (`peak_geometry.py`
re-measures both orders at the LAST snapshot to check this rather than trust it.)

Raw numbers at 256x768, tmax = 0.00345:

| lattice | sup|w_phys| end | 128 vs 256 | t_s (last 40%) | window drift | R^2 |
|---------|-----------------|-----------|----------------|--------------|------|
| (2,1)   | 3260            | -         | 0.00388        | 10.6%        | 0.964 |
| (6,5)   | 4177 (4983@512) | 19% apart | 0.00388 @512   | 1.5%         | 0.997 |
| (6,1)   | 2625            | 0.8%      | 0.00506        | 4.3%         | 0.999 |

### Two traps in reading that table

**1. `w1_ratio` is meaningless when `wamp != 0`.** It is
`sup|w1|(end)/sup|w1|(first sample)`, and a nonzero initial omega1 deflates it -
x25.8 at (6,1) against x907 at (6,5) - while the ABSOLUTE `sup_wphys_trust_end` is
the same order. Compare `sup_wphys_trust_end`, never `w1_ratio`, across different
`wamp`.

**2. t_s is NOT a class invariant, and equal-t is NOT a fair comparison.** t_s moves
continuously with the initial data, so "(6,1) blows up 30% later" says nothing about
whether it is a different singularity. Worse, at t = 0.00345 the runs sit at very
different fractional distances from their own singularities:

    (2,1): (t_s - t)/t_s = 0.11        (6,1): (t_s - t)/t_s = 0.32

(6,1) was being measured three times further out. Its better resolution (spectral
tails 1e-27 / 1e-30 against a climbing sup|w| at (6,5)), its cleaner fit and its
smaller 128-vs-256 spread are ALL what "further from the singularity" looks like.
The parsimonious reading of that table is not "a new blowup" - it is "a run that
has not gone as far". **Compare at equal (t_s - t)/t_s, never at equal t.**
Discriminator runs therefore use tmax = 0.00345 for (2,1)/(6,5) and tmax = 0.00460
for (6,1), putting all of them near 0.10 remaining.

### What actually discriminates

Not t_s and not the temporal exponent - the -1 law is FORCED by the scaling group,
so every run reproduces it by construction and it carries no information about
which fixed point the run is on. The discriminators are:

  - **argmax location**: on the corner ring (z*/L = 0, r* = 1) or off it. Boussinesq
    s=4 drifted off-corner (argmax x/pi = 0.93), which is how we knew it was a
    different STAGE, not a different singularity.
  - **spatial exponent**: peak width `ell(t) ~ (t_s - t)^p`, the genuine eigenvalue.
  - **anisotropy** `ell_z/ell_r` -> constant iff the profile has converged.

`peak_geometry.py` measures all three from checkpoints (`--ckpt-sim-dt` gives
controlled-time snapshots; the old `wall_dt`/`max_writes=2` default gave only two).

Three outcomes, decided by that script and not by t_s:
  A. same location + same p  -> one fixed point, lattice fully collapses, IC search
     closed, go to the spectrum.
  B. different p             -> a SECOND self-similar solution reachable only from
     p < 2q-1 data. Would partially vindicate Liu's degeneracy intuition in axisym
     even though CHL overturned it in Boussinesq.
  C. off-corner argmax       -> (6,1) is still in a transient stage and simply has
     not reached the attractor yet. Given the fractional-distance point above, this
     is the outcome to expect.

### GOTCHA: default run tag collides

`tag = run_id or f"axisym_N{Nz}x{Nr}_A{A:g}"`, so two runs at the same resolution
overwrite each other's `stream_*.jsonl` and `ckpt_*`. The (6,5) 512 stream was lost
this way. **Always pass `--run-id` when the IC differs.**

---

# RE-RUN AT ic_power=4 (PNAS eq 3a) — the real lattice result

`E_E21/E_E65/E_E61P/E_E61N`, 256x768, tmax=0.00345, `--ic-power 4` explicit, each with
18 controlled-time checkpoints. E21 and E65 reproduce `ax_Z1_256`/`ax_Z3_256`
**bit-for-bit** (x1360.2 / x1121.7, identical Casimir drift), which confirms both that
`ic_power` was the entire earlier discrepancy and that the new `--ckpt-sim-dt` knob does
not perturb the physics.

| lattice | run | sup abs w^theta | t_s (last 40%) | window drift | R^2 |
|---|---|---|---|---|---|
| (2,1) | E21  | 3260.1 | 0.003876 | 10.6% | 0.964 |
| (6,5) | E65  | 4176.9 | 0.004021 |  6.6% | 0.987 |
| (6,1) | E61P | 5228.1 | 0.004072 | 13.0% | 0.963 |

**All three t_s agree to ~5%.** The earlier "(6,1) blows up 30% later, and is
grid-converged" was purely the ic_power=1 artefact and is withdrawn. At the
paper-faithful IC, (6,1) has the LARGEST vorticity, not the smallest.

## The sign test was VACUOUS — by an exact symmetry

E61P (`wamp=+100`) and E61N (`wamp=-100`) came back **bit-identical** (5228.14, 435
iterations, same Casimir). Provenance confirmed `wamp:-100.0` parsed correctly, so this
is real, and the reason is a symmetry I should have seen first:

  - `Z`: shift z by half a period. `sin(kz) -> -sin(kz)`, so `(A,W) -> (-A,-W)`. Exact
    (the domain is z-periodic).
  - `S`: `u1 -> -u1` with `omega1, psi1` fixed. Exact -- the u1 equation is homogeneous
    of degree 1 in u1 and the omega1 source is `u1^2`.
  - `S o Z` maps `(A, W) -> (A, -W)`.

So the sign of `wamp` cannot matter for this IC family. The bit-identical output is an
unplanned confirmation that the engine reproduces an exact symmetry of the PDE.
**Do not re-run a wamp sign test.**

## WHAT IS ESTABLISHED, AND WHAT IS NOT

ESTABLISHED — the blowup is on the corner ring in all three lattice points:

    r* -> 1                        (1.00000 ... 0.99782)
    |z* - z_sym| / L  DECREASES monotonically to 0
        E21  0.0313 -> 0.0039      E65  0.0742 -> 0.0156      E61P 0.0508 -> 0.0078

That monotone approach only became visible after fixing the peak-locating code (see
below); it is the clean corner-blowup signature.

**NOT ESTABLISHED — the spatial self-similar exponent, hence "same fixed point?"**
`ell_z ~ (t_s-t)^p` fits give R^2 ~ 0.96-0.97, which is deceptive: the exponent SWINGS
by a factor of two depending on whether 2-4 marginal snapshots are kept.

| run | p, all points | p, unresolved points discarded |
|---|---|---|
| E21  | 1.80 | 0.90 |
| E65  | 1.70 | 1.21 |
| E61P | 1.79 | 1.38 |

Cause: at Nz=256 the z spacing is `L/256 = 6.5e-4`, and `ell_z` reaches `5e-4` -- BELOW
ONE CELL -- by t=0.0034. A width smaller than the grid is an artefact, not a
measurement, and profile collapse cannot mean anything on a 1-cell-wide peak (the
reported spreads 0.44-0.63 and the "DIFFERENT" cross-run verdicts are therefore
UNINFORMATIVE, not evidence). **The lattice's fixed-point identity is OPEN.**

To close it: either Nz >= 1024 in z (r is already fine at Nr=768 -- the structure is
z-thin), or -- better -- the dynamic-rescaling/profile route, which never has to
resolve a collapsing structure at all. That is POLAR_SPEC.md, whose prerequisites are
now all gated.

## THREE DEFECTS FOUND IN `peak_geometry.py` (all fixed) — the pattern is the lesson

1. **Uncentered collapse.** It binned `prof/pk` against raw `z/ell` with no centering on
   `z*`, comparing peaks at different z to each other. Fixed with periodic `wrap()` and
   centering on `z*`.
2. **Fixed-window order fit.** `z_order` used `z/L in [0.004, 0.06]`, which the
   sharpening structure outgrows -- it then measures the TAIL and reports a spurious
   order near 0. That is why "lattice drift" appeared to be real (ord_z u1 -> +0.13).
   Replaced by `z_order_scaled`, window `[0.03, 0.35] * ell`.
3. **No resolution guard.** Fits silently included widths below one grid cell. Added
   `ell_min = 4 cells`, and the run now PRINTS how many snapshots it discarded.

All three produced plausible, high-R^2, wrong numbers. Two of them briefly looked like
physics: (2) looked like the lattice label drifting under the dynamics, and the raw
`z*/L` jitter (0.99 -> 0.02 -> 0.49) looked like argmax migration when it was just
`|omega1|` being odd, so peaking at symmetry-EQUIVALENT z with an arbitrary tie-break.
**A geometric diagnostic on a symmetric field must be written in terms of the symmetry
(distance to the nearest symmetry point, periodic-wrapped), never raw coordinates.**
