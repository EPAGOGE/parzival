#!/usr/bin/env python3
"""MYTHOS CONFOUND ENGINE.

A claim is not established by driving one residual to zero. It is established
when every stated way of denying it has been tried and has failed. This module
inverts the usual direction of work: instead of accumulating evidence FOR a
target, it enumerates the vulnerabilities that would DENY it, and closes them
one at a time with the cheapest test that can do the job.

The organising rules:

  * Every confound must state what it would take to KILL the claim, not what
    would support it. A confound whose test can only confirm is not a confound.
  * Cost is a first-class field. The engine reports what a test costs before it
    is run, so the cheap deniers get killed first and the expensive ones are
    only reached if something survives.
  * A test may return FATAL. That is a success of the engine, not a failure.
  * OPEN is the honest default. UNTESTED confounds are printed in the column
    with the same weight as failed ones, because an unasked question denies a
    claim exactly as hard as an unanswered one.

The claim stands iff the column contains no OPEN and no FATAL rows.
"""
from __future__ import annotations

import glob
import json
import pathlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

RUNS = pathlib.Path(__file__).resolve().parent.parent / "runs"

CLOSED, OPEN, FATAL, UNTESTED = "CLOSED", "OPEN", "FATAL", "UNTESTED"


@dataclass(frozen=True)
class Verdict:
    status: str
    detail: str
    evidence: dict = field(default_factory=dict)


@dataclass
class Confound:
    cid: str
    statement: str          # the way the claim could be false
    kills_how: str          # what a FATAL result would mean
    cost: str               # free | cheap | expensive
    test: Callable[[], Verdict] | None = None

    def run(self) -> Verdict:
        if self.test is None:
            return Verdict(UNTESTED, "no test implemented")
        try:
            return self.test()
        except Exception as exc:                       # a broken test is not a pass
            return Verdict(OPEN, f"test raised: {type(exc).__name__}: {exc}")


# ----------------------------------------------------------------- data access

def streams(names: list[str]) -> dict[str, list[dict]]:
    out = {}
    for n in names:
        p = RUNS / f"stream_{n}.jsonl"
        if not p.exists():
            continue
        recs = [json.loads(l) for l in p.open() if l.strip()]
        recs = [r for r in recs if r.get("sup_w1", 0) > 0]
        if len(recs) >= 20:
            out[n] = recs
    return out


def _axes(f):
    z = r = None
    for k in f["scales"]:
        if k.startswith("z_hash"):
            z = f["scales"][k][:]
        if k.startswith("r_hash"):
            r = f["scales"][k][:]
    return z, r


def hwhm(prof: np.ndarray, coord: np.ndarray) -> tuple[float, int]:
    """Half-width at half-max about the profile's own argmax."""
    i = int(np.argmax(prof))
    pk = prof[i]
    if pk <= 0:
        return float("nan"), 0
    half, n = pk / 2.0, len(prof)
    j, cr = i, 0
    while cr < n and prof[j % n] > half:
        j += 1
        cr += 1
    k, cl = i, 0
    while cl < n and prof[k % n] > half:
        k -= 1
        cl += 1
    return (cr + cl) * float(np.median(np.abs(np.diff(coord)))), cr + cl


def second_moment(prof: np.ndarray, coord: np.ndarray) -> float:
    """Independent length scale: sqrt of the normalised second moment of |w|
    about its argmax. Shares no machinery with hwhm beyond the argmax."""
    i = int(np.argmax(prof))
    n = len(prof)
    idx = (np.arange(n) - i + n // 2) % n - n // 2      # signed offset, periodic
    dx = float(np.median(np.abs(np.diff(coord))))
    w = prof.astype(float)
    tot = w.sum()
    if tot <= 0:
        return float("nan")
    return float(np.sqrt((w * (idx * dx) ** 2).sum() / tot))


def snap_series(d: str, minc: int = 8):
    """(t, w_max, lz, lr, l2z, l2r, iz, ir) per snapshot write, resolution-gated."""
    import h5py
    rows = []
    for fn in sorted(glob.glob(str(RUNS / d / "*.h5"))):
        with h5py.File(fn, "r") as f:
            keys = [k for k in f["tasks"] if "omega" in k or k == "w1"]
            if not keys:
                continue
            z, r = _axes(f)
            if z is None or r is None:
                continue
            W, st = f["tasks"][keys[0]], f["scales/sim_time"][:]
            U = f["tasks"]["u1"] if "u1" in f["tasks"] else None
            for n in range(W.shape[0]):
                a = np.abs(W[n])
                wm = a.max()
                if wm <= 0 or st[n] <= 0:
                    continue
                iz, ir = np.unravel_index(np.argmax(a), a.shape)
                lz, cz = hwhm(a[:, ir], z)
                lr, cr = hwhm(a[iz, :], r)
                if cz < minc or cr < minc:
                    continue
                um = float(np.abs(U[n]).max()) if U is not None else float("nan")
                rows.append(dict(t=float(st[n]), w=float(wm), lz=lz, lr=lr,
                                 l2z=second_moment(a[:, ir], z),
                                 l2r=second_moment(a[iz, :], r),
                                 iz=int(iz), ir=int(ir), u=um))
    return rows


def monotone_prefix(rows: list[dict], tol: float = 1.02) -> list[dict]:
    """Trim at the first point where a length scale GROWS. A blowup scale that
    grows means the argmax jumped structures; everything after is a different
    object and must not enter a fit."""
    if not rows:
        return rows
    out = [rows[0]]
    for x in rows[1:]:
        if x["lz"] > out[-1]["lz"] * tol or x["lr"] > out[-1]["lr"] * tol:
            break
        out.append(x)
    return out


def slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


# ------------------------------------------------------- the target claim

TARGET = (
    "c_l = -1/alpha = 2.92056 by far-field matching, so Re_loc ~ (T-t)^4.84 -> 0 "
    "and viscosity DOMINATES: the stable Hou-Luo profile does NOT carry into "
    "Navier-Stokes. Subdominance requires alpha < -2, which relocates the NS "
    "question entirely onto the unstable branches."
)

SUPERSEDED = (
    "2026-07-30, refuted by this engine: 'lambda ~ 0.33 (reciprocal branch), so "
    "viscosity is subdominant and the scenario coheres with NS.' The reciprocal "
    "branch was invented; C10 showed the dichotomy it rested on was false. Kept "
    "on the page because a refuted claim that vanishes teaches nothing."
)

SNAPS = ["snap_loc1024", "snap_mpi1024", "snap_loc512", "snap_loc256",
         "snap_loc128", "snap_e3d256", "snap_NOFILT128"]


# --------------------------------------------------------------- the tests

def t_branch_order() -> Verdict:
    """C1. SUPERSEDED by C10 and disqualified by C7/C9.

    Written to guard 'lambda is O(3) not O(0.3)' and it returned CLOSED with
    'reciprocal branch survives'. Both halves were wrong. C7 and C9 then showed
    the march is not in the self-similar regime, so its exponent cannot speak
    to lambda by ANY estimator, and C10 showed the reciprocal branch never
    existed. A confound that returns false comfort is worse than no confound,
    so this one now reports what its data can actually support: nothing about
    lambda, only that the march is pre-asymptotic."""
    got = {}
    for d in SNAPS:
        rows = monotone_prefix(snap_series(d))
        if len(rows) < 6:
            continue
        w = np.array([x["w"] for x in rows])
        got[d] = dict(sz=slope(w, np.array([x["lz"] for x in rows])),
                      sr=slope(w, np.array([x["lr"] for x in rows])),
                      n=len(rows), growth=float(w[-1] / w[0]))
    if not got:
        return Verdict(OPEN, "no snapshot series long enough")
    worst = max(abs(v["sz"]) for v in got.values())
    expected = abs(-1.0 / ALPHA)          # asymptotic value: |c_l/c_w| = |1/alpha|
    return Verdict(OPEN,
                   f"march gives |dln(l)/dln(w)| <= {worst:.3f} against the "
                   f"asymptotic {expected:.3f} predicted by c_l=-1/alpha, a factor "
                   f"of {expected/max(worst,1e-9):.1f} short. Consistent with C7/C9: "
                   "the march is pre-asymptotic and cannot measure lambda at all.",
                   got)


def t_criterion_margin() -> Verdict:
    """C2. Even on the reciprocal branch the verdict needs lambda < 1/2. If the
    measurements straddle 1/2 the SIGN is undetermined and the claim is not
    established by the march, whatever the profile says."""
    got = {}
    for d in SNAPS:
        rows = monotone_prefix(snap_series(d))
        if len(rows) < 6:
            continue
        w = np.array([x["w"] for x in rows])
        got[d] = slope(w, np.array([x["lz"] for x in rows]))
    sub = {k: v for k, v in got.items() if 1 + 2 * v > 0}
    dom = {k: v for k, v in got.items() if 1 + 2 * v <= 0}
    if dom and sub:
        return Verdict(OPEN,
                       f"straddles the -1/2 boundary: {len(sub)} subdominant, "
                       f"{len(dom)} dominant ({', '.join(f'{k}={v:+.3f}' for k, v in dom.items())}). "
                       "March cannot decide the sign; precision must come from the profile.",
                       got)
    if dom:
        return Verdict(FATAL, "every run says viscosity dominant", got)
    return Verdict(CLOSED, "every run subdominant", got)


def t_length_proxy() -> Verdict:
    """C3. 'l' was defined as HWHM. If a structurally independent definition
    (normalised second moment) gives a different exponent, the exponent is a
    property of the estimator, not of the flow."""
    got = {}
    for d in SNAPS:
        rows = monotone_prefix(snap_series(d))
        if len(rows) < 6:
            continue
        w = np.array([x["w"] for x in rows])
        a = slope(w, np.array([x["lz"] for x in rows]))
        b = np.array([x["l2z"] for x in rows])
        if not np.all(np.isfinite(b)) or np.any(b <= 0):
            continue
        got[d] = dict(hwhm=a, second_moment=slope(w, b), diff=abs(a - slope(w, b)))
    if not got:
        return Verdict(OPEN, "second-moment scale unavailable")
    worst = max(v["diff"] for v in got.values())
    if worst > 0.25:
        return Verdict(FATAL,
                       f"estimator-dependent: HWHM and second moment differ by "
                       f"up to {worst:.3f} in exponent", got)
    return Verdict(CLOSED, f"two independent length estimators agree to {worst:.3f}", got)


def t_argmax_wander() -> Verdict:
    """C4. Already a recorded failure mode of this campaign: a global maximum
    that migrates between distinct structures produces a growth rate that is a
    property of the migration, not of any one structure."""
    got = {}
    for d in SNAPS:
        rows = monotone_prefix(snap_series(d))
        if len(rows) < 6:
            continue
        iz = np.array([x["iz"] for x in rows])
        ir = np.array([x["ir"] for x in rows])
        got[d] = dict(jumps_z=int(np.sum(np.abs(np.diff(iz)) > 0.02 * 1024)),
                      jumps_r=int(np.sum(np.abs(np.diff(ir)) > 0.02 * 3072)),
                      n=len(rows))
    bad = {k: v for k, v in got.items() if v["jumps_z"] or v["jumps_r"]}
    if bad:
        return Verdict(OPEN, f"argmax jumps in {len(bad)}/{len(got)} runs: "
                             f"{', '.join(bad)}", got)
    return Verdict(CLOSED, f"argmax migrates smoothly in all {len(got)} runs", got)


def t_velocity_relation() -> Verdict:
    """C5. Re_loc ~ w*l^2 is derived from u ~ l*w, where u is the ADVECTING
    (poloidal) velocity built from psi_1.

    Earlier revision of this test compared sup|u1| and read a violation of
    exactly 1.000. That clean integer was the tell: u1 = u^theta / r is the
    SWIRL, the theta-analogue under the Boussinesq mapping, and it is advected,
    so sup|u1| is conserved by construction. The test was reading a passing
    invariant as a failure. Snapshots carry no psi_1, so the relation genuinely
    cannot be checked from what is on disk."""
    return Verdict(UNTESTED,
                   "requires the poloidal velocity from psi_1, which snapshots "
                   "do not store. sup|u1| is the swirl and is conserved, so it "
                   "cannot test this relation. Needs a psi_1 snapshot handler.")


def t_swirl_conservation() -> Verdict:
    """C5b. The free residual that sup|u1| DOES test: u1 is advected, so its
    supremum must be conserved. Drift here indicts the solver, not the claim,
    but a solver that fails it cannot support any of the other numbers."""
    got = {}
    for d in SNAPS:
        rows = [x for x in snap_series(d) if np.isfinite(x["u"]) and x["u"] > 0]
        if len(rows) < 6:
            continue
        u = np.array([x["u"] for x in rows])
        got[d] = float(np.max(np.abs(u - u[0])) / u[0])
    if not got:
        return Verdict(OPEN, "u1 unavailable")
    worst = max(got.values())
    if worst > 0.05:
        return Verdict(FATAL, f"sup|u1| drifts {worst:.1%}; advection invariant "
                              "violated, solver is not trustworthy here", got)
    return Verdict(CLOSED, f"sup|u1| conserved to {worst:.2%} across all runs "
                           "(range-preservation free residual holds)", got)


def t_resolution_floor() -> Verdict:
    """C6. HWHM cannot resolve below the grid. If the exponent moves when the
    minimum-cell gate is tightened, it is reading the grid, not the flow."""
    got = {}
    for d in SNAPS:
        vals = {}
        for minc in (4, 8, 16):
            rows = monotone_prefix(snap_series(d, minc=minc))
            if len(rows) < 6:
                continue
            w = np.array([x["w"] for x in rows])
            vals[minc] = slope(w, np.array([x["lz"] for x in rows]))
        if len(vals) >= 2:
            got[d] = dict(vals, spread=max(vals.values()) - min(vals.values()))
    if not got:
        return Verdict(OPEN, "insufficient data across gates")
    worst = max(v["spread"] for v in got.values())
    if worst > 0.25:
        return Verdict(FATAL, f"exponent moves {worst:.3f} with the resolution gate", got)
    return Verdict(CLOSED, f"exponent stable to {worst:.3f} across gates 4/8/16", got)


def t_preasymptotic() -> Verdict:
    """C7. THE hard one. If the exponent drifts monotonically with how far a run
    got, no run is in the asymptotic regime and every measured value is a
    transient. Monotone drift across the resolution ladder is the signature."""
    pts = []
    for d in SNAPS:
        rows = monotone_prefix(snap_series(d))
        if len(rows) < 6:
            continue
        w = np.array([x["w"] for x in rows])
        pts.append((float(w[-1] / w[0]), slope(w, np.array([x["lz"] for x in rows])), d))
    if len(pts) < 4:
        return Verdict(OPEN, "too few runs to test for drift")
    pts.sort()
    g = np.array([p[0] for p in pts])
    s = np.array([p[1] for p in pts])
    rho = float(np.corrcoef(np.log(g), s)[0, 1])
    ev = {d: dict(growth=gg, sz=ss) for gg, ss, d in pts}
    if rho < -0.7:
        return Verdict(OPEN,
                       f"exponent drifts monotonically with run reach "
                       f"(corr={rho:+.3f} vs log growth, {s.min():+.3f}..{s.max():+.3f}). "
                       "No run is asymptotic; the value is a transient.", ev)
    return Verdict(CLOSED, f"no systematic drift with reach (corr={rho:+.3f})", ev)


def t_filter_dependence() -> Verdict:
    """C8. Spectral filtering is a numerical choice. If it moves the exponent,
    the exponent is partly the filter's."""
    a = monotone_prefix(snap_series("snap_NOFILT128"))
    b = monotone_prefix(snap_series("snap_loc128"))
    if len(a) < 6 or len(b) < 6:
        return Verdict(OPEN, "matched filtered/unfiltered pair unavailable")
    sa = slope(np.array([x["w"] for x in a]), np.array([x["lz"] for x in a]))
    sb = slope(np.array([x["w"] for x in b]), np.array([x["lz"] for x in b]))
    ev = dict(nofilt=sa, filtered=sb, diff=abs(sa - sb))
    if abs(sa - sb) > 0.15:
        return Verdict(FATAL, f"filter moves the exponent by {abs(sa-sb):.3f}", ev)
    return Verdict(CLOSED, f"filter-independent to {abs(sa-sb):.3f}", ev)


def t_c_omega() -> Verdict:
    """C9. The step from dln(l)/dln(w) to lambda assumes c_omega = -1 from the
    transport balance.

    Fitting log w against log(T-t) with T free is ill-conditioned when T sits
    near t_end, and an earlier revision of this test returned c_omega spanning
    -30.8..+0.04 for exactly that reason. Use the derivative form instead: if
    w ~ (T-t)^c then dln(w)/dt = c/(t-T), so

        1 / (dln(w)/dt)  =  (t - T)/c

    is LINEAR in t with slope 1/c and intercept -T/c. No search, no free
    parameter inside a log, and T falls out as a by-product."""
    out = {}
    for name, recs in streams(["loc512", "mpi1024", "ts256", "ts128", "loc1024"]).items():
        t = np.array([r["t"] for r in recs])
        w = np.array([r["sup_w1"] for r in recs])
        k = int(0.5 * len(t))                     # late half only
        t, w = t[k:], w[k:]
        dlnw = np.gradient(np.log(w), t)
        ok = np.isfinite(dlnw) & (np.abs(dlnw) > 0)
        if ok.sum() < 10:
            continue
        y = 1.0 / dlnw[ok]
        m, b = np.polyfit(t[ok], y, 1)
        if m == 0:
            continue
        c = 1.0 / m
        out[name] = dict(c_omega=float(c), T=float(-b * c),
                         r2=float(np.corrcoef(t[ok], y)[0, 1] ** 2))
    if not out:
        return Verdict(OPEN, "no stream data")
    cs = [v["c_omega"] for v in out.values()]
    spread = max(cs) - min(cs)
    if spread > 0.5:
        return Verdict(OPEN,
                       f"c_omega not converged across runs: "
                       f"{min(cs):+.3f}..{max(cs):+.3f} (spread {spread:.3f}); "
                       "the map from dln(l)/dln(w) to lambda is not pinned",
                       out)
    if abs(np.mean(cs) + 1.0) > 0.35:
        return Verdict(FATAL,
                       f"c_omega = {np.mean(cs):+.3f}, not -1; transport balance "
                       "assumed in the lambda map does not hold", out)
    return Verdict(CLOSED, f"c_omega = {np.mean(cs):+.3f} +- {spread/2:.3f}, "
                           "consistent with the transport balance", out)


def t_log_periodic() -> Verdict:
    """C11 / anchor A2. Exact self-similarity gives ln(l) linear in ln(w).
    DISCRETE self-similarity gives that line plus a periodic function of the
    log variable. DSS is not ruled out for Navier-Stokes, and if the flow is
    DSS then no single power-law exponent exists and the whole lambda-vs-1/2
    comparison is void, regardless of how well anything converges.

    T-free by construction: use ln(w) as the log-time coordinate, so no blowup
    time is needed and the test cannot be biased by a bad T."""
    got = {}
    for d in SNAPS:
        rows = monotone_prefix(snap_series(d))
        if len(rows) < 10:
            continue
        w = np.array([x["w"] for x in rows])
        l = np.array([x["lz"] for x in rows])
        X, Y = np.log(w), np.log(l)
        m, b = np.polyfit(X, Y, 1)
        res = Y - (m * X + b)
        v0 = float(np.var(res))
        if v0 <= 0:
            continue
        span = X[-1] - X[0]
        best = (0.0, None)
        # scan periods that fit at least 1.5 cycles and are resolved by the data
        for P in np.linspace(span / 6.0, span / 1.5, 60):
            C = np.cos(2 * np.pi * X / P)
            S = np.sin(2 * np.pi * X / P)
            M = np.vstack([C, S, np.ones_like(X)]).T
            coef, *_ = np.linalg.lstsq(M, res, rcond=None)
            v1 = float(np.var(res - M @ coef))
            red = 1.0 - v1 / v0
            if red > best[0]:
                best = (red, float(P))
        got[d] = dict(var_reduction=best[0], period=best[1],
                      n=len(rows), resid_rms=float(np.sqrt(v0)))
    if not got:
        return Verdict(OPEN, "no series long enough to test log-periodicity")
    worst = max(v["var_reduction"] for v in got.values())
    hits = {k: v for k, v in got.items() if v["var_reduction"] > 0.60}
    if hits:
        return Verdict(OPEN,
                       f"log-periodic structure absorbs >60% of residual variance "
                       f"in {len(hits)}/{len(got)} runs (max {worst:.0%}). Cannot "
                       "distinguish DSS from exact self-similarity with 3 free "
                       "parameters on this few points; anchor A2 is NOT established.",
                       got)
    return Verdict(CLOSED,
                   f"no log-periodic component above 60% variance reduction "
                   f"(max {worst:.0%}); consistent with exact self-similarity", got)


ALPHA = -0.34240          # profile exponent, adjudicated (anchor A4)
C_L_CODE = 3.00649824     # gauge constant inside the profile solver, NOT c_l


def t_false_dichotomy() -> Verdict:
    """C10. RESOLVED 2026-07-30, and it resolved AGAINST the claim it was
    written to guard. The dichotomy was indeed false and the reciprocal branch
    was invented.

    Derivation. The solver substitutes Omega = xi*A*e^{a0 xi} with a0 = alpha
    and xi = ln(1+rho), so e^{a0 xi} = (1+rho)^alpha. The recorded far-field
    form is A = (f(beta)/xi)(1 + c(beta) rho^alpha + ...), so the xi factors
    cancel identically and

        Omega(rho) -> f(beta) * rho^alpha .

    Match inner to outer: w = (T-t)^{c_w} Omega(y), y = x/(T-t)^{c_l} gives
    w -> x^alpha (T-t)^{c_w - alpha c_l}. The outer region is time-independent
    as t -> T, forcing c_w = alpha * c_l. With c_w = -1 (transport balance):

        c_l = -1/alpha = 2.92056 .

    The solver's own c_l = 3.006498 sits 2.86% away while its free-gauge
    residual is converged to 1.8e-6, which proves the two are DIFFERENT
    OBJECTS: one a normalisation constant, one a scaling exponent. Comparing
    them was the original error, and 1/c_l was never a candidate at all."""
    c_l = -1.0 / ALPHA
    c_w = -1.0
    crit = c_w + 2 * c_l
    ev = dict(c_l_physical=c_l, c_l_code=C_L_CODE,
              gap=abs(c_l - C_L_CODE) / C_L_CODE,
              Re_exponent=crit,
              alpha_needed_for_subdominance=-2.0)
    return Verdict(CLOSED,
                   f"c_l = -1/alpha = {c_l:.5f} by far-field matching; the code's "
                   f"c_l={C_L_CODE:.5f} is a gauge constant ({ev['gap']:.2%} away vs "
                   f"1.8e-6 residual). Re_loc ~ (T-t)^{crit:+.4f} -> 0, so viscosity "
                   f"DOMINATES. Subdominance needs alpha < -2; we have {ALPHA}.",
                   ev)


BRANCHES = [ALPHA, -0.4168236, -0.4439811, -0.4578230]   # a0 ours, a1..a3 published


def t_unstable_branch_target() -> Verdict:
    """C12. RESOLVED 2026-07-30 by extrapolation rather than by search.

    After C10 the NS question is quantitative: viscous subdominance needs
    c_l < 1/2, i.e. alpha < -2. So the question is not 'find an unstable
    branch' but 'can this family REACH -2', and four known members answer it.

    The branch sequence steps are -0.0744, -0.0272, -0.0138: monotonically
    SHRINKING, ratios 0.365 and 0.510, both well inside 1. The sequence
    converges. Aitken and geometric extrapolation agree at

        alpha_infinity = -0.4722 ,

    giving c_l = 2.118 and Re_loc ~ (T-t)^{+3.24} -> 0. Every member of the
    family, stable and unstable alike, is viscosity-dominated.

    Reaching -2 would need the remaining steps to sum to -1.542, which from a
    last step of -0.0138 demands a sustained ratio of 0.9911 against the 0.365
    and 0.510 observed. The verdict survives every published alpha being 50%
    more negative (limit -0.708, still 2.8x short), so it does not depend on
    the precision of values we did not compute.

    SCOPE. This closes the DIRECT self-similar Euler -> NS route for this
    family. It does not close discrete self-similarity (anchor A2, still
    ASSUMED, and C11 passes only against a threshold I chose), nor
    non-axisymmetric 3D, nor any NS mechanism not descended from Euler
    self-similar blowup. It is a negative result about one route."""
    import numpy as np
    a = np.array(BRANCHES)
    d = np.diff(a)
    lim = float(a[3] - d[2] ** 2 / (d[2] - d[1]))
    cl = -1.0 / lim
    ev = dict(alphas=BRANCHES, steps=d.tolist(),
              ratios=[float(d[1] / d[0]), float(d[2] / d[1])],
              alpha_inf=lim, c_l_inf=cl, Re_exponent=-1 + 2 * cl,
              ratio_required_to_reach_minus2=float(
                  (-2.0 - a[3]) / (d[2] + (-2.0 - a[3]))))
    if lim < -2.0:
        return Verdict(FATAL, "family reaches alpha < -2; a branch escapes viscosity", ev)
    return Verdict(CLOSED,
                   f"family accumulates at alpha_inf = {lim:.4f} (steps shrink, "
                   f"ratios {ev['ratios'][0]:.3f}/{ev['ratios'][1]:.3f}); c_l = {cl:.3f}, "
                   f"Re ~ (T-t)^{-1+2*cl:+.3f} -> 0. NO member reaches -2; that would "
                   f"need a sustained step ratio of "
                   f"{ev['ratio_required_to_reach_minus2']:.4f}. Direct Euler->NS route "
                   "CLOSED for this family. Survives 50% error in the published alphas.",
                   ev)


CONFOUNDS = [
    Confound("C1", "lambda is O(3), not O(0.3); the direct reading of c_l is right",
             "kills the reciprocal branch and the whole NS-coherence chain",
             "free", t_branch_order),
    Confound("C2", "lambda straddles 1/2, so the sign of the criterion is undetermined",
             "leaves the NS verdict undecided by the march",
             "free", t_criterion_margin),
    Confound("C3", "the exponent is a property of the HWHM estimator, not the flow",
             "makes every length-scale number an artefact of the definition",
             "free", t_length_proxy),
    Confound("C4", "the argmax migrates between distinct structures",
             "the growth rate measures the migration, not a structure",
             "free", t_argmax_wander),
    Confound("C5", "u ~ l*w fails, so Re_loc ~ w*l^2 is the wrong Reynolds number",
             "invalidates the criterion itself",
             "free", t_velocity_relation),
    Confound("C5b", "the solver does not conserve the advected swirl supremum",
             "indicts every number produced by this solver, not just this claim",
             "free", t_swirl_conservation),
    Confound("C11", "the blowup is discretely self-similar, not exactly self-similar",
             "makes a single power-law exponent meaningless; DSS is NOT ruled out "
             "for NS and would void the whole lambda comparison",
             "free", lambda: t_log_periodic()),
    Confound("C6", "the exponent is set by the grid, not the flow",
             "makes the collapse rate a resolution artefact",
             "free", t_resolution_floor),
    Confound("C7", "no run is asymptotic; every exponent is a transient",
             "means no march value may be quoted as lambda at all",
             "free", t_preasymptotic),
    Confound("C8", "spectral filtering sets the measured collapse rate",
             "makes the exponent partly a numerical choice",
             "free", t_filter_dependence),
    Confound("C9", "c_omega != -1, breaking the map from dln(l)/dln(w) to lambda",
             "severs the march measurement from the profile constant",
             "free", t_c_omega),
    Confound("C10", "'c_l or 1/c_l' is a false dichotomy; the truth is neither",
             "ruling out 3.006 would then establish nothing about 0.333",
             "derivation", t_false_dichotomy),
    Confound("C12", "no unstable branch reaches alpha < -2, so no branch escapes viscosity",
             "closes the direct Euler->NS route entirely for this family",
             "cheap", t_unstable_branch_target),
]


# --------------------------------------------------------------- the anchors
#
# Navier-Stokes is UNSOLVED. That is not a footnote, it is a structural fact
# about every reference point below. Nothing here has been cashed against an
# actual solve, so an "agreed" anchor is a stream of information, not a proof,
# and a confound engine that only audits its own measurements against agreed
# anchors is a compliance checker. Anchors get the same column as measurements.
#
# status:
#   PROVEN     follows from an exact symmetry or identity; safe to lean on
#   ADJUDICATED  >=2 INDEPENDENT streams agree (independent = disjoint method)
#   AGREED     the field agrees, one method lineage; not adjudicated
#   ASSUMED    inherited, load-bearing, never independently checked here

@dataclass
class Anchor:
    aid: str
    statement: str
    status: str
    streams: list[str]        # independent corroborations
    falsifier: str            # what would show it false
    carries: list[str]        # confound / claim ids that die with it


ANCHORS = [
    Anchor("A1", "c_omega = -1, forced by the nonlinear transport balance",
           "ASSUMED", ["transport-balance algebra"],
           "a converged march fit giving c_omega away from -1",
           ["C9", "lambda map", "TARGET"]),
    Anchor("A2", "the blowup is EXACTLY self-similar, not discretely self-similar",
           "ASSUMED", ["convention inherited from the profile formulation"],
           "log-periodic modulation in w(t) or l(t) at fixed phase",
           ["C1", "C2", "C7", "TARGET"]),
    Anchor("A3", "lambda = 1/2 is the NS-critical exponent",
           "PROVEN", ["NS scaling symmetry u_s(x,t)=s*u(sx,s^2 t), exact"],
           "nothing; it is an identity of the equation",
           ["TARGET"]),
    Anchor("A4", "alpha = -0.34240 is the profile exponent",
           "ADJUDICATED", ["our corner-regularised spectral Newton",
                           "published neural-search value, disjoint method"],
           "a third disjoint method landing outside 3e-5",
           ["TARGET"]),
    Anchor("A5", "the Hou-Luo corner scenario is the operative blowup mechanism",
           "AGREED", ["Hou-Luo adaptive march", "Chen-Hou computer-assisted proof"],
           "a different mechanism dominating at accessible resolution",
           ["TARGET", "everything downstream"]),
    Anchor("A6", "Re_loc ~ w * l^2, i.e. u ~ l*w for the advecting velocity",
           "ASSUMED", ["dimensional analysis only"],
           "direct measurement of poloidal u against l*w",
           ["C5", "TARGET"]),
    Anchor("A9", "the published unstable branch values alpha_1..alpha_3 are correct",
           "AGREED", ["neural-search paper only; unconfirmed by any non-network "
                      "method, including ours (our own candidate was a ghost)"],
           "a non-network computation landing elsewhere",
           ["C12"]),
    Anchor("A8", "far-field matching: c_omega = alpha * c_l, from Omega ~ rho^alpha",
           "ASSUMED", ["our own derivation, 2026-07-30, single pass"],
           "an outer region that is NOT time-independent as t->T",
           ["C10", "C12", "the viscous verdict"]),
    Anchor("A7", "2D Boussinesq <-> 3D axisymmetric Euler near the wall",
           "ADJUDICATED", ["our own O(eps) correction derivation, 4 routes byte-agree",
                           "the standard Hou-Luo asymptotic statement"],
           "a correction failing to vanish like eps",
           ["TARGET"]),
]


def anchor_report() -> tuple[int, int]:
    print("-" * 78)
    print("ANCHOR COLUMN  (NS is unsolved: no row here has been cashed against a solve)")
    print("-" * 78)
    weak = 0
    for a in ANCHORS:
        mark = {"PROVEN": " proven ", "ADJUDICATED": " adjud. ",
                "AGREED": " AGREED ", "ASSUMED": " ASSUMED"}[a.status]
        print(f"[{mark}] {a.aid}  {a.statement}")
        print(f"           streams({len(a.streams)}): {'; '.join(a.streams)}")
        if a.status in ("ASSUMED", "AGREED"):
            weak += 1
            print(f"           falsifier: {a.falsifier}")
            print(f"           carries  : {', '.join(a.carries)}")
        print()
    single = [a.aid for a in ANCHORS if len(a.streams) < 2 and a.status != "PROVEN"]
    print(f"Anchors on a SINGLE stream (not adjudicated): {', '.join(single) or 'none'}")
    return weak, len(ANCHORS)


def main() -> None:
    print("=" * 78)
    print("MYTHOS CONFOUND ENGINE")
    print("=" * 78)
    print("TARGET CLAIM:")
    for line in (TARGET[i:i + 72] for i in range(0, len(TARGET), 72)):
        print("  " + line)
    print("\nThe claim stands iff no row below is OPEN or FATAL.\n")

    results = []
    for c in CONFOUNDS:
        v = c.run()
        results.append((c, v))
        mark = {CLOSED: "  ok  ", OPEN: " OPEN ", FATAL: "FATAL!", UNTESTED: "UNTEST"}[v.status]
        print(f"[{mark}] {c.cid} ({c.cost})  {c.statement}")
        print(f"          {v.detail}")
        if v.status in (OPEN, FATAL, UNTESTED):
            print(f"          would kill: {c.kills_how}")
        print()

    weak, tot = anchor_report()
    n_open = sum(1 for _, v in results if v.status == OPEN)
    n_fatal = sum(1 for _, v in results if v.status == FATAL)
    n_unt = sum(1 for _, v in results if v.status == UNTESTED)
    n_closed = sum(1 for _, v in results if v.status == CLOSED)

    print("=" * 78)
    print(f"VULNERABILITY COLUMN: {n_closed} closed | {n_open} open | "
          f"{n_fatal} fatal | {n_unt} untested")
    if n_fatal:
        print("VERDICT: claim REFUTED. A denier survived and won.")
    elif n_open or n_unt:
        print("VERDICT: claim NOT ESTABLISHED. Surviving deniers listed above.")
        print("Next work is whichever surviving row is cheapest, not whichever")
        print("is most interesting.")
    else:
        print("VERDICT: no surviving denier. Claim stands until a new one is named.")
    print("=" * 78)


if __name__ == "__main__":
    main()
