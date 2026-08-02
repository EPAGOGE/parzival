#!/usr/bin/env python3
"""M3 ORBIT VERDICT -- settling / wandering / cycle, from cos_step + V,
with bootstrap CI, >= 3 independent seeds agreeing (DONE.md M3).

INSTRUMENT: the certified s-march (M1 PASS, M2 PASS) at the production
grid, root A certificate re-verified on load. Five independent admissible
perturbations (seeds 0..4; seed 0 reproduces M2's amp-1e-3 rung = built-in
regression check), amp=1e-3 (the largest basin-validated amplitude --
maximal nonlinear content, longest valid window above the noise floor),
ds=0.25, up to 240 steps, M2's stop rule (true residual < 1e-10 each step,
QN iterations <= 15).

ORBIT GEOMETRY (features.py's discriminator, ported to the march): the
DNS-side battery (features.py) separates the three fates by the DIRECTION
of successive increments, not their length. Here the orbit lives in the
REDUCED ker(Cg) coordinates v_k = _reduced_dev(R0, Z0, z0, z_k) -- the
same fixed-root projection the march itself corrects in. Per step:
    dv_k       = v_{k+1} - v_k
    cos_step_k = cos(dv_k, dv_{k+1})     increment persistence
    cos_aim_k  = cos(dv_k, -v_k)         aim at the KNOWN fixed point v=0
                                          (stronger than features.py's
                                          aim-at-final-observed-state: the
                                          march knows its target exactly)
    speed_k    = |dv_k|                  with ln|dv| vs s slope = decay rate
Plain Euclidean cosines, matching features.py's convention.

RUBRIC (features.py report(), reused verbatim, applied to the WHOLE
bootstrap CI, not the point estimate):
    SETTLING     cos_step > 0.5 AND cos_aim > 0.5   (entire CI above)
                 plus features.py's own gloss "increments point back AND
                 shrink": ln-speed slope CI entirely < 0
    WANDERING    |cos_step| < 0.25                  (entire CI inside)
    CYCLE        cos_step < -0.4 (alternating / coherent rotation --
                 features.py's OSCILLATING, DONE.md's "cycle")
    MIXED        anything else => that seed certifies NOTHING
M3 TEST: all seeds' CIs inside ONE regime.

VALID WINDOW: steps with raw |dz|_rel > 10x the M2-measured instrument
floor (3e-10, M2_AMPLADDER.out adjudication) => cut at 3e-9. Below that,
increments are solver noise and cos_step decorrelates by construction --
including them would bias TOWARD "wandering"; the cut is stated, not
silent. Bootstrap: 10,000 resamples, percentile 95% CI, fixed rng.

SEED INDEPENDENCE is checked, not assumed: pairwise cosines of the five
perturbation directions are reported (expect ~0 for random directions in
n~4758 dims).

Outputs (new files only): M3_ORBIT_VERDICT.out, m3_orbit_data.npz.
VERDICT_ORBIT.txt gains the M3 section (with the uppercase regime keyword
done.sh greps for) ONLY if the test genuinely passes.

Refusals honored: definiteness by cholesky only (certificate loader's
re-verify); no eigenvalue readouts for decisions; march_s.py untouched
(reuse via import); honest wall times; laptop only.
"""
import time

import numpy as np

from march_s import (_HERE, QuasiNewtonSMarcher, _admissible_pert,
                     _load_production_pnorm_certificate, _reduced_dev,
                     state_distance)

SEEDS = [0, 1, 2, 3, 4]
AMP = 1e-3
DS = 0.25
NSTEPS = 240
STEP_TOL = 1e-10
QN_ITER_STOP = 15
RAW_FLOOR = 3e-10                 # M2-measured instrument floor (absolute)
FLOOR_SAFETY = 10.0               # valid window: raw > FLOOR_SAFETY * RAW_FLOOR
N_BOOT = 10_000
BOOT_RNG_SEED = 12345
CI = (2.5, 97.5)

OUT = _HERE / "M3_ORBIT_VERDICT.out"
NPZ = _HERE / "m3_orbit_data.npz"
VERDICT_FILE = _HERE / "VERDICT_ORBIT.txt"
_lines = []


def log(s=""):
    _lines.append(s)
    print(s, flush=True)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def march_one_seed(cert, M, seed):
    """March one seed; return per-step orbit diagnostics (computed online:
    only the previous v and dv are held, never the whole trajectory)."""
    S, z0, R0 = cert["S"], cert["z0"], cert["R0"]
    P, Z0 = cert["P"], cert["Z0"]
    n2 = S.Nx * S.Nb

    rng = np.random.default_rng(seed)
    pert = _admissible_pert(S, n2, rng, AMP, z0)
    z = z0 + pert
    v_prev = _reduced_dev(R0, Z0, z0, z)
    V0 = float(np.real(np.vdot(v_prev, P @ v_prev)))
    raw0 = state_distance(z, z0, n2)
    dv_prev = None
    cos_steps, cos_aims, speeds, raws, s_vals = [], [], [], [raw0], []
    stop = None

    t0 = time.time()
    for i in range(NSTEPS):
        zn, r, k = M.step(z, DS, tol=STEP_TOL, maxit=25)
        s_now = (i + 1) * DS
        if zn is None or not np.all(np.isfinite(zn)):
            stop = f"correction failure at step {i+1} (s={s_now:g}), r={r:.2e}"
            break
        if r >= STEP_TOL or k > QN_ITER_STOP:
            stop = (f"nonlinear escape/stall at step {i+1} (s={s_now:g}), "
                    f"r={r:.2e} it={k}")
            break
        v = _reduced_dev(R0, Z0, z0, zn)
        dv = v - v_prev
        rawk = state_distance(zn, z0, n2)
        if dv_prev is not None:
            cos_steps.append(cosine(dv_prev, dv))
        cos_aims.append(cosine(dv, -v_prev))
        speeds.append(float(np.linalg.norm(dv)))
        raws.append(rawk)
        s_vals.append(s_now)
        v_prev, dv_prev, z = v, dv, zn
    wall = time.time() - t0
    Vend = float(np.real(np.vdot(v_prev, P @ v_prev)))
    return dict(seed=seed, pert=pert, V0=V0, Vend=Vend, raw0=raw0,
                cos_steps=np.array(cos_steps), cos_aims=np.array(cos_aims),
                speeds=np.array(speeds), raws=np.array(raws),
                s_vals=np.array(s_vals), stop=stop,
                n_done=len(speeds), wall_s=wall)


def boot_mean_ci(x, rng):
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, CI)
    return float(np.mean(x)), float(lo), float(hi)


def boot_slope_ci(s, y, rng):
    """ln-speed decay slope with pair-resampling bootstrap CI."""
    slope = float(np.polyfit(s, y, 1)[0])
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    slopes = np.array([np.polyfit(s[j], y[j], 1)[0] for j in idx])
    lo, hi = np.percentile(slopes, CI)
    return slope, float(lo), float(hi)


def classify_ci(cs_lo, cs_hi, ca_lo, ca_hi, sl_lo, sl_hi):
    """features.py rubric on the WHOLE CI. Returns (regime, reason)."""
    if cs_lo > 0.5 and ca_lo > 0.5 and sl_hi < 0.0:
        return "SETTLING", ("cos_step CI > 0.5, cos_aim CI > 0.5, "
                            "ln-speed slope CI < 0")
    if -0.25 < cs_lo and cs_hi < 0.25:
        return "WANDERING", "cos_step CI inside (-0.25, 0.25)"
    if cs_hi < -0.4:
        return "CYCLE", "cos_step CI < -0.4 (alternating/oscillating)"
    return "MIXED", "CI does not sit fully inside any regime's condition"


def main():
    t_all = time.time()
    log("=" * 78)
    log("M3 ORBIT VERDICT -- cos_step + V with bootstrap CI, 5 seeds")
    log("=" * 78)
    log(f"amp={AMP:g}  ds={DS} nsteps<={NSTEPS}  seeds={SEEDS}  "
        f"stop: r>={STEP_TOL:g} or it>{QN_ITER_STOP}")
    log(f"valid window: raw > {FLOOR_SAFETY:g} x {RAW_FLOOR:g} (M2 floor)  "
        f"bootstrap: {N_BOOT} resamples, {CI[0]:g}-{CI[1]:g}% CI, "
        f"rng seed {BOOT_RNG_SEED}")
    log("")

    cert = _load_production_pnorm_certificate()
    log(f"certificate: {cert['npz_path']}")
    log(f"grid={cert['cfg']} alpha={cert['alpha']}  n_finite="
        f"{cert['n_finite']}")
    log(f"root ||F||_max fresh = {cert['root_fmax_check']:.3e}  "
        f"cholesky(P): RE-VERIFIED "
        f"{'PASS' if cert['cholesky_ok'] else 'FAIL'}")
    if not cert["root_ok"] or not cert["cholesky_ok"]:
        log("ABORT -- certificate did not re-verify; no march attempted.")
        _finish(t_all)
        return

    M = QuasiNewtonSMarcher(cert["S"], cert["R0"], cert["Lred"], cert["Z0"],
                            sign=+1.0)
    _lu, _piv, t_fact = M.frozen_lu(DS)
    log(f"frozen LU: {t_fact:.1f}s (once, shared across seeds)")

    results = []
    for seed in SEEDS:
        log("")
        log(f"--- SEED {seed} " + ("(= M2 amp-1e-3 rung, regression check) "
                                    if seed == 0 else "") + "-" * 40)
        res = march_one_seed(cert, M, seed)
        results.append(res)
        log(f"  steps={res['n_done']}/{NSTEPS}  wall={res['wall_s']:.1f}s  "
            f"stop={res['stop'] or 'none (completed)'}")
        log(f"  V(v): {res['V0']:.6e} -> {res['Vend']:.6e}   raw: "
            f"{res['raw0']:.3e} -> {res['raws'][-1]:.3e}")

    log("")
    log("SEED INDEPENDENCE (pairwise cosines of the 5 perturbation "
        "directions; expect ~0):")
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            c = cosine(results[i]["pert"], results[j]["pert"])
            log(f"  seed {SEEDS[i]} vs {SEEDS[j]}: {c:+.4f}")

    log("")
    log("=" * 78)
    log("PER-SEED VERDICTS (rubric = features.py report(), CI-level)")
    log("=" * 78)
    rng = np.random.default_rng(BOOT_RNG_SEED)
    regimes = []
    seed_rows = []
    for res in results:
        # valid window mask on the per-increment arrays. raws[1:] aligns
        # with speeds/cos_aims (one entry per step); cos_steps starts one
        # step later (needs two increments).
        raw_steps = res["raws"][1:]
        m_aim = raw_steps > FLOOR_SAFETY * RAW_FLOOR
        m_cs = m_aim[1:]
        cs = res["cos_steps"][m_cs]
        ca = res["cos_aims"][m_aim]
        sp = res["speeds"][m_aim]
        sv = res["s_vals"][m_aim]
        n_cut = int((~m_aim).sum())
        cs_m, cs_lo, cs_hi = boot_mean_ci(cs, rng)
        ca_m, ca_lo, ca_hi = boot_mean_ci(ca, rng)
        sl, sl_lo, sl_hi = boot_slope_ci(sv, np.log(sp), rng)
        regime, reason = classify_ci(cs_lo, cs_hi, ca_lo, ca_hi, sl_lo, sl_hi)
        regimes.append(regime)
        seed_rows.append(dict(seed=res["seed"], cs=(cs_m, cs_lo, cs_hi),
                              ca=(ca_m, ca_lo, ca_hi), sl=(sl, sl_lo, sl_hi),
                              n_valid=len(ca), n_cut=n_cut, regime=regime))
        log(f"seed {res['seed']}: {len(ca)} valid increment steps "
            f"({n_cut} cut at the floor)")
        log(f"  cos_step = {cs_m:+.4f}  CI [{cs_lo:+.4f}, {cs_hi:+.4f}]")
        log(f"  cos_aim  = {ca_m:+.4f}  CI [{ca_lo:+.4f}, {ca_hi:+.4f}]")
        log(f"  dln|dv|/ds = {sl:+.4f}  CI [{sl_lo:+.4f}, {sl_hi:+.4f}]")
        log(f"  REGIME: {regime}  ({reason})")

    log("")
    log("=" * 78)
    agree = len(set(regimes)) == 1 and regimes[0] != "MIXED"
    n_agree = len(regimes)
    if agree:
        verdict = regimes[0]
        log(f"M3 TEST: PASS -- all {n_agree} seeds' CIs inside ONE regime: "
            f"{verdict}  (DONE.md requires >= 3 agreeing; have {n_agree})")
    else:
        verdict = None
        log(f"M3 TEST: FAIL -- regimes across seeds: {regimes}. "
            f"No verdict written; VERDICT_ORBIT.txt untouched.")

    np.savez(NPZ, seeds=np.array(SEEDS), amp=AMP, ds=DS,
             **{f"cos_steps_{i}": r["cos_steps"] for i, r in enumerate(results)},
             **{f"cos_aims_{i}": r["cos_aims"] for i, r in enumerate(results)},
             **{f"speeds_{i}": r["speeds"] for i, r in enumerate(results)},
             **{f"raws_{i}": r["raws"] for i, r in enumerate(results)},
             regimes=np.array(regimes),
             floor=RAW_FLOOR, floor_safety=FLOOR_SAFETY,
             n_boot=N_BOOT, boot_seed=BOOT_RNG_SEED)
    log(f"per-step data saved: {NPZ.name}")

    if agree:
        _write_verdict(seed_rows, verdict)
        log(f"VERDICT_ORBIT.txt updated with the M3 section "
            f"(keyword: {verdict}).")
    _finish(t_all)


def _write_verdict(seed_rows, verdict):
    m2_text = VERDICT_FILE.read_text()
    cut = m2_text.find("M3 (orbit regime verdict):")
    if cut >= 0:
        m2_text = m2_text[:cut].rstrip() + "\n"
    rows = "\n".join(
        f"  seed {r['seed']}: cos_step {r['cs'][0]:+.4f} "
        f"[{r['cs'][1]:+.4f},{r['cs'][2]:+.4f}]  cos_aim {r['ca'][0]:+.4f} "
        f"[{r['ca'][1]:+.4f},{r['ca'][2]:+.4f}]  dln|dv|/ds {r['sl'][0]:+.4f} "
        f"[{r['sl'][1]:+.4f},{r['sl'][2]:+.4f}]  ({r['n_valid']} valid steps)"
        f"  -> {r['regime']}" for r in seed_rows)
    m3 = (f"\nM3: ORBIT REGIME = {verdict}. {len(seed_rows)} independent "
          f"seeds (amp 1e-3, the largest basin-validated amplitude), "
          f"bootstrap 95% CIs, ALL CIs inside one regime -- the DONE.md M3 "
          f"test. Rubric = features.py report() thresholds applied to whole "
          f"CIs; aim target = the known fixed point; valid window cut at "
          f"10x the M2-measured floor.\n\n{rows}\n\n"
          f"source: M3_ORBIT_VERDICT.out / m3_orbit_data.npz\n")
    VERDICT_FILE.write_text(m2_text + m3)


def _finish(t_all):
    log("")
    log(f"total wall: {time.time() - t_all:.1f}s")
    OUT.write_text("\n".join(_lines) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
