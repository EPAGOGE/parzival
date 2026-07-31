"""N-LENS: the J-lens move, applied to Newton on the profile system.

WHAT J-LENS DOES AND WHAT CARRIES OVER.  ~/jlens computes lens(h) = softmax(W_U norm(J h)):
it takes a DIRECTION in a transformer's hidden space and projects it through the unembedding
so you can read what that direction MEANS in token space.  Alongside it, robustness.py
annotates each cell with a fragility eps* -- how large a perturbation flips the decision.
Two ideas, both portable:

  1. A direction is uninterpretable until you project it into a space with MEANING.
     For a transformer that space is tokens.  Here it is PHYSICAL SPACE: which radial band
     in xi, which angular position in beta, which field (Ot or Bt).  `localize()` below is
     the N-lens analogue of the unembedding projection, and it is the piece that turns
     "the smallest singular vector" from a 1458-dimensional object into a sentence.

  2. Annotate CONTINUOUSLY, per unit, and flag what is fragile rather than reporting one
     scalar.  J-lens does it per cell per layer; N-lens does it per Newton step per band.

WHY THIS IS THE RIGHT INSTRUMENT NOW.  Every diagnostic failure this session was a
diagnostic that reported a single number and hid where it came from:
  * ||F|| = 2.25e-13 while the open-system residual on the beta = pi/2 - eps_b column was
    3.6e-2 -- eleven orders, invisible because the norm excluded those rows.
  * cond(Cg B) = 29.3 while ||P|| = 1/sin(theta_min) = 1.43e5 -- five orders, invisible
    because a 2x2 condition number cannot see the N^2/N^4 growth of the constraint rows.
  * alpha = -0.34240009311696556 reported at ||F|| = 1.77e-2 from a run that accepted ZERO
    Newton steps -- invisible because no flag existed for "this is the seed".
  * e1 = +1.2905% at N=36 against +0.024/+0.091/-0.106% at N=44/52/64 -- a 54x outlier,
    unnoticed for weeks because nobody printed the constraint errors side by side.
Each of those is a FLAG this module raises automatically.

AND THE LIVE QUESTION IT ANSWERS.  alpha now converges cleanly to -0.3316 at N=28 and
N=44 (agreeing to 0.07% after the d1 constraint and the eps_b -> 0 extrapolation), against
Chen-Hou's -0.34240009.  A stable +3.16% offset.  Two explanations remain and they are
distinguishable: a residual discretisation systematic, OR we are converging to a DIFFERENT
ROOT, because Newton has been single-started from one seed in a basin known to be fractal
(six Cartesian configurations previously found three different wrong roots).  Deflation is
the instrument that decides it -- see polar_deflate.py.

DELIBERATELY NOT DONE HERE: nothing in this file changes the solve.  It is pure
annotation, so it cannot become another confound.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import numpy.linalg as la

HERE = pathlib.Path(__file__).parent

# Radial bands, in xi. Chosen from the measured structure, not aesthetically:
# [0,2] carries ~51% of the radial variation of Ot on 18% of the nodes; [2,15] carries
# ~48% on 39%; [15,XMAX] carries ~1.1% on 43%.  The last band being both over-resolved and
# nearly constant is the single most reported fact about this discretisation, so it gets
# its own bin and every localisation is read against it.
BANDS = ((0.0, 2.0), (2.0, 15.0), (15.0, np.inf))


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(HERE / fname))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# 1. the projection with meaning -- the analogue of J-lens's unembedding
# ---------------------------------------------------------------------------
def localize(St, v, bands=BANDS):
    """Project a direction in UNKNOWN space onto physical space.

    Returns the fraction of ||v||^2 sitting in each (field, radial band), plus the peak
    location.  This is what makes a singular vector legible: "99.35% of u_min lives in
    xi < 2" is a sentence; a 1458-vector is not."""
    S = St.S
    C = St.C
    n2 = C.nx * C.nb
    full = np.zeros(2 * n2)
    full[S.idx] = np.asarray(v).real if np.iscomplexobj(v) else np.asarray(v)
    dO = full[:n2].reshape(C.nx, C.nb)
    dB = full[n2:].reshape(C.nx, C.nb)
    tot = float((dO ** 2).sum() + (dB ** 2).sum())
    if tot <= 0.0:
        return dict(total=0.0)
    out = {"field_Ot": float((dO ** 2).sum() / tot), "field_Bt": float((dB ** 2).sum() / tot)}
    for lo, hi in bands:
        m = (C.x >= lo) & (C.x < hi)
        key = f"xi[{lo:g},{'inf' if hi == np.inf else f'{hi:g}'})"
        out[key] = float(((dO[m] ** 2).sum() + (dB[m] ** 2).sum()) / tot)
        out[key + ".nodes"] = int(m.sum())
    e = dO ** 2 + dB ** 2
    i, j = np.unravel_index(int(np.argmax(e)), e.shape)
    out["peak_xi"] = float(C.x[i])
    out["peak_beta"] = float(C.b[j])
    out["peak_r"] = float(np.exp(C.x[i]) - 1.0)
    out["peak_share"] = float(e[i, j] / tot)
    # angular concentration: is it at the wall (beta=0) or the axis (beta=pi/2)?
    eb = e.sum(axis=0)
    out["beta_wall_half"] = float(eb[: C.nb // 2].sum() / eb.sum())
    return out


def band_summary(loc):
    """One-line rendering of a localize() dict."""
    ks = [k for k in loc if k.startswith("xi[") and not k.endswith(".nodes")]
    parts = [f"{k}={100*loc[k]:.1f}%" for k in ks]
    return (f"Ot={100*loc.get('field_Ot', 0):.0f}%/Bt={100*loc.get('field_Bt', 0):.0f}% "
            + " ".join(parts)
            + f" peak@xi={loc.get('peak_xi', float('nan')):.2f}"
            f",b={loc.get('peak_beta', float('nan')):.3f}")


# ---------------------------------------------------------------------------
# 2. the flags -- every one of these corresponds to a real failure this session
# ---------------------------------------------------------------------------
class Flag:
    __slots__ = ("name", "severity", "value", "detail")

    def __init__(self, name, severity, value, detail):
        self.name, self.severity, self.value, self.detail = name, severity, value, detail

    def __repr__(self):
        return f"[{self.severity}] {self.name}={self.value:.3e} :: {self.detail}"


def flags_for(St, x, f=None, J=None, prev_x=None, cl_star=None):
    """Every curiosity detector, run at one Newton state.  Cheap ones first."""
    S, C = St.S, St.C
    out = []
    cl, cw = float(x[-2]), float(x[-1])
    if f is None:
        f, _, _ = S.F(x)
    r = float(la.norm(f) / np.sqrt(f.size))

    # --- SEED / ZERO-STEP: the phantom-result detector -----------------------
    if prev_x is not None and np.allclose(x, prev_x, rtol=0, atol=0):
        out.append(Flag("ZERO_STEP", "CRITICAL", r,
                        "state identical to the start point: any alpha here is the SEED's"))

    # --- RESIDUAL BLINDNESS: ||F|| vs what the open system sees --------------
    if hasattr(S, "open_residual"):
        od = S.open_residual(x)
        ratio = od["open_rms"] / max(r, 1e-300)
        sev = "CRITICAL" if ratio > 1e6 else ("WARN" if ratio > 1e2 else "info")
        out.append(Flag("RESIDUAL_BLIND", sev, ratio,
                        f"open_rms={od['open_rms']:.2e} vs ||F||={r:.2e}; "
                        f"axis line RMS={od['axis_rms']:.2e}, "
                        f"max|dOt| there = {100*od['axis_max_dOt_rel']:.2f}% of max|Ot|"))

    # --- FREE RESIDUAL: c_l has an exact unimposed target --------------------
    cls_ = cl_star if cl_star is not None else 2.0 * S.THXX_REF / S.WX_REF
    d_cl = (cl - cls_) / cls_
    sev = "WARN" if abs(d_cl) > 0.01 else "info"
    out.append(Flag("FREE_RESIDUAL_CL", sev, d_cl,
                    f"c_l={cl:.8f} vs the exact unimposed target {cls_:.8f} "
                    f"(= 2*THXX/WX, valid because th_xx(0)=2 v_x(0) exactly)"))

    # --- CONSTRAINT ERRORS side by side: the e1 outlier detector -------------
    Ot, Bt = S.unpack(x[:-2])
    cb = float(np.cos(C.b[0]))
    e1 = (float((C.Dx @ Ot)[0, 0]) - S.WX_REF * cb) / (S.WX_REF * cb)
    g2 = S.g2_of(Bt)
    base = S.THXX_REF if getattr(S, "constraint", "d2") == "d2" else 0.5 * S.THXX_REF
    e2 = (g2 + base - base * cb ** 2) / (base * cb ** 2)
    dlnq = e2 - 2.0 * e1
    out.append(Flag("CONSTRAINT_E1", "WARN" if abs(e1) > 3e-3 else "info", e1,
                    f"first corner constraint relative error"))
    out.append(Flag("CONSTRAINT_DLNQ", "WARN" if abs(dlnq) > 3e-3 else "info", dlnq,
                    "dln q = e2 - 2 e1 -- the ONLY projection alpha can see "
                    f"(e2={e2:+.3e})"))

    # --- SYMMETRY LEAK: the amplitude scaling is an exact symmetry -----------
    # Ot -> s Ot, Bt -> s^2 Bt, (c_l,c_w) -> s (c_l,c_w) reproduces the field residual to
    # 3.2e-12 over s in [0.5,7] -- EXCEPT that unpack() restores the UNSCALED pinned axis
    # column, which breaks it with an O(1) coefficient.  Measured defect 8.06e-4 at
    # s=1.001.  alpha is blind to this direction, so a leak here is invisible in alpha.
    s = 1.001
    xs = x.copy()
    n2 = C.nx * C.nb
    OtS, BtS = Ot * s, Bt * s * s
    xs[:-2] = S.pack(OtS, BtS)
    xs[-2], xs[-1] = cl * s, cw * s
    fs, _, _ = S.F(xs)
    # the residual should scale as s^2 for the Ot block and s^3 for Bt; compare the
    # SCALE-INVARIANT ratio instead of guessing block powers: use the alpha it implies
    leak = abs((float(xs[-1]) / float(xs[-2])) - (cw / cl))
    rs = float(la.norm(fs) / np.sqrt(fs.size))
    out.append(Flag("SYMMETRY_LEAK", "WARN" if rs / max(r, 1e-300) > 1e3 else "info",
                    rs / max(r, 1e-300),
                    f"||F|| after an exact-symmetry rescale by s={s}: {rs:.3e} "
                    f"(alpha drift {leak:.2e}; a large ratio means the pinned rows "
                    f"break the symmetry, not the physics)"))

    # --- CONDITIONING + near-null localisation (the expensive block) ---------
    if J is not None:
        try:
            U, sv, Vt = la.svd(J)
            out.append(Flag("SIGMA_MIN", "WARN" if sv[-1] < 1e-6 * sv[0] else "info",
                            sv[-1] / sv[0],
                            f"sigma_min/sigma_max of the bordered J "
                            f"(sigma_min={sv[-1]:.3e})"))
            # transversality: at a fold what governs the bordered solve is the projection
            # of the parameter derivative onto the left singular vector, NOT sigma_min.
            n = St.n
            u_min = U[:, -1]
            trans = float(np.abs(u_min[n:]).max())
            out.append(Flag("TRANSVERSALITY", "WARN" if trans < 1e-3 else "info", trans,
                            "|u_min| on the two constraint rows -- the fold diagnostic "
                            "that replaces cond(J)"))
            v_min = Vt[-1, :n]
            loc = localize(St, v_min)
            out.append(Flag("NEAR_NULL_WHERE", "info", loc.get("peak_xi", np.nan),
                            "smallest right singular vector lives at: " + band_summary(loc)))
        except la.LinAlgError:
            out.append(Flag("SVD_FAILED", "WARN", np.nan, "svd of the bordered J failed"))

    # --- STEP direction, if we have a previous state ------------------------
    if prev_x is not None and not np.allclose(x, prev_x):
        d = (x - prev_x)[:-2]
        loc = localize(St, d)
        out.append(Flag("STEP_WHERE", "info", float(la.norm(d)),
                        "the Newton step lives at: " + band_summary(loc)))
    return out


def render(flags, only=("CRITICAL", "WARN")):
    """Print the flags worth reading. `only=None` prints everything."""
    for fl in flags:
        if only is None or fl.severity in only:
            print(f"    {fl!r}", flush=True)


# ---------------------------------------------------------------------------
# 3. the continual flow -- annotate a whole Newton trajectory
# ---------------------------------------------------------------------------
def watch(St, x0, steps=40, tol=1e-11, every=1, svd_every=0, verbose=True):
    """Newton, annotated at every step.  A re-implementation of newton_exact's loop with
    the lens attached -- NOT a wrapper, because the point is to see the intermediate
    states, which newton_exact discards.

    svd_every=0 skips the O(n^3) svd block (it dominates the cost); set it to 1 to get
    conditioning and near-null localisation at every step, or e.g. 4 for a sample."""
    S = St.S
    x = x0.copy()
    f, cl, cw = S.F(x)
    r = float(la.norm(f) / np.sqrt(f.size))
    trail = []
    prev = r
    for it in range(steps):
        A = St.A_exact(x)
        Ot, Bt = S.unpack(x[:-2])
        B, Cg = St.exact_B(Ot, Bt), St.exact_Cg()
        n = St.n
        J = np.zeros((n + 2, n + 2))
        J[:n, :n], J[:n, n:], J[n:, :n] = A, B, Cg
        want_svd = svd_every and (it % svd_every == 0)
        if verbose and it % every == 0:
            print(f"  it{it:02d} ||F||={r:.4e} c_l={float(x[-2]):.6f} "
                  f"alpha={float(x[-1])/float(x[-2]):+.8f}", flush=True)
            fl = flags_for(St, x, f=f, J=J if want_svd else None,
                           prev_x=trail[-1]["x"] if trail else None)
            render(fl)
            trail.append(dict(it=it, x=x.copy(), F=r, flags=fl))
        try:
            dx = la.solve(J, -f)
        except la.LinAlgError:
            dx = -la.lstsq(J, f, rcond=None)[0]
        lam, best = 1.0, None
        for _ in range(12):
            ft, _, _ = S.F(x + lam * dx)
            rt = float(la.norm(ft) / np.sqrt(ft.size))
            if rt < prev:
                best = (x + lam * dx, ft, rt)
                break
            lam *= 0.5
        if best is None:
            if verbose:
                print(f"  it{it:02d} linesearch failed -- STOPPING with {it} accepted steps",
                      flush=True)
            break
        x, f, r = best
        prev = r
        if r < tol:
            break
    if verbose:
        print(f"  final ||F||={r:.4e}  accepted steps={len(trail)}", flush=True)
        render(flags_for(St, x, f=f), only=("CRITICAL", "WARN"))
    return x, f, r, trail
