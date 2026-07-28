"""EJA BRIDGE: one-import access to the epagoge/jump abduction engine for agents.

    sys.path.insert(0, "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
    from eja_bridge import *

Pure-python + numpy (works under the parzival venv; do NOT install anything).
READ-ONLY on /Users/epagogellc/epagoge/jump -- never write into either repo.

THE THREE IDIOMS, with the honesty gates that make them worth using:

1. TENSION (the drive signal).  Two rule systems disagreeing on a shared scenario is a
   computable scalar -- it lives in axiom space, not data space:
       w = mk_witness("solverA", 1.23, "referenceB", 1.31, scenario={"x": 0})
       w.divergence          # the gradient; tags steer what to intervene on next
   A REFERENCE-FREE internal contradiction (your own solution vs an exact identity)
   outranks disagreement with an external number.

2. QUOTIENT-AND-PROMOTE (the Jump).  Two causally distinct configurations whose
   observation traces are indistinguishable across a battery, ROBUSTLY, become one
   phenomenon -- with a fitted correspondence map, a MEASURED domain of validity, and
   the residual kept as the seed of the next theory:
       cand = mk_equivalence("famA", "famB", pairs=[(pA, pB_matched, dist), ...],
                             robust=<survived independent reseeds?>)
       cand.promotable       # r^2 > 0.999 AND all dists < eps AND robust
       ax = mk_axiom(name, statement, domain="MEASURED validity", residual="what broke")
   NEVER promote without: (a) a robustness check on data the fit never saw,
   (b) a measured domain boundary, (c) the residual NAMED, not discarded.

3. INVARIANCE (the cheap promotion).  A quantity that ignores an intervention across
   its whole range is quotiented out of the ontology:
       inv = mk_invariance("alpha_ignores_axis_column", worst_effect=3e-8, eps=1e-4)
       promote_invariance(inv)   # raises if not promotable -- refusal is a feature

DUTIES (the engine is honest or it is nothing):
 * NEGATIVE CONTROL: every session must include one merge the operator REFUSES
   (the rotation-is-not-gravity test).  If everything you try promotes, you are
   not measuring.
 * EMITTED TENSIONS: a NOVEL deduction or a surviving residual is fed back as the
   next intervention family -- write it down explicitly.
 * A conditional axiom must name its own falsifier at mint time.
"""
import sys

sys.path.insert(0, "/Users/epagogellc/epagoge/jump")

from jump.tension import RuleSystem, Witness                      # noqa: E402
from jump.abduce import EquivalenceCandidate, InvarianceResult    # noqa: E402
from jump.translate import Axiom, promote_invariance              # noqa: E402
from jump.meta import (Deduction, TensionLedger, conditional_axiom,  # noqa: E402
                       deduce, shared_constant_audit)

import numpy as np                                                # noqa: E402

__all__ = ["RuleSystem", "Witness", "EquivalenceCandidate", "InvarianceResult",
           "Axiom", "promote_invariance", "mk_witness", "mk_equivalence",
           "mk_invariance", "mk_axiom", "refuse",
           "Deduction", "deduce", "conditional_axiom", "shared_constant_audit",
           "TensionLedger"]


def mk_witness(name_a, pred_a, name_b, pred_b, scenario=None, tags=()):
    return Witness(scenario=dict(scenario or {}), system_a=name_a, system_b=name_b,
                   pred_a=float(pred_a), pred_b=float(pred_b), tags=frozenset(tags))


def mk_equivalence(family_a, family_b, pairs, robust, reseeds=0):
    """pairs: [(param_a, matched_param_b, trace_distance), ...].

    robust=True REQUIRES reseeds >= 2 -- independent re-checks on data the fit never
    saw (the N_RESEED discipline; the dh/da promotion once ran on a single walk)."""
    if robust and reseeds < 2:
        raise ValueError("refusing robust=True with reseeds < 2: robustness is a "
                         "measurement on unseen data, not a mood")
    xs = np.array([p[0] for p in pairs], float)
    ys = np.array([p[1] for p in pairs], float)
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_tot = float(np.sum((ys - ys.mean()) ** 2)) or 1e-300
    r2 = 1.0 - float(np.sum((ys - pred) ** 2)) / ss_tot
    return EquivalenceCandidate(family_a, family_b, [tuple(map(float, p)) for p in pairs],
                                float(slope), float(intercept), float(r2), bool(robust))


def mk_invariance(name, worst_effect, eps=1e-2):
    return InvarianceResult(name, float(worst_effect), float(worst_effect) < float(eps))


def mk_axiom(name, statement, domain, residual, kind="identity", corr_map=None,
             evidence=None):
    """domain and residual are MANDATORY -- an axiom without a measured domain and a
    named residual is an opinion."""
    if not domain or not residual:
        raise ValueError("refusing to mint an axiom without a measured domain AND a "
                         "named residual")
    return Axiom(name=name, kind=kind, statement=statement, corr_map=corr_map,
                 domain=domain, residual=residual, evidence=dict(evidence or {}))


def refuse(candidate_name, distance, scale, why):
    """The negative-control record: a merge the operator declined, with numbers."""
    return dict(refused=candidate_name, distance=float(distance), scale=float(scale),
                ratio=float(distance) / max(float(scale), 1e-300), why=why)


if __name__ == "__main__":
    # self-test: one of each idiom, plus a refusal
    w = mk_witness("A", 1.0, "B", 1.01, {"x": 0}, ("demo",))
    assert abs(w.divergence - 0.01 / 1.01) < 1e-12
    c = mk_equivalence("f", "g", [(1, 1.0, 1e-4), (2, 2.0, 1e-4), (3, 3.0, 1e-4)], True, reseeds=5)
    assert c.promotable
    a = mk_axiom("t", "f == g", domain="1..3 at 1e-4", residual="none seen below 1e-4",
                 corr_map={"slope": c.map_slope, "intercept": c.map_intercept,
                           "r2": c.map_r2})
    inv = mk_invariance("y_ignores_knob", 3e-8)
    promote_invariance(inv)
    r = refuse("f == h", 9.1e-3, 2e-4, "distance 46x the promotion scale")
    assert r["ratio"] > 40
    try:
        mk_axiom("bad", "x", domain="", residual="")
        raise AssertionError("should have refused")
    except ValueError:
        pass
    print("eja_bridge self-test: ALL PASS (witness, promote, invariance, refusal, "
          "axiom-hygiene)")
