# Lens instrument

Four **stable profiles** (cache-friendly prefix) + one **swappable QA block** per use.
Nothing here spins a subagent. The profiles are prompt prefixes: the long invariant
part is identical every call, so it caches; only `## SITUATION` and `## QA` change.

    [ PROFILE (invariant, cached) ]  +  [ SITUATION (swap) ]  +  [ QA (swap) ]

- `karpathy.md`  — META-FLOW. Governs *how* to attack, not what is true. Runs first
  and last: first to choose the next experiment, last to sanity-check the claim.
- `hou.md`       — numerical architecture / where the resolution goes.
- `chen.md`      — rigor, fixed points, conditioning, what would count as proof.
- `elgindi.md`   — analysis, mechanism, is-this-even-the-right-scenario.

**Rule: three separate verdicts, never pre-merged.** Each lens answers alone. Merge
only after, and record where they DISAGREE — the disagreement is the information.
A lens that always agrees with the others is not earning its slot.

**Honesty rule.** These are simulated reasoning styles built from published work,
not the people. When a profile asserts a fact about their actual results, it must be
a citation we have verified (see `project_parzival_blowup_result.md`), not an
inference. Style is simulated; facts are cited or flagged as ours.
