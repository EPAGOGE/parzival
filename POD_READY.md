# Parzival — pod-readiness manifest (2026-07-23)

Reconciliation of everything built/validated this session against the swarm
setup, so a CUDA pod run reproduces the laptop program at scale. **No pod
started** — this is the integrated, ready state.

## The engine: swarm_m1.py (the pod workload)

Port of `swarm_gpu.py` (the original bare CUDA/JAX engine), now carrying the
whole session's machinery. Device-agnostic torch — the `cuda` path is written
but validated on the pod (no CUDA on this laptop; smoke-then-scale, per the
gate discipline).

### Physics (validated on M1)
- CLM / De Gregorio / **generalized CLM** with the advection dial `--a`
  (a=0 CLM, a=1 DG). ν is an exact gauge; A is an exact gauge (Boussinesq) —
  the diagram is `(a, A/ν)`, so no ν-scan needed.
- IC families `--ic cos|cos2` (cos2 forced for DG: cos is an exact DG steady
  state — the degenerate-IC catch).
- Gates before science: exact inviscid (2.8e-15), torch-fp64 == numpy-fp64,
  fp32-vs-fp64 fate agreement. Run every launch unless `--skip-gates`.

### Hardcheck stack (the solver-ecosystem tiers, all wired)
| Tier | Flag | Type | Catches |
|---|---|---|---|
| H (search) | default | fp32 batched swarm | — |
| Q in-loop | `--audit` | M1 step-tripwire + M2 aliasing identity (CONSISTENCY) | NaN/Inf, gross state damage, operator faults |
| Q shadow | `--shadow FRAC` | independent fp64 recompute (REDUNDANCY) | consistent-ALU SDC the in-loop wire is blind to |
| P (proof) | — | interval arithmetic | unbuilt; the CAP rung |

- M1 is flux-normalized (the `|dE|` bug that false-flagged 65% is fixed).
- Shadow verdict is cohort-outlier + systematic-envelope, NOT `fp32!=fp64`
  (fp32 legitimately diverges ~7%). 0 false positives; 3% fault 32/32.
- `corrupt` count is a distinct ledger channel (integrity, not resolution).

### Pod-perf integration (this session's "while-waiting" items)
- `--device cuda` + auto-resolution (cuda > mps > cpu).
- `--tf32` (default OFF): TF32 is a distinct meter era; its bias must be
  characterized before use, same discipline as the fp32-vs-fp64 shift.
- `--sync-every K`: classify fates every K steps, cutting the host↔device
  round-trip that is the CUDA PCIe bottleneck. Audit residuals accumulate as
  a device-side running max so sensitivity is preserved. Default 1
  (bit-identical to prior behavior). Validated K=5 clean.
- `--batch`: the docstring's B=262144 is a floor on a 24GB card, not a
  ceiling — push until throughput plateaus (that is the real saturation).
- `--search-dtype float32|float64`: precision of the SEARCH tier. Default
  fp32 (throughput=statistics; near-boundary bias measured ~0.13 and fp64-
  anchored). fp64 removes that bias + the anchor step, ~2x slower on
  A100/H100, non-starter on 4090, unavailable on MPS. **Accuracy is grid-
  limited, not precision-limited** (grid error ~0.14 @N128 ≈ the fp32 bias),
  so `--grid` is the higher-leverage accuracy lever than fp64-at-fixed-N.
- Every run records a `meter` provenance block (tf32/sync/audit/shadow) into
  the summary JSON and vault note.

### Still driver-level, not engine (for the pod runbook, not code)
- Concurrent campaign points sharing one GPU (utilization to ceiling) — an
  orchestration choice in the sweep driver, not an engine flag.
- cuPyNumeric / cuFFT / Warp — pod-tier ports if throughput demands; the
  current torch path is the first thing to run and measure.

## Standalone tools (not in the hot loop, correctly)
- `audit.py` — M1/M2 + BDF cross-method auditor (shadow imports it).
- `shadow.py` — Tier-Q module (`--shadow` wires it).
- `rung1_profile.py` / `rung1b_validate.py` — edge-state Newton continuation
  (produced the ν-gauge + simple-pole result).
- `sweep_a.py` / `sweep_full.py` — campaign drivers (subprocess per point).
- `plot_portrait.py` — the phase-portrait figure.

## Boussinesq branch (separate subtree, era B2)
- `boussinesq/bq.py` (parity box, EXCLUDED from the proven basin — free-slip
  wall) and `boussinesq/bq2.py` (Chebyshev wall, θ,w free on the wall — the
  theorem's geometry class). fp64 scipy; A100-tier if ported, NOT a 4090
  workload as-is (fp64).

## First pod smoke (when we do launch)
1. `--device cuda --skip-gates` OFF: gates must pass on CUDA first.
2. Tiny `--batch 4096 --iters 200` smoke; confirm fates + `--audit`/`--shadow`
   behave (they use device-agnostic ops; shadow casts to CPU fp64).
3. Then scale `--batch` to saturation, add `--sync-every`, run the campaign.

## Pod-resilience layer (2026-07-23) — built + verified

Every piece maps to a way we got burned:
- **Checkpoint/resume** (`--checkpoint-wall N --resume`): Dedalus-native state dump
  every N wall-seconds; resume picks up from the last checkpoint. VERIFIED: killed
  a run at t=0.768, resumed from t=0.682, finished clean. Pod death costs <= one
  checkpoint interval, not the whole run.
- **Live stream** (`runs/stream_<id>.jsonl`): one JSON row per diagnostic,
  appended -> tailable mid-run (fixes the blind-run problem). `pod_ops.py probe <id>`
  reads the last line.
- **Mid-run control** (`runs/control_<id>.json`): observer drops
  `{"cmd":"stop|checkpoint|extend","stop_time":T}`; the sim consumes it and acts
  WITHOUT a kill. VERIFIED: graceful stop mid-run, output saved. `pod_ops.py send
  <id> <cmd>`.
- **SSH permanent fix**: `~/.ssh/config` RunPod block (IdentitiesOnly + the right
  key + host-key-churn handling + proxy) and `pod_ops.py push/pull/run` which build
  the ssh/rsync invocation from a pod id (proxy) or a direct string -- no hand-built
  `-i`/host-key flags ever again. ONE manual step remaining: register the pubkey
  (`~/.ssh/id_ed25519_signing.pub`) in RunPod -> Settings -> SSH Public Keys (web
  console; not exposed in the MCP), which auto-populates every new pod's
  authorized_keys. Config validated via `ssh -G`; end-to-end SSH validates on the
  first live pod.
