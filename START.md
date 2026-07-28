# Parzival — M1 lab

Local M1-capable port of the swarm engine (torch / Apple MPS), wired to the live
Obsidian vault (chain continued from head `fda00cc43ec09e60`).

## Layout

- `swarm_m1.py` — the M1 engine (validated port of `swarm_gpu.py`)
- `swarm_gpu.py` — CUDA/JAX original, reference only
- `vault/` — the live hash-chained vault (seeker edition: concepts, measurements,
  hunches, overlaps). Point Obsidian at this folder.
- `runs/` — run logs + summary JSON

## Start here

Quick smoke (~1 min, gates + short swarm):

```bash
~/parzival/.venv/bin/python ~/parzival/swarm_m1.py --smoke
```

Full M1 test (gates + B=16384 x 4000 iters, appends a vault note):

```bash
~/parzival/.venv/bin/python ~/parzival/swarm_m1.py
```

Hardchecks + pod flags (see POD_READY.md for the full manifest):

```bash
~/parzival/.venv/bin/python ~/parzival/swarm_m1.py --audit --shadow 0.02 --sync-every 5
```

- `--device cuda` (auto-selected on a pod) with `--tf32` OFF by default
  (TF32 is a distinct meter era; opt in only after characterizing its bias).
- `--audit` = in-loop M1/M2 consistency monitors; `--shadow FRAC` = Tier-Q
  independent-fp64 redundancy auditor (catches consistent-ALU SDC).
- `--sync-every K` cuts host<->device round-trips on CUDA; default 1.

The pre-registered hover hunt (De Gregorio advection term switched on —
tests `hunches/hover-requires-depletion`):

```bash
~/parzival/.venv/bin/python ~/parzival/swarm_m1.py --model dg
```

DG note (adversarial review catch, 2026-07-22): every single-mode `A*cos(kx)`
is an exact steady state of the inviscid DG nonlinearity, so from the CLM
standard IC a DG run decays as `A*exp(-nu*t)*cos(x)` identically — a false
null. DG therefore defaults to `--ic cos2` (two-mode, sup-normalized); CLM
keeps `--ic cos` for comparability with the CUDA original.

The a-dial (generalized CLM, `w_t + a*u*w_x = u_x*w - nu*Lambda*w`;
a=0 is CLM, a=1 is DG):

```bash
~/parzival/.venv/bin/python ~/parzival/swarm_m1.py --model gclm --a 0.95 --ic cos2
```

Full (a, A) phase-diagram campaign (11 swarm runs, ~30-40 min, aggregate
`gclm-phase-curve` vault note at the end):

```bash
~/parzival/.venv/bin/python ~/parzival/sweep_a.py
```

Marginal zone found by fp64 calibration 2026-07-22: hover phenomenology
(lingering blowup, T*=7.54 at a=0.95, A=24) lives at a in (0.90, 0.98);
total depletion onsets between a=0.95 and 0.98 at nu=1.

## What the gates prove before any science runs

1. **Exact-solution gate** — inviscid CLM vs the closed form
   `w = 4*w0/((2 - t*Hw0)^2 + t^2*w0^2)`; passes at ~3e-15 (machine precision).
2. **Implementation gate (2a)** — torch fp64 vs independent numpy fp64 reference:
   fates identical, T* to ~1e-5. The port is the same machine as the original.
3. **Precision gate (2b)** — device fp32 fates must match fp64. Known
   characteristic: fp32 rounding noise hastens blowup, T* ~7% early (measured
   2026-07-22, same on CPU-fp32 as MPS — platform-independent, precision-driven).
   The swarm measures fate probability vs amplitude, not T*, so this does not
   contaminate A*; tripwire at 15% catches gross breakage.

## Ground truth to expect

- Boundary cells bracketing A* = 5.5348 (CPU bisection, vault:
  `astar-bisection`); the 0.5-crossing estimate prints against it.
- `hover_candidates` ~ 0 for CLM. If it switches on under `--model dg`, that is
  the pre-registered hunch firing — do not touch the hunch note first, let the
  vault record it.

## Vault emitter

Each completed run appends `vault/measurements/swarm-m1-<model>-<stamp>.md`
continuing the hash chain (`prev` = current head). Hash rule (declared in each
note): `sha256(json sort_keys {name, values, prev, ts})[:16]`.
