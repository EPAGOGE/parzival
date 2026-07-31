#!/usr/bin/env python3
"""One sentence. No table, no hedging beyond what the CIs actually license."""
import subprocess, re, sys, glob
out = subprocess.run([sys.executable, "orbit_speed.py", "G5_11", "G5_42", "G5_77"],
                     capture_output=True, text=True).stdout
rows = []
for ln in out.splitlines():
    m = re.match(r"(G5_\d+)\s+slope\s+([-+\d.]+)\s+90% CI \[([-+\d.]+),([-+\d.]+)\].*?-> (.+)$", ln)
    if m: rows.append((m.group(1), float(m.group(2)), float(m.group(3)),
                       float(m.group(4)), m.group(5).strip()))
done = len([1 for s in (11,42,77) if glob.glob(f"/tmp/G5_{s}.log")
            and "AXISYM" in open(f"/tmp/G5_{s}.log").read()])
print("=" * 66); print("BOTTOM LINE")
if not rows:
    print("  No seed has enough post-transient data yet. Nothing is claimed."); print("=" * 66); sys.exit()
labs = set(r[4].split()[0] for r in rows)
sl = [r[1] for r in rows]
if labs == {"SETTLING"}:
    print(f"  All {len(rows)} readable seeds SETTLE (slopes {min(sl):+.2f} to {max(sl):+.2f}).")
    print("  Generic near-wall data converges onto a profile, like the engineered")
    print("  corner IC does. NO wandering detected in the resolvable window.")
    print("  SCOPE: ~1-2 e-foldings of amplitude growth only. This does not")
    print("  address behaviour at 10+ e-foldings, where wandering would live.")
elif labs == {"WANDERING"}:
    print(f"  All {len(rows)} readable seeds WANDER (slopes {min(sl):+.2f} to {max(sl):+.2f}).")
    print("  Generic orbits do NOT settle onto any profile: first measured")
    print("  footprint in the sector where no analytic tools exist.")
elif "INCONCLUSIVE" in labs:
    print(f"  {len(rows)} seeds readable, verdicts split or CIs straddle: {sorted(labs)}.")
    print("  Not enough growth to separate the regimes. See per-seed lines for")
    print("  the exact e-foldings still needed.")
else:
    print(f"  Seeds DISAGREE: {sorted(labs)}. Slopes {min(sl):+.2f} to {max(sl):+.2f}.")
    print("  Genericity itself is in question -- different data, different fate.")
print(f"  ({done}/3 runs complete, {len(rows)} readable)")
print("=" * 66)
