#!/usr/bin/env python
"""Pod ops: MCP-aware SSH/transfer + mid-run probe for RunPod pods.

Permanently fixes the recurring SSH pain by never hand-building a fragile ssh
command: the connection details come from the RunPod MCP/API (or a pasted ssh
string), and the flags come from the ~/.ssh/config RunPod block (right key,
no host-key churn). Companion to dedalus_bsq.py's checkpoint/stream/control.

The observer side of the mid-run intervention loop:
  probe   <run_id>            -- last streamed status line (local or via pull)
  send    <run_id> <cmd>      -- drop a control command (stop|checkpoint|extend)
  push    <ssh> <src> <dst>   -- rsync code UP to the pod
  pull    <ssh> <src> <dst>   -- rsync results DOWN from the pod
  run     <ssh> <cmd...>      -- exec a command on the pod

<ssh> is either a pod id (resolved to the proxy: <id>@ssh.runpod.io) or a full
"root@host -p port" direct string. All invocations use id_ed25519_signing via
the ~/.ssh/config block, so no -i / host-key flags are ever needed by hand.

Run:  ~/parzival/.venv/bin/python pod_ops.py <subcommand> ...
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

KEY = str(pathlib.Path.home() / ".ssh" / "id_ed25519_signing")
RUNS = pathlib.Path(__file__).parent / "runs"
SSH_BASE = ["ssh", "-i", KEY, "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=30"]


def _target(ssh: str) -> tuple[list, str]:
    """Return (ssh_command_prefix, rsync_e_string) for a pod id or ssh string.
    Bare id -> RunPod proxy <id>@ssh.runpod.io (stable host, needs pubkey
    registered in account settings). Full 'root@host -p PORT' -> direct."""
    if "@" in ssh:                            # explicit root@host [-p port]
        parts = ssh.split()
        host = parts[0]
        port = parts[parts.index("-p") + 1] if "-p" in parts else "22"
        pre = SSH_BASE + ["-p", port, "-o", "UserKnownHostsFile=/dev/null",
                          "-o", "LogLevel=ERROR", host]
        e = f"ssh -i {KEY} -p {port} -o IdentitiesOnly=yes " \
            f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
        return pre, e
    host = f"{ssh}@ssh.runpod.io"             # proxy by pod id
    pre = SSH_BASE + [host]
    e = f"ssh -i {KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
    return pre, e


def probe(run_id: str) -> None:
    """Last streamed status (works while the run is live)."""
    sp = RUNS / f"stream_{run_id}.jsonl"
    if not sp.exists():
        print(f"no stream for {run_id} at {sp}"); return
    lines = [ln for ln in sp.read_text().splitlines() if ln.strip()]
    if not lines:
        print("stream empty (run starting up)"); return
    last = json.loads(lines[-1])
    print(f"[{run_id}] t={last['t']:.4f} it={last['it']} "
          f"sup|grad b|={last['sup_gb']:.3e} b2_drift={last['b2_drift']:.2e} "
          f"dt={last['dt']:.1e} wall={last['wall']}s  ({len(lines)} samples)")


def send(run_id: str, cmd: str, stop_time: float | None = None) -> None:
    """Drop a control command the running sim consumes on its next check."""
    payload = {"cmd": cmd}
    if stop_time is not None:
        payload["stop_time"] = stop_time
    (RUNS / f"control_{run_id}.json").write_text(json.dumps(payload))
    print(f"sent {payload} to run {run_id}")


def _rsync(src: str, dst: str, e: str) -> None:
    subprocess.run(["rsync", "-avz", "--partial", "-e", e, src, dst], check=True)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__); return
    op = sys.argv[1]
    if op == "probe":
        probe(sys.argv[2])
    elif op == "send":
        st = float(sys.argv[4]) if len(sys.argv) > 4 else None
        send(sys.argv[2], sys.argv[3], st)
    elif op == "run":
        pre, _ = _target(sys.argv[2])
        subprocess.run(pre + sys.argv[3:])
    elif op == "push":
        _, e = _target(sys.argv[2])
        _rsync(sys.argv[3], sys.argv[4], e)   # local -> pod:path
    elif op == "pull":
        _, e = _target(sys.argv[2])
        _rsync(sys.argv[3], sys.argv[4], e)   # pod:path -> local
    else:
        print(f"unknown op {op!r}\n{__doc__}")


if __name__ == "__main__":
    main()
