"""Run an attack command and record exactly when it ran (the ground truth for labels). Owner: Nouran.

Every attack used for TRAINING data must be started through this wrapper (or write the same JSONL line),
so labels come from what actually happened, not from guessing:

    python -m fqml.live.attack_log --log runs/lab/attack_log.jsonl --label DoS --attacker 10.10.0.66 \
        -- hping3 -S -p 80 -i u1000 --count 20000 10.10.0.11

One JSON line per run:
    {"label": "DoS", "attacker": "10.10.0.66", "start": 1762300012.31, "end": 1762300031.88,
     "command": "hping3 -S ...", "exit_code": 0}

Run it on the attacker node (its clock must match the capture clock: use NTP or capture on the same host),
or on the capture host with `--start/--end` to add a line by hand for an attack launched elsewhere.
Labels used by the demo: DoS, Recon, BruteForce, ModbusWrite (anything else is kept as its own class).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


def append(log: str | Path, entry: dict) -> None:
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load(log: str | Path) -> list[dict]:
    out = []
    for i, line in enumerate(Path(log).read_text().splitlines(), 1):
        if not line.strip():
            continue
        e = json.loads(line)
        for k in ("label", "attacker", "start", "end"):
            if k not in e:
                raise ValueError(f"{log}:{i}: missing '{k}'")
        if e["end"] <= e["start"]:
            raise ValueError(f"{log}:{i}: end is not after start")
        out.append(e)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--label", required=True, help="DoS | Recon | BruteForce | ModbusWrite | ...")
    ap.add_argument("--attacker", required=True, help="attacker IP as seen in the capture")
    ap.add_argument("--target", default="")
    ap.add_argument("--start", type=float, help="manual entry: epoch start (no command run)")
    ap.add_argument("--end", type=float, help="manual entry: epoch end")
    ap.add_argument("--timeout", type=float, default=0, help="stop the command after N seconds (0 = none)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run")
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    entry = {"label": a.label, "attacker": a.attacker, "target": a.target}
    if a.start is not None and a.end is not None:
        entry.update(start=a.start, end=a.end, command="(manual entry)")
        append(a.log, entry)
        print(f"logged {a.label} {a.start:.2f} -> {a.end:.2f}")
        return 0
    if not cmd:
        ap.error("give a command after --, or --start and --end")
    start = time.time()
    try:
        rc = subprocess.run(cmd, timeout=a.timeout or None).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    except KeyboardInterrupt:
        rc = "interrupted"
    end = time.time()
    entry.update(start=start, end=end, command=shlex.join(cmd), exit_code=rc)
    append(a.log, entry)
    print(f"[attack_log] {a.label}: {end - start:.1f} s logged to {a.log}", file=sys.stderr)
    return 0 if rc in (0, "timeout", "interrupted") else 1


if __name__ == "__main__":
    sys.exit(main())
