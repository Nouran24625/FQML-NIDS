"""Attack runner for the FQML-Guard demo lab (lab only). Owner: Kareem.

Runs standard, off-the-shelf tools from the attacker container, each wrapped by fqml.live.attack_log
so its start/end time is recorded -- that is how the captured traffic gets its labels, the same way
the CICIoT2023 dataset the project uses was built. The tools only ever target the lab's own hosts
(10.10.0.0/24, no route to the internet), and rates are capped so the capture and the laptop stay
stable on camera.

    # one attack, to check it in the live view:
    python3 attacks.py one DoS --seconds 20
    # a full labelled run: normal gap, then every attack with a gap between, logged:
    python3 attacks.py run --log /work/attack_log.jsonl --repeat 1

Attacks (all from 10.10.0.66):
    DoS         rate-limited SYN flood to the camera's web port (hping3)
    Recon       TCP SYN port scan across the lab (nmap)
    BruteForce  SSH password guessing against the camera (hydra, small wordlist)
    ModbusWrite a burst of writes to the PLC's setpoint register (our Modbus client)

Run inside the attacker container:
    docker exec -it lab-attacker python3 /app/attacks.py run --log /work/attack_log.jsonl
(/work is the repo mounted into the container; mount it in compose or with -v if needed.)
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time

# lab hosts (fixed IPs from compose.yml)
CAMERA, SENSOR, PLC, HMI, BROKER = "10.10.0.11", "10.10.0.12", "10.10.0.20", "10.10.0.30", "10.10.0.40"
SUBNET = "10.10.0.0/24"
ATTACKER = "10.10.0.66"
LABELS = ["DoS", "Recon", "BruteForce", "ModbusWrite"]


def have(tool: str) -> bool:
    return subprocess.run(["sh", "-c", f"command -v {tool}"], stdout=subprocess.DEVNULL).returncode == 0


# ----- each attack returns the argv of a standard tool; the runner times and logs it -----

def dos_cmd(seconds: float, rng: random.Random) -> list[str]:
    """Rate-limited SYN flood to the camera web port. -i uN = one packet every N microseconds.
    NOT hping3 --flood: a fixed ~700-1500 pkt/s is clearly a flood but keeps capture and laptop stable."""
    us = rng.choice([700, 1000, 1400])                 # ~700-1400 packets/s
    count = int(seconds * 1_000_000 / us)
    return ["hping3", "-S", "-p", "80", "-i", f"u{us}", "-c", str(count), CAMERA]


def recon_cmd(seconds: float, rng: random.Random) -> list[str]:
    """TCP SYN scan of the lab. -T4 is brisk but not the fastest; a port range keeps it ~scan-length."""
    top = rng.choice([1000, 2000])
    return ["nmap", "-sS", "-n", "-T4", "--max-retries", "1", "-p", f"1-{top}", SUBNET]


def brute_cmd(seconds: float, rng: random.Random, wordlist: str) -> list[str]:
    """SSH password guessing against the camera login (hydra), small wordlist, few threads."""
    return ["hydra", "-l", "admin", "-P", wordlist, "-t", "4", "-f", "-I", f"ssh://{CAMERA}"]


def modbus_write_burst(seconds: float, rng: random.Random, log, label_runner) -> dict:
    """Burst of writes to the PLC setpoint register (function 6), ~3/s -- an attacker fiddling with the
    process. Uses our own Modbus client (no external tool). Logged as one run."""
    sys.path.insert(0, "/workspace")                   # repo is mounted here; fall back to vendored copy
    try:
        from demo.lab.nodes.modbus import Client
    except Exception:
        sys.path.insert(0, os.path.dirname(__file__))
        from modbus import Client                       # vendored next to this script in the image
    start = time.time()
    c = Client(PLC)
    n = 0
    while time.time() - start < seconds:
        try:
            c.write_register(1, rng.randint(200, 800))  # setpoint
            n += 1
        except OSError:
            c.close()
        time.sleep(1 / 3)
    end = time.time()
    entry = {"label": "ModbusWrite", "attacker": ATTACKER, "target": PLC, "start": start, "end": end,
             "command": f"modbus write burst ({n} writes to setpoint)", "writes": n}
    label_runner(log, entry)
    return entry


def run_logged(label: str, cmd: list[str], seconds: float, log, label_runner, timeout_pad: float = 20) -> dict:
    """Run a tool, record start/end, append one log line."""
    start = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=seconds + timeout_pad)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    except FileNotFoundError:
        print(f"  {label}: tool '{cmd[0]}' not found in this image", file=sys.stderr)
        return {}
    end = time.time()
    entry = {"label": label, "attacker": ATTACKER, "target": cmd[-1], "start": start, "end": end,
             "command": " ".join(cmd), "exit_code": rc}
    label_runner(log, entry)
    return entry


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    one = sub.add_parser("one", help="run a single attack (to check it in the live view)")
    one.add_argument("label", choices=LABELS)
    one.add_argument("--seconds", type=float, default=20)
    one.add_argument("--log", default="/workspace/runs/lab/attack_log.jsonl")
    one.add_argument("--wordlist", default="/app/wordlist.txt")
    one.add_argument("--seed", type=int, default=0)

    full = sub.add_parser("run", help="normal gap + every attack with gaps, all logged")
    full.add_argument("--log", default="/workspace/runs/lab/attack_log.jsonl")
    full.add_argument("--seconds", type=float, default=30, help="length of each attack")
    full.add_argument("--gap", type=float, default=30, help="normal-traffic gap between attacks")
    full.add_argument("--repeat", type=int, default=1, help="how many times through the attack list")
    full.add_argument("--wordlist", default="/app/wordlist.txt")
    full.add_argument("--only", default="", help="comma-separated subset, e.g. DoS,Recon")
    full.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    sys.path.insert(0, "/workspace")
    from fqml.live.attack_log import append
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)

    def do(label: str, seconds: float, rng: random.Random) -> dict:
        if label == "DoS":
            return run_logged("DoS", dos_cmd(seconds, rng), seconds, a.log, append)
        if label == "Recon":
            return run_logged("Recon", recon_cmd(seconds, rng), seconds, a.log, append)
        if label == "BruteForce":
            return run_logged("BruteForce", brute_cmd(seconds, rng, a.wordlist), seconds, a.log, append)
        if label == "ModbusWrite":
            return modbus_write_burst(seconds, rng, a.log, append)
        raise ValueError(label)

    rng = random.Random(a.seed)
    if a.mode == "one":
        print(f"running {a.label} for {a.seconds:g}s from {ATTACKER} ...", flush=True)
        e = do(a.label, a.seconds, rng)
        if e:
            print(f"logged {e['label']}: {e['end'] - e['start']:.1f}s -> {a.log}")
        return

    labels = [x for x in (a.only.split(",") if a.only else LABELS) if x in LABELS]
    print(f"full run: {a.repeat}x {labels}, {a.seconds:g}s each, {a.gap:g}s normal gaps", flush=True)
    print(f"keep normal traffic running; attacks are logged to {a.log}\n")
    time.sleep(a.gap)                                   # start with a normal gap
    for r in range(a.repeat):
        for label in labels:
            t = time.strftime("%H:%M:%S")
            print(f"[{t}] {label} ...", flush=True)
            e = do(label, a.seconds, rng)
            if e:
                print(f"       logged {e['end'] - e['start']:.1f}s", flush=True)
            time.sleep(a.gap)
    print("\ndone. Normal traffic continues. Stop the capture when you have enough.")


if __name__ == "__main__":
    main()
