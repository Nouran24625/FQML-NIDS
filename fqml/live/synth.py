"""SYNTHETIC lab traffic, for unit tests and dry runs of the pipeline ONLY. Owner: Nouran.

Never use this for the ITIDA video or any reported result: the video model is trained and shown on
REAL captures from the lab network. Everything written here is marked SYNTHETIC in its file name.

Writes a pcap plus an attack log in the same format as fqml.live.attack_log, so labelling, training,
detection and the dashboard can be exercised end to end without a lab.

    python -m fqml.live.synth --out runs/synth --minutes 6 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import struct
from pathlib import Path

from fqml.live.pcap import ACK, FIN, PSH, RST, SYN

HOSTS = {"camera": "10.10.0.11", "sensor": "10.10.0.12", "plc": "10.10.0.20", "hmi": "10.10.0.30",
         "broker": "10.10.0.40", "attacker": "10.10.0.66"}


def _ip(a: str) -> bytes:
    return bytes(int(x) for x in a.split("."))


def frame(src, dst, proto, sport=0, dport=0, flags=0, payload=b"") -> bytes:
    if proto == 6:
        l4 = struct.pack("!HHIIBBHHH", sport, dport, 1, 1, 5 << 4, flags, 65535, 0, 0) + payload
    elif proto == 17:
        l4 = struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload
    else:
        l4 = struct.pack("!BBHHH", 8, 0, 0, 1, 1) + payload
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(l4), 0, 0, 64, proto, 0, _ip(src), _ip(dst))
    eth = b"\x02\x00\x00\x00\x00\x01" + b"\x02\x00\x00\x00\x00\x02" + b"\x08\x00"
    return eth + ip + l4


class PcapWriter:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))

    def write(self, ts: float, data: bytes):
        sec = int(ts)
        self.f.write(struct.pack("<IIII", sec, int(round((ts - sec) * 1e6)) % 1000000, len(data), len(data)))
        self.f.write(data)

    def close(self):
        self.f.close()


def modbus(fc: int, tid: int) -> bytes:
    return struct.pack("!HHHB", tid, 0, 6, 1) + bytes([fc]) + struct.pack("!HH", 0, 1)


def benign(t0: float, t1: float, rng: random.Random):
    H = HOSTS
    t = t0
    tid = 0
    while t < t1:   # one "tick" ~ every 0.5 s
        for _ in range(rng.randint(2, 6)):                                  # camera -> broker video chunks
            yield t + rng.random() * 0.5, frame(H["camera"], H["broker"], 6, 40001, 8080, PSH | ACK, b"v" * rng.randint(400, 1400))
        if rng.random() < 0.6:                                              # sensor MQTT publish
            yield t + rng.random() * 0.5, frame(H["sensor"], H["broker"], 6, 40100, 1883, PSH | ACK, b"m" * rng.randint(20, 80))
            yield t + rng.random() * 0.5, frame(H["broker"], H["sensor"], 6, 1883, 40100, ACK)
        if rng.random() < 0.5:                                              # HMI reads PLC registers (fc 3)
            tid += 1
            tt = t + rng.random() * 0.4
            yield tt, frame(H["hmi"], H["plc"], 6, 41000, 502, PSH | ACK, modbus(3, tid))
            yield tt + 0.002, frame(H["plc"], H["hmi"], 6, 502, 41000, PSH | ACK, b"r" * 13)
        if rng.random() < 0.1:                                              # HMI polls camera web page
            sp = rng.randint(42000, 60000)
            tt = t + rng.random() * 0.4
            yield tt, frame(H["hmi"], H["camera"], 6, sp, 80, SYN)
            yield tt + 0.001, frame(H["camera"], H["hmi"], 6, 80, sp, SYN | ACK)
            yield tt + 0.002, frame(H["hmi"], H["camera"], 6, sp, 80, PSH | ACK, b"GET / HTTP/1.1\r\n\r\n")
            yield tt + 0.004, frame(H["camera"], H["hmi"], 6, 80, sp, PSH | ACK | FIN, b"h" * 900)
        if rng.random() < 0.05:                                             # attacker box idles normally too
            yield t + rng.random() * 0.4, frame(H["attacker"], H["broker"], 17, 5353, 53, 0, b"q" * 40)
        t += 0.5


def dos(t0, t1, rng):
    rate = rng.choice([300, 600, 1000])
    n = int((t1 - t0) * rate)
    for i in range(n):
        t = t0 + i / rate
        yield t, frame(HOSTS["attacker"], HOSTS["camera"], 6, rng.randint(1024, 65535), 80, SYN)
        if rng.random() < 0.3:
            yield t + 0.0005, frame(HOSTS["camera"], HOSTS["attacker"], 6, 80, 1024, SYN | ACK)


def recon(t0, t1, rng):
    rate = rng.choice([80, 150, 300])
    targets = [HOSTS[h] for h in ("camera", "plc", "sensor", "hmi")]
    t, i = t0, 0
    while t < t1:
        dst = targets[(i // 400) % len(targets)]
        port = 1 + (i % 1024)
        yield t, frame(HOSTS["attacker"], dst, 6, 55555, port, SYN)
        reply = SYN | ACK if port in (80, 502, 22, 1883) else RST | ACK
        yield t + 0.0004, frame(dst, HOSTS["attacker"], 6, port, 55555, reply)
        i += 1
        t += 1 / rate


def brute(t0, t1, rng):
    gap = rng.choice([0.08, 0.15, 0.3])
    t = t0
    while t < t1:
        sp = rng.randint(30000, 65000)
        a, v = HOSTS["attacker"], HOSTS["camera"]
        yield t, frame(a, v, 6, sp, 22, SYN)
        yield t + 0.001, frame(v, a, 6, 22, sp, SYN | ACK)
        yield t + 0.002, frame(a, v, 6, sp, 22, ACK)
        yield t + 0.010, frame(v, a, 6, 22, sp, PSH | ACK, b"S" * 40)
        yield t + 0.020, frame(a, v, 6, sp, 22, PSH | ACK, b"P" * rng.randint(60, 120))
        yield t + 0.030, frame(v, a, 6, 22, sp, PSH | ACK | FIN, b"F" * 30)
        yield t + 0.031, frame(a, v, 6, sp, 22, FIN | ACK)
        t += gap * rng.uniform(0.7, 1.3)


def modbus_write(t0, t1, rng):
    t, tid = t0, 1000
    while t < t1:
        tid += 1
        yield t, frame(HOSTS["attacker"], HOSTS["plc"], 6, 47000, 502, PSH | ACK, modbus(rng.choice([6, 16]), tid))
        yield t + 0.002, frame(HOSTS["plc"], HOSTS["attacker"], 6, 502, 47000, PSH | ACK, b"w" * 12)
        t += rng.uniform(0.2, 0.6)


ATTACKS = {"DoS": dos, "Recon": recon, "BruteForce": brute, "ModbusWrite": modbus_write}


def generate(out_dir: str | Path, minutes: float = 6, seed: int = 0, t0: float = 1_762_300_000.0,
             sessions_per_attack: int = 3, attack_seconds=(15, 30)):
    """Benign traffic all along, plus `sessions_per_attack` sessions of each attack spread over time."""
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    end = t0 + minutes * 60
    plan = [lab for lab in ATTACKS for _ in range(sessions_per_attack)]
    rng.shuffle(plan)
    slot = (end - t0 - 30) / len(plan)
    events, log = list(benign(t0, end, rng)), []
    for k, lab in enumerate(plan):
        s = t0 + 20 + k * slot + rng.uniform(0, max(0.0, slot - attack_seconds[1] - 5))
        e = s + rng.uniform(*attack_seconds)
        events.extend(ATTACKS[lab](s, e, rng))
        log.append({"label": lab, "attacker": HOSTS["attacker"], "start": s, "end": e,
                    "command": f"SYNTHETIC {lab}", "synthetic": True})
    events.sort(key=lambda x: x[0])
    pcap_path, log_path = out / "SYNTHETIC_lab.pcap", out / "SYNTHETIC_attack_log.jsonl"
    w = PcapWriter(pcap_path)
    for ts, data in events:
        w.write(ts, data)
    w.close()
    log_path.write_text("".join(json.dumps(r) + "\n" for r in log))
    return pcap_path, log_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/synth")
    ap.add_argument("--minutes", type=float, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p, l = generate(a.out, a.minutes, a.seed)
    print(f"wrote {p} and {l}  (SYNTHETIC: for testing the pipeline only)")
