"""Live detector: packets -> 2-second windows -> model -> detections for the dashboard. Owner: Nouran.

Live (any lab; the command before | must write a pcap stream to stdout):
    tcpdump -i <lab-iface> -n -U -s 128 -w - 'not port 22 and not port 8050' \
      | python -m fqml.live.detect --stdin --bundle models/lab_demo --state runs/live \
          --hosts 10.10.0.66=attacker,10.10.0.20=PLC,10.10.0.30=HMI --attacker 10.10.0.66
  (EVE-NG: capture on the lab bridge from the EVE host, e.g. `ssh root@eve tcpdump -i vnet0_1 ...`;
   Docker lab: `sudo tcpdump -i br-<lab network id> ...` on the laptop.)

Replay a recorded capture at real speed (rehearsal, or a fallback that is SAID to be a replay):
    python -m fqml.live.detect --pcap runs/lab/demo.pcap --realtime --bundle models/lab_demo --state runs/live

Writes to --state:
    status.json        what is running (source, model, classes, hosts), updated every window
    detections.jsonl   one line per window: {"t", "w", "pkts", "rows": [{"host", "pred", "conf", "n_out"}]}
A row is an ALERT when pred != Benign, conf >= --threshold and the host sent >= min_pkts packets.
--attacker is lab ground truth used ONLY by the dashboard to count false alarms; the model never sees it.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

from fqml.live.features import WindowAggregator
from fqml.live.model import Classifier
from fqml.live.pcap import PcapError, read_pcap

RED, GREEN, DIM, RESET = "\033[1;31m", "\033[32m", "\033[2m", "\033[0m"


def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def parse_hosts(s: str) -> dict:
    out = {}
    for part in filter(None, (p.strip() for p in s.split(","))):
        ip, _, name = part.partition("=")
        out[ip.strip()] = name.strip() or ip.strip()
    return out


class Detector:
    def __init__(self, clf: Classifier, state: Path, *, threshold: float, source: str,
                 hosts: dict, attackers: list[str], color: bool = True):
        self.clf, self.state, self.threshold = clf, state, threshold
        self.min_pkts = int(clf.meta.get("min_pkts", 3))
        self.color = color
        state.mkdir(parents=True, exist_ok=True)
        self.det_path = state / "detections.jsonl"
        self.det_path.write_text("")                       # new session: start clean
        self.status = {"source": source, "started": time.time(), "classes": clf.classes,
                       "model": clf.meta.get("description", clf.kind), "model_kind": clf.kind,
                       "federated_sites": clf.meta.get("federated_sites", 0),
                       "window": clf.meta["window"], "threshold": threshold, "min_pkts": self.min_pkts,
                       "hosts": hosts, "attackers": attackers, "windows": 0, "packets": 0, "alerts": 0,
                       "running": True}
        self.hosts = hosts
        self._write_status()

    def _write_status(self):
        _atomic_write(self.state / "status.json", json.dumps(self.status))

    def window(self, t: float, rows: list[dict], pkts: int):
        ready = [r for r in rows if r["n_out"] >= self.min_pkts]
        P = self.clf.predict_proba(ready)
        out = []
        for r, p in zip(ready, P):
            k = int(p.argmax())
            out.append({"host": r["host"], "pred": self.clf.classes[k], "conf": round(float(p[k]), 3),
                        "n_out": int(r["n_out"])})
        alerts = [o for o in out if o["pred"] != "Benign" and o["conf"] >= self.threshold]
        with open(self.det_path, "a") as f:
            f.write(json.dumps({"t": t, "w": self.clf.meta["window"], "pkts": pkts, "rows": out}) + "\n")
        self.status["windows"] += 1
        self.status["packets"] += pkts
        self.status["alerts"] += len(alerts)
        self.status["last_t"] = t
        self._write_status()
        clock = time.strftime("%H:%M:%S", time.localtime(t))
        if alerts:
            msg = "  ".join(f"{a['pred']} from {self.hosts.get(a['host'], a['host'])} ({a['conf']:.0%})" for a in alerts)
            print(f"{RED if self.color else ''}{clock}  ALERT  {msg}{RESET if self.color else ''}", flush=True)
        else:
            msg = f"{pkts:6d} pkts, {len(out)} hosts classified, all Benign" if out else f"{pkts:6d} pkts, no active hosts"
            print(f"{DIM if self.color else ''}{clock}  ok     {msg}{RESET if self.color else ''}", flush=True)

    def close(self):
        self.status["running"] = False
        self._write_status()


def run(stream, det: Detector, *, realtime: bool = False, live: bool = False):
    W = det.clf.meta["window"]
    agg = WindowAggregator(W)
    counts = {}

    def emit(done):
        for t, rows in done:
            det.window(t, rows, counts.pop(t, 0))

    def account(p):
        t = agg._start_of(p.ts)
        counts[t] = counts.get(t, 0) + 1

    if not live:                                            # file: optionally paced at real speed
        first_ts = first_wall = None
        for p in read_pcap(stream):
            if realtime:
                if first_ts is None:
                    first_ts, first_wall = p.ts, time.time()
                lag = (p.ts - first_ts) - (time.time() - first_wall)
                if lag > 0:
                    time.sleep(lag)
            account(p)
            emit(agg.feed(p))
        emit(agg.flush())
        return

    q: queue.Queue = queue.Queue(maxsize=200000)
    END = object()

    def reader():
        try:
            for p in read_pcap(stream):
                q.put(p)
        except PcapError as e:
            print(f"capture error: {e}", file=sys.stderr)
        finally:
            q.put(END)

    threading.Thread(target=reader, daemon=True).start()
    last_ts, last_wall = None, None
    while True:
        try:
            p = q.get(timeout=0.25)
        except queue.Empty:
            if last_ts is not None:                         # idle: close windows that have ended
                emit(agg.flush(now=last_ts + (time.time() - last_wall) - 0.3))
            continue
        if p is END:
            emit(agg.flush())
            return
        last_ts, last_wall = p.ts, time.time()
        account(p)
        emit(agg.feed(p))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--stdin", action="store_true", help="read a live pcap stream from stdin")
    src.add_argument("--pcap", help="read a recorded pcap file")
    ap.add_argument("--realtime", action="store_true", help="with --pcap: replay at the original speed")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--state", default="runs/live")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--hosts", default="", help="ip=name,... labels shown on the dashboard")
    ap.add_argument("--attacker", default="", help="comma-separated attacker IPs (dashboard ground truth only)")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args(argv)
    clf = Classifier(a.bundle)
    source = "live capture" if a.stdin else (f"replay of {Path(a.pcap).name}" + (" (real speed)" if a.realtime else ""))
    det = Detector(clf, Path(a.state), threshold=a.threshold, source=source, hosts=parse_hosts(a.hosts),
                   attackers=[x.strip() for x in a.attacker.split(",") if x.strip()], color=not a.no_color)
    print(f"model: {det.status['model']} | classes: {', '.join(clf.classes)} | window {clf.meta['window']} s | source: {source}",
          flush=True)
    try:
        if a.stdin:
            run(sys.stdin.buffer, det, live=True)
        else:
            with open(a.pcap, "rb") as f:
                run(f, det, realtime=a.realtime)
    except KeyboardInterrupt:
        pass
    finally:
        det.close()


if __name__ == "__main__":
    main()
