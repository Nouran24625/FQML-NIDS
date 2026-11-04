"""Watch live traffic as the model will see it: one line per active host every window. Owner: Nouran.

No model needed. Used to check the capture (lab Step 4) and that each attack looks different from
normal traffic before any training data is recorded.

    tcpdump -i br-fqml -n -U -s 160 -w - | python -m fqml.live.watch
    python -m fqml.live.watch --pcap runs/lab/train.pcap          # same view of a recorded file

Columns: pkts sent | distinct destination hosts | distinct destination ports | % bare SYN (no ACK) |
new connections | Modbus requests / writes | KB sent
"""
from __future__ import annotations

import argparse
import sys
import time

from fqml.live.detect import parse_hosts, run

LAB_HOSTS = ("10.10.0.11=camera,10.10.0.12=sensor,10.10.0.20=plc,10.10.0.30=hmi,"
             "10.10.0.40=broker,10.10.0.66=attacker,10.10.0.1=docker-gw")


class Watcher:
    """Duck-types the parts of detect.Detector that detect.run uses."""

    def __init__(self, window: float, hosts: dict, out=sys.stdout):
        self.clf = type("C", (), {"meta": {"window": window}})()
        self.hosts, self.out = hosts, out
        self.windows = 0

    def window(self, t: float, rows: list[dict], pkts: int):
        self.windows += 1
        clock = time.strftime("%H:%M:%S", time.localtime(t))
        print(f"\n{clock}  {pkts} packets", file=self.out)
        if rows:
            print(f"  {'host':22s} {'pkts':>6s} {'hosts':>5s} {'ports':>5s} {'syn%':>5s} {'conns':>5s} "
                  f"{'mb r/w':>7s} {'KB':>7s}", file=self.out)
        for r in sorted(rows, key=lambda r: -r["n_out"]):
            name = f"{self.hosts.get(r['host'], '?')} {r['host']}"
            mb = f"{r['modbus_req']}/{r['modbus_write']}"
            print(f"  {name:22s} {r['n_out']:6d} {r['n_dst_hosts']:5d} {r['n_dst_ports']:5d} "
                  f"{100 * r['syn_frac']:5.0f} {r['new_conns']:5d} {mb:>7s} {r['bytes_out'] / 1024:7.1f}",
                  file=self.out)
        self.out.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcap", help="read a recorded file instead of a stream on stdin")
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--hosts", default=LAB_HOSTS, help="ip=name,... (default: the lab's IPs)")
    a = ap.parse_args(argv)
    w = Watcher(a.window, parse_hosts(a.hosts))
    try:
        if a.pcap:
            with open(a.pcap, "rb") as f:
                run(f, w)
        else:
            run(sys.stdin.buffer, w, live=True)
    except (KeyboardInterrupt, BrokenPipeError):
        pass


if __name__ == "__main__":
    main()
