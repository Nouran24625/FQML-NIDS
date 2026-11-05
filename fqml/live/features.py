"""Per-host, per-window traffic features.

Time is cut into fixed windows (default 2 s). For every IPv4 host that SENT at least one packet in a
window we compute one feature row from the packets it sent ("out") and received ("in").
Rows are the unit the model classifies: "host 10.0.0.66 during 12:00:04-12:00:06 looks like Recon".

Why these features: each attack family in the demo leaves a different footprint
  DoS        very many packets to one host, mostly SYN or UDP, tiny or no payload
  Recon      many destination ports and/or hosts, SYN-only probes, RST replies coming back
  BruteForce many short TCP connections (new source ports) to ONE service port, with payload
  ModbusWrite Modbus/TCP write function codes sent to the PLC (normal HMI traffic only reads)
Benign IoT/HMI traffic is periodic, few destinations, few connections.

These are lab features, not the CICIoT2023 columns: the demo model is trained on lab traffic (said in
the video). No IP addresses, ports or timestamps are used as features (only counts and ratios).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from fqml.live.pcap import ACK, FIN, MODBUS_WRITE_CODES, PROTO_ICMP, PROTO_TCP, PROTO_UDP, RST, SYN, Pkt

FEATURES = [
    "n_out", "n_in", "bytes_out", "bytes_in", "len_mean", "len_std",
    "n_dst_hosts", "n_dst_ports", "n_src_ports",
    "frac_tcp", "frac_udp", "frac_icmp",
    "syn_frac", "rst_in_frac", "fin_frac", "new_conns", "top_dport_frac", "payload_frac",
    "iat_mean", "iat_std", "modbus_req", "modbus_write", "in_out_ratio",
]
# Default model inputs (one per qubit), each aimed at one footprint above. Fixed BEFORE seeing lab data.
FOOTPRINT = [
    "n_out",          # volume                       (DoS)
    "syn_frac",       # bare SYN probes / SYN flood   (DoS, Recon)
    "n_dst_ports",    # breadth of ports touched      (Recon)
    "rst_in_frac",    # refused connections coming back (Recon)
    "new_conns",      # connection churn              (BruteForce, DoS)
    "top_dport_frac", # focus on a single service     (BruteForce, DoS)
    "payload_frac",   # packets carrying data         (BruteForce vs floods/scans)
    "modbus_write",   # Modbus/TCP write requests     (ModbusWrite)
]
# Features that are already ratios in [0, 1]; all others are counts/sizes/times and get log1p.
RATIO_FEATURES = {"frac_tcp", "frac_udp", "frac_icmp", "syn_frac", "rst_in_frac", "fin_frac",
                  "top_dport_frac", "payload_frac"}


@dataclass
class _HostAcc:
    out: list[Pkt] = field(default_factory=list)
    n_in: int = 0
    bytes_in: int = 0
    tcp_in: int = 0
    rst_in: int = 0


def host_row(out: list[Pkt], n_in: int, bytes_in: int, tcp_in: int, rst_in: int) -> dict:
    n = len(out)
    lens = [p.length for p in out]
    mean = sum(lens) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in lens) / n)
    tcp = [p for p in out if p.proto == PROTO_TCP]
    n_tcp = len(tcp)
    syn_only = [p for p in tcp if p.flags & SYN and not p.flags & ACK]
    ports = defaultdict(int)
    for p in out:
        if p.proto in (PROTO_TCP, PROTO_UDP):
            ports[p.dport] += 1
    ts = sorted(p.ts for p in out)
    iats = [b - a for a, b in zip(ts, ts[1:])]
    iat_mean = sum(iats) / len(iats) if iats else 0.0
    iat_std = math.sqrt(sum((x - iat_mean) ** 2 for x in iats) / len(iats)) if iats else 0.0
    modbus = [p for p in out if p.modbus_fc >= 0]
    return {
        "n_out": n,
        "n_in": n_in,
        "bytes_out": sum(lens),
        "bytes_in": bytes_in,
        "len_mean": mean,
        "len_std": std,
        "n_dst_hosts": len({p.dst for p in out}),
        "n_dst_ports": len(ports),
        "n_src_ports": len({p.sport for p in out if p.proto in (PROTO_TCP, PROTO_UDP)}),
        "frac_tcp": n_tcp / n,
        "frac_udp": sum(p.proto == PROTO_UDP for p in out) / n,
        "frac_icmp": sum(p.proto == PROTO_ICMP for p in out) / n,
        "syn_frac": len(syn_only) / n_tcp if n_tcp else 0.0,
        "rst_in_frac": rst_in / tcp_in if tcp_in else 0.0,
        "fin_frac": sum(bool(p.flags & FIN) for p in tcp) / n_tcp if n_tcp else 0.0,
        "new_conns": len({(p.dst, p.dport, p.sport) for p in syn_only}),
        "top_dport_frac": max(ports.values()) / n if ports else 0.0,
        "payload_frac": sum(p.payload > 0 for p in out) / n,
        "iat_mean": iat_mean,
        "iat_std": iat_std,
        "modbus_req": len(modbus),
        "modbus_write": sum(p.modbus_fc in MODBUS_WRITE_CODES for p in modbus),
        "in_out_ratio": n_in / (n + 1),
    }


class WindowAggregator:
    """Feed packets in time order; get finished windows back.

    feed(pkt) -> list of (window_start, rows) for windows that closed because pkt is newer.
    flush(now) -> closes every window that ended before `now` (call on idle in live mode).
    rows = list of dicts: {"host": ip, "window_start": t, "window": W, **FEATURES}
    Packets arriving more than one window late are dropped (counted in `late`).
    """

    def __init__(self, window: float = 2.0):
        self.window = float(window)
        self.cur_start: float | None = None
        self.acc: dict[str, _HostAcc] = defaultdict(_HostAcc)
        self.packets = 0
        self.late = 0

    def _start_of(self, ts: float) -> float:
        return math.floor(ts / self.window) * self.window

    def _close(self) -> tuple[float, list[dict]]:
        rows = []
        for host, a in sorted(self.acc.items()):
            if not a.out:
                continue
            r = host_row(a.out, a.n_in, a.bytes_in, a.tcp_in, a.rst_in)
            rows.append({"host": host, "window_start": self.cur_start, "window": self.window, **r})
        out = (self.cur_start, rows)
        self.acc = defaultdict(_HostAcc)
        return out

    def feed(self, p: Pkt) -> list[tuple[float, list[dict]]]:
        done = []
        start = self._start_of(p.ts)
        if self.cur_start is None:
            self.cur_start = start
        if start < self.cur_start:
            self.late += 1
            return done
        while start > self.cur_start:          # one or more windows ended (empty ones included)
            done.append(self._close())
            self.cur_start += self.window
            if start - self.cur_start > 30 * self.window:   # long gap: jump, don't emit 1000s of empties
                self.cur_start = start
        self.packets += 1
        self.acc[p.src].out.append(p)
        d = self.acc[p.dst]
        d.n_in += 1
        d.bytes_in += p.length
        if p.proto == PROTO_TCP:
            d.tcp_in += 1
            if p.flags & RST:
                d.rst_in += 1
        return done

    def flush(self, now: float | None = None) -> list[tuple[float, list[dict]]]:
        """Close the current window if it has ended by `now` (None = close it unconditionally)."""
        if self.cur_start is None:
            return []
        if now is not None and now < self.cur_start + self.window:
            return []
        done = [self._close()]
        self.cur_start += self.window
        return done


def windows_from_packets(pkts, window: float = 2.0) -> list[dict]:
    """Offline helper: all feature rows of a packet sequence (e.g. a training pcap)."""
    agg = WindowAggregator(window)
    rows = []
    for p in pkts:
        for _, rs in agg.feed(p):
            rows.extend(rs)
    for _, rs in agg.flush():
        rows.extend(rs)
    return rows
