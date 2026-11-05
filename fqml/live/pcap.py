"""Minimal, dependency-free pcap reader.

Reads classic libpcap files or a live stream (what `tcpdump -U -w -` writes to stdout).
Supported link types: Ethernet (1, incl. 802.1Q), Linux cooked SLL (113) and SLL2 (276), raw IP (101),
BSD loopback (0). Only IPv4 is decoded; everything else (ARP, IPv6, ...) is counted and skipped.

Each packet becomes a small `Pkt` record holding only what features.py needs.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO, Iterator

MAGICS = {  # magic -> (byte order, timestamp divisor)
    0xA1B2C3D4: ("<", 1e6), 0xD4C3B2A1: (">", 1e6),
    0xA1B23C4D: ("<", 1e9), 0x4D3CB2A1: (">", 1e9),
}
PROTO_TCP, PROTO_UDP, PROTO_ICMP = 6, 17, 1
# TCP flag bits
FIN, SYN, RST, PSH, ACK = 0x01, 0x02, 0x04, 0x08, 0x10
MODBUS_PORT = 502
MODBUS_WRITE_CODES = {5, 6, 15, 16, 22, 23}   # write coil(s) / register(s), mask write, read-write


@dataclass(slots=True)
class Pkt:
    ts: float          # seconds since epoch
    src: str
    dst: str
    proto: int         # 6 TCP, 17 UDP, 1 ICMP, other IP protocol numbers as-is
    length: int        # bytes on the wire (orig_len)
    sport: int = 0
    dport: int = 0
    flags: int = 0     # TCP flags
    payload: int = 0   # transport payload bytes
    modbus_fc: int = -1  # Modbus/TCP function code of a request to port 502, else -1


class PcapError(ValueError):
    pass


def _read_exact(f: BinaryIO, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None       # end of stream (a truncated last record is dropped)
        buf += chunk
    return buf


def _ip_offset(linktype: int, data: bytes) -> int | None:
    """Byte offset of the IPv4 header inside a frame, or None if the frame is not IPv4."""
    if linktype == 1:                                   # Ethernet
        off, etype = 14, struct.unpack_from("!H", data, 12)[0] if len(data) >= 14 else 0
        while etype in (0x8100, 0x88A8) and len(data) >= off + 4:   # VLAN tags
            etype = struct.unpack_from("!H", data, off + 2)[0]
            off += 4
        return off if etype == 0x0800 else None
    if linktype == 113:                                 # Linux cooked v1
        return 16 if len(data) >= 16 and struct.unpack_from("!H", data, 14)[0] == 0x0800 else None
    if linktype == 276:                                 # Linux cooked v2
        return 20 if len(data) >= 20 and struct.unpack_from("!H", data, 0)[0] == 0x0800 else None
    if linktype in (101, 12, 14):                       # raw IP
        return 0 if data and data[0] >> 4 == 4 else None
    if linktype == 0:                                   # BSD loopback
        return 4 if len(data) >= 4 and data[0] in (2, 0) else None
    return None


def decode(linktype: int, ts: float, data: bytes, orig_len: int) -> Pkt | None:
    off = _ip_offset(linktype, data)
    if off is None or len(data) < off + 20:
        return None
    vihl = data[off]
    if vihl >> 4 != 4:
        return None
    ihl = (vihl & 0x0F) * 4
    total_len, = struct.unpack_from("!H", data, off + 2)
    proto = data[off + 9]
    src = ".".join(map(str, data[off + 12:off + 16]))
    dst = ".".join(map(str, data[off + 16:off + 20]))
    p = Pkt(ts=ts, src=src, dst=dst, proto=proto, length=orig_len)
    t = off + ihl
    frag_off = struct.unpack_from("!H", data, off + 6)[0] & 0x1FFF
    if frag_off:                                         # non-first fragment: no transport header
        return p
    if proto == PROTO_TCP and len(data) >= t + 20:
        p.sport, p.dport = struct.unpack_from("!HH", data, t)
        doff = (data[t + 12] >> 4) * 4
        p.flags = data[t + 13]
        p.payload = max(0, total_len - ihl - doff)
        if p.dport == MODBUS_PORT and p.payload >= 8 and len(data) >= t + doff + 8:
            p.modbus_fc = data[t + doff + 7]             # MBAP header is 7 bytes, then function code
    elif proto == PROTO_UDP and len(data) >= t + 8:
        p.sport, p.dport = struct.unpack_from("!HH", data, t)
        p.payload = max(0, total_len - ihl - 8)
    elif proto == PROTO_ICMP:
        p.payload = max(0, total_len - ihl - 8)
    return p


def read_pcap(f: BinaryIO, stats: dict | None = None) -> Iterator[Pkt]:
    """Yield IPv4 packets from a pcap file or stream. `stats` (optional) counts frames / skipped."""
    hdr = _read_exact(f, 24)
    if hdr is None:
        return
    magic = struct.unpack("<I", hdr[:4])[0]
    if magic not in MAGICS:
        if hdr[:4] == b"\x0a\x0d\x0d\x0a":
            raise PcapError("this is pcapng; capture with `tcpdump -w` or convert: editcap -F pcap in.pcapng out.pcap")
        raise PcapError(f"not a pcap file (magic {magic:#x})")
    order, div = MAGICS[magic]
    linktype = struct.unpack(order + "I", hdr[20:24])[0] & 0x0FFFFFFF
    rec = struct.Struct(order + "IIII")
    if stats is not None:
        stats.setdefault("frames", 0)
        stats.setdefault("non_ipv4", 0)
        stats["linktype"] = linktype
    while True:
        rh = _read_exact(f, 16)
        if rh is None:
            return
        sec, frac, incl, orig = rec.unpack(rh)
        if incl > 262144:
            raise PcapError(f"corrupt record length {incl}")
        data = _read_exact(f, incl)
        if data is None:
            return
        if stats is not None:
            stats["frames"] += 1
        pkt = decode(linktype, sec + frac / div, data, orig)
        if pkt is None:
            if stats is not None:
                stats["non_ipv4"] += 1
            continue
        yield pkt


def iter_pcap_file(path: str, stats: dict | None = None) -> Iterator[Pkt]:
    with open(path, "rb") as f:
        yield from read_pcap(f, stats)
