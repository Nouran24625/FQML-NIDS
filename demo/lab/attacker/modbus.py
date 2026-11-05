"""Minimal Modbus/TCP (standard library only), shared by the PLC, the HMI and the attack tools.

Supported function codes: 1 read coils, 3 read holding registers, 4 read input registers,
5 write single coil, 6 write single register, 16 write multiple registers.
Frame = MBAP header (transaction id, protocol 0, length, unit id) + PDU (function code + data).
"""
from __future__ import annotations

import socket
import struct

READ_COILS, READ_HOLDING, READ_INPUT = 1, 3, 4
WRITE_COIL, WRITE_REGISTER, WRITE_REGISTERS = 5, 6, 16
WRITE_CODES = {WRITE_COIL, WRITE_REGISTER, WRITE_REGISTERS}


class ModbusError(Exception):
    pass


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return buf


def read_frame(sock: socket.socket) -> tuple[int, int, bytes]:
    """Return (transaction id, unit id, pdu)."""
    tid, proto, length, unit = struct.unpack("!HHHB", recv_exact(sock, 7))
    if proto != 0 or not 2 <= length <= 254:
        raise ModbusError(f"bad MBAP header (protocol {proto}, length {length})")
    return tid, unit, recv_exact(sock, length - 1)


def frame(tid: int, unit: int, pdu: bytes) -> bytes:
    return struct.pack("!HHHB", tid, 0, len(pdu) + 1, unit) + pdu


class Client:
    """Blocking Modbus/TCP client with one persistent connection (like a real HMI)."""

    def __init__(self, host: str, port: int = 502, unit: int = 1, timeout: float = 3.0):
        self.host, self.port, self.unit, self.timeout = host, port, unit, timeout
        self.sock: socket.socket | None = None
        self.tid = 0

    def connect(self):
        self.close()
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None

    def _call(self, pdu: bytes) -> bytes:
        if self.sock is None:
            self.connect()
        self.tid = (self.tid + 1) & 0xFFFF
        try:
            self.sock.sendall(frame(self.tid, self.unit, pdu))
            tid, _, resp = read_frame(self.sock)
        except (OSError, ConnectionError, ModbusError):
            self.close()
            raise
        if resp[0] & 0x80:
            raise ModbusError(f"exception code {resp[1]} for function {pdu[0]}")
        return resp

    def read_holding(self, addr: int, count: int) -> list[int]:
        r = self._call(struct.pack("!BHH", READ_HOLDING, addr, count))
        return list(struct.unpack(f"!{r[1] // 2}H", r[2:2 + r[1]]))

    def write_register(self, addr: int, value: int) -> None:
        self._call(struct.pack("!BHH", WRITE_REGISTER, addr, value & 0xFFFF))

    def write_registers(self, addr: int, values: list[int]) -> None:
        data = struct.pack(f"!{len(values)}H", *[v & 0xFFFF for v in values])
        self._call(struct.pack("!BHHB", WRITE_REGISTERS, addr, len(values), len(data)) + data)

    def write_coil(self, addr: int, on: bool) -> None:
        self._call(struct.pack("!BHH", WRITE_COIL, addr, 0xFF00 if on else 0))
