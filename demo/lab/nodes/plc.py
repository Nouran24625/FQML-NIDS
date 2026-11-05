"""Software PLC: a Modbus/TCP server running a simulated water-tank process (lab only).

Process (updated 10 times per second):
    inflow pump fills the tank, an outlet valve drains it at a steady rate.
    PLC logic: pump ON below (setpoint - 5 %), OFF above (setpoint + 5 %).
    High-level alarm when the level passes the alarm limit; overflow when it reaches 100 %.

Holding registers (unit 1), values x10 (so 523 = 52.3 %):
    0  level            read only (writes ignored and logged)
    1  setpoint         read/write      default 500 (50.0 %)
    2  pump             0/1, read only (driven by the PLC logic unless in manual mode)
    3  mode             0 auto, 1 manual (in manual, register 2 is writable and the logic stops)
    4  alarm            0 ok, 1 high level, 2 overflow
    5  alarm limit      read/write      default 900 (90.0 %)
    6  writes counter   read only, total accepted writes since start
Coil 0 mirrors register 2 (pump); writing it in manual mode switches the pump.

Every write is printed with the client's IP, so the attack is visible in `docker logs lab-plc`.
"""
from __future__ import annotations

import os
import socket
import struct
import threading
import time

from modbus import (READ_COILS, READ_HOLDING, READ_INPUT, WRITE_COIL, WRITE_REGISTER, WRITE_REGISTERS,
                    ModbusError, frame, read_frame)

LEVEL, SETPOINT, PUMP, MODE, ALARM, ALARM_LIMIT, WRITES = range(7)
NAMES = ["level", "setpoint", "pump", "mode", "alarm", "alarm_limit", "writes"]
WRITABLE = {SETPOINT, MODE, ALARM_LIMIT}          # plus PUMP when mode = manual
N_REGS = 16


class Tank:
    FILL, DRAIN, BAND = 30.0, 15.0, 50          # tenths of a % per second (3 %/s in, 1.5 %/s out); band 5 %

    def __init__(self):
        self.lock = threading.Lock()
        self.regs = [0] * N_REGS
        self.level = 420.0
        self.regs[SETPOINT], self.regs[ALARM_LIMIT] = 500, 900

    def step(self, dt: float):
        with self.lock:
            r = self.regs
            if r[MODE] == 0:                       # automatic control
                if self.level < r[SETPOINT] - self.BAND:
                    r[PUMP] = 1
                elif self.level > r[SETPOINT] + self.BAND:
                    r[PUMP] = 0
            self.level += ((self.FILL if r[PUMP] else 0.0) - self.DRAIN) * dt
            self.level = max(0.0, min(1000.0, self.level))
            r[LEVEL] = int(round(self.level))
            r[ALARM] = 2 if self.level >= 1000 else (1 if self.level >= r[ALARM_LIMIT] else 0)

    def read(self, addr: int, count: int) -> list[int]:
        with self.lock:
            return list(self.regs[addr:addr + count])

    def write(self, addr: int, value: int, who: str) -> bool:
        with self.lock:
            ok = addr in WRITABLE or (addr == PUMP and self.regs[MODE] == 1)
            old = self.regs[addr]
            if ok:
                if addr == MODE:
                    value = 1 if value else 0
                if addr in (SETPOINT, ALARM_LIMIT):
                    value = max(0, min(1000, value))
                self.regs[addr] = value
                self.regs[WRITES] = (self.regs[WRITES] + 1) & 0xFFFF
        name = NAMES[addr] if addr < len(NAMES) else f"reg{addr}"
        stamp = time.strftime("%H:%M:%S")
        if ok:
            print(f"{stamp} WRITE from {who}: {name} {old} -> {value}", flush=True)
        else:
            print(f"{stamp} WRITE REFUSED from {who}: {name} is read-only (value {value})", flush=True)
        return ok


def exception(fc: int, code: int) -> bytes:
    return bytes([fc | 0x80, code])


def handle(pdu: bytes, tank: Tank, who: str) -> bytes:
    fc = pdu[0]
    try:
        if fc in (READ_HOLDING, READ_INPUT):
            addr, count = struct.unpack("!HH", pdu[1:5])
            if not 1 <= count <= 125 or addr + count > N_REGS:
                return exception(fc, 2)
            vals = tank.read(addr, count)
            return struct.pack(f"!BB{count}H", fc, 2 * count, *vals)
        if fc == READ_COILS:
            addr, count = struct.unpack("!HH", pdu[1:5])
            if addr != 0 or count != 1:
                return exception(fc, 2)
            return bytes([fc, 1, tank.read(PUMP, 1)[0] & 1])
        if fc == WRITE_REGISTER:
            addr, value = struct.unpack("!HH", pdu[1:5])
            if addr >= N_REGS:
                return exception(fc, 2)
            tank.write(addr, value, who)
            return pdu[:5]
        if fc == WRITE_REGISTERS:
            addr, count, nbytes = struct.unpack("!HHB", pdu[1:6])
            if addr + count > N_REGS or nbytes != 2 * count:
                return exception(fc, 2)
            for i, v in enumerate(struct.unpack(f"!{count}H", pdu[6:6 + nbytes])):
                tank.write(addr + i, v, who)
            return pdu[:5]
        if fc == WRITE_COIL:
            addr, value = struct.unpack("!HH", pdu[1:5])
            if addr != 0:
                return exception(fc, 2)
            tank.write(PUMP, 1 if value == 0xFF00 else 0, who)
            return pdu[:5]
    except struct.error:
        return exception(fc, 3)
    return exception(fc, 1)


def serve_client(conn: socket.socket, addr, tank: Tank):
    who = addr[0]
    conn.settimeout(120)
    with conn:
        try:
            while True:
                tid, unit, pdu = read_frame(conn)
                conn.sendall(frame(tid, unit, handle(pdu, tank, who)))
        except (ConnectionError, OSError, ModbusError):
            pass


def main():
    port = int(os.environ.get("PLC_PORT", "502"))
    tank = Tank()

    def process():
        last = time.monotonic()
        while True:
            time.sleep(0.1)
            now = time.monotonic()
            tank.step(now - last)
            last = now

    def status():
        while True:
            time.sleep(10)
            r = tank.read(0, 7)
            print(f"{time.strftime('%H:%M:%S')} level {r[LEVEL] / 10:5.1f} %  setpoint {r[SETPOINT] / 10:5.1f} %  "
                  f"pump {'ON ' if r[PUMP] else 'OFF'}  mode {'manual' if r[MODE] else 'auto'}  "
                  f"alarm {['ok', 'HIGH', 'OVERFLOW'][min(r[ALARM], 2)]}", flush=True)

    threading.Thread(target=process, daemon=True).start()
    threading.Thread(target=status, daemon=True).start()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(64)
    print(f"PLC running: Modbus/TCP on port {port}, water-tank process started", flush=True)
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=serve_client, args=(conn, addr, tank), daemon=True).start()


if __name__ == "__main__":
    main()
