"""Lab-node tests (no Docker needed): Modbus framing, the PLC process and control logic, the camera.
Run:  python -m pytest demo/lab/tests -q
"""
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

NODES = Path(__file__).resolve().parents[1] / "nodes"
sys.path.insert(0, str(NODES))

import modbus  # noqa: E402
from modbus import Client  # noqa: E402


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_modbus_frame_roundtrip():
    pdu = bytes([modbus.READ_HOLDING, 0, 0, 0, 5])
    raw = modbus.frame(7, 1, pdu)
    a, b = socket.socketpair()
    try:
        a.sendall(raw)
        tid, unit, got = modbus.read_frame(b)
        assert tid == 7 and unit == 1 and got == pdu
    finally:
        a.close()
        b.close()


@pytest.fixture
def plc():
    import importlib
    plc_mod = importlib.import_module("plc")
    tank = plc_mod.Tank()
    port = _free_port()
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(8)
    stop = threading.Event()

    def loop():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            threading.Thread(target=plc_mod.serve_client, args=(conn, addr, tank), daemon=True).start()

    def process():
        last = time.monotonic()
        while not stop.is_set():
            time.sleep(0.05)
            now = time.monotonic()
            tank.step(now - last)
            last = now

    threading.Thread(target=loop, daemon=True).start()
    threading.Thread(target=process, daemon=True).start()
    yield port, tank
    stop.set()
    srv.close()


def test_plc_read_and_operator_write(plc):
    port, _ = plc
    c = Client("127.0.0.1", port)
    regs = c.read_holding(0, 7)
    assert len(regs) == 7 and 0 <= regs[0] <= 1000       # level is a sane percentage x10
    c.write_register(1, 620)                              # setpoint is writable
    assert c.read_holding(1, 1)[0] == 620


def test_plc_refuses_readonly_write(plc):
    port, tank = plc
    c = Client("127.0.0.1", port)
    before = c.read_holding(6, 1)[0]                      # writes counter
    c.write_register(0, 999)                              # level is read-only: accepted frame, ignored value
    assert c.read_holding(0, 1)[0] != 999
    assert c.read_holding(6, 1)[0] == before              # refused writes do not count


def test_plc_process_moves_and_controls(plc):
    _, tank = plc
    tank.regs[1] = 500                                    # setpoint 50 %
    tank.level = 200.0
    for _ in range(400):
        tank.step(0.1)
    assert 400 <= tank.level <= 600                       # control drives the level toward the setpoint band


def test_camera_snapshot_and_login():
    import importlib
    cam = importlib.import_module("camera")
    from http.server import ThreadingHTTPServer
    cam.USER, cam.PASSWORD = "admin", "secret123"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), cam.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        img = urllib.request.urlopen(base + "/snapshot.jpg", timeout=3).read()
        assert img[:2] == b"\xff\xd8" and len(img) > 1000         # looks like a JPEG

        def login(pw):
            data = urllib.parse.urlencode({"user": "admin", "password": pw}).encode()
            try:
                return urllib.request.urlopen(base + "/login", data=data, timeout=3).status
            except urllib.error.HTTPError as e:
                return e.code

        assert login("wrong") == 401 and login("secret123") == 200
    finally:
        srv.shutdown()
