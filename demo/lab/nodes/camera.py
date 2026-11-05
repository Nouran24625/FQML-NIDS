"""IP camera (lab only): web server on port 80 with a snapshot endpoint and a login page.
SSH login (port 22) is provided by sshd, started next to this script in the container.

    GET  /                -> small status page
    GET  /snapshot.jpg    -> a 30-70 KB "JPEG" (random bytes with a JPEG header; nobody views it,
                             it only makes the traffic look like a camera's)
    POST /login           -> checks user/password like a camera admin page (for the web brute-force option)
"""
from __future__ import annotations

import os
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

USER = os.environ.get("CAM_USER", "admin")
PASSWORD = os.environ.get("CAM_PASSWORD", "admin2025")
rng = random.Random(11)
FRAMES = [b"\xff\xd8\xff\xe0" + rng.randbytes(rng.randint(30_000, 70_000)) + b"\xff\xd9" for _ in range(20)]


class Handler(BaseHTTPRequestHandler):
    server_version = "LabCam/1.2"
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/snapshot.jpg"):
            return self._send(200, FRAMES[int(time.time()) % len(FRAMES)], "image/jpeg")
        if self.path in ("/", "/index.html"):
            page = (b"<html><body><h1>LabCam 1.2</h1><form method=post action=/login>"
                    b"<input name=user><input name=password type=password><button>Login</button></form></body></html>")
            return self._send(200, page, "text/html")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0") or 0)
        form = parse_qs(self.rfile.read(min(n, 4096)).decode(errors="replace"))
        if self.path == "/login":
            ok = form.get("user", [""])[0] == USER and form.get("password", [""])[0] == PASSWORD
            return self._send(200 if ok else 401, b"welcome" if ok else b"login failed", "text/plain")
        self._send(404, b"not found", "text/plain")


if __name__ == "__main__":
    port = int(os.environ.get("CAM_PORT", "80"))
    print(f"camera running: web on port {port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
