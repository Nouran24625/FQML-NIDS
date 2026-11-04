"""Dashboard server (standard library only, works offline in the lab). Owner: Nouran.

    python -m fqml.live.dashboard --state runs/live [--port 8050]
    then open http://localhost:8050

Serves fqml/live/static/dashboard.html and two JSON endpoints that read what detect.py writes:
    /api/status          status.json
    /api/windows?from=N  detection lines N.. (the page polls this every second)
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STATIC = Path(__file__).parent / "static"


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, state: Path, **kw):
        self.state = state
        super().__init__(*args, **kw)

    def log_message(self, *_):   # keep the terminal clean for the video
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send((STATIC / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/status":
            p = self.state / "status.json"
            return self._send(p.read_bytes() if p.exists() else b"{}", "application/json")
        if u.path == "/api/windows":
            start = int(parse_qs(u.query).get("from", ["0"])[0])
            p = self.state / "detections.jsonl"
            lines = p.read_text().splitlines() if p.exists() else []
            if start > len(lines):          # detector restarted: tell the page to reset
                start = 0
            body = '{"from": %d, "next": %d, "windows": [%s]}' % (start, len(lines), ",".join(lines[start:]))
            return self._send(body.encode(), "application/json")
        self._send(b"not found", "text/plain", 404)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="runs/live")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to view from another machine")
    ap.add_argument("--port", type=int, default=8050)
    a = ap.parse_args(argv)
    srv = ThreadingHTTPServer((a.host, a.port), partial(Handler, state=Path(a.state)))
    print(f"dashboard: http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}  (state: {a.state})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
