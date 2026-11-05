"""The post-quantum channel over a real TCP connection: E7 runs and the demo (proposed decision D15). Owner: Sheghaf.

  server:  me, trust = load_server(cfg, "keys/default")
           with Listener(cfg, me, trust, host="0.0.0.0", port=9443) as lst:
               link = lst.accept()                                  # one site; handshake inside
  site:    me, server = load_site(cfg, "keys/default", "site-01")
           link = connect(cfg, me, server, "10.0.0.5", 9443)          # handshake inside
  both:    link.send_model(weights, round=r, purpose="global")       # encode -> sign -> encrypt -> send
           msg = link.recv_model(round=r, purpose="update")         # receive -> decrypt -> verify -> arrays
           link.send(b"...") / link.recv()                          # any other bytes
           link.stats                                               # bytes on the wire, crypto time (E7)

Frames on the wire: 4-byte length, 1-byte kind (H handshake, D data, A alert, C close), then the bytes.
A failed handshake sends a short alert saying why before closing, so a wrong setting or a missing key
gives a clear error on BOTH sides. After any failed check the connection is closed: a stream that
carried a rejected message cannot be trusted. One Link is used by one thread at a time.

Demo, two terminals (keys first: python -m fqml.pq.keys gen server site-01 --config default, which
writes keys/default/, the folder the demo reads for --config default):
  python -m fqml.pq.link serve   --config default
  python -m fqml.pq.link connect --config default --name site-01
"""
from __future__ import annotations

import argparse
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from fqml.pq.channel import ClientHandshake, ServerHandshake, Session, open_model, sign_model
from fqml.pq.config import ConfigError, PQConfig, load_config
from fqml.pq.keys import Identity, KeyFileError, TrustStore, default_key_dir, load_identity
from fqml.pq.primitives import HandshakeError, IntegrityError, SecurityError
from fqml.pq.serialize import dumps_weights, loads_weights

MAX_FRAME = 64 * 1024 * 1024             # refuse bigger frames (memory guard); our models are far smaller
FRAME_HEADER = 5                         # u32 length + 1-byte kind
HANDSHAKE, DATA, ALERT, CLOSE = b"H", b"D", b"A", b"C"


class LinkClosed(ConnectionError):
    """The other side closed the link."""


@dataclass
class LinkStats:
    handshake_seconds: float = 0.0
    handshake_bytes_sent: int = 0        # on the wire, frame headers included
    handshake_bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0                  # after the handshake, on the wire
    bytes_received: int = 0
    payload_bytes_sent: int = 0          # what the caller passed to send() (model envelopes included)
    payload_bytes_received: int = 0
    crypto_seconds: float = 0.0          # seal/open + sign/verify, this side only

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelMessage:
    weights: object                      # dict {name: np.ndarray} or list of np.ndarray
    round: int
    purpose: str
    signer: str
    meta: dict = field(default_factory=dict)
    encoded_bytes: int = 0


# ---- frames
def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise LinkClosed("connection closed by the other side")
        got += k
    return bytes(buf)


def _send_frame(sock: socket.socket, kind: bytes, body: bytes) -> int:
    sock.sendall(len(body).to_bytes(4, "big") + kind + body)
    return FRAME_HEADER + len(body)


def _recv_frame(sock: socket.socket, max_frame: int) -> tuple[bytes, bytes]:
    head = _read_exact(sock, FRAME_HEADER)
    n, kind = int.from_bytes(head[:4], "big"), head[4:5]
    if n > max_frame:
        raise IntegrityError(f"frame of {n:,} bytes is over the {max_frame:,}-byte limit")
    return kind, _read_exact(sock, n)


def _send_alert(sock: socket.socket, error: Exception) -> None:
    try:                                                  # best effort: the other side may be gone
        _send_frame(sock, ALERT, str(error)[:300].encode())
    except OSError:
        pass


def _expect_handshake(sock: socket.socket, max_frame: int) -> bytes:
    kind, body = _recv_frame(sock, max_frame)
    if kind == ALERT:
        raise HandshakeError(f"the other side refused the handshake: {body.decode(errors='replace')}")
    if kind != HANDSHAKE:
        raise HandshakeError(f"expected a handshake frame, got {kind!r}")
    return body


def _tune(sock: socket.socket, timeout: float | None) -> None:
    sock.settimeout(timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # small handshake frames go out at once


class Link:
    """An authenticated, encrypted connection to one peer (made by connect() or Listener.accept())."""

    def __init__(self, sock: socket.socket, session: Session, cfg: PQConfig, me, peer_identity,
                 stats: LinkStats, max_frame: int = MAX_FRAME):
        self.sock, self.session, self.cfg, self.me = sock, session, cfg, me
        self.peer_identity = peer_identity       # pinned Identity (mode none: just the name)
        self.stats, self.max_frame = stats, max_frame
        self.closed = False

    @property
    def peer(self) -> str:
        return self.session.peer

    def send(self, payload: bytes) -> None:
        t = time.perf_counter()
        sealed = self.session.seal(payload)
        self.stats.crypto_seconds += time.perf_counter() - t
        try:
            self.stats.bytes_sent += _send_frame(self.sock, DATA, sealed)
        except OSError:
            self.close(notify=False)
            raise
        self.stats.messages_sent += 1
        self.stats.payload_bytes_sent += len(payload)

    def recv(self) -> bytes:
        try:
            kind, body = _recv_frame(self.sock, self.max_frame)
            if kind == CLOSE:
                raise LinkClosed(f"{self.peer} closed the link")
            if kind == ALERT:
                raise HandshakeError(f"{self.peer} refused the connection: {body.decode(errors='replace')}")
            if kind != DATA:
                raise IntegrityError(f"unexpected frame kind {kind!r}")
            t = time.perf_counter()
            payload = self.session.open(body)
        except (SecurityError, OSError):              # OSError covers LinkClosed and timeouts
            self.close(notify=False)
            raise
        self.stats.crypto_seconds += time.perf_counter() - t
        self.stats.bytes_received += FRAME_HEADER + len(body)
        self.stats.messages_received += 1
        self.stats.payload_bytes_received += len(payload)
        return payload

    def send_model(self, weights, *, round: int, purpose: str, meta: dict | None = None) -> int:
        """Encode, sign and send model weights; returns the size of the signed envelope."""
        encoded = dumps_weights(weights)
        t = time.perf_counter()
        envelope = sign_model(self.cfg, self.me, encoded, round=round, purpose=purpose, meta=meta)
        self.stats.crypto_seconds += time.perf_counter() - t
        self.send(envelope)
        return len(envelope)

    def recv_model(self, *, round: int | None = None, purpose: str | None = None) -> ModelMessage:
        """Receive weights signed by this link's peer, for the given round and purpose."""
        envelope = self.recv()
        t = time.perf_counter()
        try:
            m = open_model(self.cfg, envelope, self.peer_identity, round=round, purpose=purpose)
        except SecurityError:
            self.close(notify=False)
            raise
        self.stats.crypto_seconds += time.perf_counter() - t
        return ModelMessage(loads_weights(m.weights), m.round, m.purpose, m.signer, m.meta, len(m.weights))

    def close(self, notify: bool = True) -> None:
        if self.closed:
            return
        self.closed = True
        if notify:
            try:
                _send_frame(self.sock, CLOSE, b"")
            except OSError:
                pass
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def connect(cfg: PQConfig, me: Identity | str, server: Identity | str | None, host: str = "127.0.0.1",
            port: int = 9443, *, timeout: float | None = 60.0, max_frame: int = MAX_FRAME) -> Link:
    """Site side: open a TCP connection to the server and run the handshake."""
    hs = ClientHandshake(cfg, me, server)                   # checks the keys before touching the network
    sock = socket.create_connection((host, port), timeout=timeout)
    stats = LinkStats()
    t0 = time.perf_counter()
    try:
        _tune(sock, timeout)
        stats.handshake_bytes_sent += _send_frame(sock, HANDSHAKE, hs.hello())
        reply = _expect_handshake(sock, max_frame)
        stats.handshake_bytes_received += FRAME_HEADER + len(reply)
        try:
            finish, session = hs.finish(reply)
        except HandshakeError as e:
            _send_alert(sock, e)
            raise
        stats.handshake_bytes_sent += _send_frame(sock, HANDSHAKE, finish)
    except BaseException:
        sock.close()
        raise
    stats.handshake_seconds = time.perf_counter() - t0
    return Link(sock, session, cfg, me, server, stats, max_frame)


class Listener:
    """Server side: accept() waits for one site, runs the handshake and returns its Link.
    A failed handshake raises HandshakeError for that connection only; keep calling accept()."""

    def __init__(self, cfg: PQConfig, me: Identity | str, trust: TrustStore | None, host: str = "127.0.0.1",
                 port: int = 9443, *, timeout: float | None = 60.0, max_frame: int = MAX_FRAME, backlog: int = 64):
        ServerHandshake(cfg, me, trust)                     # checks the keys now, not at the first site
        self.cfg, self.me, self.trust = cfg, me, trust
        self.timeout, self.max_frame = timeout, max_frame
        self._sock = socket.create_server((host, port), backlog=backlog)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._sock.getsockname()[:2]
        return host, port

    def accept(self) -> Link:
        conn, _ = self._sock.accept()
        stats = LinkStats()
        t0 = time.perf_counter()
        try:
            _tune(conn, self.timeout)
            hs = ServerHandshake(self.cfg, self.me, self.trust)
            hello = _expect_handshake(conn, self.max_frame)
            stats.handshake_bytes_received += FRAME_HEADER + len(hello)
            try:
                reply = hs.respond(hello)
            except HandshakeError as e:
                _send_alert(conn, e)
                raise
            stats.handshake_bytes_sent += _send_frame(conn, HANDSHAKE, reply)
            finish = _expect_handshake(conn, self.max_frame)
            stats.handshake_bytes_received += FRAME_HEADER + len(finish)
            try:
                session = hs.complete(finish)
            except HandshakeError as e:
                _send_alert(conn, e)
                raise
        except BaseException:
            conn.close()
            raise
        stats.handshake_seconds = time.perf_counter() - t0
        return Link(conn, session, self.cfg, self.me, hs.peer_identity, stats, self.max_frame)

    def close(self) -> None:
        self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---- loading the right keys for each side (key folders: keys/<settings name>, see keys.default_key_dir)
def load_server(cfg: PQConfig, key_dir: str | Path, name: str = "server") -> tuple[Identity | str, TrustStore | None]:
    """The server's secret key and every site's pinned public key (all other .pub files in key_dir)."""
    if not cfg.encrypted:
        return name, None
    me = load_identity(Path(key_dir) / f"{name}.key")
    return me, TrustStore.from_dir(key_dir, exclude=[name])


def load_site(cfg: PQConfig, key_dir: str | Path, name: str,
              server_name: str = "server") -> tuple[Identity | str, Identity | str]:
    """The site's secret key and the server's pinned public key."""
    if not cfg.encrypted:
        return name, server_name
    return load_identity(Path(key_dir) / f"{name}.key"), load_identity(Path(key_dir) / f"{server_name}.pub")


# ---- demo
def demo_weights(seed: int = 0) -> dict:
    """Random weights shaped like the 8-class VQC (theta [3, 8, 2], scale [8], bias [8]: 64 parameters)."""
    rng = np.random.default_rng(seed)
    return {"theta": rng.uniform(0, 2 * np.pi, (3, 8, 2)).astype(np.float32),
            "scale": np.ones(8, np.float32), "bias": np.zeros(8, np.float32)}


def _summary(link: Link) -> str:
    s = link.stats
    return (f"handshake {s.handshake_seconds * 1e3:.1f} ms, {s.handshake_bytes_sent + s.handshake_bytes_received:,} B; "
            f"after it: sent {s.bytes_sent:,} B, received {s.bytes_received:,} B; crypto {s.crypto_seconds * 1e3:.1f} ms")


def _serve_one(link: Link, rounds: int) -> None:
    try:
        weights = demo_weights()
        for r in range(rounds):
            link.send_model(weights, round=r, purpose="global")
            msg = link.recv_model(round=r, purpose="update")
            weights = msg.weights                       # one site, so FedAvg is just its update
            print(f"  [{link.peer}] round {r}: update verified (signed by {msg.signer}, meta {msg.meta})")
        print(f"  [{link.peer}] done: {_summary(link)}")
    except (SecurityError, ConnectionError, OSError) as e:
        print(f"  [{link.peer}] stopped: {e}")
    finally:
        link.close()


def main(argv=None) -> None:
    try:
        _main(argv)
    except (SecurityError, ConnectionError, KeyFileError, ConfigError, ValueError) as e:
        raise SystemExit(f"stopped: {e}") from None


def _main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m fqml.pq.link",
                                 description="Demo: a server and its sites exchange signed, encrypted models over TCP.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("serve", "connect"):
        p = sub.add_parser(cmd)
        p.add_argument("--config", default="default", help="name in configs/pq, or a YAML path")
        p.add_argument("--keys", default=None, help="key folder (default: keys/<config name>)")
        p.add_argument("--port", type=int, default=9443)
        p.add_argument("--rounds", type=int, default=3)
        if cmd == "serve":
            p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept sites from other machines")
            p.add_argument("--name", default="server")
            p.add_argument("--sites", type=int, default=0, help="stop after this many sites (0: until Ctrl+C)")
        else:
            p.add_argument("--host", default="127.0.0.1", help="the server's address")
            p.add_argument("--name", required=True, help="this site's name, e.g. site-01")
            p.add_argument("--server-name", default="server")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    key_dir = Path(a.keys) if a.keys else default_key_dir(a.config)
    print(f"settings: {cfg.describe()}")
    if a.cmd == "serve":
        me, trust = load_server(cfg, key_dir, a.name)
        if trust is not None:
            print(f"trusted sites ({len(trust)}): {', '.join(trust.names()) or 'none - add their .pub files'}")
        workers = []
        with Listener(cfg, me, trust, a.host, a.port) as lst:
            print(f"listening on {lst.address[0]}:{lst.address[1]}  (Ctrl+C to stop)")
            try:
                while not a.sites or len(workers) < a.sites:
                    try:
                        link = lst.accept()
                    except (SecurityError, ConnectionError, OSError) as e:
                        print(f"  handshake failed: {e}")
                        continue
                    print(f"  [{link.peer}] connected, session {link.session.session_id[:4].hex()}, "
                          f"handshake {link.stats.handshake_seconds * 1e3:.1f} ms")
                    w = threading.Thread(target=_serve_one, args=(link, a.rounds), daemon=True)
                    w.start()
                    workers.append(w)
            except KeyboardInterrupt:
                print("stopping")
        for w in workers:
            w.join()
    else:
        me, server = load_site(cfg, key_dir, a.name, a.server_name)
        link = connect(cfg, me, server, a.host, a.port)
        print(f"connected to {link.peer}, session {link.session.session_id[:4].hex()}, "
              f"handshake {link.stats.handshake_seconds * 1e3:.1f} ms")
        rng = np.random.default_rng()
        with link:
            for r in range(a.rounds):
                msg = link.recv_model(round=r, purpose="global")
                update = {k: v + rng.normal(0, 0.01, v.shape).astype(v.dtype) for k, v in msg.weights.items()}
                link.send_model(update, round=r, purpose="update", meta={"n_rows": 1000})
                print(f"  round {r}: global model verified (signed by {msg.signer}); update sent")
            print(f"done: {_summary(link)}")


if __name__ == "__main__":
    main()
