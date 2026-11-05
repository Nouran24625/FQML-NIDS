"""Authenticated post-quantum channel between the server and one site (proposed decision D15). Owner: Sheghaf.

Handshake, 3 messages; each side proves who it is with its pinned identity key (keys.py):

  site   -> server   ClientHello   settings fingerprint, site name, nonce, ML-KEM public key*, X25519 public key*
  server -> site     ServerHello   settings fingerprint, server name, nonce, ML-KEM ciphertext*, X25519 public key*,
                                   server signature over the transcript, key confirmation
  site   -> server   ClientFinish  site signature over the transcript, key confirmation
                                                                         (* only what the mode uses, config.py)

  session keys = HKDF(ML-KEM secret || X25519 secret, salt = hash of the messages so far)
                 -> one AES-256-GCM key per direction

The ML-KEM and X25519 key pairs are new for every connection (forward secrecy), and the hybrid stays
secure as long as EITHER of the two is unbroken ("harvest now, decrypt later" needs ML-KEM broken too).

What each check is for:
  settings fingerprint  both sides run the same mode and algorithms: nobody can downgrade a site
  nonces, fresh keys    an old handshake cannot be replayed
  signatures            server and site are who they claim (pinned keys; no certificate authority)
  key confirmation      both derived the same keys (also catches a tampered ML-KEM ciphertext)
  sequence numbers      each sealed message is numbered inside its authenticated data, so a tampered,
                        replayed, reordered or dropped message is rejected
  model signatures      sign_model/open_model: the model carries its signer, round and purpose, so a
                        stored or relayed model can be checked later, and an old round cannot be
                        replayed as a new one

Mode none runs the same three messages without keys, encryption or signatures, so E7 measures only the
cost of the cryptography. This module does no I/O (easy to test); link.py carries the bytes over TCP.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from fqml.pq import primitives as P
from fqml.pq.config import PROTOCOL_VERSION, PQConfig
from fqml.pq.keys import Identity, TrustStore
from fqml.pq.primitives import HandshakeError, IntegrityError, SecurityError, SignatureError
from fqml.pq.serialize import FormatError, pack, unpack

__all__ = ["ClientHandshake", "ServerHandshake", "Session", "SignedModel", "sign_model", "open_model",
           "SecurityError", "HandshakeError", "IntegrityError", "SignatureError", "PURPOSES"]

LABEL = b"FQML-NIDS PQ channel v1"
NONCE_BYTES = 32
SEQ_BYTES = 8
PURPOSES = ("global", "update", "final")     # global model to sites / a site's update / released model


def _name(party) -> str:
    return party.name if isinstance(party, Identity) else party


def _is(value, kind) -> bool:
    """isinstance, except that a bool never counts as an int."""
    return isinstance(value, kind) and not (kind is int and isinstance(value, bool))


def _transcript(cfg: PQConfig, *messages: bytes) -> bytes:
    """Hash of the handshake so far; each message is length-prefixed so the split is unambiguous."""
    data = bytearray(LABEL)
    for m in messages:
        data += len(m).to_bytes(4, "big") + m
    return P.digest(cfg.kdf_hash, bytes(data))


def _session_keys(cfg: PQConfig, ikm: bytes, transcript: bytes) -> dict:
    n, k = P.digest_size(cfg.kdf_hash), P.AEAD_KEY_BYTES
    okm = P.hkdf(cfg.kdf_hash, ikm, salt=transcript, info=LABEL + b" session keys", length=2 * k + 2 * n)
    return {"site_to_server": okm[:k], "server_to_site": okm[k:2 * k],
            "server_confirm": okm[2 * k:2 * k + n], "site_confirm": okm[2 * k + n:]}


def _expected(cfg: PQConfig, message: str) -> set[str]:
    """The exact fields each handshake message carries in this mode (besides 'type')."""
    if message == "client_hello":
        out = {"protocol", "settings", "site", "nonce"}
        if cfg.uses_kem:
            out.add("kem_pk")
        if cfg.uses_x25519:
            out.add("x25519_pk")
    elif message == "server_hello":
        out = {"settings", "server", "nonce"}
        if cfg.uses_kem:
            out.add("kem_ct")
        if cfg.uses_x25519:
            out.add("x25519_pk")
        if cfg.encrypted:
            out |= {"sig", "confirm"}
    else:                                                       # client_finish
        out = {"sig", "confirm"} if cfg.encrypted else set()
    return out


def _parse(cfg: PQConfig, blob: bytes, message: str) -> dict:
    try:
        f = unpack(blob)
    except FormatError as e:
        raise HandshakeError(f"malformed {message}: {e}") from None
    if f.get("type") != message:
        raise HandshakeError(f"expected {message}, got {f.get('type')!r}")
    if message != "client_finish" and f.get("settings") != cfg.fingerprint():
        side = "site" if message == "client_hello" else "server"
        raise HandshakeError(f"the {side} uses different PQ settings (configs/pq); refusing, since this could "
                             f"be a downgrade attack. Here: {cfg.describe()}")
    want, have = _expected(cfg, message), set(f) - {"type"}
    if want != have:
        raise HandshakeError(f"{message}: missing {sorted(want - have)}, unexpected {sorted(have - want)}")
    types = {"protocol": int, "site": str, "server": str}
    for name, value in f.items():
        if name != "type" and not _is(value, types.get(name, bytes)):
            raise HandshakeError(f"{message}: field {name!r} has the wrong type")
    if "nonce" in f and len(f["nonce"]) != NONCE_BYTES:
        raise HandshakeError(f"{message}: nonce must be {NONCE_BYTES} bytes")
    return f


def _check_keys(cfg: PQConfig, me, peer, peer_role: str) -> None:
    if not cfg.encrypted:
        if not isinstance(_name(me), str):
            raise ValueError("give this party's name (or Identity)")
        return
    if not isinstance(me, Identity) or me.secret is None:
        raise ValueError(f"mode {cfg.mode} needs this party's secret identity key (keys.load_identity on a .key)")
    if me.alg != cfg.sig_alg:
        raise ValueError(f"{me.name}'s key is {me.alg} but these settings sign with {cfg.sig_alg}; "
                         f"make keys for these settings with python -m fqml.pq.keys gen")
    if peer_role == "server":
        if not isinstance(peer, Identity):
            raise ValueError(f"mode {cfg.mode} needs the server's pinned public key (its .pub file)")
        if peer.alg != cfg.sig_alg:
            raise ValueError(f"the pinned server key is {peer.alg} but these settings sign with {cfg.sig_alg}")
    elif not isinstance(peer, TrustStore):
        raise ValueError(f"mode {cfg.mode} needs a TrustStore with the sites' pinned public keys")


class ClientHandshake:
    """The site side: hello() -> send; finish(server_hello) -> (client_finish to send, Session)."""

    def __init__(self, cfg: PQConfig, me: Identity | str, server: Identity | str | None):
        _check_keys(cfg, me, server, "server")
        self.cfg, self.me, self.server = cfg, me, server
        self._hello = None
        self._kem = self._x25519 = None

    def hello(self) -> bytes:
        f = {"type": "client_hello", "protocol": PROTOCOL_VERSION, "settings": self.cfg.fingerprint(),
             "site": _name(self.me), "nonce": P.random_bytes(NONCE_BYTES)}
        if self.cfg.uses_kem:
            self._kem = P.kem_keypair(self.cfg.kem)
            f["kem_pk"] = self._kem.public
        if self.cfg.uses_x25519:
            self._x25519 = P.x25519_keypair()
            f["x25519_pk"] = self._x25519.public
        self._hello = pack(f)
        return self._hello

    def finish(self, server_hello: bytes) -> tuple[bytes, Session]:
        if self._hello is None:
            raise RuntimeError("call hello() first")
        cfg = self.cfg
        f = _parse(cfg, server_hello, "server_hello")
        if self.server is not None and f["server"] != _name(self.server):
            raise HandshakeError(f"expected server {_name(self.server)!r}, got {f['server']!r}")
        th_full = _transcript(cfg, self._hello, server_hello)
        if not cfg.encrypted:
            self._hello = None                                   # one handshake per object
            return pack({"type": "client_finish"}), Session(cfg, "site", f["server"], th_full, None, None)

        core = pack({k: v for k, v in f.items() if k not in ("sig", "confirm")})
        th = _transcript(cfg, self._hello, core)
        if not P.verify(self.server.alg, self.server.public, LABEL + b" server signature" + th, f["sig"]):
            raise HandshakeError("the server's signature is invalid: wrong server key pinned, or someone in the middle")
        ikm = b""
        try:
            if cfg.uses_kem:
                ikm += P.kem_decap(cfg.kem, self._kem.secret, f["kem_ct"])
            if cfg.uses_x25519:
                ikm += P.x25519_shared(self._x25519.secret, f["x25519_pk"])
        except ValueError as e:
            raise HandshakeError(f"bad key-exchange data from the server: {e}") from None
        keys = _session_keys(cfg, ikm, th)
        if not P.mac_verify(cfg.kdf_hash, keys["server_confirm"], th, f["confirm"]):
            raise HandshakeError("key confirmation failed: the two sides derived different keys")
        finish = pack({"type": "client_finish",
                       "sig": P.sign(self.me.alg, self.me.secret, LABEL + b" site signature" + th_full),
                       "confirm": P.mac(cfg.kdf_hash, keys["site_confirm"], th_full)})
        self._kem = self._x25519 = self._hello = None            # forget the per-connection secrets
        return finish, Session(cfg, "site", f["server"], th_full, keys["site_to_server"], keys["server_to_site"])


class ServerHandshake:
    """The server side, one per connection: respond(client_hello) -> send; complete(client_finish) -> Session.
    After respond(), `peer` is the site's name and `peer_identity` its pinned key (name only in mode none)."""

    def __init__(self, cfg: PQConfig, me: Identity | str, trust: TrustStore | None):
        _check_keys(cfg, me, trust, "sites")
        self.cfg, self.me, self.trust = cfg, me, trust
        self.peer = None
        self.peer_identity = None
        self._hello = self._reply = self._keys = None

    def respond(self, client_hello: bytes) -> bytes:
        cfg = self.cfg
        f = _parse(cfg, client_hello, "client_hello")
        if f["protocol"] != PROTOCOL_VERSION:
            raise HandshakeError(f"protocol version {f['protocol']} is not supported (this side: {PROTOCOL_VERSION})")
        site = f["site"]
        if cfg.encrypted:
            ident = self.trust.get(site)
            if ident is None:
                raise HandshakeError(f"unknown site {site!r}: its .pub file is not in the server's trust store")
            if ident.alg != cfg.sig_alg:
                raise HandshakeError(f"site {site!r} is pinned with a {ident.alg} key, settings need {cfg.sig_alg}")
        else:
            ident = site
        reply = {"type": "server_hello", "settings": cfg.fingerprint(), "server": _name(self.me),
                 "nonce": P.random_bytes(NONCE_BYTES)}
        ikm = b""
        try:
            if cfg.uses_kem:
                enc = P.kem_encap(cfg.kem, f["kem_pk"])
                reply["kem_ct"] = enc.ciphertext
                ikm += enc.shared_secret
            if cfg.uses_x25519:
                mine = P.x25519_keypair()
                reply["x25519_pk"] = mine.public
                ikm += P.x25519_shared(mine.secret, f["x25519_pk"])
        except ValueError as e:
            raise HandshakeError(f"bad key-exchange data from site {site!r}: {e}") from None
        if cfg.encrypted:
            th = _transcript(cfg, client_hello, pack(reply))
            self._keys = _session_keys(cfg, ikm, th)
            reply["sig"] = P.sign(self.me.alg, self.me.secret, LABEL + b" server signature" + th)
            reply["confirm"] = P.mac(cfg.kdf_hash, self._keys["server_confirm"], th)
        self.peer, self.peer_identity = site, ident
        self._hello, self._reply = client_hello, pack(reply)
        return self._reply

    def complete(self, client_finish: bytes) -> Session:
        if self._reply is None:
            raise RuntimeError("call respond() first")
        cfg = self.cfg
        f = _parse(cfg, client_finish, "client_finish")
        th_full = _transcript(cfg, self._hello, self._reply)
        self._reply = None                                       # one handshake per object
        if not cfg.encrypted:
            return Session(cfg, "server", self.peer, th_full, None, None)
        site = self.peer_identity
        if not P.verify(site.alg, site.public, LABEL + b" site signature" + th_full, f["sig"]):
            raise HandshakeError(f"site {self.peer!r}: signature invalid (wrong key, or someone in the middle)")
        if not P.mac_verify(cfg.kdf_hash, self._keys["site_confirm"], th_full, f["confirm"]):
            raise HandshakeError(f"site {self.peer!r}: key confirmation failed")
        keys, self._keys = self._keys, None
        return Session(cfg, "server", self.peer, th_full, keys["server_to_site"], keys["site_to_server"])


class Session:
    """An open channel: seal() what you send, open() what you receive. One sender and one receiver per
    direction; messages must be opened in the order they were sealed (as over TCP)."""

    def __init__(self, cfg: PQConfig, role: str, peer: str, session_id: bytes,
                 send_key: bytes | None, recv_key: bytes | None):
        self.cfg, self.role, self.peer, self.session_id = cfg, role, peer, session_id
        self._send_key, self._recv_key = send_key, recv_key
        self.send_seq = 0
        self.recv_seq = 0

    @property
    def overhead(self) -> int:
        """Bytes added to every message: sequence number (+ nonce and tag when encrypted)."""
        return SEQ_BYTES + (P.AEAD_OVERHEAD if self.cfg.encrypted else 0)

    def seal(self, payload: bytes) -> bytes:
        header = self.send_seq.to_bytes(SEQ_BYTES, "big")
        body = bytes(payload) if self._send_key is None else \
            P.aead_seal(self._send_key, payload, LABEL + b" data" + header)
        self.send_seq += 1
        return header + body

    def open(self, message: bytes) -> bytes:
        message = bytes(message)
        if len(message) < SEQ_BYTES:
            raise IntegrityError("message too short")
        header, body = message[:SEQ_BYTES], message[SEQ_BYTES:]
        seq = int.from_bytes(header, "big")
        if seq != self.recv_seq:
            raise IntegrityError(f"expected message #{self.recv_seq}, got #{seq}: replayed, reordered or lost")
        payload = body if self._recv_key is None else \
            P.aead_open(self._recv_key, body, LABEL + b" data" + header)
        self.recv_seq += 1                       # only after the message passed every check
        return payload


# ---- signed models
@dataclass(frozen=True)
class SignedModel:
    weights: bytes                     # encoded weights; serialize.loads_weights turns them into arrays
    round: int
    purpose: str
    signer: str
    meta: dict = field(default_factory=dict)


_MODEL_TYPES = {"type": str, "purpose": str, "round": int, "signer": str, "weights": bytes, "meta": str,
                "alg": str, "sig": bytes}


def sign_model(cfg: PQConfig, me: Identity | str, weights: bytes, *, round: int, purpose: str,
               meta: dict | None = None) -> bytes:
    """Envelope with the encoded weights, signed over everything (weights, round, purpose, signer, meta).
    `meta` holds small JSON values, e.g. {"n_rows": 1234} for FedAvg weighting."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    f = {"type": "model", "purpose": purpose, "round": int(round), "signer": _name(me), "weights": bytes(weights)}
    if meta:
        f["meta"] = json.dumps(meta, sort_keys=True, separators=(",", ":"))
    if cfg.sig_alg is not None:
        if not isinstance(me, Identity) or me.secret is None or me.alg != cfg.sig_alg:
            raise ValueError(f"mode {cfg.mode} signs models with {cfg.sig_alg}: give the matching secret key")
        f["alg"] = me.alg
        f["sig"] = P.sign(me.alg, me.secret, LABEL + b" model" + pack(f))
    return pack(f)


def open_model(cfg: PQConfig, envelope: bytes, signer: Identity | str | None, *, round: int | None = None,
               purpose: str | None = None) -> SignedModel:
    """Check an envelope from sign_model: signature by the pinned `signer`, and the expected round and
    purpose when given. Raises SignatureError if anything is wrong."""
    try:
        f = unpack(envelope)
    except FormatError as e:
        raise SignatureError(f"malformed model envelope: {e}") from None
    want = {"type", "purpose", "round", "signer", "weights"} | ({"alg", "sig"} if cfg.sig_alg else set())
    have = set(f)
    if f.get("type") != "model" or not want <= have <= want | {"meta"}:
        raise SignatureError(f"not a valid model envelope for mode {cfg.mode}: fields {sorted(have)}")
    if any(not _is(v, _MODEL_TYPES[k]) for k, v in f.items()):
        raise SignatureError("model envelope has a field of the wrong type")
    if cfg.sig_alg is not None:
        if not isinstance(signer, Identity):
            raise ValueError(f"mode {cfg.mode} needs the signer's pinned public key to check a model")
        if f["signer"] != signer.name:
            raise SignatureError(f"model signed by {f['signer']!r}, expected {signer.name!r}")
        if f["alg"] != cfg.sig_alg or signer.alg != cfg.sig_alg:
            raise SignatureError(f"model signed with {f['alg']}, settings need {cfg.sig_alg}")
        sig = f.pop("sig")
        if not P.verify(signer.alg, signer.public, LABEL + b" model" + pack(f), sig):
            raise SignatureError("model signature invalid: the weights, round or purpose were changed, "
                                 "or a different key signed them")
    elif signer is not None and f["signer"] != _name(signer):
        raise SignatureError(f"model from {f['signer']!r}, expected {_name(signer)!r}")
    if round is not None and f["round"] != round:
        raise SignatureError(f"model is for round {f['round']}, expected {round}: stale or replayed")
    if purpose is not None and f["purpose"] != purpose:
        raise SignatureError(f"model purpose is {f['purpose']!r}, expected {purpose!r}")
    try:
        meta = json.loads(f["meta"]) if "meta" in f else {}
    except ValueError:
        raise SignatureError("model meta is not valid JSON") from None
    return SignedModel(f["weights"], f["round"], f["purpose"], f["signer"], meta)
