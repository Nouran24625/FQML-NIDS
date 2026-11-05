"""Post-quantum channel (fqml/pq, proposed decision D15): every mode works end to end, and every attack
the design guards against is rejected: tampering, replay, reordering, unknown or wrong keys, downgrade,
a man in the middle, and stale or re-signed models.

ML-KEM / ML-DSA tests need liboqs. It is in the Docker image, so CI always runs them; on a machine
without it they are skipped and the classical and none modes are still tested.
"""
import importlib.util
import json
import os
import socket
import sys
import threading

import numpy as np
import pytest

from fqml.pq import primitives as P
from fqml.pq.channel import (ClientHandshake, HandshakeError, IntegrityError, ServerHandshake, SignatureError,
                             open_model, sign_model)
from fqml.pq.config import CONFIG_DIR, ConfigError, PQConfig, load_config
from fqml.pq.keys import Identity, KeyFileError, TrustStore, generate_identity, load_identity, save_identity
from fqml.pq.link import Link, LinkStats, Listener, connect, load_server, load_site
from fqml.pq.serialize import FormatError, dumps_weights, loads_weights, pack, unpack

HAVE_OQS = importlib.util.find_spec("oqs") is not None
needs_oqs = pytest.mark.skipif(not HAVE_OQS, reason="liboqs-python is not installed (it is in the Docker image)")

SETTINGS = {
    "default": load_config("default"),                     # ML-KEM-768 + X25519, ML-DSA-65 (frozen interface)
    "cnsa2": load_config("cnsa2"),                         # ML-KEM-1024, ML-DSA-87, SHA-384, no hybrid
    "pq-512": PQConfig(kem="ML-KEM-512", signature="ML-DSA-44", hybrid_x25519=False),
    "classical": load_config("classical"),                 # X25519 + Ed25519 baseline
    "none": load_config("none"),                           # no protection baseline
}


def modes(*names):
    names = names or tuple(SETTINGS)
    return [pytest.param(n, marks=needs_oqs) if SETTINGS[n].mode == "pq" else n for n in names]


def parties(cfg, site="site-01"):
    """(server, site, the server's pinned public key, the server's trust store); names only in mode none."""
    if not cfg.encrypted:
        return "server", site, "server", None
    server, me = generate_identity("server", cfg), generate_identity(site, cfg)
    return server, me, server.public_only(), TrustStore([me])


def handshake(cfg, server, site, server_pub, trust):
    c, s = ClientHandshake(cfg, site, server_pub), ServerHandshake(cfg, server, trust)
    ch = c.hello()
    sh = s.respond(ch)
    cf, site_session = c.finish(sh)
    return site_session, s.complete(cf)


def edit(message: bytes, **changes) -> bytes:
    """Change fields of a packed message but keep it well-formed, so the cryptographic checks are what fail."""
    f = unpack(message)
    for k, v in changes.items():
        f[k] = v(f[k]) if callable(v) else v
    return pack(f)


def flip(b: bytes, i: int = 0) -> bytes:
    return b[:i] + bytes([b[i] ^ 0x01]) + b[i + 1:]


# ---- settings
def test_settings_files_load_and_differ(tmp_path):
    files = sorted(CONFIG_DIR.glob("*.yaml"))
    assert {f.stem for f in files} >= {"default", "classical", "cnsa2", "none"}
    cfgs = [load_config(f.stem) for f in files]
    assert len({c.fingerprint() for c in cfgs}) == len(cfgs)          # every settings file is distinguishable
    d = load_config("default")
    assert (d.mode, d.kem, d.signature, d.hybrid_x25519) == ("pq", "ML-KEM-768", "ML-DSA-65", True)
    bad = tmp_path / "typo.yaml"
    bad.write_text("mode: pq\nkem_alg: ML-KEM-768\n")
    with pytest.raises(ConfigError):
        load_config(bad)
    with pytest.raises(ConfigError):
        PQConfig(kem="ML-KEM-999")


# ---- primitives
@needs_oqs
def test_ml_kem_and_ml_dsa_all_variants():
    for alg in ("ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"):
        kp = P.kem_keypair(alg)
        enc = P.kem_encap(alg, kp.public)
        assert P.kem_decap(alg, kp.secret, enc.ciphertext) == enc.shared_secret
        assert P.kem_decap(alg, kp.secret, flip(enc.ciphertext)) != enc.shared_secret   # implicit rejection
        with pytest.raises(ValueError):
            P.kem_encap(alg, kp.public[:-1])                 # short key: refused, never zero-padded
        with pytest.raises(ValueError):
            P.kem_encap(alg, b"\xff" * len(kp.public))       # FIPS 203 input check (coefficients >= q)
    for alg in ("ML-DSA-44", "ML-DSA-65", "ML-DSA-87"):
        kp = P.sig_keypair(alg)
        sig = P.sign(alg, kp.secret, b"model")
        assert P.verify(alg, kp.public, b"model", sig)
        assert not P.verify(alg, kp.public, b"modeL", sig)
        assert not P.verify(alg, kp.public, b"model", flip(sig, 100))
        assert not P.verify(alg, kp.public, b"model", sig[:-1])
        assert not P.verify(alg, P.sig_keypair(alg).public, b"model", sig)


def test_classical_primitives_and_aead():
    kp = P.sig_keypair("Ed25519")
    assert P.verify("Ed25519", kp.public, b"m", P.sign("Ed25519", kp.secret, b"m"))
    assert not P.verify("Ed25519", kp.public, b"m", b"\x00" * 64)
    a, b = P.x25519_keypair(), P.x25519_keypair()
    assert P.x25519_shared(a.secret, b.public) == P.x25519_shared(b.secret, a.public)
    with pytest.raises(ValueError):
        P.x25519_shared(a.secret, b"\x00" * 32)             # low-order point
    assert len(P.hkdf("SHA-384", b"ikm", b"salt", b"info", 80)) == 80
    key = P.random_bytes(32)
    sealed = P.aead_seal(key, b"payload", b"aad")
    assert P.aead_open(key, sealed, b"aad") == b"payload"
    for i in (0, 12, len(sealed) - 1):                       # nonce, ciphertext, tag
        with pytest.raises(P.IntegrityError):
            P.aead_open(key, flip(sealed, i), b"aad")
    with pytest.raises(P.IntegrityError):
        P.aead_open(key, sealed, b"other aad")
    with pytest.raises(P.IntegrityError):
        P.aead_open(P.random_bytes(32), sealed, b"aad")
    assert P.aead_seal(key, b"same") != P.aead_seal(key, b"same")    # fresh nonce every time


@needs_oqs
def test_interoperates_with_openssl():
    """liboqs and OpenSSL (through `cryptography`) are independent FIPS 203/204 implementations: keys,
    ciphertexts and signatures made by one must work with the other, in both directions."""
    try:
        from cryptography.hazmat.primitives.asymmetric import mldsa, mlkem
        mlkem.MLKEM768PrivateKey.generate()
    except Exception as e:                                   # older cryptography / OpenSSL without ML-KEM
        pytest.skip(f"cryptography has no ML-KEM/ML-DSA here: {e}")
    kems = (("ML-KEM-768", mlkem.MLKEM768PrivateKey, mlkem.MLKEM768PublicKey),
            ("ML-KEM-1024", mlkem.MLKEM1024PrivateKey, mlkem.MLKEM1024PublicKey))
    for alg, private_cls, public_cls in kems:
        kp = P.kem_keypair(alg)                                          # liboqs key, OpenSSL encapsulates
        shared, ciphertext = public_cls.from_public_bytes(kp.public).encapsulate()
        assert P.kem_decap(alg, kp.secret, ciphertext) == shared
        theirs = private_cls.generate()                                  # OpenSSL key, liboqs encapsulates
        enc = P.kem_encap(alg, theirs.public_key().public_bytes_raw())
        assert theirs.decapsulate(enc.ciphertext) == enc.shared_secret
    sigs = (("ML-DSA-44", mldsa.MLDSA44PrivateKey, mldsa.MLDSA44PublicKey),
            ("ML-DSA-65", mldsa.MLDSA65PrivateKey, mldsa.MLDSA65PublicKey),
            ("ML-DSA-87", mldsa.MLDSA87PrivateKey, mldsa.MLDSA87PublicKey))
    for alg, private_cls, public_cls in sigs:
        kp = P.sig_keypair(alg)                                          # liboqs signs, OpenSSL verifies
        public_cls.from_public_bytes(kp.public).verify(P.sign(alg, kp.secret, b"m"), b"m")
        theirs = private_cls.generate()                                  # OpenSSL signs, liboqs verifies
        assert P.verify(alg, theirs.public_key().public_bytes_raw(), b"m", theirs.sign(b"m"))


# ---- encodings
def test_pack_is_canonical_and_strict():
    fields = {"z": b"\x00\xff", "a": "text", "n": -7, "flag": True, "big": 2 ** 62}
    blob = pack(fields)
    assert unpack(blob) == fields and pack(unpack(blob)) == blob
    assert list(unpack(blob)) == sorted(fields)
    swapped = b"\x00\x02" + pack({"b": b""})[2:] + pack({"a": b""})[2:]
    duplicate = b"\x00\x02" + pack({"a": b""})[2:] * 2
    bad_bool = pack({"t": True})[:-1] + b"\x02"
    bad_tag = pack({"x": b""}).replace(b"xb", b"xq")
    for bad in (blob[:-1], blob + b"\x00", swapped, duplicate, bad_bool, bad_tag):
        with pytest.raises(FormatError):
            unpack(bad)
    with pytest.raises(FormatError):
        pack({"x": 1.5})


def test_weights_round_trip_bit_exact():
    w = {"theta": np.random.default_rng(0).normal(size=(3, 8, 2)).astype(np.float32),
         "odd": np.array([np.nan, -0.0, np.inf, 1e-45], np.float32),
         "scalar": np.float64(3.25), "empty": np.zeros((0, 4), np.int64), "mask": np.array([True, False]),
         "big_endian": np.arange(5, dtype=">i4"), "transposed": np.arange(12, dtype=np.float16).reshape(3, 4).T}
    blob = dumps_weights(w)
    back = loads_weights(blob)
    assert list(back) == list(w)
    for k, v in w.items():
        v = np.asarray(v)
        assert back[k].shape == v.shape and back[k].dtype == v.dtype.newbyteorder("<")
        assert back[k].tobytes() == v.astype(v.dtype.newbyteorder("<")).tobytes()      # NaN, -0.0 kept
        assert back[k].flags.writeable
    assert dumps_weights(back) == blob                       # canonical
    as_list = loads_weights(dumps_weights([np.ones(3), np.eye(2, dtype=np.float32)]))
    assert isinstance(as_list, list) and np.array_equal(as_list[1], np.eye(2))


def test_weights_reject_bad_input():
    blob = dumps_weights({"a": np.ones(4, np.float32)})
    head_len = int.from_bytes(blob[5:9], "big")
    header = json.loads(blob[9:9 + head_len])
    header["tensors"][0][2] = [10 ** 12]                     # claims ~4 TB: must fail before allocating
    huge = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    forged = blob[:5] + len(huge).to_bytes(4, "big") + huge + blob[9 + head_len:]
    for bad in (blob[:-1], blob + b"\x00", b"XXXX" + blob[4:], forged, b""):
        with pytest.raises(FormatError):
            loads_weights(bad)
    for bad in ({"o": np.array([object()])}, {"c": np.ones(2, np.complex64)}, {1: np.ones(2)}, np.ones(3)):
        with pytest.raises(FormatError):
            dumps_weights(bad)


def test_weights_from_torch():
    torch = pytest.importorskip("torch")
    from fqml.pq.serialize import to_torch
    model = torch.nn.Linear(3, 2)
    params = dict(model.named_parameters())
    back = to_torch(loads_weights(dumps_weights(params)))
    assert all(torch.equal(back[k], params[k].detach()) for k in params)


# ---- keys
@pytest.mark.parametrize("alg", [pytest.param("ML-DSA-65", marks=needs_oqs), "Ed25519"])
def test_key_files(tmp_path, alg):
    ident = generate_identity("site-07", alg)
    key_path, pub_path = save_identity(ident, tmp_path)
    if sys.platform != "win32":
        assert (os.stat(key_path).st_mode & 0o777) == 0o600
    assert load_identity(key_path) == ident
    pub = load_identity(pub_path)
    assert pub.secret is None and pub == ident.public_only() and pub.key_id == ident.key_id
    with pytest.raises(FileExistsError):
        save_identity(ident, tmp_path)
    save_identity(generate_identity("site-07", alg), tmp_path, overwrite=True)

    data = json.loads(key_path.read_text())
    data["key_id"] = "0" * 16                               # edited file
    key_path.write_text(json.dumps(data))
    with pytest.raises(KeyFileError):
        load_identity(key_path)
    other = generate_identity("site-07", alg)              # secret from one key, public from another
    data = ident.to_json(include_secret=True)
    data.update(public=other.to_json(False)["public"], key_id=other.key_id)
    key_path.write_text(json.dumps(data))
    with pytest.raises(KeyFileError):
        load_identity(key_path)
    for name in ("", "../evil", "a" * 65, "site 1"):
        with pytest.raises(KeyFileError):
            generate_identity(name, alg)


def test_trust_store(tmp_path):
    a, b, c = (generate_identity(n, "Ed25519") for n in ("server", "site-01", "site-02"))
    for i in (a, b, c):
        save_identity(i, tmp_path)
    store = TrustStore.from_dir(tmp_path, exclude=["server"])
    assert store.names() == ["site-01", "site-02"] and "server" not in store
    assert store.get("site-01").secret is None                # only public keys are kept
    with pytest.raises(KeyFileError):
        store.add(generate_identity("site-01", "Ed25519"))    # a second, different key for the same name


# ---- the channel
@pytest.mark.parametrize("name", modes())
def test_handshake_and_messages_every_mode(name):
    cfg = SETTINGS[name]
    site_s, server_s = handshake(cfg, *parties(cfg))
    assert site_s.session_id == server_s.session_id and (site_s.peer, server_s.peer) == ("server", "site-01")
    for payload in (b"", b"global model, round 0", os.urandom(1 << 20)):
        sealed = site_s.seal(payload)
        assert server_s.open(sealed) == payload
        assert len(sealed) == len(payload) + site_s.overhead
        assert site_s.open(server_s.seal(payload)) == payload
    secret = b"benign traffic statistics of site-01"
    sealed = site_s.seal(secret)
    assert (secret in sealed) is (not cfg.encrypted)          # encrypted in every mode except none
    assert server_s.open(sealed) == secret


@pytest.mark.parametrize("name", modes("default", "classical", "none"))
def test_replayed_reordered_or_dropped_messages_are_rejected(name):
    cfg = SETTINGS[name]
    site_s, server_s = handshake(cfg, *parties(cfg))
    m0, m1, m2 = (site_s.seal(f"msg {i}".encode()) for i in range(3))
    if cfg.encrypted:                        # relabel a later message with the expected number: the number
        with pytest.raises(IntegrityError):  # is inside the authenticated data, so this fails
            server_s.open(m0[:8] + m1[8:])
    assert server_s.open(m0) == b"msg 0"
    for bad in (m0, m2):                                      # replay, then skipping m1 (dropped/reordered)
        with pytest.raises(IntegrityError):
            server_s.open(bad)
    if cfg.encrypted:
        for i in (0, 7, 8, 20, len(m1) - 1):                  # sequence number, nonce, ciphertext, tag
            with pytest.raises(IntegrityError):
                server_s.open(flip(m1, i))
        with pytest.raises(IntegrityError):
            site_s.open(m0)                  # each direction has its own key: a message cannot be reflected
    assert server_s.open(m1) == b"msg 1" and server_s.open(m2) == b"msg 2"   # failures did not skip ahead


@pytest.mark.parametrize("site_mode,server_mode", [
    pytest.param("default", "classical", marks=needs_oqs), pytest.param("classical", "default", marks=needs_oqs),
    pytest.param("default", "cnsa2", marks=needs_oqs), pytest.param("default", "none", marks=needs_oqs),
    ("classical", "none"), ("none", "classical")])
def test_different_settings_are_refused(site_mode, server_mode):
    """No downgrade: a server never answers a site with other settings, and a site never accepts a
    server reply made with other settings (e.g. a man in the middle offering a weaker mode)."""
    site_cfg, server_cfg = SETTINGS[site_mode], SETTINGS[server_mode]
    _, me, server_pub, _ = parties(site_cfg)
    server, _, _, trust = parties(server_cfg)
    site = ClientHandshake(site_cfg, me, server_pub)
    hello = site.hello()
    with pytest.raises(HandshakeError, match="different PQ settings"):
        ServerHandshake(server_cfg, server, trust).respond(hello)
    weaker = handshake_reply(server_cfg)                     # a full reply from a server with other settings
    with pytest.raises(HandshakeError):
        site.finish(weaker)


def handshake_reply(cfg):
    server, me, server_pub, trust = parties(cfg)
    return ServerHandshake(cfg, server, trust).respond(ClientHandshake(cfg, me, server_pub).hello())


@pytest.mark.parametrize("name", modes("default", "classical"))
def test_unknown_or_impostor_parties_are_refused(name):
    cfg = SETTINGS[name]
    server, site, server_pub, trust = parties(cfg)
    stranger = generate_identity("site-99", cfg)
    with pytest.raises(HandshakeError, match="unknown site"):
        ServerHandshake(cfg, server, trust).respond(ClientHandshake(cfg, stranger, server_pub).hello())

    fake_server = generate_identity("server", cfg)           # right name, wrong key
    c = ClientHandshake(cfg, site, server_pub)
    with pytest.raises(HandshakeError, match="signature"):
        c.finish(ServerHandshake(cfg, fake_server, trust).respond(c.hello()))

    fake_site = generate_identity("site-01", cfg)            # right name, wrong key
    c, s = ClientHandshake(cfg, fake_site, server_pub), ServerHandshake(cfg, server, trust)
    finish, _ = c.finish(s.respond(c.hello()))
    with pytest.raises(HandshakeError, match="signature"):
        s.complete(finish)

    c = ClientHandshake(cfg, site, server_pub)
    other_name = generate_identity("aggregator", cfg)
    with pytest.raises(HandshakeError, match="expected server"):
        c.finish(ServerHandshake(cfg, other_name, trust).respond(c.hello()))


@pytest.mark.parametrize("name", modes("default", "cnsa2", "classical"))
def test_man_in_the_middle_changes_are_detected(name):
    cfg = SETTINGS[name]
    server, site, server_pub, trust = parties(cfg)
    hello_fields = [k for k in ("nonce", "kem_pk", "x25519_pk") if k in unpack(ClientHandshake(cfg, site, server_pub).hello())]
    for field in hello_fields:                               # changed ClientHello: the server signs what it saw,
        c, s = ClientHandshake(cfg, site, server_pub), ServerHandshake(cfg, server, trust)   # the site notices
        try:
            reply = s.respond(edit(c.hello(), **{field: flip}))
        except HandshakeError:
            continue                                         # e.g. an invalid ML-KEM key is refused at once
        with pytest.raises(HandshakeError):
            c.finish(reply)
    for field in ("nonce", "kem_ct", "x25519_pk", "sig", "confirm"):
        c, s = ClientHandshake(cfg, site, server_pub), ServerHandshake(cfg, server, trust)
        reply = s.respond(c.hello())
        if field not in unpack(reply):
            continue
        with pytest.raises(HandshakeError):
            c.finish(edit(reply, **{field: flip}))
    for field in ("sig", "confirm"):
        c, s = ClientHandshake(cfg, site, server_pub), ServerHandshake(cfg, server, trust)
        finish, _ = c.finish(s.respond(c.hello()))
        with pytest.raises(HandshakeError):
            s.complete(edit(finish, **{field: flip}))
    c, s = ClientHandshake(cfg, site, server_pub), ServerHandshake(cfg, server, trust)
    with pytest.raises(HandshakeError):
        s.respond(c.hello()[:-3])                            # garbage is refused, not crashed on


def test_missing_keys_are_caught_before_connecting():
    cfg = SETTINGS["classical"]
    server, site, server_pub, trust = parties(cfg)
    with pytest.raises(ValueError, match="secret"):
        ClientHandshake(cfg, site.public_only(), server_pub)
    with pytest.raises(ValueError, match="pinned"):
        ClientHandshake(cfg, site, None)
    with pytest.raises(ValueError, match="TrustStore"):
        ServerHandshake(cfg, server, None)
    with pytest.raises(ConfigError):
        generate_identity("site-01", SETTINGS["none"])


# ---- signed models
@pytest.mark.parametrize("name", modes("default", "classical", "none"))
def test_signed_models(name):
    cfg = SETTINGS[name]
    server, _, server_pub, _ = parties(cfg)
    weights = dumps_weights({"theta": np.ones((3, 8, 2), np.float32)})
    env = sign_model(cfg, server, weights, round=4, purpose="global", meta={"n_rows": 1234})
    m = open_model(cfg, env, server_pub, round=4, purpose="global")
    assert (m.weights, m.round, m.purpose, m.signer, m.meta) == (weights, 4, "global", "server", {"n_rows": 1234})
    with pytest.raises(SignatureError, match="round"):
        open_model(cfg, env, server_pub, round=5)            # an old round's model replayed as new
    with pytest.raises(SignatureError, match="purpose"):
        open_model(cfg, env, server_pub, purpose="update")
    if not cfg.encrypted:
        return                                               # mode none: nothing below is protected
    changed = dumps_weights({"theta": np.full((3, 8, 2), 9, np.float32)})
    for bad in (edit(env, weights=changed), edit(env, round=5), edit(env, meta='{"n_rows":999999}'),
                edit(env, signer="site-01")):
        with pytest.raises(SignatureError):
            open_model(cfg, bad, server_pub)
    stripped = pack({k: v for k, v in unpack(env).items() if k not in ("sig", "alg")})
    with pytest.raises(SignatureError):                      # removing the signature is no way around it
        open_model(cfg, stripped, server_pub)
    other = generate_identity("server", cfg)
    with pytest.raises(SignatureError):
        open_model(cfg, sign_model(cfg, other, weights, round=4, purpose="global"), server_pub)


# ---- over TCP
@pytest.mark.parametrize("name", modes("default", "classical", "none"))
def test_tcp_link_end_to_end(name):
    """A server and two sites over real sockets for 2 rounds; in protected modes an unknown site is
    refused with a clear reason on both sides, and the server keeps serving the others."""
    cfg = SETTINGS[name]
    server, _, server_pub, _ = parties(cfg)
    sites = {n: (generate_identity(n, cfg) if cfg.encrypted else n) for n in ("site-01", "site-02")}
    trust = TrustStore(sites.values()) if cfg.encrypted else None
    rounds, refused, served = 2, [], {}
    start = {"w": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.zeros(2, np.float32)}
    with Listener(cfg, server, trust, port=0, timeout=10) as lst:
        port = lst.address[1]

        def serve(n_attempts):
            for _ in range(n_attempts):
                try:
                    link = lst.accept()
                except HandshakeError as e:
                    refused.append(str(e))
                    continue
                with link:
                    for r in range(rounds):
                        link.send_model(start, round=r, purpose="global")
                        msg = link.recv_model(round=r, purpose="update")
                        assert msg.signer == link.peer and msg.meta == {"n_rows": 10 * (r + 1)}
                    served[link.peer] = link.stats

        attempts = 3 if cfg.encrypted else 2
        t = threading.Thread(target=serve, args=(attempts,), daemon=True)
        t.start()
        if cfg.encrypted:
            with pytest.raises(HandshakeError, match="unknown site"):
                connect(cfg, generate_identity("site-99", cfg), server_pub, port=port, timeout=10)
        for name_, me in sites.items():
            with connect(cfg, me, server_pub, port=port, timeout=10) as link:
                for r in range(rounds):
                    msg = link.recv_model(round=r, purpose="global")
                    assert np.array_equal(msg.weights["w"], start["w"]) and msg.signer == "server"
                    link.send_model({k: v + 1 for k, v in msg.weights.items()}, round=r, purpose="update",
                                    meta={"n_rows": 10 * (r + 1)})
                stats = link.stats
            assert stats.handshake_seconds > 0 and stats.messages_sent == rounds
            if cfg.encrypted:
                assert stats.bytes_sent > stats.payload_bytes_sent    # what the protection costs on the wire
        t.join(timeout=20)
    assert sorted(served) == ["site-01", "site-02"]
    assert len(refused) == (1 if cfg.encrypted else 0)


@pytest.mark.parametrize("name", modes("default", "classical"))
def test_link_closes_on_a_tampered_frame(name):
    cfg = SETTINGS[name]
    server, site, server_pub, trust = parties(cfg)
    site_s, server_s = handshake(cfg, server, site, server_pub, trust)
    a, b = socket.socketpair()
    site_link = Link(a, site_s, cfg, site, server_pub, LinkStats())
    server_link = Link(b, server_s, cfg, server, site.public_only(), LinkStats())
    site_link.send(b"genuine")
    assert server_link.recv() == b"genuine"
    evil = flip(site_s.seal(b"forged"), 30)
    a.sendall(len(evil).to_bytes(4, "big") + b"D" + evil)
    with pytest.raises(IntegrityError):
        server_link.recv()
    assert server_link.closed
    site_link.close()


def test_link_refuses_oversized_frames():
    cfg = SETTINGS["none"]
    site_s, server_s = handshake(cfg, *parties(cfg))
    a, b = socket.socketpair()
    link = Link(b, server_s, cfg, "server", "site-01", LinkStats(), max_frame=1024)
    a.sendall((10 ** 9).to_bytes(4, "big") + b"D")           # claims 1 GB: refused before reading it
    with pytest.raises(IntegrityError, match="limit"):
        link.recv()
    a.close()
    link.close()


def test_server_refusal_after_the_handshake_reaches_the_site():
    """The site sends its last handshake message and moves on; if the server then refuses it (here: an
    impostor key), the site's next receive raises the server's reason instead of a vague error."""
    cfg = SETTINGS["classical"]
    server, site, server_pub, trust = parties(cfg)
    impostor = generate_identity("site-01", cfg)
    errors = []
    with Listener(cfg, server, trust, port=0, timeout=10) as lst:
        def serve():
            try:
                lst.accept()
            except HandshakeError as e:
                errors.append(str(e))
        t = threading.Thread(target=serve, daemon=True)
        t.start()
        link = connect(cfg, impostor, server_pub, port=lst.address[1], timeout=10)
        with pytest.raises(HandshakeError, match="refused"):
            link.recv()
        t.join(timeout=10)
    assert errors and "signature" in errors[0]


def test_load_helpers(tmp_path):
    cfg = SETTINGS["classical"]
    for n in ("server", "site-01", "site-02"):
        save_identity(generate_identity(n, cfg), tmp_path)
    me, trust = load_server(cfg, tmp_path)
    assert me.name == "server" and trust.names() == ["site-01", "site-02"]
    me, server = load_site(cfg, tmp_path, "site-02")
    assert me.secret is not None and server.secret is None and server.name == "server"
    assert load_server(SETTINGS["none"], tmp_path) == ("server", None)
    assert isinstance(load_site(cfg, tmp_path, "site-01")[0], Identity)
