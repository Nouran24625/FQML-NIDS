"""Cryptographic building blocks of the post-quantum channel (proposed decision D15). Owner: Sheghaf.

One small function per operation, so channel.py never calls a crypto library directly:

  kem_keypair(alg) / kem_encap(alg, pk) / kem_decap(alg, sk, ct)     ML-KEM (NIST FIPS 203), liboqs
  sig_keypair(alg) / sign(alg, sk, msg) / verify(alg, pk, msg, sig)   ML-DSA (FIPS 204), liboqs; or Ed25519
  x25519_keypair() / x25519_shared(sk, peer_pk)                       X25519 (RFC 7748), cryptography
  hkdf(hash, ikm, salt, info, length) / mac(hash, key, data) / digest(hash, data)
  aead_seal(key, data, aad) / aead_open(key, blob, aad)               AES-256-GCM, random 96-bit nonce

Keys, secrets and outputs are plain bytes; key pairs come back as KeyPair(public, secret).
Everything that can arrive from the network is length-checked here before it reaches liboqs (its
Python wrapper silently zero-pads a short buffer). liboqs is imported only when an ML-KEM or ML-DSA
function runs, so the classical and none modes also work on machines without it.
"""
from __future__ import annotations

import functools
import hashlib
import hmac as _hmac
import os
from typing import NamedTuple

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from fqml.pq.config import KDF_HASHES, KEMS, SIGNATURES

SIG_ALGS = SIGNATURES + ("Ed25519",)
AEAD_KEY_BYTES = 32            # AES-256
AEAD_NONCE_BYTES = 12          # 96-bit nonce, random per message
AEAD_TAG_BYTES = 16
AEAD_OVERHEAD = AEAD_NONCE_BYTES + AEAD_TAG_BYTES


# ---- errors: anything that means "this data must not be used"
class SecurityError(Exception):
    """A security check failed; the data that caused it must be thrown away."""


class IntegrityError(SecurityError):
    """A sealed message was tampered with, replayed, reordered, or opened with the wrong key."""


class SignatureError(SecurityError):
    """A signature did not verify, or it was made by a different key than the one expected."""


class HandshakeError(SecurityError):
    """The handshake failed: unknown peer, different settings (possible downgrade), or a bad proof."""


class KeyPair(NamedTuple):
    public: bytes
    secret: bytes


class Encapsulation(NamedTuple):
    ciphertext: bytes          # sent to the owner of the public key
    shared_secret: bytes       # kept; the owner recovers it with kem_decap


# ---- liboqs (ML-KEM, ML-DSA)
def _oqs():
    try:
        import oqs
    except ImportError as e:
        raise RuntimeError("liboqs-python is not installed: run inside the project Docker image "
                           "(docker/Dockerfile builds liboqs 0.16.0), or use mode classical/none") from e
    return oqs


def _check(name: str, value, size: int) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"{name} must be bytes, got {type(value).__name__}")
    value = bytes(value)
    if len(value) != size:
        raise ValueError(f"{name} must be {size} bytes, got {len(value)}")
    return value


@functools.lru_cache(maxsize=None)
def kem_sizes(alg: str) -> dict:
    """Byte sizes of an ML-KEM variant: public_key, secret_key, ciphertext, shared_secret."""
    if alg not in KEMS:
        raise ValueError(f"unknown KEM {alg!r}; allowed: {KEMS}")
    with _oqs().KeyEncapsulation(alg) as k:
        return {"public_key": int(k.length_public_key), "secret_key": int(k.length_secret_key),
                "ciphertext": int(k.length_ciphertext), "shared_secret": int(k.length_shared_secret)}


@functools.lru_cache(maxsize=None)
def sig_sizes(alg: str) -> dict:
    """Byte sizes of a signature algorithm: public_key, secret_key, signature."""
    if alg == "Ed25519":
        return {"public_key": 32, "secret_key": 32, "signature": 64}
    if alg not in SIGNATURES:
        raise ValueError(f"unknown signature algorithm {alg!r}; allowed: {SIG_ALGS}")
    with _oqs().Signature(alg) as s:
        return {"public_key": int(s.length_public_key), "secret_key": int(s.length_secret_key),
                "signature": int(s.length_signature)}


def kem_keypair(alg: str) -> KeyPair:
    sizes = kem_sizes(alg)
    with _oqs().KeyEncapsulation(alg) as k:
        public = k.generate_keypair()
        secret = k.export_secret_key()
    return KeyPair(_check("KEM public key", public, sizes["public_key"]),
                   _check("KEM secret key", secret, sizes["secret_key"]))


def kem_encap(alg: str, pk: bytes) -> Encapsulation:
    """Make a fresh shared secret for the owner of `pk`. Raises ValueError for a malformed key."""
    pk = _check("KEM public key", pk, kem_sizes(alg)["public_key"])
    with _oqs().KeyEncapsulation(alg) as k:
        try:
            ct, ss = k.encap_secret(pk)
        except RuntimeError as e:            # FIPS 203 input check failed (coefficients out of range)
            raise ValueError(f"{alg}: invalid public key") from e
    return Encapsulation(ct, ss)


def kem_decap(alg: str, sk: bytes, ct: bytes) -> bytes:
    """Recover the shared secret. A tampered ciphertext gives an unrelated secret (implicit rejection),
    which the handshake then catches with its key-confirmation check."""
    sizes = kem_sizes(alg)
    sk = _check("KEM secret key", sk, sizes["secret_key"])
    ct = _check("KEM ciphertext", ct, sizes["ciphertext"])
    with _oqs().KeyEncapsulation(alg, secret_key=sk) as k:
        try:
            return k.decap_secret(ct)
        except RuntimeError as e:
            raise ValueError(f"{alg}: decapsulation failed") from e


# ---- signatures (ML-DSA via liboqs, Ed25519 via cryptography for the classical baseline)
def sig_keypair(alg: str) -> KeyPair:
    if alg == "Ed25519":
        sk = Ed25519PrivateKey.generate()
        return KeyPair(sk.public_key().public_bytes_raw(), sk.private_bytes_raw())
    sizes = sig_sizes(alg)
    with _oqs().Signature(alg) as s:
        public = s.generate_keypair()
        secret = s.export_secret_key()
    return KeyPair(_check("signature public key", public, sizes["public_key"]),
                   _check("signature secret key", secret, sizes["secret_key"]))


def sign(alg: str, sk: bytes, msg: bytes) -> bytes:
    sk = _check("signature secret key", sk, sig_sizes(alg)["secret_key"])
    if alg == "Ed25519":
        return Ed25519PrivateKey.from_private_bytes(sk).sign(bytes(msg))
    with _oqs().Signature(alg, secret_key=sk) as s:
        return s.sign(bytes(msg))


def verify(alg: str, pk: bytes, msg: bytes, sig: bytes) -> bool:
    """True only for a valid signature. A malformed signature gives False; a malformed public key is a
    setup error (keys come from the trust store, not the network) and raises ValueError."""
    sizes = sig_sizes(alg)
    pk = _check("signature public key", pk, sizes["public_key"])
    if not isinstance(sig, (bytes, bytearray, memoryview)) or len(sig) != sizes["signature"]:
        return False
    if alg == "Ed25519":
        try:
            Ed25519PublicKey.from_public_bytes(pk).verify(bytes(sig), bytes(msg))
            return True
        except InvalidSignature:
            return False
    with _oqs().Signature(alg) as s:
        return bool(s.verify(bytes(msg), bytes(sig), pk))


def check_public_key(alg: str, pk: bytes) -> bytes:
    """Validate the length of a signature public key (used when loading key files)."""
    return _check(f"{alg} public key", pk, sig_sizes(alg)["public_key"])


# ---- X25519 (classical key exchange; also the classical half of the hybrid)
def x25519_keypair() -> KeyPair:
    sk = X25519PrivateKey.generate()
    return KeyPair(sk.public_key().public_bytes_raw(), sk.private_bytes_raw())


def x25519_shared(sk: bytes, peer_pk: bytes) -> bytes:
    """Raises ValueError for a malformed or low-order peer key (all-zero shared secret)."""
    priv = X25519PrivateKey.from_private_bytes(_check("X25519 secret key", sk, 32))
    return priv.exchange(X25519PublicKey.from_public_bytes(_check("X25519 public key", peer_pk, 32)))


# ---- hashing and key derivation
def _hash(name: str):
    if name not in KDF_HASHES:
        raise ValueError(f"unknown hash {name!r}; allowed: {KDF_HASHES}")
    return {"SHA-256": hashes.SHA256, "SHA-384": hashes.SHA384}[name]()


def _hashlib_name(name: str) -> str:
    _hash(name)
    return name.replace("-", "").lower()          # SHA-256 -> sha256


def digest(hash_name: str, data: bytes) -> bytes:
    return hashlib.new(_hashlib_name(hash_name), bytes(data)).digest()


def digest_size(hash_name: str) -> int:
    return hashlib.new(_hashlib_name(hash_name)).digest_size


def hkdf(hash_name: str, ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """HKDF (RFC 5869): extract with `salt`, expand with `info` to `length` bytes."""
    return HKDF(algorithm=_hash(hash_name), length=length, salt=bytes(salt), info=bytes(info)).derive(bytes(ikm))


def mac(hash_name: str, key: bytes, data: bytes) -> bytes:
    return _hmac.new(bytes(key), bytes(data), _hashlib_name(hash_name)).digest()


def mac_verify(hash_name: str, key: bytes, data: bytes, tag: bytes) -> bool:
    return _hmac.compare_digest(mac(hash_name, key, data), bytes(tag))   # constant time


# ---- AES-256-GCM
def aead_seal(key: bytes, data: bytes, aad: bytes = b"") -> bytes:
    """Encrypt and authenticate `data`; `aad` is authenticated but not encrypted (and not included).
    Output: nonce (12 B) || ciphertext || tag (16 B)."""
    nonce = os.urandom(AEAD_NONCE_BYTES)
    return nonce + AESGCM(_check("AES-256 key", key, AEAD_KEY_BYTES)).encrypt(nonce, bytes(data), bytes(aad))


def aead_open(key: bytes, blob: bytes, aad: bytes = b"") -> bytes:
    """Inverse of aead_seal. Raises IntegrityError if anything was changed or the key/aad is wrong."""
    aes = AESGCM(_check("AES-256 key", key, AEAD_KEY_BYTES))
    blob = bytes(blob)
    if len(blob) < AEAD_OVERHEAD:
        raise IntegrityError("sealed message is too short")
    try:
        return aes.decrypt(blob[:AEAD_NONCE_BYTES], blob[AEAD_NONCE_BYTES:], bytes(aad))
    except InvalidTag:
        raise IntegrityError("authentication failed: message tampered with, or wrong key") from None


def random_bytes(n: int) -> bytes:
    return os.urandom(n)
