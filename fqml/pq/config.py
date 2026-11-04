"""Settings of the post-quantum channel (proposed decision D15). Owner: Sheghaf.

The settings live in configs/pq/<name>.yaml. Server and sites must load the SAME settings: the
handshake compares a fingerprint of them and stops if they differ, so a site can never be talked
down to a weaker mode by someone in the middle.

  mode           pq         ML-KEM (+ X25519 if hybrid) for the session keys, ML-DSA signatures  (default)
                 classical  X25519 for the session keys, Ed25519 signatures       (baseline for E7)
                 none       no encryption, no signatures                          (baseline for E7)
  kem            ML-KEM-512 | ML-KEM-768 | ML-KEM-1024   NIST FIPS 203, used when mode = pq
  signature      ML-DSA-44  | ML-DSA-65  | ML-DSA-87     NIST FIPS 204, used when mode = pq
  hybrid_x25519  true = also mix a classical X25519 secret into the session keys (proposal Table 6)
  kdf_hash       SHA-256 | SHA-384  hash used by HKDF to turn the shared secrets into keys
Every message is encrypted with AES-256-GCM in every mode except none.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

MODES = ("pq", "classical", "none")
KEMS = ("ML-KEM-512", "ML-KEM-768", "ML-KEM-1024")
SIGNATURES = ("ML-DSA-44", "ML-DSA-65", "ML-DSA-87")
KDF_HASHES = ("SHA-256", "SHA-384")
PROTOCOL_VERSION = 1

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "pq"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PQConfig:
    mode: str = "pq"
    kem: str = "ML-KEM-768"
    signature: str = "ML-DSA-65"
    hybrid_x25519: bool = True
    kdf_hash: str = "SHA-256"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ConfigError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.kem not in KEMS:
            raise ConfigError(f"kem must be one of {KEMS}, got {self.kem!r}")
        if self.signature not in SIGNATURES:
            raise ConfigError(f"signature must be one of {SIGNATURES}, got {self.signature!r}")
        if self.kdf_hash not in KDF_HASHES:
            raise ConfigError(f"kdf_hash must be one of {KDF_HASHES}, got {self.kdf_hash!r}")
        if not isinstance(self.hybrid_x25519, bool):
            raise ConfigError("hybrid_x25519 must be true or false")

    # ---- what the settings mean in practice
    @property
    def encrypted(self) -> bool:
        return self.mode != "none"

    @property
    def uses_kem(self) -> bool:
        return self.mode == "pq"

    @property
    def uses_x25519(self) -> bool:
        return self.mode == "classical" or (self.mode == "pq" and self.hybrid_x25519)

    @property
    def sig_alg(self) -> str | None:
        """Algorithm of the identity keys and of the model signatures (None = no signatures)."""
        return {"pq": self.signature, "classical": "Ed25519", "none": None}[self.mode]

    def effective(self) -> dict:
        """Only the settings that change the protocol (unused fields are left out)."""
        out = {"protocol_version": PROTOCOL_VERSION, "mode": self.mode}
        if self.mode == "pq":
            out.update(kem=self.kem, signature=self.signature, hybrid_x25519=self.hybrid_x25519)
        if self.mode != "none":
            out.update(kdf_hash=self.kdf_hash, aead="AES-256-GCM")
        return out

    def fingerprint(self) -> bytes:
        """32 bytes that both sides compare in the handshake."""
        return hashlib.sha256(json.dumps(self.effective(), sort_keys=True).encode()).digest()

    def describe(self) -> str:
        if self.mode == "none":
            return "NONE: no encryption, no signatures (baseline only)"
        if self.mode == "classical":
            return f"classical: X25519 -> HKDF-{self.kdf_hash} -> AES-256-GCM; signatures Ed25519"
        kex = f"{self.kem} + X25519 (hybrid)" if self.hybrid_x25519 else self.kem
        return f"post-quantum: {kex} -> HKDF-{self.kdf_hash} -> AES-256-GCM; signatures {self.signature}"


def load_config(name_or_path: str | Path = "default") -> PQConfig:
    """Load configs/pq/<name>.yaml, or a YAML file given by path. Unknown keys are an error (typos)."""
    p = Path(name_or_path)
    if not p.suffix:
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise ConfigError(f"no PQ settings file at {p}")
    data = yaml.safe_load(p.read_text()) or {}
    known = {f.name for f in fields(PQConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"{p.name}: unknown setting(s) {unknown}; allowed: {sorted(known)}")
    return PQConfig(**data)
