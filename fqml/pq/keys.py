"""Long-term identity keys of the server and the sites (proposed decision D15). Owner: Sheghaf.

Every party (the server and each site) has ONE signing key pair: ML-DSA in mode pq, Ed25519 in mode
classical (mode none uses no keys). The secret half never leaves its machine. The public half is
copied to the other side once, before training: the server keeps every site's .pub file and each site
keeps the server's ("key pinning", like SSH known_hosts). A fixed federation of known sites needs no
certificate authority, and in mode pq no classical signature is left to break.

  keys/<settings>/<name>.key   secret key (JSON, file mode 600). NEVER commit or share it (git-ignored)
  keys/<settings>/<name>.pub   public key (JSON). Give this one to the other side

  python -m fqml.pq.keys gen server site-01 site-02 site-03 --config default     # -> keys/default/
  python -m fqml.pq.keys show keys/default/server.pub keys/default/site-01.pub
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from fqml.pq.config import ConfigError, PQConfig, load_config
from fqml.pq.primitives import SIG_ALGS, check_public_key, digest, sig_keypair, sig_sizes, sign, verify

KEY_FORMAT = "fqml-pq-key/1"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")      # also a safe file name


class KeyFileError(ValueError):
    """A key file is missing, malformed, or does not match its own contents."""


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _unb64(text, what: str) -> bytes:
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, TypeError, ValueError):
        raise KeyFileError(f"{what} is not valid base64") from None


@dataclass(frozen=True)
class Identity:
    """A named signing key. `secret` is None for a pinned (public-only) identity."""
    name: str
    alg: str
    public: bytes
    secret: bytes | None = field(default=None, repr=False)

    def __post_init__(self):
        if not isinstance(self.name, str) or not NAME_RE.match(self.name):
            raise KeyFileError(f"invalid name {self.name!r}: use letters, digits, '.', '_' or '-' (max 64)")
        if self.alg not in SIG_ALGS:
            raise KeyFileError(f"unknown signature algorithm {self.alg!r}; allowed: {SIG_ALGS}")
        check_public_key(self.alg, self.public)
        if self.secret is not None and len(self.secret) != sig_sizes(self.alg)["secret_key"]:
            raise KeyFileError(f"{self.name}: secret key has the wrong length for {self.alg}")

    @property
    def key_id(self) -> str:
        """Short fingerprint to compare by eye or over the phone (16 hex digits)."""
        return digest("SHA-256", b"fqml-pq key id\x00" + self.alg.encode() + b"\x00" + self.public)[:8].hex()

    def public_only(self) -> Identity:
        return Identity(self.name, self.alg, self.public)

    def to_json(self, include_secret: bool) -> dict:
        out = {"format": KEY_FORMAT, "name": self.name, "alg": self.alg, "key_id": self.key_id,
               "public": _b64(self.public)}
        if include_secret:
            if self.secret is None:
                raise KeyFileError(f"{self.name}: no secret key to save")
            out["secret"] = _b64(self.secret)
        return out


def default_key_dir(config_name: str | Path) -> Path:
    """keys/<settings name>: each settings file gets its own keys (ML-DSA-65 keys cannot sign for cnsa2)."""
    return Path("keys") / Path(str(config_name)).stem


def sig_alg_of(alg_or_cfg) -> str:
    if isinstance(alg_or_cfg, PQConfig):
        if alg_or_cfg.sig_alg is None:
            raise ConfigError("mode none uses no keys")
        return alg_or_cfg.sig_alg
    return alg_or_cfg


def generate_identity(name: str, alg_or_cfg) -> Identity:
    alg = sig_alg_of(alg_or_cfg)
    pair = sig_keypair(alg)
    return Identity(name, alg, pair.public, pair.secret)


def save_identity(ident: Identity, directory: str | Path = "keys", overwrite: bool = False) -> tuple[Path, Path]:
    """Write <name>.key (secret, mode 600) and <name>.pub. Refuses to replace existing files unless
    overwrite=True, because a replaced key locks this party out until the other side re-pins it."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    key_path, pub_path = d / f"{ident.name}.key", d / f"{ident.name}.pub"
    if not overwrite:
        for p in (key_path, pub_path):
            if p.exists():
                raise FileExistsError(f"{p} exists; pass overwrite=True (CLI: --force) to replace it")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    fd = os.open(key_path, flags, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(ident.to_json(include_secret=True), f, indent=1)
    try:
        os.chmod(key_path, 0o600)                     # also when the file already existed
    except OSError:
        pass                                          # e.g. some Windows file systems
    pub_path.write_text(json.dumps(ident.to_json(include_secret=False), indent=1))
    return key_path, pub_path


def load_identity(path: str | Path) -> Identity:
    """Load a .key (with secret) or .pub file. The stored key ID must match the key, and a secret key
    must sign something its public key verifies, so a corrupted or mixed-up file fails here."""
    p = Path(path)
    try:
        data = json.loads(p.read_text())
    except FileNotFoundError:
        raise KeyFileError(f"no key file at {p}") from None
    except (UnicodeDecodeError, ValueError):
        raise KeyFileError(f"{p}: not a JSON key file") from None
    if not isinstance(data, dict) or data.get("format") != KEY_FORMAT:
        raise KeyFileError(f"{p}: not a {KEY_FORMAT} file")
    secret = _unb64(data["secret"], f"{p}: secret") if "secret" in data else None
    try:
        ident = Identity(data.get("name"), data.get("alg"), _unb64(data.get("public"), f"{p}: public"), secret)
    except ValueError as e:
        raise KeyFileError(f"{p}: {e}") from None
    if data.get("key_id") != ident.key_id:
        raise KeyFileError(f"{p}: key ID does not match the key (file edited or corrupted)")
    if secret is not None:
        probe = b"fqml-pq key self-test"
        if not verify(ident.alg, ident.public, probe, sign(ident.alg, secret, probe)):
            raise KeyFileError(f"{p}: secret and public key do not belong together")
    return ident


class TrustStore:
    """The pinned public keys this party accepts, by name."""

    def __init__(self, identities=()):
        self._by_name: dict[str, Identity] = {}
        for ident in identities:
            self.add(ident)

    def add(self, ident: Identity) -> None:
        old = self._by_name.get(ident.name)
        if old is not None and (old.alg, old.public) != (ident.alg, ident.public):
            raise KeyFileError(f"two different keys for {ident.name!r}")
        self._by_name[ident.name] = ident.public_only()

    def get(self, name: str) -> Identity | None:
        return self._by_name.get(name)

    def __contains__(self, name) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)

    def names(self) -> list[str]:
        return sorted(self._by_name)

    @classmethod
    def from_dir(cls, directory: str | Path = "keys", exclude=()) -> TrustStore:
        """Every *.pub file in `directory`, except the names in `exclude` (usually your own)."""
        d = Path(directory)
        if not d.is_dir():
            raise KeyFileError(f"no key folder at {d}")
        return cls(i for i in (load_identity(p) for p in sorted(d.glob("*.pub"))) if i.name not in set(exclude))


# ---- command line
def main(argv=None) -> None:
    try:
        _main(argv)
    except (KeyFileError, ConfigError, FileExistsError) as e:
        raise SystemExit(f"stopped: {e}") from None


def _main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m fqml.pq.keys", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="create identity key pairs")
    g.add_argument("names", nargs="+", help="e.g. server site-01 site-02")
    g.add_argument("--config", default="default", help="PQ settings: name in configs/pq or a YAML path")
    g.add_argument("--dir", default=None, help="key folder (default: keys/<config name>)")
    g.add_argument("--force", action="store_true", help="replace existing key files")
    s = sub.add_parser("show", help="print algorithm and key ID of key files")
    s.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "gen":
        cfg = load_config(a.config)
        key_dir = Path(a.dir) if a.dir else default_key_dir(a.config)
        for name in a.names:
            ident = generate_identity(name, cfg)
            key_path, pub_path = save_identity(ident, key_dir, overwrite=a.force)
            print(f"{name:12s} {ident.alg:10s} key ID {ident.key_id}   {key_path} (secret)  {pub_path}")
        print("Copy each .pub file to the other side; never share a .key file.")
    else:
        for f in a.files:
            ident = load_identity(f)
            kind = "secret+public" if ident.secret is not None else "public"
            print(f"{ident.name:12s} {ident.alg:10s} key ID {ident.key_id}   {kind}   {f}")


if __name__ == "__main__":
    main()
