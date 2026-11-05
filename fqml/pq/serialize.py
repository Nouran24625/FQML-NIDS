"""Byte encodings for the post-quantum channel (proposed decision D15). Owner: Sheghaf.

  pack(fields) / unpack(blob)           handshake messages and model envelopes: {name: bytes | str | int | bool}
  dumps_weights(w) / loads_weights(b)   model weights <-> bytes (this is what gets signed and encrypted)

No pickle: whatever arrives from the network can only become bytes, strings, integers and numeric
arrays, never code. Both formats are canonical (the same input always gives the same bytes, and
pack(unpack(b)) == b), and parsing is strict: truncated data, trailing bytes, unsorted or duplicate
names, unknown types and unsupported dtypes all raise FormatError.

pack layout:  u16 count, then per field (sorted by name): u8 name length, name (UTF-8),
              1-byte type (b bytes, s str, i int64, t bool), u32 value length, value
weights:      b"FQMW", u8 version, u32 header length, JSON header (names, dtypes, shapes), raw
              little-endian C-order array bytes back to back
"""
from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping

import numpy as np

WEIGHTS_MAGIC = b"FQMW"
WEIGHTS_VERSION = 1
DTYPES = ("<f2", "<f4", "<f8", "|i1", "<i2", "<i4", "<i8", "|u1", "<u2", "<u4", "<u8", "|b1")
MAX_DIMS = 32


class FormatError(ValueError):
    """The bytes are not a valid encoding (corrupted, truncated, or not ours)."""


# ---- fields (handshake messages, envelopes)
def pack(fields: Mapping) -> bytes:
    if len(fields) > 0xFFFF:
        raise FormatError("too many fields")
    out = [struct.pack(">H", len(fields))]
    for name in sorted(fields):
        if not isinstance(name, str):
            raise FormatError(f"field names must be str, got {name!r}")
        key = name.encode()
        if not 0 < len(key) <= 255:
            raise FormatError(f"field name length must be 1..255 bytes: {name!r}")
        value = fields[name]
        if isinstance(value, bool):                    # before int: bool is a subclass of int
            tag, raw = b"t", b"\x01" if value else b"\x00"
        elif isinstance(value, int):
            tag, raw = b"i", value.to_bytes(8, "big", signed=True)
        elif isinstance(value, str):
            tag, raw = b"s", value.encode()
        elif isinstance(value, (bytes, bytearray, memoryview)):
            tag, raw = b"b", bytes(value)
        else:
            raise FormatError(f"field {name!r}: unsupported type {type(value).__name__}")
        if len(raw) > 0xFFFFFFFF:
            raise FormatError(f"field {name!r} is too large")
        out += [struct.pack(">B", len(key)), key, tag, struct.pack(">I", len(raw)), raw]
    return b"".join(out)


def unpack(blob: bytes) -> dict:
    view = memoryview(bytes(blob))
    pos = 0

    def take(n: int) -> bytes:
        nonlocal pos
        if n < 0 or pos + n > len(view):
            raise FormatError("truncated message")
        chunk = view[pos:pos + n].tobytes()
        pos += n
        return chunk

    (count,) = struct.unpack(">H", take(2))
    fields: dict = {}
    previous = None
    for _ in range(count):
        (klen,) = struct.unpack(">B", take(1))
        try:
            name = take(klen).decode()
        except UnicodeDecodeError:
            raise FormatError("field name is not UTF-8") from None
        if not name or (previous is not None and name <= previous):
            raise FormatError("field names must be non-empty, sorted and unique")
        previous = name
        tag = take(1)
        (vlen,) = struct.unpack(">I", take(4))
        raw = take(vlen)
        if tag == b"b":
            fields[name] = raw
        elif tag == b"s":
            try:
                fields[name] = raw.decode()
            except UnicodeDecodeError:
                raise FormatError(f"field {name!r} is not UTF-8") from None
        elif tag == b"i":
            if vlen != 8:
                raise FormatError(f"field {name!r}: int must be 8 bytes")
            fields[name] = int.from_bytes(raw, "big", signed=True)
        elif tag == b"t":
            if raw not in (b"\x00", b"\x01"):
                raise FormatError(f"field {name!r}: bool must be 0 or 1")
            fields[name] = raw == b"\x01"
        else:
            raise FormatError(f"field {name!r}: unknown type {tag!r}")
    if pos != len(view):
        raise FormatError(f"{len(view) - pos} unexpected trailing bytes")
    return fields


# ---- model weights
def _as_array(name, value) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):        # torch.Tensor, without importing torch
        try:
            value = value.detach().cpu().numpy()
        except TypeError as e:                                     # e.g. bfloat16 has no numpy dtype
            raise FormatError(f"tensor {name!r}: {e}") from None
    a = np.asarray(value)
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    if a.dtype.str not in DTYPES:
        raise FormatError(f"tensor {name!r}: dtype {a.dtype} is not supported; allowed: {DTYPES}")
    if a.ndim > MAX_DIMS:
        raise FormatError(f"tensor {name!r}: more than {MAX_DIMS} dimensions")
    return a                       # tobytes(order="C") below handles non-contiguous arrays; 0-d stays 0-d


def dumps_weights(weights) -> bytes:
    """Encode a dict {name: array or tensor} (a model's named parameters) or a list of arrays (Flower's
    NDArrays). Order is kept. Torch tensors are copied to CPU first; values are stored bit-exactly."""
    if isinstance(weights, Mapping):
        named = True
        items = list(weights.items())
        for name, _ in items:
            if not isinstance(name, str):
                raise FormatError(f"weight names must be str, got {name!r}")
    elif isinstance(weights, (list, tuple)):
        named = False
        items = [(None, v) for v in weights]
    else:
        raise FormatError("weights must be a dict of named arrays or a list of arrays")
    arrays = [(name, _as_array(name if named else f"#{i}", v)) for i, (name, v) in enumerate(items)]
    header = {"named": named, "tensors": [[name, a.dtype.str, list(a.shape)] for name, a in arrays]}
    head = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return b"".join([WEIGHTS_MAGIC, struct.pack(">BI", WEIGHTS_VERSION, len(head)), head]
                    + [a.tobytes(order="C") for _, a in arrays])


def loads_weights(blob: bytes):
    """Inverse of dumps_weights: dict {name: np.ndarray} or list of np.ndarray (writable copies)."""
    blob = bytes(blob)
    if len(blob) < 9 or blob[:4] != WEIGHTS_MAGIC:
        raise FormatError("not an encoded set of model weights")
    version, hlen = struct.unpack(">BI", blob[4:9])
    if version != WEIGHTS_VERSION:
        raise FormatError(f"weights format version {version} is not supported (expected {WEIGHTS_VERSION})")
    if 9 + hlen > len(blob):
        raise FormatError("truncated weights header")
    try:
        header = json.loads(blob[9:9 + hlen].decode())
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise FormatError("weights header is not valid JSON") from None
    if not isinstance(header, dict) or set(header) != {"named", "tensors"} \
            or not isinstance(header["named"], bool) or not isinstance(header["tensors"], list):
        raise FormatError("weights header has the wrong structure")
    named = header["named"]
    specs = []
    for spec in header["tensors"]:
        if not (isinstance(spec, list) and len(spec) == 3):
            raise FormatError("bad tensor entry in weights header")
        name, dtype, shape = spec
        if (named and not isinstance(name, str)) or (not named and name is not None):
            raise FormatError("bad tensor name in weights header")
        if dtype not in DTYPES:
            raise FormatError(f"dtype {dtype!r} is not supported")
        if not (isinstance(shape, list) and len(shape) <= MAX_DIMS
                and all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape)):
            raise FormatError(f"bad shape {shape!r} in weights header")
        nbytes = math.prod(shape) * np.dtype(dtype).itemsize
        specs.append((name, np.dtype(dtype), tuple(shape), nbytes))
    if named and len({s[0] for s in specs}) != len(specs):
        raise FormatError("duplicate tensor names in weights header")
    offset = 9 + hlen
    if offset + sum(s[3] for s in specs) != len(blob):            # checked before allocating anything
        raise FormatError("weights data length does not match the header")
    arrays = []
    for name, dtype, shape, nbytes in specs:
        count = nbytes // dtype.itemsize
        arrays.append((name, np.frombuffer(blob, dtype=dtype, count=count, offset=offset).reshape(shape).copy()))
        offset += nbytes
    return {name: a for name, a in arrays} if named else [a for _, a in arrays]


def to_torch(weights):
    """Convenience for the FL code: numpy weights (dict or list) -> torch tensors."""
    import torch
    if isinstance(weights, Mapping):
        return {k: torch.from_numpy(v) for k, v in weights.items()}
    return [torch.from_numpy(v) for v in weights]
