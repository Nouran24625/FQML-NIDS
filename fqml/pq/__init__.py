"""Post-quantum channel between the federated server and its sites (proposed decision D15). Owner: Sheghaf.

  config.py      settings in configs/pq/*.yaml: mode pq | classical | none, algorithms, fingerprint
  primitives.py  ML-KEM, ML-DSA (liboqs), X25519, Ed25519, HKDF, AES-256-GCM: one function each
  serialize.py   pickle-free encodings: handshake fields and model weights <-> bytes
  keys.py        identity keys of the server and the sites, pinned public keys (TrustStore), key CLI
  channel.py     3-message authenticated handshake, encrypted sessions, signed models (no I/O)
  link.py        the channel over TCP, with byte and time counters for E7, and a two-terminal demo
  smoke.py       Phase 0 smoke test (python -m fqml.pq.smoke must pass on every machine)

Guide: docs/PQ_CHANNEL.md. Tests: tests/test_pq_channel.py. Cost measurements: spikes/s6_pq_overhead.py.
"""
