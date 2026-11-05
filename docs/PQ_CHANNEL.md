# Post-quantum channel (`fqml/pq`) — guide

How the federated server and its sites talk in E7: every model travels encrypted and signed, using the NIST
post-quantum standards (ML-KEM, FIPS 203; ML-DSA, FIPS 204). Decision: D15 (proposed). Interface:
INTERFACES section 4. Code map: `fqml/pq/__init__.py`.

## 1. Settings (`configs/pq/`)

| File | Session keys | Signatures | Use |
|---|---|---|---|
| `default.yaml` | ML-KEM-768 + X25519 (hybrid) | ML-DSA-65 | the main setting |
| `cnsa2.yaml` | ML-KEM-1024 | ML-DSA-87 (SHA-384) | higher-security option |
| `classical.yaml` | X25519 | Ed25519 | E7 baseline: today's cryptography |
| `none.yaml` | — | — | E7 baseline: no protection |

Server and sites must use the **same** file. If they differ, the handshake stops on both sides
("uses different PQ settings"): that is the downgrade protection, not a bug.

## 2. What happens on a connection

```
 site                                                    server
  | ClientHello: settings, name, nonce,                     |
  |   ML-KEM public key, X25519 public key  --------------> |  encapsulate to the ML-KEM key
  |                                                         |  sign the transcript (ML-DSA)
  | <------------ ServerHello: ML-KEM ciphertext, X25519,   |
  |               signature, key confirmation               |
  | check signature with the pinned server key             |
  | decapsulate, derive keys, check confirmation           |
  | ClientFinish: signature, key confirmation  -----------> |  check with the site's pinned key
  |                                                         |
  | <====== AES-256-GCM, one key per direction, numbered ===> |
  |   global model (signed by server, round r)  <---------  |
  |   update (signed by site, round r, n_rows)  --------->  |
```

Session keys = HKDF(ML-KEM secret ‖ X25519 secret, salt = hash of the handshake). Fresh ML-KEM and
X25519 keys are made for every connection, so a stolen identity key does not reveal old sessions.

## 3. Quick start (inside the Docker image, which has liboqs)

```bash
docker compose -f docker/compose.yml up -d dev
docker compose -f docker/compose.yml exec dev bash

# inside the container:
python -m pytest tests/test_pq_channel.py -q            # inside Docker: all pass, none skipped
python -m fqml.pq.keys gen server site-01 site-02 --config default   # writes keys/default/
python -m fqml.pq.keys show keys/default/server.pub      # prints the key ID
```

Demo in two terminals (second terminal: another `docker compose -f docker/compose.yml exec dev bash`):

```bash
python -m fqml.pq.link serve   --config default                      # terminal 1
python -m fqml.pq.link connect --config default --name site-01       # terminal 2
```

The server prints each site, its handshake time, and each verified update; the site prints each verified
global model and the bytes it used. Try `--config classical` or `none` (make keys for `classical` first;
`none` needs no keys), or connect as `site-03` (no key pinned: refused, with the reason on both sides).

**Several machines.** Make each party's keys on its own machine and copy only the `.pub` files: the
server's folder holds `server.key` and every `site-XX.pub`; each site's folder holds its own `site-XX.key`
and `server.pub`. Compare key IDs (`keys show`) by phone or in person the first time. Publish the port:

```bash
docker compose -f docker/compose.yml run --rm -p 9443:9443 dev \
    python -m fqml.pq.link serve --config default --host 0.0.0.0
# on a site:   python -m fqml.pq.link connect --config default --name site-01 --host <server IP>
```

## 4. Using it from code (E7)

```python
from fqml.pq.config import load_config
from fqml.pq.link import Listener, connect, load_server, load_site

cfg = load_config("default")

# server
me, trust = load_server(cfg, "keys/default")
with Listener(cfg, me, trust, host="0.0.0.0", port=9443) as lst:
    links = [lst.accept() for _ in range(n_sites)]          # one handshake per site
    for r in range(rounds):
        for link in links:
            link.send_model(global_params, round=r, purpose="global")
        updates = [link.recv_model(round=r, purpose="update") for link in links]
        # updates[k].weights: dict of numpy arrays, updates[k].meta["n_rows"]: FedAvg weight
        global_params = fedavg_of(updates)
    for link in links:
        link.close()

# site
me, server = load_site(cfg, "keys/default", "site-01")
with connect(cfg, me, server, "10.0.0.5", 9443) as link:
    for r in range(rounds):
        msg = link.recv_model(round=r, purpose="global")
        new_params = local_train(msg.weights)
        link.send_model(new_params, round=r, purpose="update", meta={"n_rows": n_rows})
    print(link.stats)    # handshake ms and bytes, bytes sent/received, crypto time
```

`weights` may be a dict of named tensors/arrays (like `FedResult.params`) or a list of arrays (Flower
NDArrays); numpy comes back, `fqml.pq.serialize.to_torch` converts. For Flower, the same `Link` can carry
the bytes of each Flower message; that adapter is an open E7 design question (D15).

Without sockets (tests, custom transports): `channel.ClientHandshake` / `ServerHandshake` give a `Session`
with `seal()` / `open()`; `channel.sign_model` / `open_model` handle model envelopes.

## 5. What it protects, and what it does not

Protected against anyone on the network, including a future quantum computer reading recorded traffic:
reading the models, changing them, replaying or reordering messages, pretending to be the server or a
site, and forcing a weaker mode. Every rejection raises a `SecurityError` and closes the connection.

**Not** protected (other parts of the project, or out of scope):
- a real site that sends a poisoned update with a valid signature: that is the job of the robust
  aggregators and attack experiments (Phase 1–3); the signatures do tell you *which* site sent it;
- message sizes and timing, which stay visible on the network;
- the machines themselves: `.key` files are plain files (mode 600). Whoever can read a site's key file can
  act as that site, so keep keys off shared folders, never commit them (`keys/` and `*.key` are
  git-ignored), and make new ones if a machine is compromised (`gen --force`, then re-pin the `.pub`).

## 6. Measuring and testing

- `python -m spikes.s6_pq_overhead` (about a minute): handshake time and bytes, bytes per model message,
  send/receive time for 64, 10k and 1M parameters, per setting → `results/spikes/S6_pq_overhead.json`.
- `tests/test_pq_channel.py`: every mode; tampering, replay, reordering, unknown/impostor keys,
  downgrade, man-in-the-middle edits, stale models; interoperability with OpenSSL's ML-KEM / ML-DSA.
  Without liboqs (outside Docker) the post-quantum tests are skipped; classical and none still run.

## 7. Troubleshooting

| Message | Meaning / fix |
|---|---|
| `liboqs-python is not installed` | run inside the Docker image, or use `classical` / `none` |
| `uses different PQ settings` | server and site loaded different `configs/pq` files |
| `unknown site 'site-03'` | copy `site-03.pub` into the server's key folder, restart the server |
| `the server's signature is invalid` | the site pinned an old or wrong `server.pub`: compare key IDs |
| `key is ML-DSA-65 but these settings sign with ML-DSA-87` | make keys for this settings file (its own folder) |
| `model is for round 3, expected 4` | a stale or replayed model; check both sides count rounds the same way |
| `connection closed by the other side` | the other side stopped; its terminal shows why |
