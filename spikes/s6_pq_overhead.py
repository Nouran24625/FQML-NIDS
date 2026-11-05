"""SPIKE S6 - What does the post-quantum channel cost? (owner: Sheghaf; evidence for proposed decision D15)

Question : E7 compares federated training over the PQ channel with the classical baseline and with no
           protection. Before E7: how much time and how many bytes do the handshake, each model message
           and the model signatures add, for each choice of settings?
Setup    : fqml.pq, 1 thread (D2). Settings: none, classical (X25519 + Ed25519), pq-512 (ML-KEM-512 +
           ML-DSA-44), pq-768 (ML-KEM-768 + ML-DSA-65, no X25519), default (pq-768 + X25519 hybrid),
           cnsa2 (ML-KEM-1024 + ML-DSA-87, SHA-384).
           (a) handshake in memory, both sides, median of --reps (CPU cost only, no network);
           (b) bytes of each handshake message;
           (c) one model message = encode -> sign -> encrypt, and decrypt -> verify -> decode, for the
               8-class VQC (64 float32 parameters), a 10k-parameter MLP and a 1M-parameter model;
           (d) handshake + one VQC round over loopback TCP (fqml.pq.link), median of --tcp-reps.
Output   : table, results/spikes/S6_pq_overhead.json, MLflow run in 'phase0-spikes'.

Run: docker compose -f docker/compose.yml exec dev python -m spikes.s6_pq_overhead        (~1 min)
"""
import argparse
import json
import os
import platform
import statistics
import threading
import time
from pathlib import Path

import numpy as np

from fqml.pq import primitives as P
from fqml.pq.channel import ClientHandshake, ServerHandshake, open_model, sign_model
from fqml.pq.config import PQConfig, load_config
from fqml.pq.keys import TrustStore, generate_identity
from fqml.pq.link import FRAME_HEADER, Listener, connect
from fqml.pq.serialize import dumps_weights, loads_weights

SETTINGS = {
    "none": load_config("none"),
    "classical": load_config("classical"),
    "pq-512": PQConfig(kem="ML-KEM-512", signature="ML-DSA-44", hybrid_x25519=False),
    "pq-768": PQConfig(kem="ML-KEM-768", signature="ML-DSA-65", hybrid_x25519=False),
    "default": load_config("default"),
    "cnsa2": load_config("cnsa2"),
}
MODELS = {"vqc-64": 64, "mlp-10k": 10_000, "model-1M": 1_000_000}


def parties(cfg):
    if not cfg.encrypted:
        return "server", "site-01", "server", None
    server, site = generate_identity("server", cfg), generate_identity("site-01", cfg)
    return server, site, server.public_only(), TrustStore([site])


def median_ms(fn, reps):
    fn()                                                            # warm-up (first calls load liboqs)
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1e3)
    return statistics.median(times)


def handshake_once(cfg, server, site, server_pub, trust):
    c, s = ClientHandshake(cfg, site, server_pub), ServerHandshake(cfg, server, trust)
    ch = c.hello()
    sh = s.respond(ch)
    cf, site_session = c.finish(sh)
    return ch, sh, cf, site_session, s.complete(cf)


def tcp_round(cfg, server, site, server_pub, trust, weights):
    """Wall time of connect (handshake) and of one global-model + update round, measured at the site."""
    out = {}
    with Listener(cfg, server, trust, port=0, timeout=30) as lst:
        def serve():
            with lst.accept() as link:
                link.send_model(weights, round=0, purpose="global")
                link.recv_model(round=0, purpose="update")
        t = threading.Thread(target=serve, daemon=True)
        t.start()
        t0 = time.perf_counter()
        link = connect(cfg, site, server_pub, port=lst.address[1], timeout=30)
        out["connect_ms"] = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        msg = link.recv_model(round=0, purpose="global")
        link.send_model(msg.weights, round=0, purpose="update")
        out["round_ms"] = (time.perf_counter() - t1) * 1e3
        out["handshake_wire_bytes"] = link.stats.handshake_bytes_sent + link.stats.handshake_bytes_received
        link.close()
        t.join(timeout=30)
    return out


def measure(name, cfg, reps, tcp_reps):
    server, site, server_pub, trust = parties(cfg)
    ch, sh, cf, site_s, server_s = handshake_once(cfg, server, site, server_pub, trust)
    row = {"settings": name, "describe": cfg.describe(),
           "handshake_ms": median_ms(lambda: handshake_once(cfg, server, site, server_pub, trust), reps),
           "client_hello_bytes": len(ch), "server_hello_bytes": len(sh), "client_finish_bytes": len(cf),
           "handshake_bytes": len(ch) + len(sh) + len(cf),
           "message_overhead_bytes": FRAME_HEADER + site_s.overhead}
    if cfg.sig_alg:
        sizes = P.sig_sizes(cfg.sig_alg)
        row.update(identity_public_key_bytes=sizes["public_key"], signature_bytes=sizes["signature"])
    rng = np.random.default_rng(0)
    for model, n in MODELS.items():
        weights = {"w": rng.normal(size=n).astype(np.float32)}
        r = max(5, reps // 20) if n >= 1_000_000 else reps
        _, _, _, site_s, server_s = handshake_once(cfg, server, site, server_pub, trust)   # fresh counters
        send_ms, receive_ms, frames = [], [], []
        for _ in range(r + 1):                                      # server: encode -> sign -> encrypt
            t = time.perf_counter()
            envelope = sign_model(cfg, server, dumps_weights(weights), round=1, purpose="global")
            frames.append(server_s.seal(envelope))
            send_ms.append((time.perf_counter() - t) * 1e3)
        for frame in frames:                                        # site: decrypt -> verify -> decode
            t = time.perf_counter()
            loads_weights(open_model(cfg, site_s.open(frame), server_pub, round=1, purpose="global").weights)
            receive_ms.append((time.perf_counter() - t) * 1e3)
        wire = FRAME_HEADER + len(frames[-1])
        row[model] = {"parameters": n, "raw_bytes": 4 * n, "envelope_bytes": len(envelope), "wire_bytes": wire,
                      "overhead_bytes_vs_raw": wire - 4 * n,
                      "send_ms": statistics.median(send_ms[1:]),      # first call = warm-up
                      "receive_ms": statistics.median(receive_ms[1:])}
    tcp = [tcp_round(cfg, server, site, server_pub, trust, {"w": rng.normal(size=64).astype(np.float32)})
           for _ in range(tcp_reps)]
    row["tcp"] = {"connect_ms": statistics.median(t["connect_ms"] for t in tcp),
                  "round_ms": statistics.median(t["round_ms"] for t in tcp),
                  "handshake_wire_bytes": tcp[0]["handshake_wire_bytes"]}       # bytes never vary
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--tcp-reps", type=int, default=20)
    ap.add_argument("--out", default="results/spikes/S6_pq_overhead.json")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")                     # decision D2
    rows = []
    print(f"{'settings':10s} {'handshake':>9s} {'hs bytes':>8s} {'TCP connect':>11s} "
          f"{'VQC msg':>8s} {'VQC send/recv':>14s} {'1M send/recv':>14s}   (times: median ms, 1 thread)")
    for name, cfg in SETTINGS.items():
        try:
            row = measure(name, cfg, a.reps, a.tcp_reps)
        except RuntimeError as e:                                       # liboqs missing: skip pq rows
            print(f"{name:10s} skipped: {e}")
            continue
        rows.append(row)
        v, m = row["vqc-64"], row["model-1M"]
        print(f"{name:10s} {row['handshake_ms']:7.2f}ms {row['handshake_bytes']:8,d} "
              f"{row['tcp']['connect_ms']:9.2f}ms {v['wire_bytes']:7,d}B "
              f"{v['send_ms']:6.3f}/{v['receive_ms']:.3f}ms {m['send_ms']:6.2f}/{m['receive_ms']:.2f}ms")
    try:
        import oqs
        liboqs = oqs.oqs_version()
    except Exception:
        liboqs = None
    import cryptography
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"machine": platform.platform(), "python": platform.python_version(),
                               "liboqs": liboqs, "cryptography": cryptography.__version__,
                               "args": vars(a), "rows": rows}, indent=2))
    print(f"\nsaved {out}")
    try:
        import mlflow
        mlflow.set_experiment("phase0-spikes")
        with mlflow.start_run(run_name=f"S6-{platform.node()}"):
            mlflow.log_params(vars(a))
            for r in rows:
                mlflow.log_metric(f"handshake_ms_{r['settings']}", r["handshake_ms"])
                mlflow.log_metric(f"handshake_bytes_{r['settings']}", r["handshake_bytes"])
                mlflow.log_metric(f"vqc_wire_bytes_{r['settings']}", r["vqc-64"]["wire_bytes"])
            mlflow.log_artifact(str(out))
        print("logged to MLflow experiment 'phase0-spikes'")
    except Exception as e:
        print(f"(MLflow logging skipped: {e})")


if __name__ == "__main__":
    main()
