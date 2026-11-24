"""Live federation demo: one server and N sites as separate programs, real training, real network. Owner: Kareem.

What the audience sees: a server window and one window per site. Every site holds only its own slice of real
CICIoT2023 traffic (the FED-3 splits: a site sees only some attack types), trains the shared model on it, and sends
its update to the server over the team's post-quantum channel (`fqml.pq.link`, D15: ML-KEM-768 + X25519 key
exchange, ML-DSA-65 signatures, AES-256-GCM). The server checks every signature, combines the updates (FedAvg or a
robust aggregator, `fqml.fl.robust`) and prints, each round, how well the global model detects each attack family,
how far each site's update was from the global model, and the bytes on the wire.

It reuses the experiment code, so what runs here is what the experiments measure:
  models       experiments.fed4_trial.make_model, the VQC variants and the MLP size ladder
               (vqc, vqc_lin, vqc_noent, mlp, mlp_p<params> e.g. mlp_p3000)
  splits       fqml.fl.partition                                              (iid, dir0.5, sites, sitesfam)
  local step   fqml.fl.batched._local_sequential: the reference that batched FedAvg is tested against
  aggregators  fqml.fl.robust.get_aggregator                                  (fedavg, median, trimmed_mean)
  attacks      label flipping / class suppression, chosen ON THE SITE (the server is not told), docs/ATTACKS.md
New models, aggregators, attacks or splits added to those modules appear here through the same options.

Demo shortcuts (say them when showing it):
  - every site reads the same training sample and keeps only its own rows; in a real deployment each site has
    only its own data
  - sites tell the server which attack types they hold, so the server can score "attack types this site never
    saw"; a real site would not reveal that (docs/THREAT_MODEL.md section 4)
  - features: the stand-in top-8 used by FED-3/4/5/6 until DATA-6; scores are on the validation block

Run (inside the project image; demo/federated_live.ps1 opens all the windows):
  python -m demo.federated_live keys   --sites 3
  python -m demo.federated_live server --sites 3 --rounds 10 --split sites --aggregator fedavg
  python -m demo.federated_live site   --name site-01
  python -m demo.federated_live site   --name site-02 --attack class_suppression --target Web
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 3600.0                     # a site may train for minutes between messages (VQC)
FAMILY_SHORT = {"Benign": "Ben", "DDoS": "DDoS", "DoS": "DoS", "Recon": "Rec", "Web": "Web",
                "BruteForce": "Brute", "Spoofing": "Spoof", "Mirai": "Mirai"}


# ---- shared: data, model, split (deterministic, so the server and every site build the same thing)
def load_sample(fold: int, cap: int, val_cap: int):
    from experiments.fed3_partitions import GAP, REPORT, label_ranges_from_report
    from experiments.fed6_zero_attacker import fit_inputs, load_block
    from fqml.data.folds import build_time_blocked, load_manifest, manifest_path
    report = json.loads(REPORT.read_text())
    feats = [c for c in report["features"] if c != "IAT"]                     # D9
    ranges = label_ranges_from_report(report)
    path = manifest_path("ciciot2023", "A")
    m = load_manifest(path) if path.exists() else build_time_blocked(ranges, dataset="ciciot2023", n_folds=5, gap=GAP)
    Ftr, native_tr, ytr = load_block(m, fold, "train", ranges, cap, feats)
    Fva, native_va, yva = load_block(m, fold, "val", ranges, val_cap, feats)
    top, transform = fit_inputs(Ftr, ytr)                                      # stand-in for DATA-6
    return {"families": report["families"], "features": [feats[i] for i in top],
            "Xtr": transform(Ftr), "ytr": ytr, "native_tr": native_tr,
            "Xva": transform(Fva), "yva": yva, "native_va": native_va}


def make_model(kind: str, n_classes: int):
    from experiments.fed4_trial import make_model as fed4_model
    from fqml.quantum.torch_sim import TorchVQC
    if kind == "vqc_lin":
        return TorchVQC(8, n_classes, 3, seed=0, readout="linear")
    if kind == "vqc_noent":
        return TorchVQC(8, n_classes, 3, seed=0, entangler="none", readout="linear")
    if kind.startswith("mlp_p"):                                               # size ladder, e.g. mlp_p512, mlp_p3000
        from fqml.models.mlp import MatchedMLP
        return MatchedMLP(8, n_classes, target_params=int(kind[5:]), seed=0)
    return fed4_model(kind, n_classes)                                         # "vqc", "mlp" (VQC-sized)


def make_split(name: str, native, y, n_sites: int, seed: int):
    from fqml.data.label_maps import load_label_map
    from fqml.fl.partition import dirichlet_partition, iid_partition, sites_partition
    n_types = len(set(native.tolist()) - {"BenignTraffic"})
    if name == "iid":
        return iid_partition(len(y), n_sites, seed)
    if name.startswith("dir"):
        return dirichlet_partition(y, n_sites, float(name[3:]), seed)
    if name == "sites":                       # each site holds benign + some attack types, every type somewhere
        return sites_partition(native, n_sites, max(8, math.ceil(n_types / n_sites)), seed)[0]
    if name == "sitesfam":                    # each site holds whole families
        _, fam_of = load_label_map("ciciot2023")
        groups = {t: fam_of[t] for t in fam_of if t != "BenignTraffic"}
        return sites_partition(native, n_sites, max(3, math.ceil(len(set(groups.values())) / n_sites)), seed,
                               groups=groups)[0]
    raise SystemExit(f"unknown split {name!r}: iid, dir0.5, sites, sitesfam")


def label_attack(y, attack: str, families: list[str], target: str):
    """Label attacks of docs/ATTACKS.md applied by a malicious site to its own rows."""
    if attack == "none":
        return y, 0
    if attack == "label_flip":
        try:
            from fqml.attacks.poison import flip_map                           # FED-2 (#24), once merged
            fmap = flip_map(len(families))
        except ImportError:
            fmap = (np.arange(len(families)) + 1) % len(families)
        return fmap[y], int(len(y))
    if attack == "class_suppression":
        t = families.index(target)
        out = y.copy()
        out[y == t] = 0                                                        # target family -> Benign
        return out, int((y == t).sum())
    raise SystemExit(f"unknown attack {attack!r}: none, label_flip, class_suppression")


def to_np(params):
    return {n: v.detach().cpu().numpy().astype(np.float32) for n, v in params.items()}


def to_torch(weights):
    return {n: torch.as_tensor(np.asarray(v), dtype=torch.float32) for n, v in weights.items()}


def pq_setup(config: str, keys: str | None):
    from fqml.pq.config import load_config
    from fqml.pq.keys import default_key_dir
    cfg = load_config(config)
    return cfg, Path(keys) if keys else default_key_dir(config)


# ---- browser dashboard (standard library only; the page polls /api/state every second)
class Live:
    """What the dashboard shows: settings, each site's state, every round's scores and an event log."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = {"phase": "starting", "round": 0, "rounds": 0, "settings": {}, "families": [], "sites": {},
                      "history": [], "events": [], "started": time.time()}

    def update(self, **kw):
        with self.lock:
            self.state.update(kw)

    def site(self, name, **kw):
        with self.lock:
            s = self.state["sites"].setdefault(name, {"name": name, "bytes_in": 0, "bytes_out": 0})
            for k, v in kw.items():
                s[k] = s.get(k, 0) + v if k in ("bytes_in", "bytes_out") else v

    def event(self, text):
        with self.lock:
            self.state["events"] = (self.state["events"] + [{"t": time.time(), "text": text}])[-60:]

    def snapshot(self) -> bytes:
        with self.lock:
            return json.dumps({**self.state, "now": time.time()}, default=float).encode()


def start_gui(port: int) -> Live:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    live = Live()
    page = Path(__file__).parent / "static" / "federated_live.html"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path.startswith("/api/state"):
                body, ctype = live.snapshot(), "application/json"
            elif self.path in ("/", "/index.html"):
                body, ctype = page.read_bytes(), "text/html; charset=utf-8"
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[server] dashboard: open http://localhost:{port} in a browser", flush=True)
    return live


# ---- keys
def cmd_keys(a):
    from fqml.pq.keys import main as keys_main
    names = ["server"] + [f"site-{k + 1:02d}" for k in range(a.sites)]
    keys_main(["gen", *names, "--config", a.config, "--force"])
    print(f"keys for {', '.join(names)} written (config {a.config}); secret keys stay out of git (.gitignore)")


# ---- server
def cmd_server(a):
    from sklearn.metrics import f1_score, recall_score

    from fqml.fl.batched import predict_proba
    from fqml.fl.equivalence import sqrt_class_weights
    from fqml.fl.robust import get_aggregator
    from fqml.fl.unseen import per_client_unseen, summary
    from fqml.pq.link import Listener, load_server

    torch.set_num_threads(2)
    live = start_gui(a.gui_port) if a.gui_port else Live()
    cfg, key_dir = pq_setup(a.config, a.keys)
    me, trust = load_server(cfg, key_dir)
    aggregate = get_aggregator(a.aggregator)
    print(f"[server] channel: {cfg.describe()}", flush=True)
    live.update(phase="loading data", rounds=a.rounds, channel=cfg.describe(),
                settings={k: v for k, v in vars(a).items() if k != "keys"})
    live.event(f"Server started. Channel: {cfg.describe()}")
    print("[server] loading the validation block for the scores (demo shortcut: the stand-in features are fitted on "
          "the shared training sample) ...", flush=True)
    data = load_sample(a.fold, a.cap, a.val_cap)
    fams = data["families"]
    C = len(fams)
    Xva = torch.as_tensor(data["Xva"], dtype=torch.float32)
    yva = data["yva"]
    model = make_model(a.model, C)
    gparams = {n: p.detach().clone() for n, p in model.named_parameters()}
    n_par = sum(v.numel() for v in gparams.values())
    print(f"[server] model {a.model} ({n_par} parameters), split {a.split}, aggregator {a.aggregator}, "
          f"{a.rounds} rounds x {a.local_steps} local steps", flush=True)
    live.update(phase=f"waiting for {a.sites} sites", families=fams, n_params=n_par)

    links = []
    with Listener(cfg, me, trust, a.host, a.port, timeout=TIMEOUT) as lst:
        print(f"[server] listening on {a.host}:{a.port}, waiting for {a.sites} sites ...", flush=True)
        while len(links) < a.sites:
            link = lst.accept()
            links.append(link)
            print(f"[server]   {link.peer} connected: identity verified, handshake "
                  f"{link.stats.handshake_seconds * 1e3:.0f} ms ({len(links)}/{a.sites})", flush=True)
            live.site(link.peer, status="connected, loading its data",
                      handshake_ms=link.stats.handshake_seconds * 1e3, active=time.time(), dir="up")
            live.event(f"{link.peer} connected: identity verified (ML-DSA), handshake "
                       f"{link.stats.handshake_seconds * 1e3:.0f} ms")
    links.sort(key=lambda l: l.peer)
    pool = ThreadPoolExecutor(len(links))

    # setup: each site gets its place in the split; class counts are summed into one class weighting (D11)
    for k, link in enumerate(links):
        link.send(json.dumps({"index": k, "n_sites": len(links), "model": a.model, "split": a.split,
                              "fold": a.fold, "cap": a.cap, "val_cap": a.val_cap, "seed": a.seed,
                              "local_steps": a.local_steps, "batch_size": a.batch_size, "lr": a.lr}).encode())
    hellos = list(pool.map(lambda l: json.loads(l.recv()), links))
    counts = np.sum([h["counts"] for h in hellos], axis=0)
    w = sqrt_class_weights(np.repeat(np.arange(C), counts.astype(int)), C).float()
    for link in links:
        link.send(json.dumps({"class_weights": w.tolist()}).encode())
    held = [set(h["held_types"]) for h in hellos]
    for link, h in zip(links, hellos):
        live.site(link.peer, rows=h["rows"], n_types=len(h["held_types"]), status="ready",
                  families_held=[f for f, c in zip(fams, h["counts"]) if c and f != "Benign"])
    live.update(phase="training")
    live.event("All sites ready; class weights agreed from the sites' summed class counts")
    print("\n[server] sites and what they hold:", flush=True)
    for link, h in zip(links, hellos):
        print(f"           {link.peer}: {h['rows']:,} rows, {len(h['held_types'])} attack types, families "
              f"{', '.join(f for f, c in zip(fams, h['counts']) if c and f != 'Benign')}", flush=True)

    history = []
    header = f"{'round':>5} {'macro-F1':>8} {'att@1%':>6} | " + " ".join(f"{FAMILY_SHORT[f]:>5}" for f in fams)
    for r in range(a.rounds):
        t0 = time.perf_counter()
        live.update(round=r + 1)

        def send_global(link):
            n = link.send_model(to_np(gparams), round=r, purpose="global")
            live.site(link.peer, bytes_out=n, status="training on its own rows", active=time.time(), dir="down")
            return n

        def recv_update(link):
            m = link.recv_model(round=r, purpose="update")
            live.site(link.peer, bytes_in=m.encoded_bytes, status="update received, signature verified",
                      verified_round=r + 1, active=time.time(), dir="up")
            return m

        sent = list(pool.map(send_global, links))
        msgs = list(pool.map(recv_update, links))
        stacked = {n: torch.stack([to_torch(m.weights)[n] for m in msgs]) for n in gparams}
        n_rows = torch.tensor([float(m.meta["n_rows"]) for m in msgs])
        flat_g = torch.cat([v.flatten() for v in gparams.values()])
        dist = [float((torch.cat([to_torch(m.weights)[n].flatten() for n in gparams]) - flat_g).norm()) for m in msgs]
        gparams = aggregate(stacked, n_rows)
        proba = predict_proba(model, gparams, Xva)
        pred = proba.argmax(1)
        att = 1 - proba[:, 0]
        thr = float(np.quantile(att[yva == 0], 0.99))
        rec = recall_score(yva, pred, labels=range(C), average=None, zero_division=0)
        f1 = float(f1_score(yva, pred, average="macro"))
        unseen = summary(per_client_unseen(yva, data["native_va"], held, [pred] * len(links)), "family_recall_types")
        entry = {"round": r + 1, "macro_f1": f1, "attack_recall_at_1pct_fpr": float((att[yva != 0] > thr).mean()),
                 "recall": dict(zip(fams, map(float, rec))), "unseen_family_recall": unseen,
                 "update_distance": dict(zip([l.peer for l in links], dist)),
                 "bytes_global": int(sum(sent)), "bytes_updates": int(sum(m.encoded_bytes for m in msgs)),
                 "seconds": time.perf_counter() - t0}
        history.append(entry)
        for link, d in zip(links, dist):
            live.site(link.peer, distance=d)
        live.update(history=history)
        if r % 10 == 0:
            print(f"\n{header}", flush=True)
        print(f"{r + 1:>5} {f1:>8.3f} {entry['attack_recall_at_1pct_fpr']:>6.2f} | " +
              " ".join(f"{x:>5.2f}" for x in rec) +
              f"   unseen-type family {unseen['mean']:.2f} | update distance " +
              " ".join(f"{l.peer[-2:]}:{d:.2f}" for l, d in zip(links, dist)) +
              f" | {entry['bytes_global'] + entry['bytes_updates']:,} B, {entry['seconds']:.1f}s", flush=True)
        live.event(f"Round {r + 1}: {len(msgs)} signed updates verified and combined ({a.aggregator}); "
                   f"macro-F1 {f1:.3f}")
        if a.round_delay:
            time.sleep(a.round_delay)

    for link in links:
        s = link.stats
        print(f"[server] {link.peer}: sent {s.bytes_sent:,} B, received {s.bytes_received:,} B, "
              f"crypto {s.crypto_seconds * 1e3:.0f} ms", flush=True)
        link.close()
    out = ROOT / f"results/demo/federated_live_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"settings": vars(a), "channel": cfg.describe(), "families": fams,
                               "features": data["features"], "sites": [l.peer for l in links],
                               "history": history}, indent=1, default=str))
    print(f"[server] done; history written to {out.relative_to(ROOT)}", flush=True)
    live.update(phase="finished")
    for link in links:
        live.site(link.peer, status="session closed")
    live.event(f"Finished after {a.rounds} rounds; history saved to {out.relative_to(ROOT)}")
    if a.gui_port and a.keep_open:
        print("[server] dashboard stays open; Ctrl+C to stop", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


# ---- site
def cmd_site(a):
    from fqml.fl.batched import _local_sequential, client_batches
    from fqml.fl.unseen import held_types
    from fqml.pq.link import connect, load_site
    from fqml.pq.primitives import SecurityError

    torch.set_num_threads(1)
    cfg, key_dir = pq_setup(a.config, a.keys)
    me, server = load_site(cfg, key_dir, a.name)
    print(f"[{a.name}] connecting to the server at {a.host}:{a.port} ...", flush=True)
    link = None
    for _ in range(60):                                   # the server may still be starting
        try:
            link = connect(cfg, me, server, a.host, a.port, timeout=TIMEOUT)
            break
        except ConnectionRefusedError:
            time.sleep(2)
    if link is None:
        raise SystemExit("server not reachable")
    print(f"[{a.name}] connected: server identity verified, handshake {link.stats.handshake_seconds * 1e3:.0f} ms",
          flush=True)
    with link:
        conf = json.loads(link.recv())
        print(f"[{a.name}] I am site {conf['index'] + 1} of {conf['n_sites']}; loading my data ...", flush=True)
        data = load_sample(conf["fold"], conf["cap"], conf["val_cap"])
        fams = data["families"]
        parts = make_split(conf["split"], data["native_tr"], data["ytr"], conf["n_sites"], conf["seed"])
        rows = np.asarray(parts[conf["index"]])
        X = torch.as_tensor(data["Xtr"], dtype=torch.float32)
        y_clean = data["ytr"]
        y, changed = label_attack(y_clean.copy(), a.attack, fams, a.target)
        held = sorted(held_types(data["native_tr"], [rows])[0])
        counts = np.bincount(y_clean[rows], minlength=len(fams))
        print(f"[{a.name}] my data: {len(rows):,} rows, {len(held)} attack types; families "
              + ", ".join(f"{f} {c:,}" for f, c in zip(fams, counts) if c), flush=True)
        if a.attack != "none":
            nmine = int(((y != y_clean)[rows]).sum())
            print(f"[{a.name}] *** MALICIOUS: {a.attack}"
                  + (f" (target {a.target} -> Benign)" if a.attack == "class_suppression" else "")
                  + f", {nmine:,} of my labels changed"
                  + (f" from round {a.attack_from}" if a.attack_from > 1 else "")
                  + "; the server is not told ***", flush=True)
        link.send(json.dumps({"rows": int(len(rows)), "counts": counts.tolist(), "held_types": held}).encode())
        w = torch.tensor(json.loads(link.recv())["class_weights"], dtype=torch.float32)
        model = make_model(conf["model"], len(fams))
        yt_attack = torch.as_tensor(y, dtype=torch.long)
        yt_clean = torch.as_tensor(y_clean, dtype=torch.long)
        rng = np.random.default_rng(conf["seed"] * 1000 + conf["index"])
        r = 0
        while True:
            try:
                msg = link.recv_model(purpose="global")
            except (ConnectionError, OSError, SecurityError):
                break
            t0 = time.perf_counter()
            gp = to_torch(msg.weights)
            attacking = a.attack != "none" and msg.round + 1 >= a.attack_from
            yt = yt_attack if attacking else yt_clean
            if a.attack != "none" and msg.round + 1 == a.attack_from:
                print(f"[{a.name}] *** round {a.attack_from}: switching to poisoned labels now ***", flush=True)
            batches = torch.as_tensor(client_batches(rows, conf["local_steps"], conf["batch_size"], rng))[None]
            upd = {n: v[0] for n, v in _local_sequential(model, gp, X, yt, batches, conf["lr"], w).items()}
            with torch.no_grad():
                idx = torch.as_tensor(rows[:4096])
                from torch.func import functional_call
                loss = float(torch.nn.functional.cross_entropy(functional_call(model, upd, (X[idx],)), yt[idx],
                                                               weight=w))
            nbytes = link.send_model(to_np(upd), round=msg.round, purpose="update", meta={"n_rows": int(len(rows))})
            r = msg.round + 1
            print(f"[{a.name}] round {r:>3}: global model received (signature by {msg.signer} verified), trained "
                  f"{conf['local_steps']} steps on my rows, loss {loss:.3f}, update sent ({nbytes:,} B, signed and "
                  f"encrypted) {time.perf_counter() - t0:.1f}s", flush=True)
        print(f"[{a.name}] server closed the session after {r} rounds", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m demo.federated_live", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("keys", "server", "site"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="default", help="post-quantum settings (configs/pq): default, classical, none")
        p.add_argument("--keys", default=None, help="key folder (default keys/<config>)")
        if name in ("keys", "server"):
            p.add_argument("--sites", type=int, default=3)
        if name in ("server", "site"):
            p.add_argument("--port", type=int, default=9443)
    s = sub.choices["server"]
    s.add_argument("--host", default="0.0.0.0", help="0.0.0.0 accepts sites from other machines")
    s.add_argument("--rounds", type=int, default=10)
    s.add_argument("--model", default="mlp", help="mlp (VQC-sized), mlp_p512, mlp_p3000, vqc, vqc_lin, vqc_noent")
    s.add_argument("--split", default="sites", help="iid, dir0.5, sites, sitesfam")
    s.add_argument("--aggregator", default="fedavg", help="fedavg, median, trimmed_mean, trimmed_mean0.1")
    s.add_argument("--local-steps", type=int, default=50)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--lr", type=float, default=0.01)
    s.add_argument("--fold", type=int, default=0)
    s.add_argument("--cap", type=int, default=5000)
    s.add_argument("--val-cap", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--gui-port", type=int, default=8050, help="browser dashboard port (0: no dashboard)")
    s.add_argument("--round-delay", type=float, default=0.0, help="pause between rounds, so people can follow")
    s.add_argument("--keep-open", action="store_true", help="keep the dashboard up after the last round")
    t = sub.choices["site"]
    t.add_argument("--name", required=True, help="site-01, site-02, ...")
    t.add_argument("--host", default="127.0.0.1", help="the server's address")
    t.add_argument("--attack", default="none", help="none, label_flip, class_suppression")
    t.add_argument("--target", default="Web", help="family hidden by class_suppression")
    t.add_argument("--attack-from", type=int, default=1, help="behave honestly until this round, then attack")
    a = ap.parse_args(argv)
    {"keys": cmd_keys, "server": cmd_server, "site": cmd_site}[a.cmd](a)


if __name__ == "__main__":
    main()
