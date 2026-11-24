"""FED-5: histogram-based federated XGBoost (XGBoost's own federated mode), one process per site. Owner: Kareem.

Each site keeps its rows; per tree level, sites send gradient histograms to a federated server, which sums them
and sends the totals back, so every site grows the same trees. The script checks that this runs in the project
image and how close it comes to pooled XGBoost (central) on the same sample and the same splits as
experiments/fed5_trees.py. Same model settings as fed5_trees: 200 trees, depth 6, learning rate 0.1, hist.

Writes results/fed5/xgb_federated_<protocol>_f<fold>.json.

    python -m experiments.fed5_xgb_federated --fold 0
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"objective": "multi:softprob", "max_depth": 6, "eta": 0.1, "tree_method": "hist", "nthread": 1}
ROUNDS = 200


def _worker(rank, world, port, X, y, n_classes, Xva, q):
    import xgboost as xgb
    with xgb.collective.CommunicatorContext(dmlc_communicator="federated",
                                            federated_server_address=f"localhost:{port}",
                                            federated_world_size=world, federated_rank=rank):
        bst = xgb.train({**PARAMS, "num_class": n_classes}, xgb.DMatrix(X, label=y), ROUNDS)
        if rank == 0:
            q.put(bst.predict(xgb.DMatrix(Xva)))


def federated_xgb(parts, X, y, n_classes, Xva, port):
    from xgboost import federated
    ctx = mp.get_context("spawn")
    server = ctx.Process(target=federated.run_federated_server, kwargs={"n_workers": len(parts), "port": port})
    server.start()
    time.sleep(2)
    q = ctx.Queue()
    workers = [ctx.Process(target=_worker, args=(k, len(parts), port, X[p], y[p], n_classes, Xva, q))
               for k, p in enumerate(parts)]
    for w in workers:
        w.start()
    proba = q.get(timeout=3600)
    for w in workers:
        w.join()
    server.terminate()
    return proba


def main():
    from experiments.fed3_partitions import GAP, REPORT, label_ranges_from_report
    from experiments.fed5_trees import score
    from experiments.fed6_zero_attacker import fit_inputs, load_block, splits
    from fqml.data.folds import build_time_blocked, load_manifest, manifest_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--protocol", default="A")
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--val-cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="sites,sitesfam")
    a = ap.parse_args()
    t0 = time.perf_counter()
    report = json.loads(REPORT.read_text())
    families = report["families"]
    feats = [c for c in report["features"] if c != "IAT"]
    ranges = label_ranges_from_report(report)
    path = manifest_path("ciciot2023", a.protocol)
    m = load_manifest(path) if path.exists() else build_time_blocked(ranges, dataset="ciciot2023", n_folds=5, gap=GAP)
    print(f"[data] reading protocol {a.protocol} fold {a.fold} ...", flush=True)
    Ftr, native, ytr = load_block(m, a.fold, "train", ranges, a.cap, feats)
    Fva, _, yva = load_block(m, a.fold, "val", ranges, a.val_cap, feats)
    _, transform = fit_inputs(Ftr, ytr)
    Xtr, Xva = transform(Ftr).astype(np.float32), transform(Fva).astype(np.float32)
    print(f"[data] train {len(ytr):,}, validation {len(yva):,} ({time.perf_counter() - t0:.0f}s)", flush=True)

    out = {"task": "FED-5 histogram-federated XGBoost", "protocol": a.protocol, "fold": a.fold, "params": PARAMS,
           "rounds": ROUNDS, "families": families, "splits": {}}
    parts_by = splits(native, ytr, a.seed)
    for i, split in enumerate(a.splits.split(",")):
        parts = parts_by[split]
        print(f"\n[{split}] {len(parts)} sites, one process each; server on port {9091 + i} ...", flush=True)
        t1 = time.perf_counter()
        s = score(yva, federated_xgb(parts, Xtr, ytr, len(families), Xva, 9091 + i), families)
        s["seconds"] = time.perf_counter() - t1
        out["splits"][split] = s
        print(f"  fed_xgb_hist          macro-F1 {s['macro_f1']:.3f} | " +
              " ".join(f"{f[:5]} {r:.2f}" for f, r in s["recall"].items()) + f"  ({s['seconds']:.0f}s)", flush=True)

    dest = ROOT / f"results/fed5/xgb_federated_{a.protocol}_f{a.fold}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest.relative_to(ROOT)}  (total {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
