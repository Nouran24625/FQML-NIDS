"""SPIKE S5 - Does the D6/D7 circuit change cost accuracy? (owners: Sheghaf + Kareem)

Question : version 2 (RZ between the RYs, last-qubit readout; no dead parameters) and the chain
           entangler (fits IBM chips with 0 extra 2-qubit gates, S4) change the model. Do they lose
           accuracy compared with today's ring / version 1?
Setup    : centralised training (no federation) on the cleaned CICIoT2023 parquet, 8 features chosen
           by mutual information, quantile-scaled to [0, pi]; TorchVQC, Adam, batch 64; macro-F1 on a
           held-out set. Tasks: 8-class (family) and binary (benign vs attack). Seeds 0, 1, 2.
           Same features, data, steps and seeds for every variant; only the circuit changes.
Output   : table, results/spikes/S5_circuit_variants.json, MLflow run in 'phase0-spikes'.

Run: docker compose -f docker/compose.yml exec dev python -m spikes.s5_circuit_variants
     (needs data/interim/ciciot2023_pcapcsv_clean.parquet; ~10-15 min on 1 CPU thread)
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.preprocessing import QuantileTransformer
from torch import nn

from fqml.quantum.torch_sim import TorchVQC

VARIANTS = [("ring", 1), ("ring", 2), ("chain", 2)]


def load(path, n_train, n_test, seed):
    df = pd.read_parquet(path)
    print(f"loaded {len(df):,} rows; columns: {list(df.columns)[:6]} ... ({df.shape[1]} total)")
    if "family" not in df.columns:
        raise SystemExit("expected a 'family' column (8 attack families incl. benign)")
    fam = df["family"].astype(str)
    feats = [c for c in df.columns if c not in ("family", "y", "label", "Label")
             and pd.api.types.is_numeric_dtype(df[c])]
    df = df.sample(n=n_train + n_test, random_state=seed)
    X = df[feats].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    classes = sorted(fam.unique())
    y8 = df["family"].astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
    benign = [i for i, c in enumerate(classes) if "benign" in c.lower()]
    if not benign:
        raise SystemExit(f"no benign family found in {classes}")
    y2 = (y8 != benign[0]).astype(int)                                   # 1 = attack
    return X, y8, y2, feats, classes


def prepare(X, y, feats, n_train, k=8, seed=0):
    Xtr, Xte, ytr, yte = X[:n_train], X[n_train:], y[:n_train], y[n_train:]
    sub = np.random.default_rng(seed).choice(n_train, min(20000, n_train), replace=False)
    mi = mutual_info_classif(Xtr[sub], ytr[sub], random_state=seed)
    top = np.argsort(mi)[::-1][:k]
    q = QuantileTransformer(n_quantiles=1000, output_distribution="uniform", subsample=200000, random_state=seed)
    Xtr = q.fit_transform(Xtr[:, top]) * np.pi
    Xte = q.transform(Xte[:, top]) * np.pi
    return (torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr), torch.tensor(Xte, dtype=torch.float32),
            torch.tensor(yte), [feats[i] for i in top])


def train_eval(Xtr, ytr, Xte, yte, n_classes, entangler, version, seed, steps, batch, lr):
    torch.manual_seed(seed)
    m = TorchVQC(8, n_classes, 3, seed=seed, entangler=entangler, version=version)
    counts = torch.bincount(ytr, minlength=n_classes).float().clamp(min=1)
    w = (1 / counts.sqrt()) / (1 / counts.sqrt()).mean()
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    for _ in range(steps):
        idx = torch.randint(0, len(Xtr), (batch,), generator=g)
        loss = lossf(m(Xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    secs = time.perf_counter() - t0
    with torch.no_grad():
        pred = torch.cat([m(Xte[i:i + 4096]).argmax(1) for i in range(0, len(Xte), 4096)])
    return {"macro_f1": float(f1_score(yte.numpy(), pred.numpy(), average="macro")),
            "accuracy": float((pred == yte).float().mean()), "train_s": secs,
            "samples_per_s": steps * batch / secs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/interim/ciciot2023_pcapcsv_clean.parquet")
    ap.add_argument("--train", type=int, default=100_000)
    ap.add_argument("--test", type=int, default=20_000)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(1)                                             # decision D2
    X, y8, y2, feats, classes = load(a.data, a.train, a.test, seed=0)
    print(f"classes: {classes}\n")
    rows = []
    print(f"{'task':7s} {'entangler':9s} {'ver':>3s} {'macro-F1 mean':>14s} {'(per seed)':24s} {'samples/s':>10s}")
    for task, y, nc in (("8-class", y8, len(classes)), ("binary", y2, 2)):
        Xtr, ytr, Xte, yte, chosen = prepare(X, y, feats, a.train)
        print(f"[{task}] features: {chosen}")
        for ent, ver in VARIANTS:
            res = [train_eval(Xtr, ytr, Xte, yte, nc, ent, ver, s, a.steps, a.batch, a.lr) for s in range(a.seeds)]
            f1 = [r["macro_f1"] for r in res]
            sps = statistics.mean(r["samples_per_s"] for r in res)
            rows.append({"task": task, "entangler": ent, "version": ver, "features": chosen,
                         "macro_f1_mean": statistics.mean(f1), "macro_f1_std": statistics.pstdev(f1),
                         "macro_f1_per_seed": f1, "samples_per_s": sps})
            print(f"{task:7s} {ent:9s} {ver:3d} {statistics.mean(f1):14.4f} {str([round(v, 3) for v in f1]):24s} {sps:10,.0f}")
    out = Path("results/spikes/S5_circuit_variants.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"machine": platform.platform(), "torch": torch.__version__, "args": vars(a),
                               "classes": classes, "rows": rows}, indent=2))
    print(f"\nsaved {out}")
    try:
        import mlflow
        mlflow.set_experiment("phase0-spikes")
        with mlflow.start_run(run_name=f"S5-{platform.node()}"):
            mlflow.log_params(vars(a))
            for r in rows:
                mlflow.log_metric(f"f1_{r['task']}_{r['entangler']}_v{r['version']}", r["macro_f1_mean"])
            mlflow.log_artifact(str(out))
        print("logged to MLflow experiment 'phase0-spikes'")
    except Exception as e:
        print(f"(MLflow logging skipped: {e})")


if __name__ == "__main__":
    main()
