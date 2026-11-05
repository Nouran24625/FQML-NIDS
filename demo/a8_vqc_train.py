"""Video shot A8: the 8-qubit VQC training on a GPU, next to its parameter-matched classical twin.
Run in Google Colab (GPU runtime) with demo/A8_vqc_gpu_colab.ipynb, or anywhere:

    python -m demo.a8_vqc_train --data /content/drive/MyDrive/fqml/ciciot2023_pcapcsv_clean.parquet

What appears on screen, in order: device, data and split, the 8 input features, parameter counts,
loss going down while training, then macro-F1 and per-family F1 for both models on held-out rows.
Honesty rules (from the team brief): the numbers are printed exactly as measured; the split type is
printed (random row split until the Phase 1 capture-disjoint split is merged); no targets, no claims of
quantum advantage. Results are also saved to results/demo/a8_vqc.json so the video can be checked.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from torch import nn

from fqml.models.mlp import MatchedMLP
from fqml.quantum.torch_sim import TorchVQC

FAMILIES = ["Benign", "DDoS", "DoS", "Recon", "Web", "BruteForce", "Spoofing", "Mirai"]


def load(path, n_train, n_test, seed):
    df = pd.read_parquet(path)
    if "y" not in df.columns:
        df["y"] = df["family"].astype(str).map({f: i for i, f in enumerate(FAMILIES)})
    feats = [c for c in df.columns if c not in ("family", "y", "native_label", "label", "capture", "sample_rate")
             and pd.api.types.is_numeric_dtype(df[c])]
    total = len(df)
    df = df.sample(n=min(n_train + n_test, total), random_state=seed).reset_index(drop=True)
    X = np.nan_to_num(df[feats].to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = df["y"].to_numpy(np.int64)
    n_tr = len(df) - n_test
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], feats, total


def preprocess(Xtr, ytr, Xte, feats, k, seed):
    """INTERFACES.md 1.3: fitted on train only. Top-k mutual information, 1st-99th pct clip, min-max to [0, pi]."""
    sub = np.random.default_rng(seed).choice(len(Xtr), min(20000, len(Xtr)), replace=False)
    mi = mutual_info_classif(Xtr[sub], ytr[sub], random_state=seed)
    top = np.argsort(mi)[::-1][:k]
    lo, hi = np.percentile(Xtr[:, top], 1, axis=0), np.percentile(Xtr[:, top], 99, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    f = lambda X: ((np.clip(X[:, top], lo, hi) - lo) / span * math.pi).astype(np.float32)  # noqa: E731
    return f(Xtr), f(Xte), [feats[i] for i in top], mi[top]


def train(name, model, X, y, Xte, yte, n_classes, *, steps, batch, lr, seed, device, every):
    torch.manual_seed(seed)
    model = model.to(device)
    counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1)
    w = (1 / counts.sqrt()) / (1 / counts.sqrt()).mean()
    lossf = nn.CrossEntropyLoss(weight=w.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    print(f"\n--- training {name} ({steps} steps x batch {batch}) ---")
    t0, run = time.perf_counter(), 0.0
    for s in range(1, steps + 1):
        idx = torch.randint(0, len(X), (batch,), generator=g).to(device)
        loss = lossf(model(X[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        run = loss.item() if s == 1 else 0.95 * run + 0.05 * loss.item()
        if s % every == 0 or s == 1:
            el = time.perf_counter() - t0
            print(f"  step {s:5d}/{steps}   loss {run:.4f}   {s * batch / el:9,.0f} samples/s", flush=True)
    secs = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(Xte[i:i + 8192]).argmax(1) for i in range(0, len(Xte), 8192)]).cpu().numpy()
    yt = yte.cpu().numpy()
    labels = sorted(set(yt.tolist()))
    return {"macro_f1": float(f1_score(yt, pred, labels=labels, average="macro", zero_division=0)),
            "accuracy": float((pred == yt).mean()),
            "per_family_f1": dict(zip([FAMILIES[i] for i in labels],
                                      map(float, f1_score(yt, pred, labels=labels, average=None, zero_division=0)))),
            "train_seconds": secs, "samples_per_s": steps * batch / secs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/interim/ciciot2023_pcapcsv_clean.parquet")
    ap.add_argument("--train", type=int, default=200_000)
    ap.add_argument("--test", type=int, default=50_000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=100, help="print the loss every N steps")
    ap.add_argument("--out", default="results/demo/a8_vqc.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "none (CPU)"
    print(f"FQML-NIDS · step A8 · device: {gpu} · torch {torch.__version__}")

    Xtr, ytr, Xte, yte, feats, total = load(a.data, a.train, a.test, a.seed)
    print(f"data: CICIoT2023 cleaned, {total:,} rows, 8 families · using {len(Xtr):,} train / {len(Xte):,} test rows")
    print("split: random row split (prototype; the capture-disjoint split of Phase 1 is not used here yet)")
    Ztr, Zte, chosen, mi = preprocess(Xtr, ytr, Xte, feats, 8, a.seed)
    print("8 input features (one per qubit, top mutual information on train): " + ", ".join(chosen))

    T = lambda v, dt=torch.float32: torch.as_tensor(v, dtype=dt, device=device)  # noqa: E731
    X, y, Xt, yt = T(Ztr), T(ytr, torch.long), T(Zte), T(yte, torch.long)
    vqc = TorchVQC(n_qubits=8, n_classes=8, n_layers=3, seed=a.seed, entangler="chain", version=2)
    n_v = sum(p.numel() for p in vqc.parameters())
    mlp = MatchedMLP(8, 8, target_params=n_v)
    n_m = sum(p.numel() for p in mlp.parameters())
    print(f"models: VQC 8 qubits x 3 layers, CNOT chain, our GPU simulator: {n_v} parameters | "
          f"classical twin MLP 8-{mlp.hidden}-8: {n_m} parameters")

    res = {}
    for name, m in (("VQC (quantum, simulated)", vqc), ("MLP (classical, same size)", mlp)):
        res[name] = train(name, m, X, y, Xt, yt, 8, steps=a.steps, batch=a.batch, lr=a.lr, seed=a.seed,
                          device=device, every=a.every)

    print("\n=== held-out results (prototype, as measured) ===")
    names = list(res)
    print(f"{'':12s}" + "".join(f"{n.split(' ')[0]:>10s}" for n in names))
    print(f"{'macro-F1':12s}" + "".join(f"{res[n]['macro_f1']:10.3f}" for n in names))
    print(f"{'accuracy':12s}" + "".join(f"{res[n]['accuracy']:10.3f}" for n in names))
    for fam in res[names[0]]["per_family_f1"]:
        print(f"  {fam:10s}" + "".join(f"{res[n]['per_family_f1'][fam]:10.3f}" for n in names))
    print(f"{'train time':12s}" + "".join(f"{res[n]['train_seconds']:9.1f}s" for n in names))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": gpu, "machine": platform.platform(), "torch": torch.__version__,
                               "args": vars(a), "features": chosen, "split": "random rows",
                               "params": {"vqc": n_v, "mlp": n_m}, "results": res}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
