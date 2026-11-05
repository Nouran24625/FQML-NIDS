"""Train the live-demo model on labelled LAB windows and report it honestly.

    python -m fqml.live.train --windows runs/lab/windows.parquet --out models/lab_demo [--deploy vqc]

Steps
  1. Session-disjoint split: for every attack label with >= 2 logged runs, whole runs are held out for
     test (about a third, at least one). Benign rows are split by 60 s time blocks. Labels with only
     one run stay in training and are reported as "not tested on an unseen run".
     A validation set is carved from training the same way (used only to pick the seed).
  2. Features (8 = one per qubit). Default --features footprint: the 8 features that each target one
     attack footprint (features.FOOTPRINT, chosen before seeing any lab data; easy to explain on video).
     --features mrmr: chosen on TRAIN rows by class-balanced mutual information minus redundancy.
     Mutual information of the used features is reported either way.
  3. Preprocessing on TRAIN rows only: log1p, 1st-99th percentile clip, min-max to [0, pi].
  4. Models, same inputs for all:
       vqc  TorchVQC, 8 qubits, 3 layers, CNOT chain, circuit v2 (decisions D1, D6, D7)
       mlp  MatchedMLP with the VQC's parameter count (the project's fair classical twin)
       rf   RandomForest, a strong classical reference (not parameter-matched)
     Optional --federated-sites K: the VQC is trained with the project's batched FedAvg
     (fqml.fl.batched.run_fedavg) by K simulated sites, each holding a share of the training rows.
  5. Test report per model: macro-F1, per-class F1 / recall, false-positive rate on benign windows,
     confusion matrix -> <out>/report.json. The --deploy model is saved as the bundle.
Numbers printed here are the numbers that go on screen; nothing is rounded up or replaced by targets.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from fqml.live.features import FEATURES, FOOTPRINT
from fqml.live.labels import BENIGN, class_order
from fqml.live.model import Preprocess, build_torch_model, save_bundle


# ---------------------------------------------------------------- split
def session_split(df: pd.DataFrame, frac: float, seed: int, block_s: float = 60.0):
    """Return (keep_mask, held_mask, untested_labels). Whole attack runs / benign time blocks move together."""
    rng = np.random.default_rng(seed)
    held = np.zeros(len(df), dtype=bool)
    untested = []
    att = df[df["session"] >= 0]
    for lab, g in att.groupby("label"):
        sess = np.sort(g["session"].unique())
        if len(sess) < 2:
            untested.append(lab)
            continue
        n_hold = max(1, int(round(frac * len(sess))))
        chosen = rng.choice(sess, n_hold, replace=False)
        held |= df["session"].isin(chosen).to_numpy()
    ben = (df["session"] < 0).to_numpy()
    blocks = (df["window_start"] // block_s).astype(np.int64).to_numpy()
    ub = np.unique(blocks[ben])
    hb = rng.choice(ub, max(1, int(round(frac * len(ub)))), replace=False) if len(ub) > 1 else []
    held |= ben & np.isin(blocks, hb)
    return ~held, held, untested


# ---------------------------------------------------------------- training
def class_weights(y, n_classes):
    counts = np.maximum(np.bincount(y, minlength=n_classes), 1)
    w = 1.0 / np.sqrt(counts)
    return w / w.mean()


def train_torch(kind, Xtr, ytr, n_classes, *, seed, steps, batch, lr, target_params=None):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    m = build_torch_model(kind, Xtr.shape[1], n_classes, seed=seed, target_params=target_params)
    X = torch.from_numpy(Xtr)
    y = torch.from_numpy(ytr.astype(np.int64))
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(ytr, n_classes), dtype=torch.float32))
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (batch,), generator=g)
        loss = lossf(m(X[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    return m


def train_vqc_federated(Xtr, ytr, n_classes, *, seed, sites, rounds, local_steps, batch, lr):
    import torch
    from fqml.fl.batched import run_fedavg
    from fqml.fl.partition import iid_partition
    m = build_torch_model("vqc", Xtr.shape[1], n_classes, seed=seed)
    parts = iid_partition(len(Xtr), sites, seed)
    res = run_fedavg(m, Xtr, ytr, parts, rounds=rounds, clients_per_round=sites, local_steps=local_steps,
                     batch_size=batch, lr=lr, seed=seed,
                     class_weights=torch.tensor(class_weights(ytr, n_classes), dtype=torch.float32),
                     on_round=lambda e: print(f"    round {e['round'] + 1:3d}/{rounds}  {e['seconds']:.2f} s"))
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(res.params[n])
    m.eval()
    return m


def predict(kind, model, X):
    if kind == "rf":
        return model.predict(X)
    import torch
    with torch.no_grad():
        return model(torch.from_numpy(X)).argmax(1).numpy()


def report(y, pred, classes) -> dict:
    labels = list(range(len(classes)))
    present = sorted(set(y.tolist()))
    f1 = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    ben = y == 0
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "per_class": {classes[i]: {"f1": float(f1[i]), "recall": float(rec[i]), "support": int((y == i).sum())}
                      for i in present},
        "benign_fpr": float((pred[ben] != 0).mean()) if ben.any() else None,
        "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def select_features(X: np.ndarray, y: np.ndarray, k: int, seed: int = 0, per_class: int = 400):
    """mRMR on TRAIN rows: relevance = mutual information with the label, measured on a class-balanced
    sample (so rare attack classes count as much as Benign); redundancy = mean |Spearman correlation|
    with the features already chosen. Plain top-k MI picks k near-copies of 'how much traffic'."""
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), min(per_class, int((y == c).sum())), replace=False)
                          for c in np.unique(y)])
    Z = np.log1p(np.maximum(X[idx], 0))
    mi = mutual_info_classif(Z, y[idx], random_state=seed)
    ranks = pd.DataFrame(Z).rank().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.abs(np.corrcoef(ranks, rowvar=False)), nan=0.0)
    chosen = [int(np.argmax(mi))]
    while len(chosen) < min(k, X.shape[1]):
        score = mi - corr[:, chosen].mean(axis=1)
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return chosen, mi


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True, help="output of fqml.live.labels (.parquet or .csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="vqc,mlp,rf")
    ap.add_argument("--deploy", default="vqc", help="which model the detector uses")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--features", choices=["footprint", "mrmr"], default="footprint")
    ap.add_argument("--min-pkts", type=int, default=3, help="rows with fewer sent packets are not classified")
    ap.add_argument("--test-frac", type=float, default=0.34)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--federated-sites", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--split-seed", type=int, default=0)
    a = ap.parse_args(argv)
    kinds = [k.strip() for k in a.models.split(",") if k.strip()]
    if a.deploy not in kinds:
        ap.error("--deploy must be one of --models")

    df = pd.read_parquet(a.windows) if a.windows.endswith(".parquet") else pd.read_csv(a.windows)
    df = df[~df["ambiguous"] & (df["n_out"] >= a.min_pkts)].reset_index(drop=True)
    classes = class_order(df["label"])
    if len(classes) > a.k:
        raise SystemExit(f"{len(classes)} classes but only {a.k} qubits/readouts")
    y_all = df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()

    keep, test, untested = session_split(df, a.test_frac, a.split_seed)
    sub = df[keep].reset_index(drop=True)
    k_tr, k_val, _ = session_split(sub, 0.25, a.split_seed + 1)
    tr_idx, val_idx = np.flatnonzero(keep)[k_tr], np.flatnonzero(keep)[k_val]
    te_idx = np.flatnonzero(test)
    print(f"classes: {classes}")
    print(f"rows: train {len(tr_idx):,} | val {len(val_idx):,} | test {len(te_idx):,}  (whole attack runs held out)")
    if untested:
        print(f"WARNING: only one run of {untested}: kept in training, NOT tested on an unseen run")

    Xraw = df[FEATURES].to_numpy(dtype=np.float64)
    top, mi = select_features(Xraw[tr_idx], y_all[tr_idx], a.k)
    if a.features == "footprint":
        top = [FEATURES.index(f) for f in FOOTPRINT[:a.k]]
    feats = [FEATURES[i] for i in top]
    print(f"features ({a.features}; mutual information on train): " + ", ".join(f"{FEATURES[i]} {mi[i]:.2f}" for i in top))
    pre = Preprocess.fit(Xraw[tr_idx][:, top], feats)
    X = pre.transform(Xraw[:, top])
    Xtr, ytr, Xval, yval, Xte, yte = X[tr_idx], y_all[tr_idx], X[val_idx], y_all[val_idx], X[te_idx], y_all[te_idx]
    C = len(classes)

    results, fitted = {}, {}
    vqc_params = None
    for kind in kinds:
        t0 = time.perf_counter()
        best = None
        for s in range(a.seeds if kind != "rf" else 1):
            if kind == "rf":
                m = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=s, n_jobs=-1).fit(Xtr, ytr)
                init = {}
            elif kind == "vqc" and a.federated_sites:
                print(f"  vqc seed {s}: federated, {a.federated_sites} sites x {a.rounds} rounds")
                m = train_vqc_federated(Xtr, ytr, C, seed=s, sites=a.federated_sites, rounds=a.rounds,
                                        local_steps=max(1, a.steps // a.rounds), batch=a.batch, lr=a.lr)
                init = {"seed": s}
            else:
                tp = None
                if kind == "mlp":
                    tp = vqc_params or n_params(build_torch_model("vqc", a.k, C))
                m = train_torch(kind, Xtr, ytr, C, seed=s, steps=a.steps, batch=a.batch, lr=a.lr, target_params=tp)
                init = {"seed": s} if kind == "vqc" else {"target_params": tp}
            pv = predict(kind, m, Xval) if len(Xval) else predict(kind, m, Xtr)
            yv = yval if len(Xval) else ytr
            f1v = f1_score(yv, pv, labels=sorted(set(yv.tolist())), average="macro", zero_division=0)
            if best is None or f1v > best[0]:
                best = (f1v, s, m, init)
        f1v, s, m, init = best
        if kind == "vqc":
            vqc_params = n_params(m)
        r = report(yte, predict(kind, m, Xte), classes)
        r.update(val_macro_f1=float(f1v), seed=s, train_seconds=time.perf_counter() - t0,
                 n_params=(n_params(m) if kind != "rf" else None))
        results[kind], fitted[kind] = r, (m, init)
        pc = "  ".join(f"{c} {v['f1']:.2f}" for c, v in r["per_class"].items())
        print(f"{kind:4s} params={str(r['n_params'] or '-'):>4s}  test macro-F1 {r['macro_f1']:.3f}  "
              f"benign FPR {r['benign_fpr']:.3f}  | {pc}")

    m, init = fitted[a.deploy]
    info = {
        "window": float(df["window"].iloc[0]), "min_pkts": a.min_pkts,
        "trained_on": str(a.windows), "feature_method": a.features, "federated_sites": a.federated_sites if a.deploy == "vqc" else 0,
        "mutual_information": {FEATURES[i]: float(mi[i]) for i in top},
        "untested_labels": untested, "test_report": results[a.deploy],
        "description": {"vqc": f"8-qubit VQC (TorchVQC simulator, chain, v2), {results['vqc']['n_params'] if 'vqc' in results else ''} parameters",
                        "mlp": "parameter-matched MLP", "rf": "random forest (classical reference)"}[a.deploy],
    }
    save_bundle(a.out, classes=classes, pre=pre, kind=a.deploy, model=m, init=init, extra=info)
    rep = {"classes": classes, "features": feats, "untested_labels": untested, "split_seed": a.split_seed,
           "rows": {"train": len(tr_idx), "val": len(val_idx), "test": len(te_idx)}, "models": results,
           "args": vars(a)}
    (Path(a.out) / "report.json").write_text(json.dumps(rep, indent=2))
    print(f"saved bundle ({a.deploy}) and report.json to {a.out}")


if __name__ == "__main__":
    main()
