"""FED-5 evidence for decision D12: can the tree models (XGBoost, Random Forest) be trained federated? Owner: Kareem.

Same training sample, features and splits as the FED-6 control (protocol A fold 0, D11 sample, top-8 stand-in for
DATA-6). Scored on the validation block (macro-F1 over the 8 families and recall per family):

  central_xgb, central_rf   pooled data: the ceiling. Histogram-based federated XGBoost (XGBoost's own federated
                            mode, NVFlare's histogram recipe) sends gradient histograms instead of rows and grows the
                            same trees as pooled training apart from the feature-quantile sketch, so central_xgb is
                            also the expected result of that option.
  local_rf                  every client trains a forest alone; mean and worst client (the local-only baseline)
  fed_rf                    one round: every client trains a forest on its own rows, the server averages the
                            clients' class probabilities, weighted by row count (no data leaves a client)
  fed_xgb_ens               the same with one XGBoost model per client (close to Flower's bagging strategy run
                            for one round)

Writes results/fed5/trees_<protocol>_f<fold>.json.

    python -m experiments.fed5_trees --fold 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

from experiments.fed3_partitions import GAP, REPORT, label_ranges_from_report
from experiments.fed6_zero_attacker import fit_inputs, load_block, splits
from fqml.data.folds import build_time_blocked, load_manifest, manifest_path

ROOT = Path(__file__).resolve().parents[1]


def full_proba(model, X, n_classes):
    """predict_proba over all classes, also when a client's model saw only some of them."""
    p = np.zeros((len(X), n_classes))
    p[:, model.classes_ if hasattr(model, "classes_") else np.arange(n_classes)] = model.predict_proba(X)
    return p


def rf(seed):
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed, class_weight="balanced_subsample")


class Xgb:
    """XGBoost on a client that may hold only some classes: labels re-indexed to 0..k-1, mapped back."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            return self
        idx = np.searchsorted(self.classes_, y)
        self.m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
                               n_jobs=-1, random_state=self.seed)
        self.m.fit(X, idx)
        return self

    def predict_proba(self, X):
        if len(self.classes_) == 1:
            return np.ones((len(X), 1))
        return self.m.predict_proba(X)


def score(y, proba, families):
    pred = proba.argmax(axis=1)
    rec = recall_score(y, pred, labels=range(len(families)), average=None, zero_division=0)
    return {"macro_f1": float(f1_score(y, pred, average="macro")), "recall": dict(zip(families, map(float, rec)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--protocol", default="A")
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--val-cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="iid,sites,sitesfam")
    a = ap.parse_args()
    t0 = time.perf_counter()
    report = json.loads(REPORT.read_text())
    families = report["families"]
    C = len(families)
    feats = [c for c in report["features"] if c != "IAT"]
    ranges = label_ranges_from_report(report)
    path = manifest_path("ciciot2023", a.protocol)
    m = load_manifest(path) if path.exists() else build_time_blocked(ranges, dataset="ciciot2023", n_folds=5, gap=GAP)
    print(f"[data] reading protocol {a.protocol} fold {a.fold} ...", flush=True)
    Ftr, native, ytr = load_block(m, a.fold, "train", ranges, a.cap, feats)
    Fva, _, yva = load_block(m, a.fold, "val", ranges, a.val_cap, feats)
    top, transform = fit_inputs(Ftr, ytr)
    Xtr, Xva = transform(Ftr), transform(Fva)
    print(f"[data] train {len(ytr):,}, validation {len(yva):,} ({time.perf_counter() - t0:.0f}s)", flush=True)

    out = {"task": "FED-5 / D12 evidence", "protocol": a.protocol, "fold": a.fold, "cap": a.cap, "seed": a.seed,
           "families": families, "features": [feats[i] for i in top], "central": {}, "splits": {}}

    def show(name, s):
        print(f"  {name:22s} macro-F1 {s['macro_f1']:.3f} | " +
              " ".join(f"{f[:5]} {r:.2f}" for f, r in s["recall"].items()), flush=True)

    print("\n[central] pooled training (ceiling)", flush=True)
    for name, model in (("central_xgb", Xgb(a.seed)), ("central_rf", rf(a.seed))):
        t1 = time.perf_counter()
        model.fit(Xtr, ytr)
        s = score(yva, full_proba(model, Xva, C), families)
        s["seconds"] = time.perf_counter() - t1
        out["central"][name] = s
        show(name, s)

    parts_by = splits(native, ytr, a.seed)
    for split in a.splits.split(","):
        parts = parts_by[split]
        sizes = np.array([len(p) for p in parts], dtype=float)
        print(f"\n[{split}] {len(parts)} clients", flush=True)
        res = {}
        for kind, make in (("rf", rf), ("xgb", Xgb)):
            t1 = time.perf_counter()
            probas = []
            for k, p in enumerate(parts):
                probas.append(full_proba(make(a.seed + k).fit(Xtr[p], ytr[p]), Xva, C))
                print(f"    {kind} client {k + 1}/{len(parts)}", end="\r", flush=True)
            fed = np.tensordot(sizes / sizes.sum(), np.stack(probas), axes=1)
            res[f"fed_{kind}" if kind == "rf" else "fed_xgb_ens"] = s = score(yva, fed, families)
            s["seconds"] = time.perf_counter() - t1
            show(f"fed_{kind}" if kind == "rf" else "fed_xgb_ens", s)
            if kind == "rf":
                loc = [score(yva, pr, families)["macro_f1"] for pr in probas]
                res["local_rf"] = {"macro_f1_mean": float(np.mean(loc)), "macro_f1_min": float(np.min(loc)),
                                   "macro_f1_max": float(np.max(loc))}
                print(f"  {'local_rf':22s} macro-F1 mean {np.mean(loc):.3f}  worst {np.min(loc):.3f}  "
                      f"best {np.max(loc):.3f}", flush=True)
        out["splits"][split] = res

    dest = ROOT / f"results/fed5/trees_{a.protocol}_f{a.fold}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dest.relative_to(ROOT)}  (total {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
