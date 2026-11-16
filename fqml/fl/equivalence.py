"""Batched FedAvg = one-client-at-a-time FedAvg, checked on REAL data (Phase 0 amendment 8). Owner: Kareem.

Decision D3 rests on one claim: training the sampled clients together in one batch gives the same
global model as training them one by one. Phase 0 tested that on random numbers only
(tests/test_batched_fedavg.py). Real traffic is different in ways that matter to the loop: a few
classes dominate, many rows are identical, some clients are tiny (their rows repeat inside a round)
and some hold a single class. `compare_modes` runs both loops on the same data and reports the largest
difference; the federated trial run (task FED-4) calls it before its first round, and this file runs it
on a cleaned dataset directly:

    python -m fqml.fl.equivalence --data data/interim/ciciot2023_v1/ciciot2023_clean_v1.parquet

Writes results/checks/fedavg_equivalence.json and exits with status 1 if any case differs.
Everything runs in float64 (see the note in tests/test_batched_fedavg.py: in float32 Adam turns
rounding noise into full-size steps on weights whose true gradient is ~0).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

from fqml.fl.batched import run_fedavg
from fqml.fl.partition import dirichlet_partition, iid_partition

ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = 1e-7


def sqrt_class_weights(y, n_classes: int) -> torch.Tensor:
    """Same weights as fqml.train.loop.class_weights (1 / sqrt(count), mean 1), in float64."""
    counts = np.maximum(np.bincount(np.asarray(y), minlength=n_classes), 1)
    w = 1.0 / np.sqrt(counts)
    return torch.tensor(w / w.mean(), dtype=torch.float64)


def compare_modes(model, X, y, parts, *, class_weights=None, rounds=2, clients_per_round=3, local_steps=3,
                  batch_size=8, lr=0.05, seed=1) -> dict:
    """Run FedAvg twice (mode="batched" and mode="sequential") from the same start and compare.

    Returns the largest absolute difference between the two global models (weights and logits on X),
    whether the same clients were sampled, and whether training moved the weights at all."""
    X = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(y), dtype=torch.long)
    kw = dict(rounds=rounds, clients_per_round=clients_per_round, local_steps=local_steps, batch_size=batch_size,
              lr=lr, class_weights=class_weights, seed=seed, dtype=torch.float64)
    a = run_fedavg(model, X, y, parts, mode="batched", **kw)
    b = run_fedavg(model, X, y, parts, mode="sequential", **kw)
    ref = copy.deepcopy(model).double()
    with torch.no_grad():
        la = functional_call(ref, a.params, (X,))
        lb = functional_call(ref, b.params, (X,))
    start = {n: p.detach() for n, p in ref.named_parameters()}
    weight_diff = max(float((a.params[n] - b.params[n]).abs().max()) for n in a.params)
    logit_diff = float((la - lb).abs().max())
    return {
        "same_clients": [h["clients"] for h in a.history] == [h["clients"] for h in b.history],
        "max_weight_diff": weight_diff,
        "max_logit_diff": logit_diff,
        "weights_moved": any(not torch.allclose(a.params[n], start[n]) for n in a.params),
        "equal": weight_diff <= TOLERANCE and logit_diff <= TOLERANCE,
    }


def passed(result: dict) -> bool:
    return bool(result["same_clients"] and result["weights_moved"] and result["equal"])


# ---- real data -----------------------------------------------------------------------------------
def sample_parquet(path, columns: list[str], n_rows: int):
    """About n_rows rows spread evenly over the whole file (every k-th row), read batch by batch so the
    full file is never in memory. Returns a DataFrame with `columns`."""
    import pandas as pd
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    step = max(1, total // max(1, n_rows))
    parts, offset = [], 0
    for batch in pf.iter_batches(batch_size=1 << 18, columns=columns):
        take = np.flatnonzero((offset + np.arange(batch.num_rows)) % step == 0)
        offset += batch.num_rows
        if len(take):
            parts.append(batch.to_pandas().iloc[take])
    return pd.concat(parts, ignore_index=True)


def to_model_inputs(F: np.ndarray, y: np.ndarray, n_features: int = 8, seed: int = 0):
    """Quick stand-in for the per-fold preprocessing (task DATA-6): the n_features columns with the
    highest mutual information with y, clipped to their 1st-99th percentile and scaled to [0, pi].
    Returns (X float64 [n, n_features], indices of the chosen columns)."""
    from sklearn.feature_selection import mutual_info_classif
    F = np.asarray(F, dtype=np.float64)
    sub = np.random.default_rng(seed).choice(len(F), min(20_000, len(F)), replace=False)
    varying = np.flatnonzero(F[sub].std(axis=0) > 0)
    mi = mutual_info_classif(F[sub][:, varying], np.asarray(y)[sub], random_state=seed)
    top = np.sort(varying[np.argsort(mi)[::-1][:n_features]])
    lo, hi = np.percentile(F[:, top], [1, 99], axis=0)
    hi = np.where(hi > lo, hi, lo + 1.0)
    X = (np.clip(F[:, top], lo, hi) - lo) / (hi - lo) * np.pi
    return X, top


def real_data_cases(X: np.ndarray, y: np.ndarray, n_classes: int, seed: int = 0) -> list[dict]:
    """The cases the check runs: VQC (all classes), VQC (binary), the entanglement-free control and MLP,
    each on an IID split and on a
    strongly skewed split (Dirichlet 0.1, with clients small enough that rows repeat inside a round)."""
    from fqml.models.mlp import MatchedMLP
    from fqml.quantum.torch_sim import TorchVQC
    y = np.asarray(y)
    yb = (y != 0).astype(np.int64)
    splits = {"iid": iid_partition(len(y), 20, seed),
              "dir0.1": dirichlet_partition(y, 20, 0.1, seed, min_size=5)}
    cases = []
    for split, parts in splits.items():
        for name, model, labels, k in (
                (f"vqc_c{n_classes}", TorchVQC(8, n_classes, 3, seed=0), y, n_classes),
                ("vqc_c2", TorchVQC(8, 2, 3, seed=0), yb, 2),
                (f"vqc_noent_c{n_classes}", TorchVQC(8, n_classes, 3, seed=0, entangler="none", readout="linear"),
                 y, n_classes),
                (f"mlp_c{n_classes}", MatchedMLP(8, n_classes, target_params=64, seed=0), y, n_classes)):
            res = compare_modes(model, X, labels, parts, class_weights=sqrt_class_weights(labels, k),
                                rounds=2, clients_per_round=5, local_steps=5, batch_size=16)
            cases.append({"model": name, "split": split, "clients": len(parts),
                          "smallest_client_rows": int(min(len(p) for p in parts)), **res,
                          "passed": passed(res)})
    return cases


def main():
    ap = argparse.ArgumentParser(description="Batched vs sequential FedAvg on a cleaned dataset.")
    ap.add_argument("--data", default=str(ROOT / "data/interim/ciciot2023_v1/ciciot2023_clean_v1.parquet"))
    ap.add_argument("--report", default=str(ROOT / "results/data_reports/ciciot2023_clean_v1.json"),
                    help="cleaning report: gives the feature column names")
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--exclude", nargs="*", default=["IAT"], help="feature columns never used as inputs")
    ap.add_argument("--out", default=str(ROOT / "results/checks/fedavg_equivalence.json"))
    a = ap.parse_args()
    torch.set_num_threads(1)                                             # decision D2
    report = json.loads(Path(a.report).read_text())
    feats = [c for c in report["features"] if c not in set(a.exclude)]
    df = sample_parquet(a.data, feats + ["y"], a.rows)
    y = df["y"].to_numpy(np.int64)
    n_classes = len(report["families"])
    X, top = to_model_inputs(df[feats].to_numpy(np.float64), y)
    cases = real_data_cases(X, y, n_classes)
    out = {"data": a.data, "rows": int(len(df)), "features": [feats[i] for i in top],
           "class_counts": np.bincount(y, minlength=n_classes).tolist(),
           "exact_duplicate_rows_in_sample": int(len(X) - len(np.unique(X, axis=0))),
           "tolerance": TOLERANCE, "torch": torch.__version__, "cases": cases,
           "passed": all(c["passed"] for c in cases)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    for c in cases:
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['model']:14s} {c['split']:7s} "
              f"max weight diff {c['max_weight_diff']:.2e}  max logit diff {c['max_logit_diff']:.2e}")
    print(f"wrote {a.out}")
    raise SystemExit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
