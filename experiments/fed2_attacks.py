"""FED-2 / FED-6: label attacks against FedAvg and the robust aggregators. Owner: Kareem.

The run grid of docs/ATTACKS.md section 6, on protocol A fold 0 (a first pass; reported runs use 5 folds x 3 seeds):
  attacks      none (the zero-attacker control), label_flip (c -> c+1 mod C), class suppression of Web and of
               Mirai (target family -> Benign); 20% of the clients are malicious, fixed for the whole run and drawn
               with their own seed; for class suppression only among clients that hold the target family
  aggregators  fedavg, median, trimmed_mean (beta 0.2)
  splits       iid (50 clients), sites (10 x 8 attack types), sitesfam (10 x 3 families)
  model        --model mlp_p3000 (default: a 3,000-parameter MLP, large enough to learn Web; the VQC-sized model
               does not, see the FED-4 trial), mlp_p<N>, mlp (VQC-sized) or vqc; --device cuda on a GPU machine

Scores on the clean validation block: macro-F1, recall per family, attack recall at 1% benign FPR (threshold set on
the same block: a first-pass shortcut) and, for class suppression, the target family's recall and the share of its
rows called Benign. Each attacked run is compared with the same split and aggregator with no attacker.

Writes results/fed2/attacks_<protocol>_f<fold>_<model>_s<seed>.json.

    python -m experiments.fed2_attacks --fold 0 --rounds 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score

from experiments.fed3_partitions import GAP, REPORT, label_ranges_from_report
from experiments.fed6_zero_attacker import fit_inputs, load_block, splits
from fqml.data.folds import build_time_blocked, load_manifest, manifest_path
from fqml.fl.batched import predict_proba, run_fedavg
from fqml.fl.equivalence import sqrt_class_weights
from fqml.fl.robust import get_aggregator
from fqml.models.mlp import MatchedMLP
from fqml.quantum.torch_sim import TorchVQC

ROOT = Path(__file__).resolve().parents[1]
ATTACKS = {"none": None, "lf20": ("label_flip", None), "cs20web": ("class_suppression", "Web"),
           "cs20mirai": ("class_suppression", "Mirai")}


def make_model(kind: str, n_classes: int):
    if kind == "vqc":
        return TorchVQC(8, n_classes, 3, seed=0)
    if kind == "mlp":                                     # the VQC's parameter count (MatchedMLP)
        return MatchedMLP(8, n_classes, target_params=sum(p.numel() for p in TorchVQC(8, n_classes, 3, seed=0)
                                                          .parameters()), seed=0)
    if kind.startswith("mlp_p"):
        return MatchedMLP(8, n_classes, target_params=int(kind[5:]), seed=0)
    raise SystemExit(f"unknown model {kind!r}: mlp_p<N>, mlp, vqc")


def choose_malicious(parts, frac, seed, eligible=None):
    """Same rule as fqml.attacks.poison.choose_malicious (FED-2, #24): round(frac * N) clients, drawn among the
    eligible ones; all eligible ones if fewer."""
    pool = np.arange(len(parts)) if eligible is None else np.flatnonzero(eligible)
    k = min(int(round(frac * len(parts))), len(pool))
    mask = np.zeros(len(parts), bool)
    if k:
        mask[np.random.default_rng(seed).choice(pool, k, replace=False)] = True
    return mask


def poison(y, parts, kind, target, C, frac, seed):
    """Training labels with the malicious clients' rows changed (validation labels and class weights stay clean)."""
    if kind == "label_flip":
        mal = choose_malicious(parts, frac, seed)
        fmap = (np.arange(C) + 1) % C
        rows = np.concatenate([parts[k] for k in np.flatnonzero(mal)])
        out = y.copy()
        out[rows] = fmap[y[rows]]
    else:
        holds = np.array([bool((y[p] == target).any()) for p in parts])
        mal = choose_malicious(parts, frac, seed, holds)
        rows = np.concatenate([parts[k] for k in np.flatnonzero(mal)]) if mal.any() else np.zeros(0, int)
        out = y.copy()
        hit = rows[y[rows] == target]
        out[hit] = 0
    return out, mal, int((out != y).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--protocol", default="A")
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--val-cap", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--local-steps", type=int, default=50)
    ap.add_argument("--model", default="mlp_p3000", help="mlp_p<N>, mlp (VQC-sized) or vqc")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attack-seed", type=int, default=100)
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--splits", default="iid,sites,sitesfam")
    ap.add_argument("--attacks", default="none,lf20,cs20web,cs20mirai")
    ap.add_argument("--aggregators", default="fedavg,median,trimmed_mean")
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t0 = time.perf_counter()

    report = json.loads(REPORT.read_text())
    fams = report["families"]
    C = len(fams)
    feats = [c for c in report["features"] if c != "IAT"]
    ranges = label_ranges_from_report(report)
    path = manifest_path("ciciot2023", a.protocol)
    m = load_manifest(path) if path.exists() else build_time_blocked(ranges, dataset="ciciot2023", n_folds=5, gap=GAP)
    print(f"[data] protocol {a.protocol} fold {a.fold} ...", flush=True)
    Ftr, native, ytr = load_block(m, a.fold, "train", ranges, a.cap, feats)
    Fva, _, yva = load_block(m, a.fold, "val", ranges, a.val_cap, feats)
    _, transform = fit_inputs(Ftr, ytr)
    Xtr, Xva = transform(Ftr), transform(Fva)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32)
    w = sqrt_class_weights(ytr, C).float()                                # clean weights (D11): attacker can't change
    parts_by = splits(native, ytr, a.seed)
    print(f"[data] train {len(ytr):,}, validation {len(yva):,}; model {a.model} "
          f"({sum(p.numel() for p in make_model(a.model, C).parameters())} parameters), {a.rounds} rounds, "
          f"{a.device} ({time.perf_counter() - t0:.0f}s)", flush=True)

    runs = []
    todo = [(s, at, g) for s in a.splits.split(",") for at in a.attacks.split(",") for g in a.aggregators.split(",")]
    for i, (split, atk, agg) in enumerate(todo, 1):
        parts = parts_by[split]
        y = ytr
        mal, changed = np.zeros(len(parts), bool), 0
        if ATTACKS[atk]:
            kind, target = ATTACKS[atk]
            y, mal, changed = poison(ytr, parts, kind, fams.index(target) if target else None, C, a.frac,
                                     a.attack_seed)
        t1 = time.perf_counter()
        res = run_fedavg(make_model(a.model, C), Xtr, y, parts, rounds=a.rounds,
                         clients_per_round=min(20, len(parts)), local_steps=a.local_steps, class_weights=w,
                         seed=a.seed, aggregator=get_aggregator(agg), malicious=mal, device=a.device)
        params = {n: v.cpu() for n, v in res.params.items()}
        proba = predict_proba(make_model(a.model, C), params, Xva_t)
        pred = proba.argmax(1)
        att = 1 - proba[:, 0]
        thr = float(np.quantile(att[yva == 0], 0.99))
        rec = recall_score(yva, pred, labels=range(C), average=None, zero_division=0)
        run = {"split": split, "attack": atk, "aggregator": agg, "n_clients": len(parts),
               "malicious": np.flatnonzero(mal).tolist(), "labels_changed": changed,
               "macro_f1": float(f1_score(yva, pred, average="macro")), "recall": dict(zip(fams, map(float, rec))),
               "attack_recall_at_1pct_fpr": float((att[yva != 0] > thr).mean()),
               "seconds": time.perf_counter() - t1}
        if ATTACKS[atk] and ATTACKS[atk][0] == "class_suppression":
            t = fams.index(ATTACKS[atk][1])
            run["target"] = ATTACKS[atk][1]
            run["target_called_benign"] = float((pred[yva == t] == 0).mean())
        runs.append(run)
        extra = (f" | {run['target']} recall {run['recall'][run['target']]:.2f}, called Benign "
                 f"{run['target_called_benign']:.2f}") if "target" in run else ""
        print(f"[{i:>2}/{len(todo)}] {split:9s} {atk:10s} {agg:13s} malicious {len(run['malicious']):>2} | macro-F1 "
              f"{run['macro_f1']:.3f} | caught@1% {run['attack_recall_at_1pct_fpr']:.2f}{extra} "
              f"({run['seconds']:.0f}s)", flush=True)

    # effect of each attack against the same split and aggregator with no attacker
    base = {(r["split"], r["aggregator"]): r for r in runs if r["attack"] == "none"}
    print("\n=== Effect of each attack (attacked minus no attacker, same split and aggregator) ===")
    print(f"{'split':9s} {'attack':10s} {'aggregator':13s} {'dF1':>7s} {'dTarget':>8s}")
    for r in runs:
        b = base.get((r["split"], r["aggregator"]))
        if r["attack"] == "none" or b is None:
            continue
        r["delta_macro_f1"] = r["macro_f1"] - b["macro_f1"]
        if "target" in r:
            r["delta_target_recall"] = r["recall"][r["target"]] - b["recall"][r["target"]]
        print(f"{r['split']:9s} {r['attack']:10s} {r['aggregator']:13s} {r['delta_macro_f1']:+7.3f} "
              + (f"{r['delta_target_recall']:+8.2f}" if "target" in r else f"{'':>8s}"))

    out = ROOT / f"results/fed2/attacks_{a.protocol}_f{a.fold}_{a.model}_s{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": "FED-2 attack grid (first pass)", **{k: v for k, v in vars(a).items()},
                               "families": fams, "runs": runs}, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}  (total {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
