"""FED-4: federated trial run on real data. Owner: Kareem. Reviewer: Nouran.

A pipeline check, not a reported result (Phase 1 plan v3). For each model (VQC and MatchedMLP with the VQC's
parameter count) and each split (IID 50 clients; sites 10 x 8 attack types; whole-family sites 10 x 3 families):

  1. compare_modes on this run's own data, before round 1: batched FedAvg must equal one-client-at-a-time
     (INTERFACES 3.2, Phase 0 amendment 8). The run stops if it does not.
  2. fedavg       run_fedavg, about 20 rounds, seed 0
  3. local        mode="local" (FED-7): every client trains alone, rounds x local_steps steps
  4. centralised  the same model on all training rows, the same number of steps (one client holding every row)

Scores on the validation block (thresholds and test blocks belong to the runner, EXP-0):
  - 8-class macro-F1 and recall per family; for local-only, the mean and worst client
  - recall on attack types a client never held (fqml.fl.unseen): local-only model of each client, against the
    FedAvg and centralised models on the same rows. Federation earns its place where it raises this number.
  - with fqml.eval.metrics present (EVAL-1, Sheghaf), the full EVAL-1 scores via evaluate(..., families=) (FED-1)

Inputs: DATA-6 (fqml.data.preprocess) as soon as it is in the checkout, otherwise the stand-in used by FED-3/5/6
(top-8 by mutual information on the training sample, clipped and scaled to [0, pi]); `--features` forces one
(experiments.fed6_zero_attacker.prepare_inputs). The result file records which was used.

Writes results/fed4/trial_<protocol>_f<fold>_r<rounds>[_<models>][_xNumber].json (_xNumber: Number excluded, the default
since 24 Nov; files without it kept Number as a candidate input).

    python -m experiments.fed4_trial --fold 0 --rounds 20
    python -m experiments.fed4_trial --rounds 2 --local-steps 5 --models mlp --splits sites   (quick check)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score

from experiments.fed3_partitions import GAP, REPORT, label_ranges_from_report
from experiments.fed6_zero_attacker import (EXCLUDE, LR, exclude_tag, input_features, load_block, make_model,
                                            parse_exclude, prepare_inputs, run_tag, splits)
from fqml.data.folds import build_time_blocked, load_manifest, manifest_path
from fqml.fl.batched import predict_proba, run_fedavg
from fqml.fl.equivalence import compare_modes, passed, sqrt_class_weights
from fqml.fl.unseen import held_types, per_client_unseen, summary

ROOT = Path(__file__).resolve().parents[1]
HAVE_EVAL1 = importlib.util.find_spec("fqml.eval.metrics") is not None


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


FPR = 0.01


def scores(y, proba, families):
    """Returns (pred, attack_score, threshold at 1% benign FPR, scores). The threshold is set on the same validation
    block it is applied to: acceptable for a pipeline check, not for a reported result (the runner sets it on
    validation and applies it to test)."""
    pred = proba.argmax(axis=1)
    attack = 1.0 - proba[:, 0]
    thr = float(np.quantile(attack[y == 0], 1 - FPR)) if (y == 0).any() else 0.5
    rec = recall_score(y, pred, labels=range(len(families)), average=None, zero_division=0)
    return pred, attack, thr, {"macro_f1": float(f1_score(y, pred, average="macro")),
                               "recall": dict(zip(families, map(float, rec))),
                               "attack_recall_at_1pct_fpr": float((attack[y != 0] > thr).mean())}


def unseen(y, native, held, preds, attacks, thrs, everywhere=None):
    """Attack types a client never held. With `everywhere` (every type of the training sample, protocol B), also the
    types that NO client held: the protocol-B question, since federation cannot pass on a type nobody has."""
    pc = per_client_unseen(y, native, held, preds, scores=attacks, thresholds=thrs)
    out = {"detection_at_1pct_fpr": summary(pc, "detection_at_threshold_types"),
           "family_recall": summary(pc, "family_recall_types")}
    if everywhere is not None:
        pe = per_client_unseen(y, native, [everywhere] * len(preds), preds, scores=attacks, thresholds=thrs)
        out["nowhere_detection_at_1pct_fpr"] = summary(pe, "detection_at_threshold_types")
        out["nowhere_family_recall"] = summary(pe, "family_recall_types")
        out["nowhere_types"] = sorted(set(np.asarray(native).tolist()) - everywhere - {"BenignTraffic"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--protocol", default="A")
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--val-cap", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="mlp,vqc", help="vqc, vqc_lin, vqc_noent, mlp (= mlp_p64), mlp_p<N>, rff, logreg")
    ap.add_argument("--splits", default="iid,sites,sitesfam")
    ap.add_argument("--exclude", default=",".join(EXCLUDE), help="features never used as inputs (IAT always)")
    ap.add_argument("--lr", type=float, default=LR, help="local learning rate (the VQC needs a tuned one)")
    ap.add_argument("--features", default="auto", help="auto (DATA-6 when present, else stand-in), stand-in, data6")
    a = ap.parse_args()
    exclude = parse_exclude(a.exclude)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t0 = time.perf_counter()

    report = json.loads(REPORT.read_text())
    families = report["families"]
    C = len(families)
    feats = input_features(report, exclude)                          # D9 and its Number addition
    ranges = label_ranges_from_report(report)
    path = manifest_path("ciciot2023", a.protocol)
    m = load_manifest(path) if path.exists() else build_time_blocked(ranges, dataset="ciciot2023", n_folds=5, gap=GAP)
    print(f"[data] protocol {a.protocol} fold {a.fold}: reading the training sample and validation block ...", flush=True)
    Ftr, native_tr, ytr = load_block(m, a.fold, "train", ranges, a.cap, feats)
    Fva, native_va, yva = load_block(m, a.fold, "val", ranges, a.val_cap, feats)
    names, transform, source = prepare_inputs(m, a.fold, ranges, report, feats, exclude, Ftr, ytr,
                                              protocol=a.protocol, source=a.features)
    Xtr, Xva = transform(Ftr), transform(Fva)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32)
    print(f"[data] train {len(ytr):,} rows, validation {len(yva):,}; features {names} [{source}]; "
          f"EVAL-1 metrics module {'found' if HAVE_EVAL1 else 'not yet available'} ({time.perf_counter() - t0:.0f}s)",
          flush=True)
    w = sqrt_class_weights(ytr, C).float()                              # D11 part 1
    parts_by = splits(native_tr, ytr, a.seed)
    # protocol B: validation attack types that are in no client's training rows (None under protocol A)
    everywhere = set(np.unique(native_tr).tolist()) - {"BenignTraffic"} if a.protocol == "B" else None
    out = {"task": "FED-4 federated trial run (pipeline check)", "protocol": a.protocol, "fold": a.fold,
           "cap": a.cap, "val_cap": a.val_cap, "rounds": a.rounds, "local_steps": a.local_steps, "seed": a.seed,
           "families": families, "exclude": sorted({"IAT", *exclude}), "lr": a.lr,
           "features": names, "features_source": source,
           "eval1": HAVE_EVAL1, "runs": []}

    for kind in a.models.split(","):
        for split in a.splits.split(","):
            parts = parts_by[split]
            cpr = min(20, len(parts))
            held = held_types(native_tr, parts)
            model = make_model(kind, C)
            print(f"\n=== {kind} ({n_params(model)} parameters) | {split}: {len(parts)} clients, {cpr} per round ===",
                  flush=True)
            eq = compare_modes(model, Xtr, ytr, parts, class_weights=w.double(), rounds=2,
                               clients_per_round=min(3, len(parts)), local_steps=3, batch_size=8)
            print(f"  [check] batched = sequential on this data: {passed(eq)} "
                  f"(max weight diff {eq['max_weight_diff']:.1e}, logit diff {eq['max_logit_diff']:.1e})", flush=True)
            if not passed(eq):
                raise SystemExit("batched FedAvg differs from the sequential reference on this data; stopping")
            run = {"model": kind, "params": n_params(model), "split": split, "n_clients": len(parts),
                   "clients_per_round": cpr, "equivalence": eq}

            def progress(name):
                def on_round(e):
                    f1 = e.get("val_macro_f1", e.get("val_macro_f1_mean"))
                    print(f"    {name:11s} round {e['round'] + 1:>3}/{a.rounds}  {e['seconds']:6.1f}s"
                          + (f"  val macro-F1 {f1:.3f}" if f1 is not None else ""), flush=True)
                return on_round

            common = dict(rounds=a.rounds, local_steps=a.local_steps, lr=a.lr, class_weights=w, seed=a.seed,
                          X_val=Xva, y_val=yva, families=families if HAVE_EVAL1 else None)

            # FedAvg
            t1 = time.perf_counter()
            fa = run_fedavg(make_model(kind, C), Xtr, ytr, parts, clients_per_round=cpr,
                            on_round=progress("fedavg"), **common)
            pred, att, thr, s = scores(yva, predict_proba(make_model(kind, C), fa.params, Xva_t), families)
            s["unseen"] = unseen(yva, native_va, held, [pred] * len(parts), [att] * len(parts), [thr] * len(parts),
                                 everywhere)
            s["seconds"] = time.perf_counter() - t1
            if fa.final_val is not None:
                s["eval1"] = fa.final_val
            run["fedavg"] = s

            # local-only (FED-7)
            t1 = time.perf_counter()
            lo = run_fedavg(make_model(kind, C), Xtr, ytr, parts, clients_per_round=cpr, mode="local",
                            on_round=progress("local-only"), **common)
            preds, atts, thrs, f1s = [], [], [], []
            for k in range(len(parts)):
                pk, ak, tk, sk = scores(yva, predict_proba(make_model(kind, C),
                                                           {n: v[k] for n, v in lo.client_params.items()}, Xva_t),
                                        families)
                preds.append(pk)
                atts.append(ak)
                thrs.append(tk)
                f1s.append(sk["macro_f1"])
            run["local"] = {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_min": float(np.min(f1s)),
                            "macro_f1_max": float(np.max(f1s)),
                            "unseen": unseen(yva, native_va, held, preds, atts, thrs, everywhere),
                            "seconds": time.perf_counter() - t1}

            # centralised: one client holding every training row, same number of steps
            t1 = time.perf_counter()
            ce = run_fedavg(make_model(kind, C), Xtr, ytr, [np.arange(len(ytr))], clients_per_round=1, mode="local",
                            on_round=progress("central"), **common)
            pred, att, thr, s = scores(yva, predict_proba(make_model(kind, C),
                                                          {n: v[0] for n, v in ce.client_params.items()}, Xva_t),
                                       families)
            s["unseen"] = unseen(yva, native_va, held, [pred] * len(parts), [att] * len(parts), [thr] * len(parts),
                                 everywhere)
            s["seconds"] = time.perf_counter() - t1
            run["central"] = s
            out["runs"].append(run)

            def u(d, key):
                return d["unseen"][key]["mean"]
            print(f"  -> macro-F1   fedavg {run['fedavg']['macro_f1']:.3f} | local mean {run['local']['macro_f1_mean']:.3f}"
                  f" (worst {run['local']['macro_f1_min']:.3f}) | central {run['central']['macro_f1']:.3f}", flush=True)
            for key, label in (("detection_at_1pct_fpr", "detected at 1% FPR"), ("family_recall", "right family")):
                print(f"  -> attack types a client never held, {label:18s}: fedavg {u(run['fedavg'], key):.3f} | "
                      f"local {u(run['local'], key):.3f} | central {u(run['central'], key):.3f}", flush=True)
                if everywhere is not None:
                    key = "nowhere_" + key
                    print(f"  -> attack types NO client held,     {label:18s}: fedavg {u(run['fedavg'], key):.3f} | "
                          f"local {u(run['local'], key):.3f} | central {u(run['central'], key):.3f}", flush=True)

    mtag = "" if a.models == "mlp,vqc" else "_" + a.models.replace(",", "-")
    dest = ROOT / f"results/fed4/trial_{a.protocol}_f{a.fold}_r{a.rounds}{mtag}{exclude_tag(exclude)}{run_tag(a.lr, source)}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {dest.relative_to(ROOT)}  (total {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
