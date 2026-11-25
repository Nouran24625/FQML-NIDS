"""FED-4 follow-up (Kareem): why did the VQC collapse under protocol B (fold 0)?

Centralised VQC, same data and steps as FED-4's 'central',
varying (1) learning rate and (2) a log1p on heavy-tailed features before clipping (DATA-6 spec step 1).
Scored on the validation block: macro-F1 and attack recall at 1% FPR. A probe, not a reported result.

    python -m experiments.fed4_vqc_lr_probe          # writes results/fed4/vqc_lr_probe_B_f0.json (about 40 min)
"""
import json
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

from experiments.fed3_partitions import REPORT, label_ranges_from_report
from experiments.fed6_zero_attacker import fit_inputs, input_features, load_block, make_model
from fqml.data.folds import load_manifest, manifest_path
from fqml.fl.batched import predict_proba, run_fedavg
from fqml.fl.equivalence import sqrt_class_weights

torch.set_num_threads(4)
report = json.loads(REPORT.read_text())
fams, C = report["families"], len(report["families"])
feats = input_features(report)
ranges = label_ranges_from_report(report)
m = load_manifest(manifest_path("ciciot2023", "B"))
Ftr, _, ytr = load_block(m, 0, "train", ranges, 5000, feats)
Fva, _, yva = load_block(m, 0, "val", ranges, 1000, feats)
w = sqrt_class_weights(ytr, C).float()


def log_heavy(F_fit, *blocks):
    """DATA-6 step 1: log1p on a feature with min >= 0 and p99 > 100 x median (median 0: heavy if p99 > 0 and the
    feature is not binary)."""
    mn, med, p99 = F_fit.min(0), np.median(F_fit, 0), np.percentile(F_fit, 99, 0)
    binary = np.array([len(np.unique(F_fit[:5000, j])) <= 2 for j in range(F_fit.shape[1])])
    heavy = (mn >= 0) & np.where(med > 0, p99 > 100 * med, (p99 > 0) & ~binary)
    return heavy, [np.where(heavy, np.log1p(np.clip(B, 0, None)), B) for B in blocks]


def run(lr, use_log, rounds=100):
    if use_log:
        heavy, (A, V) = log_heavy(Ftr, Ftr, Fva)
    else:
        heavy, A, V = None, Ftr, Fva
    top, tf = fit_inputs(A, ytr)
    Xtr, Xva = tf(A), tf(V)
    Xv = torch.as_tensor(Xva, dtype=torch.float32)
    res = run_fedavg(make_model("vqc", C), Xtr, ytr, [np.arange(len(ytr))], rounds=rounds, clients_per_round=1,
                     mode="local", lr=lr, class_weights=w, seed=0, X_val=Xva, y_val=yva)
    p = predict_proba(make_model("vqc", C), {n: v[0] for n, v in res.client_params.items()}, Xv)
    att = 1 - p[:, 0]
    thr = np.quantile(att[yva == 0], 0.99)
    return {"lr": lr, "log": use_log, "features": [feats[i] for i in top],
            "log_applied": None if heavy is None else [feats[i] for i in top if heavy[i]],
            "macro_f1": float(f1_score(yva, p.argmax(1), average="macro")),
            "attack_recall_at_1pct_fpr": float((att[yva != 0] > thr).mean())}


out = []
for lr, lg in [(0.01, False), (0.003, False), (0.001, False), (0.01, True), (0.003, True)]:
    t = time.time()
    r = run(lr, lg)
    out.append(r)
    print(json.dumps(r), f"({time.time() - t:.0f}s)", flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results/fed4/vqc_lr_probe_B_f0.json", "w"), indent=1)
