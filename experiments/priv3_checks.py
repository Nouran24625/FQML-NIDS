"""PRIV-3 on real rows: check 1 (our DP-SGD against Opacus, same MLP) and check 3 (one-run canary audit). Owner: Nouran.

Data: CICIoT2023 protocol A, fold 0. Training: at most --train-per-type rows per attack type, spread over time
(fqml.data.rows.cap_per_label); validation: at most 500 rows per type of the fold's validation block. 35 features
(no IAT, no Number). Scaling to [0, pi]: clip to the training 1st-99th percentile, then min-max (a stand-in until the
DATA-6 preprocessing is built; both runs of every check use the same inputs, so the comparison is unaffected).

Check 1, for each epsilon: MatchedMLP (256 parameters) trained (a) with fqml.privacy.dpsgd.dp_fit and (b) with Opacus's
PrivacyEngine.make_private_with_epsilon, same seed, Poisson sampling, batch 256, clip 1.0, delta 1e-6, same epochs,
Adam, unweighted cross-entropy. Reported: noise multiplier and epsilon of each, validation macro-F1 of each.
Pass: noise multipliers within 5% and epsilons within 10% (the two use the same RDP analysis; ours searches more
orders, so it may be slightly tighter); macro-F1 within 0.05 (different random batches, same mechanism).

Check 3, for each epsilon in {inf, 8, 1}: 1,000 canaries, MLP-3000, one training run each, empirical lower bound on
epsilon at 95% confidence. Pass: every private run's bound is at most its accountant epsilon.

    docker compose -f docker/compose.yml exec dev python -m experiments.priv3_checks          # about 20-40 min on CPU
Writes results/priv3/checks_A_f0.json.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fqml.data.rows import CICIOT2023_V1, cap_per_label, read_rows
from fqml.eval.checkpoint1 import REPORT, label_ranges, split_rows
from fqml.models.classical import build
from fqml.privacy.audit import run_audit
from fqml.privacy.dpsgd import dp_fit

DROP = ("IAT", "Number")


def load(train_per_type, val_per_type, seed=0):
    report = json.loads(Path(REPORT).read_text())
    feats = [f for f in report["features"] if f not in DROP]
    pool, val_ids, _ = split_rows("A-gap100", report, 0, seed)
    ranges = label_ranges(report)
    tr = read_rows(CICIOT2023_V1, cap_per_label(pool, ranges, train_per_type), ["y", *feats])
    va = read_rows(CICIOT2023_V1, cap_per_label(np.sort(val_ids), ranges, val_per_type), ["y", *feats])
    Xtr, Xva = tr[feats].to_numpy(np.float64), va[feats].to_numpy(np.float64)
    lo, hi = np.percentile(Xtr, 1, axis=0), np.percentile(Xtr, 99, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    scale = lambda X: (np.pi * (np.clip(X, lo, hi) - lo) / span).astype(np.float32)   # noqa: E731
    return scale(Xtr), tr["y"].to_numpy(np.int64), scale(Xva), va["y"].to_numpy(np.int64), feats


def macro_f1(model, X, y):
    from sklearn.metrics import f1_score
    model.eval()
    with torch.no_grad():
        p = model(torch.as_tensor(X)).argmax(1).numpy()
    model.train()
    return float(f1_score(y, p, average="macro", labels=list(range(8)), zero_division=0))


def opacus_run(X, y, *, eps, delta, epochs, lr, clip, batch, seed):
    from opacus import PrivacyEngine
    torch.manual_seed(seed)
    model = build("mlp_p256", n_in=X.shape[1], n_classes=8, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = torch.utils.data.TensorDataset(torch.as_tensor(X), torch.as_tensor(y))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, generator=torch.Generator().manual_seed(seed))
    engine = PrivacyEngine(accountant="rdp")
    model, opt, loader = engine.make_private_with_epsilon(module=model, optimizer=opt, data_loader=loader,
                                                          target_epsilon=eps, target_delta=delta, epochs=epochs,
                                                          max_grad_norm=clip, poisson_sampling=True)
    for _ in range(epochs):
        for xb, yb in loader:
            if len(xb) == 0:
                continue
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
    return model, float(opt.noise_multiplier), float(engine.get_epsilon(delta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-type", type=int, default=2000)
    ap.add_argument("--audit-train-per-type", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--audit-epochs", type=int, default=30)
    ap.add_argument("--canaries", type=int, default=1000)
    ap.add_argument("--out", default="results/priv3/checks_A_f0.json")
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    delta, clip, batch, lr = 1e-6, 1.0, 256, 0.01
    t0 = time.time()
    X, y, Xv, yv, feats = load(a.train_per_type, 500)
    print(f"check 1 data: {len(X):,} training rows, {len(Xv):,} validation rows, {len(feats)} features")
    check1 = []
    for eps in (8.0, 1.0):
        m = build("mlp_p256", n_in=X.shape[1], n_classes=8, seed=0)
        r = dp_fit(m, X, y, 8, target_epsilon=eps, delta=delta, clip=clip, batch_size=batch, max_epochs=a.epochs,
                   lr=lr, seed=0, class_weighted=False)
        mo, sig_o, eps_o = opacus_run(X, y, eps=eps, delta=delta, epochs=a.epochs, lr=lr, clip=clip, batch=batch, seed=0)
        row = {"target_epsilon": eps, "ours": {"noise_multiplier": r.noise_multiplier, "epsilon": r.epsilon,
                                               "epsilon_prv": r.epsilon_prv, "val_macro_f1": macro_f1(m, Xv, yv)},
               "opacus": {"noise_multiplier": sig_o, "epsilon": eps_o, "val_macro_f1": macro_f1(mo, Xv, yv)}}
        row["passes"] = bool(abs(r.noise_multiplier / sig_o - 1) < 0.05 and abs(r.epsilon / eps_o - 1) < 0.10
                             and abs(row["ours"]["val_macro_f1"] - row["opacus"]["val_macro_f1"]) < 0.05)
        check1.append(row)
        print(f"  eps {eps}: sigma ours {r.noise_multiplier:.3f} / opacus {sig_o:.3f}; eps ours {r.epsilon:.3f} / "
              f"opacus {eps_o:.3f}; F1 ours {row['ours']['val_macro_f1']:.3f} / opacus {row['opacus']['val_macro_f1']:.3f}"
              f"  -> {'PASS' if row['passes'] else 'CHECK'}")

    Xa, ya, _, _, _ = load(a.audit_train_per_type, 1)
    print(f"check 3 data: {len(Xa):,} training rows + up to {a.canaries} canaries")
    check3 = []
    for eps in (float("inf"), 8.0, 1.0):
        r = run_audit(lambda s: build("mlp_p3000", n_in=Xa.shape[1], n_classes=8, seed=s), Xa, ya, 8,
                      m=a.canaries, target_epsilon=eps, delta=delta, seed=0, clip=clip, batch_size=batch,
                      max_epochs=a.audit_epochs, lr=lr)
        check3.append(r)
        print(f"  eps {eps}: accountant {r['accountant_epsilon']:.3f}, audit lower bound "
              f"{r['empirical_epsilon_lower_bound']:.3f} ({r['correct']}/{r['guesses']} correct) -> "
              f"{'PASS' if r['passes'] else 'FAIL'}")

    out = {"task": "PRIV-3 checks 1 and 3", "data": "CICIoT2023 protocol A fold 0, 35 features, min-max stand-in",
           "machine": platform.platform(), "torch": torch.__version__, "args": vars(a), "delta": delta, "clip": clip,
           "batch": batch, "lr": lr, "check1_vs_opacus": check1, "check3_canary_audit": check3,
           "all_pass": all(r["passes"] for r in check1 + check3), "seconds": round(time.time() - t0)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=lambda v: None if isinstance(v, float) and math.isinf(v) else v))
    print(f"\nall pass: {out['all_pass']}   wrote {a.out}")


if __name__ == "__main__":
    main()
