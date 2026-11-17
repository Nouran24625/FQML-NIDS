"""Checkpoint 1 (task DATA-5): how much does the evaluation split change the score? Owner: Ganna.

Phase 0 scored models on a random row split. Under protocol A (time-blocked folds, INTERFACES 1.2) a
model is tested on a later stretch of the same attack, which is closer to how an IDS is used. This
script trains ONE strong classical model (XGBoost, fixed settings, no tuning) under:

  split     R            old random row split (60 / 20 / 20), the leakage reference
            A-gap0       protocol A, fold `--fold`, no rows dropped at block boundaries
            A-gap100     protocol A with the proposed gap (D8a)
            A-gap1000    protocol A with a 10x larger gap
  features  all          the 37 features of cleaner v1
            no_iat       without IAT (decision D9)

and scores every setting on the VALIDATION block only, so the test blocks stay unused for the real
experiments. Scores are given on all validation rows, on rows with no exact twin among the training rows
the model saw, and on rows with conflict == 0 (D8), for the 8-class task, the merged 7-class task (c7m,
decision D17) and the binary task (attack vs benign).

What each comparison answers:
  R vs A-gap100            how much a random split inflates scores (the Phase 0 question)
  A-gap0 / 100 / 1000      whether the gap is big enough (the score should stop falling)
  all vs no_iat            what dropping IAT costs (D9)
  8-class vs c7m           how much of the error is the DDoS / DoS confusion (D17)

Training uses at most `--train-per-type` rows of each attack type from the training side (spread over
time for protocol A, random for R); validation uses every row of the block. Fixed seed, fixed settings.

    python -m fqml.eval.checkpoint1                      # all 8 settings, fold 0, about 1 hour on a laptop
    python -m fqml.eval.checkpoint1 --splits A-gap100 --features all     # one setting

Writes results/checkpoint1.json (and prints a table). Needs the cleaned file v1, its report, xgboost.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from fqml.data.folds import build_time_blocked, fold_rows, ranges_to_rows, twin_mask
from fqml.data.rows import CICIOT2023_V1, cap_per_label, read_rows

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/data_reports/ciciot2023_clean_v1.json"
OUT = ROOT / "results/checkpoint1.json"
SPLITS = ("R", "A-gap0", "A-gap100", "A-gap1000")
FEATURE_SETS = ("all", "no_iat")
XGB_PARAMS = dict(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", max_bin=256, early_stopping_rounds=20, eval_metric="mlogloss")


# ---- splits ---------------------------------------------------------------------------------------
def label_ranges(report: dict) -> dict[str, tuple[int, int]]:
    return {lab: (d["row_start"], d["row_stop"]) for lab, d in report["per_label"].items() if d["rows_written"] > 0}


def split_rows(split: str, report: dict, fold: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(training pool, validation rows) as sorted row_ids."""
    ranges = label_ranges(report)
    if split == "R":
        n = report["totals"]["rows_written"]
        u = np.random.default_rng(seed).random(n, dtype=np.float32)
        return np.flatnonzero(u < 0.6), np.flatnonzero((u >= 0.6) & (u < 0.8))
    if not split.startswith("A-gap"):
        raise ValueError(f"unknown split {split!r}")
    m = build_time_blocked(ranges, dataset="ciciot2023", n_folds=5, gap=int(split[5:]))
    return fold_rows(m, fold, "train"), fold_rows(m, fold, "val")


# ---- scoring --------------------------------------------------------------------------------------
def _scores(y, p, n_classes):
    labels = list(range(n_classes))
    return {"macro_f1": float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
            "per_class_f1": [float(v) for v in f1_score(y, p, labels=labels, average=None, zero_division=0)],
            "rows": int(len(y))}


def merge_ddos_dos(y: np.ndarray, families: list[str]) -> tuple[np.ndarray, list[str]]:
    """8 families -> c7m: DoS joins DDoS, later families move down by one."""
    i_ddos, i_dos = families.index("DDoS"), families.index("DoS")
    y7 = np.where(y == i_dos, i_ddos, y)
    y7 = np.where(y7 > i_dos, y7 - 1, y7)
    fam7 = [f if f != "DDoS" else "DDoS_DoS" for f in families if f != "DoS"]
    return y7, fam7


def score(y: np.ndarray, p: np.ndarray, families: list[str]) -> dict:
    """8-class, c7m and binary scores, plus the DDoS <-> DoS confusion."""
    k = len(families)
    y7, fam7 = merge_ddos_dos(y, families)
    p7, _ = merge_ddos_dos(p, families)
    yb, pb = (y != 0).astype(int), (p != 0).astype(int)
    i_ddos, i_dos = families.index("DDoS"), families.index("DoS")
    cm = confusion_matrix(y, p, labels=list(range(k)))
    dos_rows, ddos_rows = max(cm[i_dos].sum(), 1), max(cm[i_ddos].sum(), 1)
    benign = max(int((yb == 0).sum()), 1)
    return {
        "c8": _scores(y, p, k),
        "c7m": _scores(y7, p7, len(fam7)),
        "binary": {"f1": float(f1_score(yb, pb, zero_division=0)),
                   "attack_recall": float(recall_score(yb, pb, zero_division=0)),
                   "precision": float(precision_score(yb, pb, zero_division=0)),
                   "false_positive_rate": float(((pb == 1) & (yb == 0)).sum() / benign)},
        "dos_predicted_ddos": float(cm[i_dos, i_ddos] / dos_rows),
        "ddos_predicted_dos": float(cm[i_ddos, i_dos] / ddos_rows),
        "confusion_c8": cm.tolist(),
    }


# ---- one setting ----------------------------------------------------------------------------------
def run_setting(split, feature_set, *, data, report, fold, seed, train_per_type, threads, log=print) -> dict:
    import xgboost as xgb

    families = report["families"]
    feats = [f for f in report["features"] if not (feature_set == "no_iat" and f == "IAT")]
    t0 = time.time()
    pool, val_ids = split_rows(split, report, fold, seed)
    rng = np.random.default_rng(seed) if split == "R" else None
    train_ids = cap_per_label(pool, label_ranges(report), train_per_type, rng)
    cols = ["y", "dup_group", "conflict", *feats]
    tr = read_rows(data, train_ids, cols)
    log(f"  {split:9s} {feature_set:6s}: {len(tr):,} training rows from a pool of {len(pool):,}; "
        f"validation {len(val_ids):,} rows")

    # balanced families: every family carries the same total weight
    y = tr["y"].to_numpy(np.int64)
    counts = np.bincount(y, minlength=len(families)).astype(float)
    if (counts == 0).any():
        raise RuntimeError(f"{split}: a family has no training rows: {dict(zip(families, counts))}")
    w = (len(y) / (len(families) * counts))[y]
    # early stopping on 10% of the training rows (never on the validation block)
    inner = np.random.default_rng(seed + 1).random(len(y)) < 0.1
    X = tr[feats].to_numpy(np.float32)
    model = xgb.XGBClassifier(**XGB_PARAMS, n_jobs=threads, random_state=seed)
    model.fit(X[~inner], y[~inner], sample_weight=w[~inner], eval_set=[(X[inner], y[inner])],
              sample_weight_eval_set=[w[inner]], verbose=False)
    t_train = time.time() - t0
    train_dups = np.unique(tr["dup_group"].to_numpy())
    del tr, X

    # validation, read and scored in chunks so 9.35 M rows never sit in memory twice
    ys, ps, twins, conf = [], [], [], []
    for chunk in np.array_split(val_ids, max(1, len(val_ids) // 2_000_000)):
        va = read_rows(data, chunk, cols)
        ys.append(va["y"].to_numpy(np.int64))
        ps.append(model.predict(va[feats].to_numpy(np.float32)).astype(np.int64))
        twins.append(np.isin(va["dup_group"].to_numpy(), train_dups))
        conf.append(va["conflict"].to_numpy())
    yv, pv, tw, cf = (np.concatenate(a) for a in (ys, ps, twins, conf))
    res = {
        "split": split, "features": feature_set, "n_features": len(feats), "fold": fold, "seed": seed,
        "train_rows": int(len(y)), "train_pool_rows": int(len(pool)), "val_rows": int(len(yv)),
        "trees": int(model.best_iteration + 1) if getattr(model, "best_iteration", None) is not None else None,
        "val_twin_share": float(tw.mean()), "val_conflict_share": float((cf != 0).mean()),
        "scores_all": score(yv, pv, families),
        "scores_no_twin": score(yv[~tw], pv[~tw], families),
        "scores_no_conflict": score(yv[cf == 0], pv[cf == 0], families),
        "seconds_train": round(t_train, 1), "seconds_total": round(time.time() - t0, 1),
    }
    if feature_set == "all":
        imp = model.feature_importances_
        res["top_features"] = [[feats[i], float(imp[i])] for i in np.argsort(imp)[::-1][:10]]
    return res


def _row(r):
    a, n, c = r["scores_all"], r["scores_no_twin"], r["scores_no_conflict"]
    return (f"{r['split']:10s}{r['features']:8s}{a['c8']['macro_f1']:8.3f}{n['c8']['macro_f1']:8.3f}"
            f"{c['c8']['macro_f1']:8.3f}{a['c7m']['macro_f1']:8.3f}{a['binary']['f1']:8.3f}"
            f"{a['dos_predicted_ddos']:9.2f}{r['val_twin_share']:7.1%}")


def main():
    ap = argparse.ArgumentParser(description="Checkpoint 1: XGBoost under the random split and protocol A.")
    ap.add_argument("--data", default=str(CICIOT2023_V1))
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--features", nargs="+", default=list(FEATURE_SETS), choices=FEATURE_SETS)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-per-type", type=int, default=50_000)
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    import xgboost as xgb
    report = json.loads(Path(a.report).read_text())
    out = Path(a.out)
    done = json.loads(out.read_text())["runs"] if out.exists() else []
    def key(r):
        return r["split"], r["features"], r["fold"], r["seed"]
    have = {key(r) for r in done}
    threads = a.threads or None
    for split in a.splits:
        for fs in a.features:
            if (split, fs, a.fold, a.seed) in have:  # finished runs are kept, so a stopped run can resume
                print(f"  {split} {fs}: already in {out.name}, skipped")
                continue
            r = run_setting(split, fs, data=a.data, report=report, fold=a.fold, seed=a.seed,
                            train_per_type=a.train_per_type, threads=threads)
            done = [d for d in done if key(d) != key(r)] + [r]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "task": "DATA-5 Checkpoint 1", "model": "XGBoost, fixed settings", "xgb_params": XGB_PARAMS,
                "xgboost": xgb.__version__, "machine": platform.platform(), "train_per_type": a.train_per_type,
                "scored_on": "validation block only", "runs": done}, indent=1))
            print("    " + _row(r) + f"   ({r['seconds_total']:.0f} s)", flush=True)
    print(f"\n{'split':10s}{'feat':8s}{'c8':>8s}{'c8 -tw':>8s}{'c8 -cf':>8s}{'c7m':>8s}{'bin F1':>8s}"
          f"{'DoS>DDoS':>9s}{'twins':>7s}")
    order = {s: i for i, s in enumerate(SPLITS)}
    for r in sorted(done, key=lambda r: (order.get(r["split"], 9), r["features"])):
        print(_row(r))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
