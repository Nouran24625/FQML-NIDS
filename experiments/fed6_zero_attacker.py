"""FED-6 zero-attacker control: what a robust aggregator costs when nobody attacks. Owner: Kareem.

docs/ATTACKS.md section 5: before any attack run, every robust aggregator is trained with no malicious client and
compared with FedAvg on per-family recall. Under the sites split BruteForce is held by 1 site of 10, so median or
trimmed mean may discard the only site that knows it. A defence is judged by this cost plus what it saves under
attack, never by the second number alone.

A pipeline check on protocol A fold 0, not a reported result:
  - training sample as in E0 (D11): at most --cap rows per attack type, spread over time; validation capped the same way
  - features: a stand-in for DATA-6 (top-8 by mutual information, clipped to the 1st-99th percentile, scaled to
    [0, pi]), fitted on the training sample only and applied unchanged to validation
  - splits: iid and dir0.5 (50 clients), sites (10 sites x 8 attack types), sitesfam (10 sites x 3 families)
  - aggregators: fedavg, median, trimmed_mean (beta 0.2)
  - scores on the validation block: macro-F1 and recall per family (8 classes); class weights 1/sqrt(count)

Writes results/fed6/zero_attacker_<protocol>_f<fold>_<model>_r<rounds>[_s<seed>][_xNumber].json
(_xNumber: Number excluded, the default since 24 Nov; files without it kept Number as a candidate input).

    python -m experiments.fed6_zero_attacker --fold 0 --rounds 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score

from experiments.fed3_partitions import GAP, REPORT, label_ranges_from_report, native_of
from fqml.data.folds import build_time_blocked, fold_rows, load_manifest, manifest_path
from fqml.data.label_maps import family_index, load_label_map
from fqml.data.rows import cap_per_label, read_rows
from fqml.fl.batched import predict_proba, run_fedavg
from fqml.fl.equivalence import sqrt_class_weights
from fqml.fl.partition import dirichlet_partition, iid_partition, sites_partition
from fqml.fl.robust import get_aggregator
from fqml.models.classical import build, is_torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/interim/ciciot2023_v1/ciciot2023_clean_v1.parquet"
AGGREGATORS = ("fedavg", "median", "trimmed_mean")
SPLITS = ("iid", "dir0.5", "sites", "sitesfam")
# D9 (IAT, accepted) and its proposed addition (Number: the packet-window size, a recording setting that is a
# shortcut under protocol B, results/checkpoint1_no_number.json). Same default as the DATA-6 spec's `exclude`.
EXCLUDE = ("IAT", "Number")


def input_features(report, exclude=EXCLUDE):
    """Candidate input columns: the cleaner's features minus the excluded ones (IAT is always excluded, D9)."""
    drop = {"IAT", *exclude}
    unknown = drop - set(report["features"])
    if unknown:
        raise SystemExit(f"--exclude names unknown features: {sorted(unknown)}")
    return [c for c in report["features"] if c not in drop]


def exclude_tag(exclude):
    """File-name tag: empty for IAT only (the first runs' names), '_xNumber' when Number is dropped too."""
    extra = sorted(set(exclude) - {"IAT"})
    return "".join(f"_x{c.replace(' ', '')}" for c in extra)


def parse_exclude(text):
    return tuple(c.strip() for c in text.split(",") if c.strip())


def load_block(m, fold, role, ranges, cap, feats):
    rows = cap_per_label(fold_rows(m, fold, role), ranges, cap)
    native = native_of(rows, ranges)
    F = read_rows(DATA, rows, feats)[feats].to_numpy(np.float64)
    return F, native, family_index(native, "ciciot2023")


def fit_inputs(F, y, n_features=8, seed=0):
    """Stand-in for DATA-6, fitted on training rows only. Returns a transform for any block."""
    from sklearn.feature_selection import mutual_info_classif
    sub = np.random.default_rng(seed).choice(len(F), min(20_000, len(F)), replace=False)
    varying = np.flatnonzero(F[sub].std(axis=0) > 0)
    mi = mutual_info_classif(F[sub][:, varying], y[sub], random_state=seed)
    top = np.sort(varying[np.argsort(mi)[::-1][:n_features]])
    lo, hi = np.percentile(F[:, top], [1, 99], axis=0)
    hi = np.where(hi > lo, hi, lo + 1.0)
    return top, lambda G: (np.clip(G[:, top], lo, hi) - lo) / (hi - lo) * np.pi


LR = 0.01                       # the first runs' learning rate; the VQC needs a tuned one (results/FED_FIRST_PASS.md 1b)
FEATURE_SOURCES = ("auto", "stand-in", "data6")


def preprocess_path(protocol, fold):
    return ROOT / f"results/preprocess/ciciot2023_{protocol}_f{fold}.json"


def prepare_inputs(m, fold, ranges, report, feats, exclude, Ftr, ytr, *, protocol, source="auto"):
    """The 8 model inputs of this fold. Returns (names, transform, source_label); transform maps a block read with
    `feats` columns ([n, len(feats)]) to [n, 8] in [0, pi].

    source "auto": DATA-6 (fqml.data.preprocess, Ganna) when it exists, else the stand-in `fit_inputs`.
    With DATA-6 the fitted settings are read from results/preprocess/ciciot2023_<protocol>_f<fold>.json when that
    file exists; otherwise they are fitted here exactly as the spec says (docs/specs/DATA-6_preprocessing.md §2: at
    most 20,000 training rows per attack type, spread over time, every feature of the report, `exclude` dropped).
    Every model of a run gets the same inputs, which is what keeps the comparison fair."""
    import importlib.util
    if source not in FEATURE_SOURCES:
        raise SystemExit(f"--features must be one of {FEATURE_SOURCES}")
    have = importlib.util.find_spec("fqml.data.preprocess") is not None
    if source == "data6" and not have:
        raise SystemExit("--features data6: fqml.data.preprocess (DATA-6) is not in this checkout yet")
    if source == "stand-in" or not have:
        top, transform = fit_inputs(Ftr, ytr)
        return [feats[i] for i in top], transform, "stand-in (DATA-6 pending)"

    import pandas as pd

    from fqml.data.preprocess import Preprocessor, fit
    all_feats = list(report["features"])
    path = preprocess_path(protocol, fold)
    if path.exists():
        pre, origin = Preprocessor.from_json(path), str(path.relative_to(ROOT))
    else:
        rows = cap_per_label(fold_rows(m, fold, "train"), ranges, 20_000)          # the spec's fitting sample
        df = read_rows(DATA, rows, all_feats)[all_feats]
        y_fit = family_index(native_of(rows, ranges), "ciciot2023")
        pre, origin = fit(df, y_fit, features=all_feats, exclude=tuple(exclude)), "fitted here from the spec"

    def transform(F):
        df = pd.DataFrame(np.asarray(F), columns=feats)
        for c in all_feats:                    # excluded columns are never read by the preprocessor; present as 0
            if c not in df:
                df[c] = 0.0
        return np.asarray(pre.transform(df[all_feats], feature_set="top8"), dtype=np.float64)

    return list(pre.top8), transform, f"DATA-6 ({origin})"


def run_tag(lr, source_label):
    """File-name tag for settings that differ from the first runs: '_lr0.003' and '_d6' (DATA-6 inputs)."""
    return (f"_lr{lr:g}" if lr != LR else "") + ("_d6" if source_label.startswith("DATA-6") else "")


def splits(native, y, seed):
    _, fam_of = load_label_map("ciciot2023")
    return {"iid": iid_partition(len(y), 50, seed),
            "dir0.5": dirichlet_partition(y, 50, 0.5, seed),
            "sites": sites_partition(native, 10, 8, seed)[0],
            "sitesfam": sites_partition(native, 10, 3, seed,
                                        groups={t: fam_of[t] for t in fam_of if t != "BenignTraffic"})[0]}


# Names the federated runs accept besides the E0 names: "mlp" is the VQC-sized MLP of the first runs (= mlp_p64).
ALIASES = {"mlp": "mlp_p64"}


def make_model(kind, n_classes, **settings):
    """Every federated model is built by the E0 model factory (fqml.models.classical.build, CLS-1), so a federated
    `vqc` or `mlp_p64` is exactly the model E0 trains centrally: vqc, vqc_lin, vqc_noent, mlp_p<N>, mlp_large, rff,
    logreg. Tree and scikit-learn models (lr, rf, xgb) have no weights to average: FED-5 (D12) handles them."""
    name = ALIASES.get(kind, kind)
    if not is_torch(name):
        raise SystemExit(f"unknown or non-PyTorch model {kind!r} (federated models: vqc, vqc_lin, vqc_noent, "
                         "mlp_p<N>, mlp_large, rff, logreg; trees: experiments/fed5_trees.py)")
    return build(name, n_in=8, n_classes=n_classes, seed=0, **settings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--protocol", default="A")
    ap.add_argument("--cap", type=int, default=5000, help="rows per attack type in the training sample")
    ap.add_argument("--val-cap", type=int, default=1000)
    ap.add_argument("--model", default="mlp", help="vqc, vqc_lin, vqc_noent, mlp (= mlp_p64), mlp_p<N>, rff, logreg")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--aggregators", default=",".join(AGGREGATORS))
    ap.add_argument("--exclude", default=",".join(EXCLUDE), help="features never used as inputs (IAT always)")
    ap.add_argument("--lr", type=float, default=LR, help="local learning rate (the VQC needs a tuned one)")
    ap.add_argument("--features", default="auto", help="auto (DATA-6 when present, else stand-in), stand-in, data6")
    a = ap.parse_args()
    exclude = parse_exclude(a.exclude)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    t0 = time.perf_counter()
    report = json.loads(REPORT.read_text())
    families = report["families"]
    feats = input_features(report, exclude)                          # D9 and its Number addition
    ranges = label_ranges_from_report(report)
    path = manifest_path("ciciot2023", a.protocol)
    if path.exists():
        m = load_manifest(path)
    elif a.protocol == "A":                    # same builder and settings as data/folds/ciciot2023_A.json (D8a, #18)
        m = build_time_blocked(ranges, dataset="ciciot2023", n_folds=5, gap=GAP)
    else:
        raise SystemExit(f"{path} not found")
    print(f"[data] reading protocol {a.protocol} fold {a.fold}: train cap {a.cap}, validation cap {a.val_cap} ...",
          flush=True)
    Ftr, native, ytr = load_block(m, a.fold, "train", ranges, a.cap, feats)
    Fva, _, yva = load_block(m, a.fold, "val", ranges, a.val_cap, feats)
    names, transform, source = prepare_inputs(m, a.fold, ranges, report, feats, exclude, Ftr, ytr,
                                              protocol=a.protocol, source=a.features)
    Xtr, Xva = transform(Ftr), transform(Fva)
    print(f"[data] train {len(ytr):,} rows, validation {len(yva):,} rows; features {names} [{source}] "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    w = sqrt_class_weights(ytr, len(families)).float()
    parts_by = splits(native, ytr, a.seed)

    runs = []
    todo = [(s, g) for s in a.splits.split(",") for g in a.aggregators.split(",")]
    for i, (split, agg) in enumerate(todo, 1):
        parts = parts_by[split]
        cpr = min(20, len(parts))
        print(f"\n[{i}/{len(todo)}] split={split} ({len(parts)} clients, {cpr} per round)  aggregator={agg}",
              flush=True)

        def on_round(e, total=a.rounds):
            f1 = e.get("val_macro_f1")
            print(f"    round {e['round'] + 1:>3}/{total}  {e['seconds']:5.1f}s"
                  + (f"  val macro-F1 {f1:.3f}" if f1 is not None else ""), flush=True)

        t1 = time.perf_counter()
        res = run_fedavg(make_model(a.model, len(families)), Xtr, ytr, parts, rounds=a.rounds, lr=a.lr,
                         clients_per_round=cpr, local_steps=a.local_steps, class_weights=w, seed=a.seed,
                         aggregator=get_aggregator(agg), X_val=Xva, y_val=yva, on_round=on_round)
        pred = predict_proba(make_model(a.model, len(families)), res.params,
                             torch.as_tensor(Xva, dtype=torch.float32)).argmax(axis=1)
        rec = recall_score(yva, pred, labels=range(len(families)), average=None, zero_division=0)
        run = {"split": split, "aggregator": agg, "n_clients": len(parts), "clients_per_round": cpr,
               "macro_f1": float(f1_score(yva, pred, average="macro")),
               "recall": {f: float(r) for f, r in zip(families, rec)},
               "seconds": time.perf_counter() - t1}
        runs.append(run)
        print(f"    -> macro-F1 {run['macro_f1']:.3f} | " +
              " ".join(f"{f} {r:.2f}" for f, r in run["recall"].items()), flush=True)

    # cost of each robust aggregator against FedAvg on the same split
    print("\n=== Recall change against FedAvg on the same split (negative = the defence loses detection) ===")
    print(f"{'split':10s} {'aggregator':13s} {'dF1':>7s} " + " ".join(f"{f[:6]:>7s}" for f in families))
    base = {r["split"]: r for r in runs if r["aggregator"] == "fedavg"}
    for r in runs:
        if r["aggregator"] == "fedavg" or r["split"] not in base:
            continue
        b = base[r["split"]]
        r["delta_vs_fedavg"] = {"macro_f1": r["macro_f1"] - b["macro_f1"],
                                **{f: r["recall"][f] - b["recall"][f] for f in families}}
        d = r["delta_vs_fedavg"]
        print(f"{r['split']:10s} {r['aggregator']:13s} {d['macro_f1']:+7.3f} " +
              " ".join(f"{d[f]:+7.2f}" for f in families))

    seed_tag = f"_s{a.seed}" if a.seed else ""          # seed 0 keeps the first runs' file names
    out = ROOT / f"results/fed6/zero_attacker_{a.protocol}_f{a.fold}_{a.model}_r{a.rounds}{seed_tag}{exclude_tag(exclude)}{run_tag(a.lr, source)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": "FED-6 zero-attacker control", "protocol": a.protocol, "fold": a.fold,
                               "model": a.model, "cap": a.cap, "val_cap": a.val_cap, "rounds": a.rounds,
                               "local_steps": a.local_steps, "seed": a.seed, "families": families,
                               "exclude": sorted({"IAT", *exclude}), "lr": a.lr, "features": names,
                               "features_source": source, "runs": runs}, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}  (total {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
