"""Summary of the federated grid over folds and seeds (results/fedgrid_25nov, workstation run of 25 Nov).
Owner: Kareem. Means, standard deviations and win counts per split, aggregator and model; prints a text report.

    python -m experiments.fed_grid_summary [results folder]
"""
import glob
import json
import sys

import numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else "results/fedgrid_25nov"
def ms(v): v = np.array(v, float); return f"{v.mean():.2f} ± {v.std(ddof=1) if len(v)>1 else 0:.2f} (n={len(v)})"
print("== FED-2 attack grid, mlp_p3000, protocol A, 5 folds x 3 seeds")
runs = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/fed2/attacks_A_f*_mlp_p3000_s*_xNumber.json"))]
print("files", len(runs))
def get(d, split, attack, agg):
    return next(r for r in d["runs"] if r["split"] == split and r["attack"] == attack and r["aggregator"] == agg)
for split in ("iid", "sites", "sitesfam"):
    for agg in ("fedavg", "median", "trimmed_mean"):
        none = [get(d, split, "none", agg)["recall"]["Web"] for d in runs]
        cs = [get(d, split, "cs20web", agg)["recall"]["Web"] for d in runs]
        lf = [get(d, split, "lf20", agg)["macro_f1"] - get(d, split, "none", agg)["macro_f1"] for d in runs]
        mi = [get(d, split, "cs20mirai", agg)["recall"]["Mirai"] for d in runs]
        print(f"{split:9s} {agg:13s} Web none {ms(none)} | Web under suppression {ms(cs)} | label-flip dF1 {ms(lf)} | Mirai supp {np.mean(mi):.2f}")
    # paired: robust beats fedavg on Web under suppression
    for agg in ("median", "trimmed_mean"):
        d_ = [get(d, split, "cs20web", agg)["recall"]["Web"] - get(d, split, "cs20web", "fedavg")["recall"]["Web"] for d in runs]
        print(f"   {split} {agg} - fedavg, Web under suppression: mean {np.mean(d_):+.2f}, better in {sum(x > 0.02 for x in d_)}/{len(d_)}, worse in {sum(x < -0.02 for x in d_)}")
print()
print("== FED-6 zero-attacker control, protocol A, 5 folds x 3 seeds")
for model in ("mlp", "mlp_p3000"):
    fs = sorted(glob.glob(f"{R}/fed6/zero_attacker_A_f*_{model}_r100*_xNumber.json"))
    ds = [json.load(open(f)) for f in fs]
    print(model, "files", len(ds))
    for split in ("iid", "dir0.5", "sites", "sitesfam"):
        for agg in ("median", "trimmed_mean"):
            dd = [next(r for r in d["runs"] if r["split"] == split and r["aggregator"] == agg)["delta_vs_fedavg"]["macro_f1"] for d in ds]
            print(f"   {split:9s} {agg:13s} dF1 vs fedavg {ms(dd)}; lower (< -0.01) in {sum(x < -0.01 for x in dd)}/{len(dd)}, higher (> 0.01) in {sum(x > 0.01 for x in dd)}")
print()
for P in ("A", "B"):
    print(f"== FED-4 protocol {P}, 5 folds, seed 0")
    fs = sorted(glob.glob(f"{R}/fed4/trial_{P}_f*_r100_mlp_p3000-mlp_xNumber.json"))
    ds = [json.load(open(f)) for f in fs]
    print("files", len(ds))
    for model in ("mlp_p3000", "mlp"):
        for split in ("sites", "sitesfam", "iid"):
            rr = [next(r for r in d["runs"] if r["model"] == model and r["split"] == split) for d in ds]
            F = lambda r, m: r[m].get("macro_f1", r[m].get("macro_f1_mean"))
            f1 = {m: [F(r, m) for r in rr] for m in ("fedavg", "local", "central")}
            line = f"  {model:9s} {split:8s} F1 fa {ms(f1['fedavg'])} lo {ms(f1['local'])} ce {ms(f1['central'])}"
            if split != "iid" or P == "B":
                key = "nowhere_family_recall" if P == "B" else "family_recall"
                dk = "nowhere_detection_at_1pct_fpr" if P == "B" else "detection_at_1pct_fpr"
                fam = {m: [r[m]["unseen"][key]["mean"] for r in rr] for m in ("fedavg", "local", "central")}
                det = {m: [r[m]["unseen"][dk]["mean"] for r in rr] for m in ("fedavg", "local", "central")}
                gain = [a - b for a, b in zip(fam["fedavg"], fam["local"])]
                line += (f"\n            family fa {ms(fam['fedavg'])} lo {ms(fam['local'])} ce {ms(fam['central'])}; fa>lo in {sum(g > 0.02 for g in gain)}/{len(gain)}, fa<lo in {sum(g < -0.02 for g in gain)}"
                         f"\n            detect fa {ms(det['fedavg'])} lo {ms(det['local'])} ce {ms(det['central'])}")
            print(line)
