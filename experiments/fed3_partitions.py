"""FED-3: build the federated client splits on one fold's training sample and chart them. Owner: Kareem.

Needs no data file: in the cleaned CICIoT2023 file every attack type is a contiguous row_id range
(INTERFACES 1.1), so a row's native label follows from its row_id and the cleaning report.

Training sample = the same one every E0 model uses (D11): protocol A fold `--fold`, at most N = 5,000 rows per
attack type, spread evenly over time (`fqml.data.rows.cap_per_label`). Partitions are positions into that sample.

Schemes (INTERFACES 3.1):
  iid          50 clients, equal random shares
  dir1, dir0.5, dir0.1   50 clients, Dirichlet label skew over the 8 families
  type         one client per attack type (33), benign spread over all (the most extreme skew)
  sites        10 sites x 8 attack types each, every type held by >= 1 site
  sitesfam     10 sites x 3 whole families each (a site misses entire families)

Writes results/partitions/fed3_<protocol>_f<fold>.json (per-client class counts, sites' attack types, summary)
and figures/fed3_*.png. With --save-npz also data/partitions/<dataset>_<protocol>_f<fold>_<scheme>_n<N>_s<seed>.npz.

    python -m experiments.fed3_partitions --fold 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fqml.data.folds import build_time_blocked, fold_rows, load_manifest, manifest_path
from fqml.data.label_maps import family_index, load_label_map
from fqml.data.rows import cap_per_label
from fqml.fl.partition import dirichlet_partition, iid_partition, save_partition, sites_partition, type_partition

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/data_reports/ciciot2023_clean_v1.json"
GAP = 100                                     # D8a

# Categorical slots in fixed order (dataviz reference palette); Benign is the neutral gray.
COLORS = ["#9b9a97", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def label_ranges_from_report(report: dict) -> dict[str, tuple[int, int]]:
    return {lab: (d["row_start"], d["row_stop"]) for lab, d in report["per_label"].items() if d["rows_written"] > 0}


def native_of(row_ids: np.ndarray, ranges: dict[str, tuple[int, int]]) -> np.ndarray:
    labs = sorted(ranges, key=lambda k: ranges[k][0])
    starts = np.array([ranges[k][0] for k in labs])
    stops = np.array([ranges[k][1] for k in labs])
    i = np.searchsorted(starts, row_ids, side="right") - 1
    if (i < 0).any() or (row_ids >= stops[i]).any():
        raise ValueError("row_id outside every attack type's range")
    return np.array(labs, dtype=object)[i]


def training_sample(fold: int, cap: int, protocol: str = "A"):
    report = json.loads(REPORT.read_text())
    ranges = label_ranges_from_report(report)
    path = manifest_path("ciciot2023", protocol)
    if path.exists():
        m = load_manifest(path)
    elif protocol == "A":                     # same builder and settings as data/folds/ciciot2023_A.json (D8a)
        m = build_time_blocked(ranges, dataset="ciciot2023", n_folds=5, gap=GAP)
    else:
        raise SystemExit(f"{path} not found")
    rows = cap_per_label(fold_rows(m, fold, "train"), ranges, cap)
    native = native_of(rows, ranges)
    return rows, native, family_index(native, "ciciot2023")


def build(native, y, seed: int, n_clients: int, n_sites: int, types_per_site: int, fams_per_site: int):
    _, fam_of = load_label_map("ciciot2023")
    out = {"iid": (iid_partition(len(y), n_clients, seed), None)}
    for a in (1.0, 0.5, 0.1):
        out[f"dir{a:g}"] = (dirichlet_partition(y, n_clients, a, seed), None)
    out["type"] = (type_partition(native, seed), None)
    out["sites"] = sites_partition(native, n_sites, types_per_site, seed)
    out["sitesfam"] = sites_partition(native, n_sites, fams_per_site, seed,
                                      groups={t: fam_of[t] for t in fam_of if t != "BenignTraffic"})
    return out


def summarise(schemes, native, y, families):
    res = {}
    web, bf = families.index("Web"), families.index("BruteForce")
    for name, (parts, site_types) in schemes.items():
        counts = np.stack([np.bincount(y[p], minlength=len(families)) for p in parts])
        attack_types = sorted(set(native.tolist()) - {"BenignTraffic"})
        held = [set(native[p].tolist()) - {"BenignTraffic"} for p in parts]
        unseen = [len(attack_types) - len(h) for h in held]
        fam_missing = [int(((c[1:]) == 0).sum()) for c in counts]
        res[name] = {
            "n_clients": len(parts),
            "rows_per_client": {"min": int(counts.sum(1).min()), "median": float(np.median(counts.sum(1))),
                                "max": int(counts.sum(1).max())},
            "clients_with_any_web": int((counts[:, web] > 0).sum()),
            "clients_with_any_bruteforce": int((counts[:, bf] > 0).sum()),
            "attack_types_unseen_per_client": {"min": int(min(unseen)), "mean": float(np.mean(unseen)),
                                               "max": int(max(unseen))},
            "attack_families_missing_per_client": {"min": int(min(fam_missing)), "mean": float(np.mean(fam_missing)),
                                                   "max": int(max(fam_missing))},
            "class_counts": counts.tolist(),
            "site_types": site_types,
        }
    return res


def plot(summary, families, out_dir: Path, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
                         "axes.spines.right": False})
    names = list(summary)
    titles = {"iid": "IID", "dir1": "Dirichlet α = 1", "dir0.5": "Dirichlet α = 0.5", "dir0.1": "Dirichlet α = 0.1",
              "type": "One attack type per client", "sites": "Sites: 8 attack types each",
              "sitesfam": "Sites: 3 whole families each"}
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2.0 * len(names)), constrained_layout=True)
    for ax, name in zip(axes, names):
        c = np.array(summary[name]["class_counts"], dtype=float)
        order = np.argsort(-c.sum(1), kind="stable")
        c = c[order]
        x = np.arange(len(c))
        bottom = np.zeros(len(c))
        for k, fam in enumerate(families):
            ax.bar(x, c[:, k], bottom=bottom, width=0.8, color=COLORS[k], edgecolor="#fcfcfb", linewidth=0.6,
                   label=fam)
            bottom += c[:, k]
        ax.set_title(f"{titles.get(name, name)}  ·  {len(c)} clients", loc="left", fontsize=9.5, color="#0b0b0b")
        ax.set_xlim(-0.6, len(c) - 0.4)
        ax.set_xticks([])
        ax.set_ylabel("rows")
        ax.yaxis.grid(True, color="#ecebe8", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(ncol=8, loc="lower left", bbox_to_anchor=(0, 1.25), frameon=False, fontsize=8.5,
                   handlelength=1.0, columnspacing=1.2)
    axes[-1].set_xlabel("client (sorted by size)")
    fig.savefig(out_dir / f"fed3_rows_per_client_{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 3.4), constrained_layout=True)
    x = np.arange(len(names))
    share_web = [summary[n]["clients_with_any_web"] / summary[n]["n_clients"] for n in names]
    share_bf = [summary[n]["clients_with_any_bruteforce"] / summary[n]["n_clients"] for n in names]
    b1 = ax.bar(x - 0.2, share_web, 0.38, color=COLORS[4], label="Web", edgecolor="#fcfcfb", linewidth=1)
    b2 = ax.bar(x + 0.2, share_bf, 0.38, color=COLORS[5], label="BruteForce", edgecolor="#fcfcfb", linewidth=1)
    for bars, key in ((b1, "clients_with_any_web"), (b2, "clients_with_any_bruteforce")):
        for bar, n in zip(bars, names):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{summary[n][key]}/{summary[n]['n_clients']}", ha="center", va="bottom", fontsize=7.5,
                    color="#52514e")
    short = {"iid": "IID", "dir1": "Dir α=1", "dir0.5": "Dir α=0.5", "dir0.1": "Dir α=0.1",
             "type": "1 type\nper client", "sites": "Sites\n8 types each", "sitesfam": "Sites\n3 families each"}
    ax.set_xticks(x, [short.get(n, n) for n in names], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("share of clients holding any rows")
    ax.yaxis.grid(True, color="#ecebe8", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.set_title("Clients that hold any Web or BruteForce rows", loc="left", fontsize=9.5, color="#0b0b0b")
    fig.savefig(out_dir / f"fed3_rare_families_{tag}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="FED-3 client splits and charts.")
    ap.add_argument("--protocol", default="A")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cap", type=int, default=5000, help="rows per attack type (D11 part 2: N = 5,000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-clients", type=int, default=50)
    ap.add_argument("--n-sites", type=int, default=10)
    ap.add_argument("--types-per-site", type=int, default=8)
    ap.add_argument("--families-per-site", type=int, default=3)
    ap.add_argument("--save-npz", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    rows, native, y = training_sample(a.fold, a.cap, a.protocol)
    families, _ = load_label_map("ciciot2023")
    schemes = build(native, y, a.seed, a.n_clients, a.n_sites, a.types_per_site, a.families_per_site)
    summary = summarise(schemes, native, y, families)
    tag = f"{a.protocol}_f{a.fold}"
    out = {"task": "FED-3", "dataset": "ciciot2023", "protocol": a.protocol, "fold": a.fold, "cap": a.cap,
           "seed": a.seed, "sample_rows": int(len(rows)), "families": families,
           "sample_class_counts": np.bincount(y, minlength=len(families)).tolist(), "schemes": summary}
    rdir = ROOT / "results/partitions"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / f"fed3_{tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    if not a.no_plot:
        fdir = ROOT / "results/partitions/figures"
        fdir.mkdir(parents=True, exist_ok=True)
        plot(summary, families, fdir, tag)
    if a.save_npz:
        for name, (parts, _) in schemes.items():
            n = len(parts)
            save_partition(parts, ROOT / f"data/partitions/ciciot2023_{a.protocol}_f{a.fold}_{name}_n{n}_s{a.seed}.npz")
    print(f"{len(rows):,} sample rows, fold {a.fold}, protocol {a.protocol}")
    print(f"{'scheme':10s} {'clients':>7s} {'rows min/med/max':>20s} {'Web':>6s} {'BF':>6s} {'types unseen':>14s} {'fams missing':>13s}")
    for n, s in summary.items():
        r, u, f = s["rows_per_client"], s["attack_types_unseen_per_client"], s["attack_families_missing_per_client"]
        print(f"{n:10s} {s['n_clients']:7d} {r['min']:>6d}/{r['median']:>6.0f}/{r['max']:>6d} "
              f"{s['clients_with_any_web']:6d} {s['clients_with_any_bruteforce']:6d} "
              f"{u['min']:>4d}-{u['mean']:4.1f}-{u['max']:<4d} {f['min']:>4d}-{f['mean']:3.1f}-{f['max']:<3d}")


if __name__ == "__main__":
    main()
