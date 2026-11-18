"""Fold manifests (INTERFACES 1.2; Phase 0 amendment 3). Owner: Ganna.

A fold manifest says, for every fold of an evaluation protocol, which rows of a cleaned dataset are
train, validation and test. Rows are named by `row_id` (cleaner v1). Because the cleaned file keeps
every attack type's rows together and in recorded order, a manifest is just a short list of row_id
ranges [start, stop): small enough to commit (data/folds/<dataset>_<protocol>.json) and to read by eye.

Protocol A, time-blocked (built here): each attack type's rows (and the benign capture's), in recorded
order, are cut into `n_folds` equal blocks. Fold k tests on block k, validates on block k+1 (wrapping
to block 0), trains on the others. At every boundary between two blocks with different roles, `gap`
rows are dropped from the side with the lower priority (test > validation > train), so neighbouring
packet windows never sit on both sides. Test blocks are never trimmed: over the folds every row is a
test row exactly once.

Protocol B, unseen attack type (built here): inside every family with two or more attack types, the types
are dealt into `n_folds` groups (largest first, round-robin). Fold k TESTS on the types of group k and
VALIDATES on the types of group k+1, so both are attack types the model never saw in training; it trains on
all other types. Benign and the single-type families (BruteForce in CICIoT2023) cannot be held out: they are
split in time exactly as in protocol A, so every fold still has benign rows to measure false alarms on. A
family may have no unseen type in a fold (Spoofing has 2 types, so only folds 0 and 1 test one).

Protocol C (independent dataset) uses the same file format; its builder belongs to task DATA-8b. Every
manifest must pass `check_manifest`.

    python -m fqml.data.folds build-a --report results/data_reports/ciciot2023_clean_v1.json --gap 100
    python -m fqml.data.folds build-b --report results/data_reports/ciciot2023_clean_v1.json --gap 100
    python -m fqml.data.folds check data/folds/ciciot2023_A.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FORMAT = "fqml-folds/1"
ROLES = ("train", "val", "test")
_PRIORITY = {"train": 0, "val": 1, "test": 2}


# ---- ranges: lists of [start, stop) row_id pairs --------------------------------------------------
def merge_ranges(ranges) -> list[list[int]]:
    """Sorted, with touching or overlapping ranges joined and empty ones removed."""
    out: list[list[int]] = []
    for a, b in sorted((int(a), int(b)) for a, b in ranges if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def ranges_len(ranges) -> int:
    return sum(b - a for a, b in ranges)


def ranges_to_rows(ranges) -> np.ndarray:
    """All row_ids in the ranges, sorted (int64)."""
    parts = [np.arange(a, b, dtype=np.int64) for a, b in merge_ranges(ranges)]
    return np.concatenate(parts) if parts else np.zeros(0, np.int64)


def clip_ranges(ranges, start: int, stop: int) -> list[list[int]]:
    return merge_ranges([max(a, start), min(b, stop)] for a, b in ranges)


def _overlap(r1, r2) -> int:
    return sum(max(0, min(b, d) - max(a, c)) for a, b in r1 for c, d in r2)


def _distance(r1, r2) -> int:
    """Smallest number of rows lying between a range of r1 and a range of r2 (0 = they touch)."""
    return min((max(c - b, a - d) for a, b in r1 for c, d in r2), default=10 ** 18)


# ---- protocol A -----------------------------------------------------------------------------------
def build_time_blocked(label_ranges: dict[str, tuple[int, int]], *, dataset: str, n_folds: int = 5,
                       gap: int = 0, max_gap_frac: float = 0.1, source: dict | None = None) -> dict:
    """Protocol A manifest from {native label: (row_start, row_stop)} (clean_ciciot2023_v1.label_ranges).

    `gap` rows are dropped at each boundary between blocks of different roles; for a small attack type
    the gap is reduced to at most `max_gap_frac` of its block length (the value used is recorded).
    """
    if n_folds < 3:
        raise ValueError("n_folds must be >= 3 (train, validation and test blocks)")
    if gap < 0 or not 0 <= max_gap_frac < 0.5:
        raise ValueError("gap must be >= 0 and max_gap_frac in [0, 0.5)")
    labels, folds = {}, [{"fold": k, **{r: [] for r in ROLES}} for k in range(n_folds)]
    for lab, (start, stop) in sorted(label_ranges.items(), key=lambda kv: kv[1]):
        n = stop - start
        if n < n_folds:
            raise ValueError(f"{lab}: {n} rows cannot be cut into {n_folds} blocks")
        edges = [start + (n * i) // n_folds for i in range(n_folds + 1)]
        blocks = [[edges[i], edges[i + 1]] for i in range(n_folds)]
        g = int(min(gap, max_gap_frac * min(b - a for a, b in blocks)))
        labels[lab] = {"range": [int(start), int(stop)], "gap": g, "blocks": blocks}
        for k in range(n_folds):
            role = ["train"] * n_folds
            role[k], role[(k + 1) % n_folds] = "test", "val"
            for i, (a, b) in enumerate(blocks):
                if i > 0 and _PRIORITY[role[i - 1]] > _PRIORITY[role[i]]:
                    a += g
                if i < n_folds - 1 and _PRIORITY[role[i + 1]] > _PRIORITY[role[i]]:
                    b -= g
                folds[k][role[i]].append([a, b])
    for f in folds:
        for r in ROLES:
            f[r] = merge_ranges(f[r])
    manifest = {"format": FORMAT, "dataset": dataset, "protocol": "A", "n_folds": n_folds, "gap": int(gap),
                "max_gap_frac": max_gap_frac, "source": source or {},
                "n_rows": int(sum(b - a for a, b in label_ranges.values())), "labels": labels, "folds": folds}
    check_manifest(manifest)
    return manifest


# ---- protocol B -----------------------------------------------------------------------------------
def unseen_type_groups(label_family: dict[str, str], label_rows: dict[str, int], n_folds: int) -> dict[str, int]:
    """{attack type: group} for every type of a family with >= 2 types. Inside a family the types are taken
    largest first and dealt round-robin, so group 0 holds each family's largest type."""
    by_family: dict[str, list[str]] = {}
    for lab, fam in label_family.items():
        by_family.setdefault(fam, []).append(lab)
    groups = {}
    for fam, labs in by_family.items():
        if len(labs) < 2:
            continue
        for i, lab in enumerate(sorted(labs, key=lambda x: (-label_rows[x], x))):
            groups[lab] = i % n_folds
    return groups


def build_unseen_type(label_ranges: dict[str, tuple[int, int]], label_family: dict[str, str], *, dataset: str,
                      n_folds: int = 5, gap: int = 0, max_gap_frac: float = 0.1, source: dict | None = None) -> dict:
    """Protocol B manifest. `label_family` maps every attack type to its family (benign included)."""
    if set(label_family) != set(label_ranges):
        raise ValueError("label_family must name the family of every attack type in label_ranges")
    rows = {lab: b - a for lab, (a, b) in label_ranges.items()}
    groups = unseen_type_groups(label_family, rows, n_folds)
    timed = {lab: r for lab, r in label_ranges.items() if lab not in groups}
    if not timed:
        raise ValueError("protocol B needs at least one time-blocked type (benign)")
    tb = build_time_blocked(timed, dataset=dataset, n_folds=n_folds, gap=gap, max_gap_frac=max_gap_frac)
    labels = {lab: {"range": [int(a), int(b)], "gap": 0, "group": groups[lab], "family": label_family[lab]}
              for lab, (a, b) in label_ranges.items() if lab in groups}
    for lab, v in tb["labels"].items():
        labels[lab] = {**v, "family": label_family[lab], "group": None}
    folds = []
    for k in range(n_folds):
        f = {"fold": k, **{r: list(tb["folds"][k][r]) for r in ROLES},
             "test_types": sorted(l for l, g in groups.items() if g == k),
             "val_types": sorted(l for l, g in groups.items() if g == (k + 1) % n_folds)}
        for lab, g in groups.items():
            role = "test" if g == k else "val" if g == (k + 1) % n_folds else "train"
            f[role].append(list(label_ranges[lab]))
        for r in ROLES:
            f[r] = merge_ranges(f[r])
        folds.append(f)
    manifest = {"format": FORMAT, "dataset": dataset, "protocol": "B", "n_folds": n_folds, "gap": int(gap),
                "max_gap_frac": max_gap_frac, "source": source or {},
                "n_rows": int(sum(rows.values())), "labels": labels, "folds": folds,
                "time_blocked_types": sorted(timed)}
    check_manifest(manifest)
    return manifest


# ---- checks every manifest must pass ----------------------------------------------------------------
def check_manifest(m: dict) -> None:
    """Raises ValueError unless: the format is known; inside every fold no row has two roles and every
    range lies inside an attack type's range; inside every attack type, rows of different roles are at
    least that type's `gap` apart; no row is a test row in two folds. For protocol A also: every row is
    a test row in exactly one fold."""
    if m.get("format") != FORMAT:
        raise ValueError(f"unknown manifest format {m.get('format')!r} (expected {FORMAT})")
    if len(m["folds"]) != m["n_folds"] or [f["fold"] for f in m["folds"]] != list(range(m["n_folds"])):
        raise ValueError("folds must be numbered 0 .. n_folds-1")
    label_rng = merge_ranges(v["range"] for v in m["labels"].values())
    if ranges_len(label_rng) != sum(v["range"][1] - v["range"][0] for v in m["labels"].values()):
        raise ValueError("attack-type ranges overlap")
    tested: list[list[int]] = []
    for f in m["folds"]:
        k = f["fold"]
        for r in ROLES:
            if f[r] != merge_ranges(f[r]):
                raise ValueError(f"fold {k} {r}: ranges must be sorted and merged")
            if _overlap(f[r], label_rng) != ranges_len(f[r]):
                raise ValueError(f"fold {k} {r}: rows outside every attack type's range")
        for r1, r2 in (("train", "val"), ("train", "test"), ("val", "test")):
            if _overlap(f[r1], f[r2]):
                raise ValueError(f"fold {k}: {r1} and {r2} share rows")
            for lab, v in m["labels"].items():
                a, b = clip_ranges(f[r1], *v["range"]), clip_ranges(f[r2], *v["range"])
                if a and b and _distance(a, b) < v.get("gap", 0):
                    raise ValueError(f"fold {k}, {lab}: {r1} and {r2} closer than the gap of {v['gap']} rows")
        if not f["test"] or not f["train"]:
            raise ValueError(f"fold {k}: empty train or test set")
        if _overlap(f["test"], tested):
            raise ValueError(f"fold {k}: test rows already tested in an earlier fold")
        tested = merge_ranges(tested + f["test"])
    if m.get("protocol") == "A" and tested != label_rng:
        raise ValueError("protocol A: the test blocks of all folds must cover every row exactly once")


def fold_ranges(m: dict, fold: int, role: str) -> list[list[int]]:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    return m["folds"][fold][role]


def fold_rows(m: dict, fold: int, role: str) -> np.ndarray:
    return ranges_to_rows(fold_ranges(m, fold, role))


def twin_mask(test_dup_group, train_dup_group) -> np.ndarray:
    """True for test rows that have an exact twin (same `dup_group`) among the training rows. Scores are
    reported on all test rows and again on the rows where this is False."""
    return np.isin(np.asarray(test_dup_group), np.unique(np.asarray(train_dup_group)))


def manifest_path(dataset: str, protocol: str) -> Path:
    return ROOT / "data" / "folds" / f"{dataset}_{protocol}.json"


def save_manifest(m: dict, path: str | Path) -> None:
    check_manifest(m)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=1) + "\n")


def load_manifest(path: str | Path) -> dict:
    m = json.loads(Path(path).read_text())
    check_manifest(m)
    return m


def main():
    ap = argparse.ArgumentParser(description="Build or check fold manifests.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-a", help="protocol A (time-blocked) from a cleaning report")
    b.add_argument("--report", required=True, help="results/data_reports/<dataset>_clean_v1.json")
    b.add_argument("--dataset", default="ciciot2023")
    b.add_argument("--n-folds", type=int, default=5)
    b.add_argument("--gap", type=int, required=True, help="rows dropped at each boundary (decision D8a)")
    b.add_argument("--max-gap-frac", type=float, default=0.1)
    b.add_argument("--out", default=None)
    bb = sub.add_parser("build-b", help="protocol B (unseen attack type) from a cleaning report")
    for arg, kw in (("--report", dict(required=True)), ("--dataset", dict(default="ciciot2023")),
                    ("--n-folds", dict(type=int, default=5)),
                    ("--gap", dict(type=int, required=True, help="gap for the time-blocked types (D8a)")),
                    ("--max-gap-frac", dict(type=float, default=0.1)), ("--out", dict(default=None))):
        bb.add_argument(arg, **kw)
    c = sub.add_parser("check", help="check a manifest file")
    c.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "check":
        m = load_manifest(a.path)
        print(f"OK {a.path}: protocol {m['protocol']}, {m['n_folds']} folds, {m['n_rows']:,} rows")
        return
    report = json.loads(Path(a.report).read_text())
    ranges = {lab: (d["row_start"], d["row_stop"]) for lab, d in report["per_label"].items()
              if d["rows_written"] > 0}
    src = {"report": a.report, "cleaner": report.get("cleaner"), "settings": report.get("settings"),
           "rows_written": report["totals"]["rows_written"]}
    if a.cmd == "build-b":
        fam = {lab: d["family"] for lab, d in report["per_label"].items() if d["rows_written"] > 0}
        m = build_unseen_type(ranges, fam, dataset=a.dataset, n_folds=a.n_folds, gap=a.gap,
                              max_gap_frac=a.max_gap_frac, source=src)
    else:
        m = build_time_blocked(ranges, dataset=a.dataset, n_folds=a.n_folds, gap=a.gap, max_gap_frac=a.max_gap_frac,
                               source=src)
    out = Path(a.out) if a.out else manifest_path(a.dataset, m["protocol"])
    save_manifest(m, out)
    for f in m["folds"]:
        print(f"fold {f['fold']}: " + ", ".join(f"{r} {ranges_len(f[r]):,}" for r in ROLES))
        if m["protocol"] == "B":
            print(f"    unseen in test: {', '.join(f['test_types']) or '-'}")
            print(f"    unseen in validation: {', '.join(f['val_types']) or '-'}")
    reduced = {lab: v["gap"] for lab, v in m["labels"].items() if v.get("group") is None and v["gap"] < a.gap}
    if reduced:
        print(f"gap reduced for small attack types: {reduced}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
