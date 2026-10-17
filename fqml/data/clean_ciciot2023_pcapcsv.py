"""CICIoT2023 cleaning, per-capture CSV variant (CIC "CSV.zip"). Owner: Ganna.

Why a second cleaner: the CIC download has two CSV variants.
  - MERGED_CSV.zip : 46 features + 'label' column (the standard benchmark)  -> clean_ciciot2023.py
  - CSV.zip        : one CSV per capture file (e.g. 'DDoS-ICMP_Flood3.pcap.csv'),
                     no 'label' column, a subset of the 46 features         -> THIS file
Here the attack label comes from the FILE NAME, and only the standard features present in
every file are kept. Everything else follows Plan section 6 exactly as the main cleaner:
  1. read every CSV (macOS '._' junk files and __MACOSX folders are skipped)
  2. label = file name -> one of the 34 native labels -> 8 families (unknown name = hard error)
  3. keep the standard CICIoT2023 features present in ALL files (extras are reported, not used)
  4. drop rows with NaN / +-inf
  5. drop exact duplicate rows across all files (first copy kept)
  6. stratified cap to <= cap rows, rare families kept whole
  7. write parquet + a JSON report with counts after every step

    python -m fqml.data.clean_ciciot2023_pcapcsv --check      # only show file -> label mapping
    python -m fqml.data.clean_ciciot2023_pcapcsv              # clean
Memory: pass 1 keeps a 64-bit hash + label per row; pass 2 re-reads and keeps selected rows.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from fqml.data.label_maps import load_label_map
from fqml.data.schema import CICIOT2023_FEATURES

ROOT = Path(__file__).resolve().parents[2]


def _norm(s: str) -> str:
    """'DDoS-ICMP_Flood3.pcap.csv' -> 'ddosicmpflood' (lowercase letters only)."""
    s = s.lower().replace(".csv", "").replace(".pcap", "")
    return re.sub(r"[^a-z]", "", s)


def list_csvs(raw: Path) -> list[Path]:
    return sorted(p for p in raw.rglob("*.csv")
                  if not p.name.startswith("._") and "__MACOSX" not in p.parts)


def label_for(path: Path, lookup: dict[str, str]) -> str | None:
    """Native label from the file name, else from a parent folder name."""
    for candidate in [path.name, *[p for p in reversed(path.parent.parts)]]:
        key = _norm(candidate)
        if key in lookup:
            return lookup[key]
    return None


def map_files(files: list[Path]):
    families, mapping = load_label_map("ciciot2023")
    lookup = {_norm(native): native for native in mapping}
    labels, unknown = {}, []
    for f in files:
        lab = label_for(f, lookup)
        (unknown.append(f.name) if lab is None else labels.__setitem__(f, lab))
    return families, mapping, labels, unknown


def _read(f: Path, feats: list[str]) -> pd.DataFrame:
    df = pd.read_csv(f, usecols=feats, low_memory=False)
    return df.apply(pd.to_numeric, errors="coerce")


def clean(files: list[Path], *, cap: int = 1_000_000, seed: int = 0):
    families, mapping, labels, unknown = map_files(files)
    if unknown:
        raise SystemExit("Could not match these file names to a CICIoT2023 label:\n  "
                         + "\n  ".join(unknown[:30]) + "\nSend this list to Claude.")
    fam_idx = {f: i for i, f in enumerate(families)}

    # features present in every file
    cols_per_file = [set(pd.read_csv(f, nrows=0).columns) for f in files]
    common = set.intersection(*cols_per_file)
    feats = [c for c in CICIOT2023_FEATURES if c in common]
    missing = [c for c in CICIOT2023_FEATURES if c not in common]
    extra = sorted(set.union(*cols_per_file) - set(CICIOT2023_FEATURES))
    print(f"using {len(feats)} of 46 standard features; missing {missing}; ignored extra {extra}")

    raw_n, finite_n, hashes, ys = Counter(), Counter(), [], []
    for i, f in enumerate(files, 1):
        print(f"  pass 1/2  [{i}/{len(files)}] {f.name}", flush=True)
        df = _read(f, feats)
        raw_n[labels[f]] += len(df)
        ok = np.isfinite(df.to_numpy(np.float64)).all(axis=1)
        df = df.loc[ok]
        finite_n[labels[f]] += len(df)
        df = df.assign(native_label=labels[f])
        hashes.append(pd.util.hash_pandas_object(df, index=False).to_numpy(np.uint64))
        ys.append(np.full(len(df), fam_idx[mapping[labels[f]]], dtype=np.int8))
    h, y = np.concatenate(hashes), np.concatenate(ys)
    _, first = np.unique(h, return_index=True)
    keep = np.zeros(len(h), bool)
    keep[first] = True

    # stratified cap (water-filling: small families kept whole)
    uniq = np.flatnonzero(keep)
    counts = np.bincount(y[uniq], minlength=len(families))
    budget, alloc = min(cap, len(uniq)), {}
    remaining = int((counts > 0).sum())
    for c in np.argsort(counts, kind="stable"):
        if counts[c] == 0:
            continue
        alloc[c] = int(min(counts[c], budget // remaining))
        budget -= alloc[c]
        remaining -= 1
    rng = np.random.default_rng(seed)
    selected = np.zeros(len(h), bool)
    for c, n in alloc.items():
        idx = uniq[y[uniq] == c]
        selected[idx if n >= len(idx) else rng.choice(idx, n, replace=False)] = True

    parts, offset = [], 0
    for i, f in enumerate(files, 1):
        print(f"  pass 2/2  [{i}/{len(files)}] {f.name}", flush=True)
        df = _read(f, feats)
        df = df.loc[np.isfinite(df.to_numpy(np.float64)).all(axis=1)].reset_index(drop=True)
        sel = selected[offset:offset + len(df)]
        offset += len(df)
        df = df.loc[sel].astype("float32")
        df["native_label"] = labels[f]
        parts.append(df)
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    out = out.reset_index(drop=True)
    out["family"] = out["native_label"].map(mapping)
    out["y"] = out["family"].map(fam_idx).astype("int64")

    def by_family(native_counts):
        s = Counter()
        for k, v in native_counts.items():
            s[mapping[k]] += v
        return dict(sorted(s.items()))

    report = {
        "variant": "CSV.zip per-capture files; label from file name",
        "n_files": len(files), "features_used": feats, "features_missing": missing,
        "extra_columns_ignored": extra,
        "native_label_counts_raw": dict(sorted(raw_n.items())),
        "steps": [
            {"step": "raw", "rows": sum(raw_n.values()), "per_class": by_family(raw_n)},
            {"step": "drop_nan_inf", "rows": sum(finite_n.values()), "per_class": by_family(finite_n)},
            {"step": "drop_duplicates", "rows": int(keep.sum()),
             "per_class": {families[c]: int(n) for c, n in enumerate(counts) if n}},
            {"step": "stratified_cap", "rows": len(out),
             "per_class": {k: int(v) for k, v in out["family"].value_counts().sort_index().items()}},
        ],
    }
    return out, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw/ciciot2023"))
    ap.add_argument("--out", default=str(ROOT / "data/interim/ciciot2023_pcapcsv_clean.parquet"))
    ap.add_argument("--cap", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="only show the file -> label mapping")
    a = ap.parse_args()
    files = list_csvs(Path(a.raw))
    if not files:
        raise SystemExit(f"no CSVs under {a.raw}")
    if a.check:
        families, mapping, labels, unknown = map_files(files)
        per_label = Counter(labels.values())
        print(f"{len(files)} files, {len(per_label)} of 34 labels found")
        for lab, n in sorted(per_label.items()):
            print(f"  {lab:28s} -> {mapping[lab]:11s} {n} files")
        print("unmatched:", unknown or "none")
        missing_labels = sorted(set(mapping) - set(per_label))
        print("labels with no file:", missing_labels or "none")
        return
    print(f"cleaning {len(files)} CSV files (two passes)...")
    out, rep = clean(files, cap=a.cap, seed=a.seed)
    rdir = ROOT / "results/data_reports"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "ciciot2023_pcapcsv_clean_v0.json").write_text(json.dumps(rep, indent=2))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    for s in rep["steps"]:
        print(f"{s['step']:>16}: {s['rows']:>11,} rows")
    print("per family after cleaning:", rep["steps"][-1]["per_class"])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
