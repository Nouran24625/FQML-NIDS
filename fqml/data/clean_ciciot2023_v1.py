"""CICIoT2023 cleaner v1 (Phase 0 amendment 1, task DATA-2). Owner: Ganna.

Replaces the Phase 0 cleaned file (clean_ciciot2023_pcapcsv.py, "v0"). v0 capped the data to 1,000,000
rows BEFORE any split, deleted exact duplicates, shuffled the rows and kept neither the file name nor
the row position, so no leakage-free split could be built from it. v1 keeps what a split needs:

  input    the per-capture CSVs (CIC "CSV.zip": one CSV per capture file, e.g. DDoS-ICMP_Flood3.pcap.csv).
           The merged benchmark CSV cannot be used: it has no file name and no row order.
  order    native labels in label-map order, then files by their number (no number = first), then
           rows as they are in the file. NOTHING is shuffled. `row_id` is the position in this order.
  rows     rows with NaN / +-inf are removed (counted per file). Nothing else is deleted by default.
  dups     exact duplicates are MARKED, not deleted: rows with identical feature values share one
           `dup_group` id (a 64-bit hash of the float32 feature values, over all files and labels).
           A test row whose dup_group also occurs in training is a "twin" (fqml.data.folds.twin_mask).
  labels   identical feature rows that carry different labels are counted and marked in `conflict`
           (0 none, 1 different attack types of one family, 2 different families). Policy --conflicts
           keep (default) or drop-cross-family: decision D8.
  sampling attack types with more than --per-label-cap rows are thinned evenly (every k-th row, same
           rate in every file of that type, order kept). The rate is stored in `sample_rate`; metrics
           weight rows by 1 / sample_rate to recover the real class mix. --per-label-cap 0 keeps all
           rows. The value is decision D8, so the argument has no default.

Output columns (INTERFACES 1.1):
  row_id, file, file_seq, row_index, native_label, family, y, dup_group, conflict, sample_rate,
  [ts if every CSV has it], then the feature columns (float32).

    python -m fqml.data.clean_ciciot2023_v1 --check                    # file -> label -> order, no cleaning
    python -m fqml.data.clean_ciciot2023_v1 --per-label-cap 0          # clean, keep every row
    python -m fqml.data.checksums record interim/ciciot2023_v1         # fingerprint the result

Writes data/interim/ciciot2023_v1/ciciot2023_clean_v1.parquet (one row group per source file) and the
report results/data_reports/ciciot2023_clean_v1.json (committed: counts after every step, duplicate and
conflict rates, and the row_id range of every attack type and file, which the fold manifests are built
from). Two passes over the CSVs; pass 1 keeps a 64-bit hash and a label code per row (about 3 GB of
memory at the sort for the full 46.7 M rows).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fqml.data.clean_ciciot2023_pcapcsv import list_csvs, map_files
from fqml.data.schema import CICIOT2023_FEATURES, CICIOT2023_KEY_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
VERSION = "v1"
OUT = ROOT / "data/interim/ciciot2023_v1/ciciot2023_clean_v1.parquet"
REPORT = ROOT / "results/data_reports/ciciot2023_clean_v1.json"
CONFLICT_POLICIES = ("keep", "drop-cross-family")
HASH_NAME = "splitmix64 chain over the float32 bit patterns of the feature columns, in column order (-0.0 = 0.0)"

_SEQ = re.compile(r"(\d*)\.pcap\.csv$", re.IGNORECASE)
_GOLD = 0x9E3779B97F4A7C15
_M1, _M2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31 = np.uint64(30), np.uint64(27), np.uint64(31)


def file_seq(name: str) -> int:
    """'DDoS-ICMP_Flood3.pcap.csv' -> 3; no number -> 0 (tcpdump names the first file without one)."""
    m = _SEQ.search(name)
    return int(m.group(1)) if m and m.group(1) else 0


def _mix(z: np.ndarray) -> np.ndarray:
    z = (z ^ (z >> _S30)) * _M1
    z = (z ^ (z >> _S27)) * _M2
    return z ^ (z >> _S31)


def row_hash(features: np.ndarray) -> np.ndarray:
    """64-bit hash of every row of a float32 [n, d] array. Equal rows always get equal hashes; different
    rows collide with probability about n^2 / 2^65 over the whole dataset (6e-5 for 46.7 M rows)."""
    a = np.ascontiguousarray(features, dtype=np.float32) + np.float32(0.0)     # -0.0 -> +0.0
    bits = a.view(np.uint32)
    h = np.full(a.shape[0], _GOLD, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(a.shape[1]):
            h = _mix(h ^ (bits[:, j].astype(np.uint64) + np.uint64((_GOLD * (j + 1)) % 2 ** 64)))
    return h


def order_files(files: list[Path], labels: dict[Path, str], label_order: list[str]) -> list[Path]:
    rank = {lab: i for i, lab in enumerate(label_order)}
    return sorted(files, key=lambda f: (rank[labels[f]], file_seq(f.name), f.name))


def _read(f: Path, cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(f, usecols=cols, low_memory=False)
    return df[cols].apply(pd.to_numeric, errors="coerce")


def _finite(df: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(float32 feature array, mask of rows whose features are all finite as float64 AND as float32)."""
    a64 = df[feats].to_numpy(np.float64)
    with np.errstate(over="ignore"):
        a32 = a64.astype(np.float32)
    return a32, np.isfinite(a64).all(axis=1) & np.isfinite(a32).all(axis=1)


def even_sample(n: int, cap: int) -> np.ndarray:
    """Boolean mask over n ordered rows keeping exactly min(n, cap) of them, evenly spaced. cap <= 0: all."""
    if cap <= 0 or n <= cap:
        return np.ones(n, bool)
    i = np.arange(n, dtype=np.int64)
    return ((i + 1) * cap) // n != (i * cap) // n


def duplicate_stats(h: np.ndarray, lab: np.ndarray, fam_of_label: np.ndarray):
    """Per-row conflict code and global counts, from the row hashes and native-label codes.

    conflict: 0 = every row with these feature values has the same native label; 1 = several native
    labels of ONE family; 2 = several families."""
    n = len(h)
    order = np.argsort(h, kind="stable")
    hs, ls = h[order], lab[order]
    fs = fam_of_label[ls]
    starts = np.flatnonzero(np.r_[True, hs[1:] != hs[:-1]])
    size = np.diff(np.r_[starts, n])
    code = np.where(np.minimum.reduceat(fs, starts) != np.maximum.reduceat(fs, starts), 2,
                    np.where(np.minimum.reduceat(ls, starts) != np.maximum.reduceat(ls, starts), 1, 0)).astype(np.int8)
    benign_attack = (np.minimum.reduceat(fs, starts) == 0) & (np.maximum.reduceat(fs, starts) > 0)
    conflict = np.empty(n, np.int8)
    conflict[order] = np.repeat(code, size)
    in_dup = np.empty(n, bool)
    in_dup[order] = np.repeat(size > 1, size)
    totals = {
        "distinct_feature_rows": int(len(starts)),
        "rows_with_an_exact_copy": int(in_dup.sum()),
        "duplicate_share": round(1 - len(starts) / n, 6) if n else 0.0,
        "conflict_groups_same_family": int((code == 1).sum()),
        "conflict_groups_cross_family": int((code == 2).sum()),
        "conflict_groups_benign_vs_attack": int(benign_attack.sum()),
        "rows_conflict_same_family": int(size[code == 1].sum()),
        "rows_conflict_cross_family": int(size[code == 2].sum()),
        "rows_conflict_benign_vs_attack": int(size[benign_attack].sum()),
    }
    return conflict, in_dup, totals


class ParquetSink:
    """Streams DataFrames into ONE parquet file, one row group (or more) per call. Written to a
    temporary name and renamed on close, so a crashed run never leaves a file that looks complete."""

    def __init__(self, path: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self._pa, self._pq = pa, pq
        self.path = Path(path)
        self._tmp = self.path.with_name(self.path.name + ".partial")
        self._writer, self._schema = None, None

    def write(self, df: pd.DataFrame) -> None:
        table = self._pa.Table.from_pandas(df, schema=self._schema, preserve_index=False)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._schema = table.schema
            self._writer = self._pq.ParquetWriter(self._tmp, self._schema)
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._tmp.replace(self.path)


def clean(files: list[Path], sink, *, per_label_cap: int, conflicts: str = "keep", log=print) -> dict:
    """Clean `files` into `sink` (an object with write(DataFrame)); returns the report dict."""
    if conflicts not in CONFLICT_POLICIES:
        raise ValueError(f"conflicts must be one of {CONFLICT_POLICIES}")
    families, mapping, labels, unknown = map_files(files)
    if unknown:
        raise SystemExit("Could not match these file names to a CICIoT2023 label:\n  " + "\n  ".join(unknown[:30]))
    label_order = list(mapping)                                   # label-map order: BenignTraffic first
    fam_idx = {f: i for i, f in enumerate(families)}
    lab_idx = {lab: i for i, lab in enumerate(label_order)}
    fam_of_label = np.array([fam_idx[mapping[lab]] for lab in label_order], dtype=np.int16)
    files = order_files(files, labels, label_order)

    cols_per_file = [set(pd.read_csv(f, nrows=0).columns) for f in files]
    common = set.intersection(*cols_per_file)
    feats = [c for c in CICIOT2023_FEATURES if c in common]
    missing = [c for c in CICIOT2023_FEATURES if c not in common]
    keys = [c for c in CICIOT2023_KEY_COLUMNS if c in common]
    extra = sorted(set.union(*cols_per_file) - set(CICIOT2023_FEATURES) - set(keys))
    log(f"using {len(feats)} of {len(CICIOT2023_FEATURES)} standard features; missing {missing}; "
        f"order keys kept {keys}; ignored extra {extra}")

    # ---- pass 1: one hash and one label code per finite row
    info, hashes, labs = [], [], []
    for i, f in enumerate(files, 1):
        log(f"  pass 1/2  [{i}/{len(files)}] {f.name}")
        df = _read(f, feats + keys)
        a32, ok = _finite(df, feats)
        entry = {"file": f.name, "native_label": labels[f], "file_seq": file_seq(f.name),
                 "rows_raw": int(len(df)), "rows_finite": int(ok.sum())}
        for k in keys:                                            # is the file in time order?
            v = df.loc[ok, k].to_numpy(np.float64)
            v = v[np.isfinite(v)]
            if len(v):
                entry[f"{k}_min"], entry[f"{k}_max"] = float(v.min()), float(v.max())
                entry[f"{k}_backward_steps"] = int((np.diff(v) < 0).sum())
        info.append(entry)
        hashes.append(row_hash(a32[ok]))
        labs.append(np.full(int(ok.sum()), lab_idx[labels[f]], dtype=np.int16))
    h = np.concatenate(hashes) if hashes else np.zeros(0, np.uint64)
    lab = np.concatenate(labs) if labs else np.zeros(0, np.int16)
    del hashes, labs
    if not len(h):
        raise SystemExit("no finite rows found")
    conflict, in_dup, totals = duplicate_stats(h, lab, fam_of_label)

    # ---- which rows are written: conflict policy first, then even thinning per attack type
    eligible = conflict != 2 if conflicts == "drop-cross-family" else np.ones(len(h), bool)
    selected = np.zeros(len(h), bool)
    rate = np.ones(len(label_order), np.float64)
    per_label = {}
    for lab_name, code in lab_idx.items():
        rows = np.flatnonzero(lab == code)                        # already in output order
        if not len(rows):
            continue
        cand = rows[eligible[rows]]
        keep = cand[even_sample(len(cand), per_label_cap)]
        selected[keep] = True
        rate[code] = len(keep) / len(cand) if len(cand) else 1.0
        per_label[lab_name] = {
            "family": mapping[lab_name], "files": sum(e["native_label"] == lab_name for e in info),
            "rows_raw": sum(e["rows_raw"] for e in info if e["native_label"] == lab_name),
            "rows_finite": int(len(rows)),
            "distinct_feature_rows": int(len(np.unique(h[rows]))),
            "rows_with_an_exact_copy": int(in_dup[rows].sum()),
            "rows_conflict_same_family": int((conflict[rows] == 1).sum()),
            "rows_conflict_cross_family": int((conflict[rows] == 2).sum()),
            "rows_eligible": int(len(cand)), "rows_written": int(len(keep)),
            "sample_rate": float(rate[code]),
        }

    # ---- pass 2: write, in the same order
    offset = row_id = 0
    columns = None
    for i, (f, entry) in enumerate(zip(files, info), 1):
        log(f"  pass 2/2  [{i}/{len(files)}] {f.name}")
        df = _read(f, feats + keys)
        a32, ok = _finite(df, feats)
        n_ok = int(ok.sum())
        if n_ok != entry["rows_finite"]:
            raise SystemExit(f"{f.name} changed between the two passes")
        sl = slice(offset, offset + n_ok)
        sel = selected[sl]
        offset += n_ok
        code = lab_idx[labels[f]]
        n = int(sel.sum())
        entry.update(rows_written=n, row_start=int(row_id), row_stop=int(row_id + n))
        if not n:
            continue
        out = pd.DataFrame({
            "row_id": np.arange(row_id, row_id + n, dtype=np.int64),
            "file": f.name,
            "file_seq": np.int32(entry["file_seq"]),
            "row_index": np.flatnonzero(ok)[sel].astype(np.int64),   # 0-based row of the source CSV
            "native_label": labels[f],
            "family": mapping[labels[f]],
            "y": np.int64(fam_of_label[code]),
            "dup_group": h[sl][sel].view(np.int64),
            "conflict": conflict[sl][sel],
            "sample_rate": np.float32(rate[code]),
        })
        for k in keys:
            out[k] = df.loc[ok, k].to_numpy(np.float64)[sel]
        out = pd.concat([out, pd.DataFrame(a32[ok][sel], columns=feats)], axis=1)
        columns = list(out.columns)
        sink.write(out)
        row_id += n

    # ---- row_id range of every attack type (contiguous, because files are ordered by label)
    for lab_name, d in per_label.items():
        mine = [e for e in info if e["native_label"] == lab_name]
        d["row_start"], d["row_stop"] = min(e["row_start"] for e in mine), max(e["row_stop"] for e in mine)
        assert d["row_stop"] - d["row_start"] == d["rows_written"]
    totals = {"files": len(files), "rows_raw": sum(e["rows_raw"] for e in info), "rows_finite": int(len(h)),
              "rows_removed_nan_inf": sum(e["rows_raw"] for e in info) - int(len(h)), **totals,
              "rows_written": int(row_id),
              "written_rows_with_an_exact_copy_among_written": int(_copies_within(h[selected]))}
    return {
        "cleaner": f"fqml.data.clean_ciciot2023_{VERSION}", "version": VERSION,
        "source": "CIC CSV.zip per-capture files; label from the file name; rows in file order",
        "settings": {"per_label_cap": int(per_label_cap), "conflicts": conflicts},
        "hash": HASH_NAME, "families": families,
        "features": feats, "features_missing": missing, "key_columns": keys, "extra_columns_ignored": extra,
        "columns": columns, "totals": totals, "per_label": per_label, "files": info,
    }


def _copies_within(h: np.ndarray) -> int:
    _, counts = np.unique(h, return_counts=True)
    return int(counts[counts > 1].sum())


def label_ranges(report: dict) -> dict[str, tuple[int, int]]:
    """{native label: (row_start, row_stop)} from a cleaning report; input of fqml.data.folds."""
    return {lab: (d["row_start"], d["row_stop"]) for lab, d in report["per_label"].items()
            if d["rows_written"] > 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", default=str(ROOT / "data/raw/ciciot2023"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--per-label-cap", type=int, default=None,
                    help="max rows kept per attack type, thinned evenly; 0 = keep all (decision D8, no default)")
    ap.add_argument("--conflicts", choices=CONFLICT_POLICIES, default="keep")
    ap.add_argument("--check", action="store_true", help="only show file -> label -> order")
    a = ap.parse_args()
    files = list_csvs(Path(a.raw))
    if not files:
        raise SystemExit(f"no CSVs under {a.raw}")
    if a.check:
        families, mapping, labels, unknown = map_files(files)
        for f in order_files([f for f in files if f in labels], labels, list(mapping)):
            print(f"  {labels[f]:28s} {mapping[labels[f]]:11s} seq {file_seq(f.name):3d}  {f.name}")
        print(f"{len(files)} files, {len(set(labels.values()))} of {len(mapping)} labels found")
        print("unmatched:", unknown or "none")
        print("labels with no file:", sorted(set(mapping) - set(labels.values())) or "none")
        return
    if a.per_label_cap is None:
        raise SystemExit("--per-label-cap is required (0 = keep every row). Its value is decision D8.")
    print(f"cleaning {len(files)} CSV files (two passes)...")
    sink = ParquetSink(Path(a.out))
    report = clean(files, sink, per_label_cap=a.per_label_cap, conflicts=a.conflicts,
                   log=lambda m: print(m, flush=True))
    sink.close()
    report["output"] = str(Path(a.out).resolve().relative_to(ROOT)) if Path(a.out).resolve().is_relative_to(ROOT) else a.out
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2))
    t = report["totals"]
    print(f"raw {t['rows_raw']:,} -> finite {t['rows_finite']:,} -> written {t['rows_written']:,} rows")
    print(f"distinct feature rows {t['distinct_feature_rows']:,} (duplicate share {t['duplicate_share']:.1%}); "
          f"rows in cross-family conflicts {t['rows_conflict_cross_family']:,}")
    print(f"wrote {a.out}\nwrote {a.report}\nnext: python -m fqml.data.checksums record interim/ciciot2023_v1")


if __name__ == "__main__":
    main()
