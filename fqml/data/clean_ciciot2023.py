"""CICIoT2023 cleaning v0 (Phase 0). Owner: Ganna.

Steps (Plan §6), with per-class counts logged after every step -> results/data_reports/:
  1. read the CSVs (full release ~46M rows -> streamed file by file, see clean_streaming)
  2. coerce numeric, drop NaN / +-inf rows
  3. drop exact duplicate rows (global, across files; first occurrence kept)
  4. keep only the 46 known features (identifier/shortcut columns are reported, never used)
  5. map 34 native labels -> 8 families via label_maps/ciciot2023.yaml (unknown label = hard error)
  6. stratified cap to <= cap_rows, keeping ALL rows of classes below the per-class cap
  7. write parquet + a JSON step report

Memory: pass 1 keeps only a 64-bit hash + family id per row (~0.5 GB for 46M rows);
pass 2 re-reads each file and keeps the selected rows. `clean()` is the in-memory equivalent
used by tests; tests assert both give identical output.

    python -m fqml.data.clean_ciciot2023 [--raw data/raw/ciciot2023] [--cap 1000000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fqml.data.label_maps import load_label_map
from fqml.data.schema import CICIOT2023_FEATURES, CICIOT2023_LABEL, SHORTCUT_COLUMNS

ROOT = Path(__file__).resolve().parents[2]


def _counts(s: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}


def read_raw(raw_dir: Path, max_files: int | None = None) -> pd.DataFrame:
    files = sorted(raw_dir.rglob("*.csv"))
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no CSVs under {raw_dir}")
    dtypes = {c: "float32" for c in CICIOT2023_FEATURES}
    parts = [pd.read_csv(f, dtype=dtypes, low_memory=False) for f in files]
    return pd.concat(parts, ignore_index=True)


def stratified_cap(df: pd.DataFrame, label: str, cap: int, seed: int) -> pd.DataFrame:
    """Water-filling cap: small classes are kept whole, large ones share the remaining budget."""
    if len(df) <= cap:
        return df
    counts = df[label].value_counts().sort_values()
    budget, remaining, alloc = cap, len(counts), {}
    for cls, n in counts.items():
        share = budget // remaining
        alloc[cls] = int(min(n, share))
        budget -= alloc[cls]
        remaining -= 1
    rng = np.random.default_rng(seed)
    keep = [g.iloc[rng.choice(len(g), alloc[c], replace=False)] if alloc[c] < len(g) else g
            for c, g in df.groupby(label, sort=True)]
    return pd.concat(keep).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def clean(df: pd.DataFrame, *, cap: int = 1_000_000, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    families, mapping = load_label_map("ciciot2023")
    rep: dict = {"steps": []}

    def log(step: str, frame: pd.DataFrame, col: str = CICIOT2023_LABEL):
        rep["steps"].append({"step": step, "rows": int(len(frame)), "per_class": _counts(frame[col])})

    log("raw", df)
    missing = [c for c in CICIOT2023_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"unexpected schema; missing columns: {missing}")
    present_shortcuts = SHORTCUT_COLUMNS & set(df.columns)
    rep["shortcut_columns_found"] = sorted(present_shortcuts)
    df = df[CICIOT2023_FEATURES + [CICIOT2023_LABEL]].copy()

    feats = df[CICIOT2023_FEATURES].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(feats.to_numpy(np.float64)).all(axis=1)
    df = df.loc[finite].copy()
    df[CICIOT2023_FEATURES] = feats.loc[finite].astype("float32")
    log("drop_nan_inf", df)

    df = df.drop_duplicates(ignore_index=True)
    log("drop_duplicates", df)

    unknown = sorted(set(df[CICIOT2023_LABEL].unique()) - set(mapping))
    if unknown:
        raise ValueError(f"labels not in label map: {unknown}")
    df["native_label"] = df[CICIOT2023_LABEL]
    df["family"] = df[CICIOT2023_LABEL].map(mapping)
    df["y"] = df["family"].map({f: i for i, f in enumerate(families)}).astype("int64")
    log("family_map", df, "family")

    df = stratified_cap(df, "family", cap, seed)
    log("stratified_cap", df, "family")

    rep["families"] = families
    rep["n_features"] = len(CICIOT2023_FEATURES)
    return df.drop(columns=[CICIOT2023_LABEL]), rep


def _prep_file(df: pd.DataFrame, mapping: dict[str, str], fam_idx: dict[str, int]):
    """Per-file part of steps 2, 4, 5: returns cleaned frame (+ family cols) and raw counts."""
    raw_counts = df[CICIOT2023_LABEL].value_counts()
    missing = [c for c in CICIOT2023_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"unexpected schema; missing columns: {missing}")
    shortcuts = sorted(SHORTCUT_COLUMNS & set(df.columns))
    df = df[CICIOT2023_FEATURES + [CICIOT2023_LABEL]]
    feats = df[CICIOT2023_FEATURES].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(feats.to_numpy(np.float64)).all(axis=1)
    out = feats.loc[finite].astype("float32").reset_index(drop=True)
    labels = df[CICIOT2023_LABEL].loc[finite].reset_index(drop=True)
    unknown = sorted(set(labels.unique()) - set(mapping))
    if unknown:
        raise ValueError(f"labels not in label map: {unknown}")
    out["native_label"] = labels
    out["family"] = labels.map(mapping)
    out["y"] = out["family"].map(fam_idx).astype("int64")
    return out, raw_counts, shortcuts


def _row_hash(df: pd.DataFrame) -> np.ndarray:
    cols = CICIOT2023_FEATURES + ["native_label"]
    return pd.util.hash_pandas_object(df[cols], index=False).to_numpy(np.uint64)


def clean_streaming(files: list[Path], *, cap: int = 1_000_000, seed: int = 0
                    ) -> tuple[pd.DataFrame, dict]:
    families, mapping = load_label_map("ciciot2023")
    fam_idx = {f: i for i, f in enumerate(families)}
    raw_total: pd.Series | None = None
    finite_total: pd.Series | None = None
    hashes, ys, shortcuts = [], [], set()
    # ---- pass 1: hashes + family ids only
    for i, f in enumerate(files, 1):
        print(f"  pass 1/2  [{i}/{len(files)}] {f.name}", flush=True)
        df, rc, sc = _prep_file(pd.read_csv(f, low_memory=False), mapping, fam_idx)
        raw_total = rc if raw_total is None else raw_total.add(rc, fill_value=0)
        fc = df["native_label"].value_counts()
        finite_total = fc if finite_total is None else finite_total.add(fc, fill_value=0)
        hashes.append(_row_hash(df))
        ys.append(df["y"].to_numpy(np.int8))
        shortcuts.update(sc)
    h, y = np.concatenate(hashes), np.concatenate(ys)
    _, first = np.unique(h, return_index=True)
    keep = np.zeros(len(h), dtype=bool)
    keep[first] = True
    # stratified cap over the de-duplicated rows (same water-filling as stratified_cap)
    rng = np.random.default_rng(seed)
    uniq_idx = np.flatnonzero(keep)
    counts = np.bincount(y[uniq_idx], minlength=len(families))
    alloc, budget, order = {}, min(cap, len(uniq_idx)), np.argsort(counts, kind="stable")
    remaining = int((counts > 0).sum())
    for c in order:
        if counts[c] == 0:
            continue
        alloc[c] = int(min(counts[c], budget // remaining))
        budget -= alloc[c]
        remaining -= 1
    selected = np.zeros(len(h), dtype=bool)
    for c, n in alloc.items():
        idx = uniq_idx[y[uniq_idx] == c]
        selected[idx if n >= len(idx) else rng.choice(idx, n, replace=False)] = True
    # ---- pass 2: re-read and keep selected rows
    parts, offset = [], 0
    for i, f in enumerate(files, 1):
        print(f"  pass 2/2  [{i}/{len(files)}] {f.name}", flush=True)
        df, _, _ = _prep_file(pd.read_csv(f, low_memory=False), mapping, fam_idx)
        parts.append(df.loc[selected[offset:offset + len(df)]])
        offset += len(df)
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    out = out.reset_index(drop=True)

    def fam_counts(native: pd.Series) -> dict[str, int]:
        s = native.groupby(native.index.map(mapping)).sum()
        return {k: int(v) for k, v in s.sort_index().items()}

    dedup_fam = {families[c]: int(n) for c, n in enumerate(counts) if n}
    rep = {"steps": [
        {"step": "raw", "rows": int(raw_total.sum()), "per_class": fam_counts(raw_total)},
        {"step": "drop_nan_inf", "rows": int(finite_total.sum()), "per_class": fam_counts(finite_total)},
        {"step": "drop_duplicates", "rows": int(keep.sum()), "per_class": dedup_fam},
        {"step": "stratified_cap", "rows": int(len(out)),
         "per_class": {k: int(v) for k, v in out["family"].value_counts().sort_index().items()}},
    ], "families": families, "n_features": len(CICIOT2023_FEATURES),
        "shortcut_columns_found": sorted(shortcuts), "n_files": len(files),
        "native_label_counts_raw": {k: int(v) for k, v in raw_total.sort_index().items()}}
    return out, rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw/ciciot2023"))
    ap.add_argument("--out", default=str(ROOT / "data/interim/ciciot2023_clean.parquet"))
    ap.add_argument("--cap", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-files", type=int, default=None, help="debug: read only N CSVs")
    a = ap.parse_args()
    files = sorted(Path(a.raw).rglob("*.csv"))[: a.max_files] if a.max_files else \
        sorted(Path(a.raw).rglob("*.csv"))
    if not files:
        raise SystemExit(f"no CSVs under {a.raw}; download CICIoT2023 (CSV files) there first")
    print(f"cleaning {len(files)} CSV files (two passes)...")
    df, rep = clean_streaming(files, cap=a.cap, seed=a.seed)
    rdir = ROOT / "results/data_reports"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "ciciot2023_clean_v0.json").write_text(json.dumps(rep, indent=2))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    for s in rep["steps"]:
        print(f"{s['step']:>16}: {s['rows']:>10,} rows")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
