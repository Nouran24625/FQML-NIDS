"""Data quality report for the cleaned CICIoT2023 file (Phase 1 tasks DATA-4 and DATA-1b). Owner: Ganna.

Reads the cleaned file v1, its cleaning report and the fold manifests (INTERFACES 1.1, 1.2) and measures what
limits every model trained on them, before E0 starts:

  rows        rows per family and per attack type, overall and in every fold and role of protocols A and B
  duplicates  per family: rows that have an exact copy somewhere in the file, and 1 - distinct / rows
  conflicts   rows whose exact feature values also occur with another label (another attack type of the same
              family, or another family), per family and per fold and role
  twins       per fold: validation and test rows with an exact twin among that fold's training rows
  ceiling     the reference score D8 asks for: every group of identical feature rows is predicted as its most
              common label in that block (8-class, c7m and binary). No function of these features is more
              accurate on that block; for macro-F1 it is the agreed reference, close to but not strictly the maximum
  screen      window-size screen: how well each feature alone separates the families recorded with 100-packet
              windows (DDoS, DoS, Mirai; Neto et al., Sensors 2023) from those recorded with 10-packet windows,
              and how well it separates families on the SAME side. A window-size marker separates the two sides
              but not the families inside a side. Plus the full distribution of `Number` per attack type
  ttl         what `Time_To_Live` is: its values per attack type, read from the raw CSVs (cleaner v1 skips it)
  iat         DATA-1b: distinct `IAT` values per file (the published "few values per capture" finding), and
              whether IAT moves in one direction inside a file and across the numbered files of an attack

Duplicates, conflicts, twins and the ceiling are computed for up to three feature sets: all 37 features of
cleaner v1 (the file's own `dup_group`), the 36 model inputs after D9 (no IAT), and 35 (no IAT, no Number).

    python -m fqml.data.quality                            # everything; raw CSVs are read if present
    python -m fqml.data.quality --raw none                 # no raw CSVs: no Time_To_Live, IAT from the float32 file
    python -m fqml.data.quality --feature-sets all,no_iat  # fewer passes over the file

Writes results/data/ciciot2023_quality_v1.json (committed) and prints a summary. Memory peak about 2-3 GB:
one 64-bit key per row of the 46.8 M rows while groups are numbered, plus small per-row codes. Every stage
prints its time.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fqml.data.clean_ciciot2023_pcapcsv import list_csvs
from fqml.data.clean_ciciot2023_v1 import row_hash
from fqml.data.folds import ROLES, load_manifest
from fqml.data.label_maps import family_index, load_label_map

ROOT = Path(__file__).resolve().parents[2]
DATASET = "ciciot2023"
PARQUET = ROOT / "data/interim/ciciot2023_v1/ciciot2023_clean_v1.parquet"
REPORT = ROOT / "results/data_reports/ciciot2023_clean_v1.json"
FOLDS = (ROOT / "data/folds/ciciot2023_A.json", ROOT / "data/folds/ciciot2023_B.json")
RAW = ROOT / "data/raw/ciciot2023"
OUT = ROOT / "results/data/ciciot2023_quality_v1.json"
VERSION = 1
W100_FAMILIES = ("DDoS", "DoS", "Mirai")            # features averaged over 100-packet windows (dataset paper)
FEATURE_SETS = {"all": (), "no_iat": ("IAT",), "no_iat_no_number": ("IAT", "Number")}    # name -> features left out
RAW_COLUMNS = ("IAT", "Time_To_Live", "IGMP")
BATCH_ROWS = 1 << 20
ABOUT = {
    "duplicates": "rows whose exact feature values (float32) occur more than once in the whole file",
    "conflicts": "rows whose exact feature values also occur with another attack type of the same family "
                 "(same_family) or with another family (cross_family)",
    "twins": "validation / test rows whose exact feature values also occur in that fold's training rows",
    "ceiling": "scores of the rule 'every group of identical feature rows gets its most common label in the block'; "
               "no model using these features can be more accurate on that block (for macro-F1 a reference, not a "
               "strict maximum). Macro-F1 over classes with true rows (EVAL-1 2.1)",
    "screen": "AUC of one feature for 100-packet-window families (DDoS, DoS, Mirai) against the others, "
              "family-balanced sample; separation = |AUC - 0.5| * 2 (0 none, 1 perfect); within_side = the same "
              "between families that share a window size",
    "counts": "rows, shares and twins count rows of the cleaned file; only the ceiling weights rows by "
              "1 / sample_rate (all 1.0 with D8's --per-label-cap 0)",
}


# ---- where the rows come from ----------------------------------------------------------------------
class ParquetSource:
    """Column batches of the cleaned file, in row_id order, never the whole file in memory."""

    def __init__(self, path, batch_rows: int = BATCH_ROWS):
        import pyarrow.parquet as pq
        self.path = Path(path)
        self._pf = pq.ParquetFile(self.path)
        self.n_rows = self._pf.metadata.num_rows
        self.columns = list(self._pf.schema_arrow.names)
        self.batch_rows = batch_rows

    def batches(self, columns):
        for b in self._pf.iter_batches(batch_size=self.batch_rows, columns=list(columns)):
            yield {c: b.column(c).to_numpy(zero_copy_only=False) for c in columns}


class FrameSource:
    """The same interface over a pandas DataFrame (tests, small files)."""

    def __init__(self, df: pd.DataFrame, batch_rows: int = BATCH_ROWS):
        self.df, self.n_rows, self.columns, self.batch_rows = df, len(df), list(df.columns), batch_rows

    def batches(self, columns):
        for a in range(0, self.n_rows, self.batch_rows):
            part = self.df.iloc[a:a + self.batch_rows]
            yield {c: part[c].to_numpy() for c in columns}


class Layout:
    """What the cleaning report alone says about every row. INTERFACES 1.1: every attack type is one contiguous
    row_id range, so the attack type, family and c7m class of a row follow from its row_id."""

    def __init__(self, report: dict, dataset: str = DATASET):
        per = report["per_label"]
        self.types = sorted((t for t, d in per.items() if d["rows_written"] > 0), key=lambda t: per[t]["row_start"])
        self.start = np.array([per[t]["row_start"] for t in self.types], np.int64)
        self.stop = np.array([per[t]["row_stop"] for t in self.types], np.int64)
        self.n = int(report["totals"]["rows_written"])
        if self.start[0] != 0 or self.stop[-1] != self.n or (self.start[1:] != self.stop[:-1]).any():
            raise ValueError("the attack-type row ranges of the report do not tile 0 .. rows_written")
        self.families = list(report["families"])
        self.families7 = load_label_map(dataset, "c7m")[0]
        self.type_family = [per[t]["family"] for t in self.types]
        self.fam_of_type = np.array([self.families.index(f) for f in self.type_family], np.int8)
        if not (family_index(self.types, dataset) == self.fam_of_type).all():
            raise ValueError("the report's families do not match the label map")
        self.c7m_of_type = family_index(self.types, dataset, "c7m").astype(np.int8)
        self.w100_of_type = np.array([f in W100_FAMILIES for f in self.type_family])

    def type_code(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.types), dtype=np.int16), self.stop - self.start)


# ---- small helpers ---------------------------------------------------------------------------------
def _r(x, nd=6):
    """JSON-friendly number: NaN stays NaN (EVAL-1 rule 6), floats are rounded."""
    x = float(x)
    return x if np.isnan(x) else round(x, nd)


def _share(num, den):
    return _r(num / den) if den else float("nan")


def _by_name(names, values):
    return {n: (int(v) if isinstance(v, (int, np.integer)) else _r(v)) for n, v in zip(names, values)}


def _counts(ranges, codes, k, mask=None) -> np.ndarray:
    """bincount of `codes` over the rows in `ranges` (only where `mask`, if given), length k."""
    out = np.zeros(k, np.int64)
    for a, b in ranges:
        c = codes[a:b] if mask is None else codes[a:b][mask[a:b]]
        out += np.bincount(c, minlength=k)[:k]
    return out


def _rows_of(ranges, mask) -> np.ndarray:
    """row_ids inside `ranges` where `mask` is True, sorted."""
    parts = [np.flatnonzero(mask[a:b]) + a for a, b in ranges]
    return np.concatenate(parts) if parts else np.zeros(0, np.int64)


def even_positions(start: int, stop: int, k: int) -> np.ndarray:
    """k positions spread evenly over [start, stop) in recorded order; all of them if there are at most k
    (the same rows as fqml.data.rows.even_pick(np.arange(start, stop), k))."""
    n = stop - start
    if k <= 0 or n <= k:
        return np.arange(start, stop, dtype=np.int64)
    return start + np.linspace(0, n - 1, k).round().astype(np.int64)


def dense_ids(keys: np.ndarray) -> tuple[np.ndarray, int]:
    """Group number 0 .. G-1 (int32) for every row; rows with equal keys share one number."""
    if not len(keys):
        return np.zeros(0, np.int32), 0
    order = np.argsort(keys)
    ks = keys[order]
    new = np.empty(len(ks), bool)
    new[0] = True
    np.not_equal(ks[1:], ks[:-1], out=new[1:])
    del ks
    ranked = np.cumsum(new, dtype=np.int32)
    del new
    ranked -= 1
    gid = np.empty(len(keys), np.int32)
    gid[order] = ranked
    return gid, int(ranked[-1]) + 1


def group_stats(gid: np.ndarray, n_groups: int, y8, y7, tcode) -> dict:
    """Per group of identical feature rows: size, families present (bit mask), and which labels it mixes."""
    def lo_hi(v, dtype):
        lo = np.full(n_groups, np.iinfo(dtype).max, dtype)
        hi = np.full(n_groups, np.iinfo(dtype).min, dtype)
        np.minimum.at(lo, gid, v)
        np.maximum.at(hi, gid, v)
        return lo, hi

    f_lo, f_hi = lo_hi(y8, np.int8)
    s_lo, s_hi = lo_hi(y7, np.int8)
    t_lo, t_hi = lo_hi(tcode, np.int16)
    fam_bits = np.zeros(n_groups, np.uint16)
    np.bitwise_or.at(fam_bits, gid, np.left_shift(np.uint16(1), y8.astype(np.uint16)))
    return {"size": np.bincount(gid, minlength=n_groups),
            "code": np.where(f_lo != f_hi, 2, np.where(t_lo != t_hi, 1, 0)).astype(np.int8),
            "c7m_mixed": s_lo != s_hi, "benign_attack": (f_lo == 0) & (f_hi > 0), "fam_bits": fam_bits}


def majority_labels(g: np.ndarray, y: np.ndarray, k: int, w: np.ndarray | None = None) -> np.ndarray:
    """For every row, the most common label (by weight) among the rows with the same g. Ties go to the
    smaller label index, so the result is deterministic."""
    if not len(g):
        return np.zeros(0, np.int64)
    key = g.astype(np.int64) * k + y.astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    cnt = np.bincount(inv, weights=w, minlength=len(uk)).astype(np.float64)
    ug, uc = uk // k, uk % k
    grp = np.cumsum(np.r_[True, ug[1:] != ug[:-1]]) - 1        # group number of every (g, label) pair
    order = np.lexsort((uc, -cnt, ug))                           # by group, most rows first, then smaller label
    first = order[np.r_[True, ug[order][1:] != ug[order][:-1]]]
    winner = np.empty(grp[-1] + 1, np.int64)
    winner[grp[first]] = uc[first]
    return winner[grp[inv]]


def scores_from_confusion(conf: np.ndarray, classes: list[str]) -> dict:
    """Accuracy, macro-F1 over the classes with true rows (EVAL-1 2.1), per-class F1 and recall (NaN = no true
    rows; such classes are listed in `skipped`). Rows = true class, columns = predicted class."""
    support, predicted, tp = conf.sum(1), conf.sum(0), np.diag(conf)
    present = support > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(present, 2 * tp / (support + predicted), np.nan)
        recall = np.where(present, tp / support, np.nan)
    return {"accuracy": _share(tp.sum(), conf.sum()),
            "macro_f1": _r(np.nanmean(f1)) if present.any() else float("nan"),
            "per_class_f1": _by_name(classes, f1), "per_class_recall": _by_name(classes, recall),
            "skipped": [c for c, p in zip(classes, present) if not p], "rows": _r(conf.sum(), 3)}


def weighted_auc(x: np.ndarray, pos: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """(AUC, best balanced accuracy of one threshold, that threshold) for telling `pos` rows apart by x alone.
    AUC = P(x of a positive row > x of a negative row) + 0.5 P(equal), with row weights: 0.5 = no separation,
    1.0 or 0.0 = perfect (positives above, or below). The threshold rule is `x > t` when AUC >= 0.5, else
    `x <= t`. Rows with a non-finite x are left out."""
    x = np.asarray(x, np.float64)
    ok = np.isfinite(x)
    x, pos, w = x[ok], np.asarray(pos, bool)[ok], np.asarray(w, np.float64)[ok]
    if not pos.any() or pos.all():
        return float("nan"), float("nan"), float("nan")
    vals, inv = np.unique(x, return_inverse=True)
    inv = inv.reshape(-1)
    wp = np.bincount(inv, weights=w * pos, minlength=len(vals))
    wn = np.bincount(inv, weights=w * ~pos, minlength=len(vals))
    tot_p, tot_n = wp.sum(), wn.sum()
    below_n = np.cumsum(wn) - wn
    auc = float((wp * (below_n + 0.5 * wn)).sum() / (tot_p * tot_n))
    tpr = 1.0 - np.cumsum(wp) / tot_p                            # share of positives with x > vals[i]
    fpr = 1.0 - np.cumsum(wn) / tot_n
    j = tpr - fpr if auc >= 0.5 else fpr - tpr
    i = int(np.argmax(j))
    return auc, float((1.0 + max(j[i], 0.0)) / 2.0), float(vals[i])


def summary_from_counts(vals: np.ndarray, cnt: np.ndarray, top: int = 5) -> dict:
    """Distribution of one column from its distinct values and their counts: quantiles (inverted CDF),
    distinct values, share of whole numbers, the most common values with their share."""
    if not len(vals) or cnt.sum() == 0:
        return {"rows": 0}
    vals, cnt = np.asarray(vals, np.float64), np.asarray(cnt, np.int64)
    n, cum = int(cnt.sum()), np.cumsum(cnt)
    q = {name: _r(vals[min(int(np.searchsorted(cum, p * n, side="left")), len(vals) - 1)])
         for name, p in (("p01", 0.01), ("median", 0.5), ("p99", 0.99))}
    best = np.argsort(-cnt, kind="stable")[:top]
    return {"rows": n, "distinct": int(len(vals)), "mean": _r((vals * cnt).sum() / n), "min": _r(vals[0]), **q,
            "max": _r(vals[-1]), "integer_share": _share(cnt[vals == np.round(vals)].sum(), n),
            "top_values": [[_r(vals[i]), _share(cnt[i], n)] for i in best]}


def counts_of(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = v[np.isfinite(v)]
    return np.unique(v, return_counts=True)


def merge_counts(parts) -> tuple[np.ndarray, np.ndarray]:
    vals = np.concatenate([p[0].astype(np.float64) for p in parts])
    cnt = np.concatenate([p[1] for p in parts])
    u, inv = np.unique(vals, return_inverse=True)
    return u, np.bincount(inv.reshape(-1), weights=cnt, minlength=len(u)).astype(np.int64)


def order_summary(v: np.ndarray) -> dict:
    """Distinct values and direction of change of one column inside one file, in row order."""
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"rows": 0}
    vals, cnt = np.unique(v, return_counts=True)
    d = np.diff(v)
    return {"rows": int(len(v)), "distinct": int(len(vals)), "top_value": _r(vals[np.argmax(cnt)]),
            "top_share": _share(cnt.max(), len(v)), "min": _r(v.min()), "median": _r(np.median(v)),
            "max": _r(v.max()), "first": _r(v[0]), "last": _r(v[-1]),
            "steps_down_share": _share(np.count_nonzero(d < 0), len(d)),
            "steps_up_share": _share(np.count_nonzero(d > 0), len(d))}


def family_balanced_weights(tcode: np.ndarray, lay: Layout) -> np.ndarray:
    """Row weights for a per-type sample: inside each side (100- or 10-packet windows) every family weighs the
    same, and inside a family every attack type weighs the same, whatever its number of rows."""
    present = [int(t) for t in np.unique(tcode)]
    n_rows = np.bincount(tcode, minlength=len(lay.types))
    w_type = np.zeros(len(lay.types))
    for side in (True, False):
        fams = {lay.type_family[t] for t in present if lay.w100_of_type[t] == side}
        for fam in fams:
            types = [t for t in present if lay.type_family[t] == fam]
            for t in types:
                w_type[t] = 1.0 / (len(fams) * len(types) * n_rows[t])
    return w_type[tcode]


# ---- stage 1: rows per family, type and fold --------------------------------------------------------
def rows_section(lay: Layout, tcode, y8, manifests) -> dict:
    out = {"per_family": _by_name(lay.families, np.bincount(y8, minlength=len(lay.families))),
           "per_type": {t: {"family": lay.type_family[i], "rows": int(lay.stop[i] - lay.start[i])}
                        for i, t in enumerate(lay.types)},
           "per_fold": {}}
    for name, m in manifests.items():
        folds = []
        for f in m["folds"]:
            entry = {"fold": f["fold"]}
            for role in ROLES:
                types = _counts(f[role], tcode, len(lay.types))
                entry[role] = {"rows": int(types.sum()),
                               "per_family": _by_name(lay.families, _counts(f[role], y8, len(lay.families))),
                               "per_type": {t: int(c) for t, c in zip(lay.types, types) if c}}
            for key in ("test_types", "val_types"):
                if key in f:
                    entry[key] = f[key]
            folds.append(entry)
        out["per_fold"][name] = folds
    return out


# ---- stage 2: one pass for the file's own groups (dup_group), the screen sample and per-file columns --
def base_pass(src, lay: Layout, features: list[str], per_type: int, log) -> dict:
    cols = ["row_id", "dup_group", "conflict", "sample_rate"] + features
    missing = [c for c in cols if c not in src.columns]
    if missing:
        raise SystemExit(f"the cleaned file has no column(s) {missing}")
    if src.n_rows != lay.n:
        raise SystemExit(f"the cleaned file has {src.n_rows:,} rows, the cleaning report says {lay.n:,}: "
                         f"they are not from the same run")
    keep = {c: np.empty(lay.n, np.float32) for c in ("IAT", "Number") if c in features}
    dup = np.empty(lay.n, np.int64)
    conflict = np.empty(lay.n, np.int8)
    rate = np.empty(lay.n, np.float32)
    sample_ids = np.concatenate([even_positions(a, b, per_type) for a, b in zip(lay.start, lay.stop)])
    sample = np.empty((len(sample_ids), len(features)), np.float32)
    max_abs = np.zeros(len(features))
    off = 0
    for b in src.batches(cols):
        m = len(b["row_id"])
        if not np.array_equal(b["row_id"], np.arange(off, off + m)):
            raise SystemExit(f"row_id is not 0, 1, 2, ... near row {off:,}: the file is not in cleaner v1 order")
        dup[off:off + m] = b["dup_group"]
        conflict[off:off + m] = b["conflict"]
        rate[off:off + m] = b["sample_rate"]
        for c in keep:
            keep[c][off:off + m] = b[c]
        lo, hi = np.searchsorted(sample_ids, [off, off + m])
        local = sample_ids[lo:hi] - off
        for j, c in enumerate(features):
            v = np.asarray(b[c], np.float32)
            sample[lo:hi, j] = v[local]
            if m:
                max_abs[j] = max(max_abs[j], float(np.max(np.abs(v))))
        off += m
        log(f"    {off:,} / {lay.n:,} rows", end="\r")
    log("")
    if off != lay.n:
        raise SystemExit(f"read {off:,} rows, expected {lay.n:,}")
    return {"dup": dup, "conflict": conflict, "rate": rate, "keep": keep, "sample_ids": sample_ids,
            "sample": sample, "max_abs": max_abs}


def feature_set_keys(src, lay: Layout, columns: list[str], log) -> np.ndarray:
    """One 64-bit key per row over `columns` (the cleaner's row_hash: equal rows always get equal keys)."""
    keys = np.empty(lay.n, np.uint64)
    off = 0
    for b in src.batches(columns):
        m = len(b[columns[0]])
        keys[off:off + m] = row_hash(np.column_stack([np.asarray(b[c], np.float32) for c in columns]))
        off += m
        log(f"    {off:,} / {lay.n:,} rows", end="\r")
    log("")
    if off != lay.n:
        raise SystemExit(f"read {off:,} rows, expected {lay.n:,}")
    return keys


# ---- stage 3: duplicates, conflicts, twins and the ceiling for one feature set ----------------------
def analyse_set(gid, n_groups, lay: Layout, tcode, labels: dict, w, manifests, stored_conflict=None) -> dict:
    y8, y7 = labels["c8"][0], labels["c7m"][0]
    K = len(lay.families)
    gs = group_stats(gid, n_groups, y8, y7, tcode)
    has_copy = (gs["size"] > 1)[gid]
    code = gs["code"][gid]
    cross, same, ba = code == 2, code == 1, gs["benign_attack"][gid]
    rows = np.bincount(y8, minlength=K)
    with_copy = np.bincount(y8[has_copy], minlength=K)
    distinct = [int(((gs["fam_bits"] >> k) & 1).sum()) for k in range(K)]
    dup = {"rows_with_an_exact_copy": int(has_copy.sum()), "share_with_an_exact_copy": _share(has_copy.sum(), lay.n),
           "duplicate_share": _share(lay.n - n_groups, lay.n), "per_family": {}}
    for k, fam in enumerate(lay.families):
        dup["per_family"][fam] = {"rows": int(rows[k]), "rows_with_an_exact_copy": int(with_copy[k]),
                                  "share_with_an_exact_copy": _share(with_copy[k], rows[k]),
                                  "distinct": distinct[k], "duplicate_share": _share(rows[k] - distinct[k], rows[k])}
    cross_t = np.bincount(tcode[cross], minlength=len(lay.types))
    conflicts = {"rows_cross_family": int(cross.sum()), "rows_same_family": int(same.sum()),
                 "rows_benign_vs_attack": int(ba.sum()), "rows_c7m": int(gs["c7m_mixed"][gid].sum()),
                 "share_cross_family": _share(cross.sum(), lay.n), "share_same_family": _share(same.sum(), lay.n),
                 "share_benign_vs_attack": _share(ba.sum(), lay.n),
                 "per_family_cross": _by_name(lay.families, np.bincount(y8[cross], minlength=K) / np.maximum(rows, 1)),
                 "per_family_same": _by_name(lay.families, np.bincount(y8[same], minlength=K) / np.maximum(rows, 1)),
                 "per_type_cross": {t: _share(c, lay.stop[i] - lay.start[i])
                                    for i, (t, c) in enumerate(zip(lay.types, cross_t))},
                 "per_fold": {}}
    if stored_conflict is not None:                              # the cleaner's own codes for the same groups
        conflicts["rows_differing_from_stored_conflict"] = int(np.count_nonzero(stored_conflict != code))
    del has_copy, code
    out = {"groups": int(n_groups), "duplicates": dup, "conflicts": conflicts, "twins": {},
           "ceiling": {"whole_file": ceiling([[0, lay.n]], gid, cross, ba, labels, w)}}
    for name, m in manifests.items():
        conf_folds, twin_folds, ceil_folds = [], [], []
        for f in m["folds"]:
            in_train = np.zeros(n_groups, bool)
            for a, b in f["train"]:
                in_train[gid[a:b]] = True
            c_entry, t_entry, x_entry = {"fold": f["fold"]}, {"fold": f["fold"]}, {"fold": f["fold"]}
            for role in ("val", "test"):
                r = _counts(f[role], y8, K)
                n = int(r.sum())
                tw = np.zeros(K, np.int64)
                for a, b in f[role]:
                    tw += np.bincount(y8[a:b][in_train[gid[a:b]]], minlength=K)
                cc, cs = _counts(f[role], y8, K, cross), _counts(f[role], y8, K, same)
                per_fam = lambda c: _by_name(lay.families, np.where(r > 0, c / np.maximum(r, 1), np.nan))
                t_entry[role] = {"rows": n, "share": _share(tw.sum(), n), "per_family": per_fam(tw)}
                c_entry[role] = {"rows": n, "share_cross_family": _share(cc.sum(), n),
                                 "share_same_family": _share(cs.sum(), n), "per_family_cross": per_fam(cc)}
                x_entry[role] = ceiling(f[role], gid, cross, ba, labels, w)
            conf_folds.append(c_entry)
            twin_folds.append(t_entry)
            ceil_folds.append(x_entry)
        out["conflicts"]["per_fold"][name] = conf_folds
        out["twins"][name] = twin_folds
        out["ceiling"][name] = ceil_folds
        out["ceiling"][f"{name}_test_mean"] = {
            task: {s: _r(np.mean([c["test"][task][s] for c in ceil_folds])) for s in ("macro_f1", "accuracy")}
            for task in labels}
    return out


def ceiling(ranges, gid, cross, ba, labels: dict, w) -> dict:
    """Scores of the majority-label rule on one block (the rows in `ranges`). Only rows of cross-family conflict
    groups can be wrong under this rule (for c8, c7m and binary alike), so only they are grouped; every other
    row is predicted as its own label."""
    idx = _rows_of(ranges, cross)
    g = gid[idx]
    wr = None if w is None else w[idx]
    out = {}
    for task, (y, classes) in labels.items():
        k = len(classes)
        allc = np.zeros(k)
        for a, b in ranges:
            allc += np.bincount(y[a:b], weights=None if w is None else w[a:b], minlength=k)[:k]
        yr = y[idx].astype(np.int64)
        pr = majority_labels(g, yr, k, wr)
        conf = np.bincount(yr * k + pr, weights=wr, minlength=k * k).astype(np.float64).reshape(k, k)
        conf += np.diag(allc - np.bincount(yr, weights=wr, minlength=k)[:k])
        out[task] = scores_from_confusion(conf, classes)
    # attack rows identical to a benign row of the SAME block cannot be flagged without flagging that benign row
    jdx = _rows_of(ranges, ba)
    is_attack = labels["binary"][0][jdx] == 1
    _, local = np.unique(gid[jdx], return_inverse=True)
    local = local.reshape(-1)
    size = int(local.max()) + 1 if len(local) else 0
    has_b, has_a = np.zeros(size, bool), np.zeros(size, bool)
    has_b[local[~is_attack]] = True
    has_a[local[is_attack]] = True
    mixed = (has_b & has_a)[local]
    n_att = sum(int(np.count_nonzero(labels["binary"][0][a:b])) for a, b in ranges)
    n_ben = sum(b - a for a, b in ranges) - n_att
    out["binary"]["attack_rows_identical_to_a_benign_row"] = _share(np.count_nonzero(mixed & is_attack), n_att)
    out["binary"]["benign_rows_identical_to_an_attack_row"] = _share(np.count_nonzero(mixed & ~is_attack), n_ben)
    return out


# ---- stage 4: window-size screen and Number -------------------------------------------------------
def _screen_row(name, source, x, tt, lay: Layout) -> dict:
    w, pos = family_balanced_weights(tt, lay), lay.w100_of_type[tt]
    auc, bal, thr = weighted_auc(x, pos, w)
    fam = lay.fam_of_type[tt]
    within = []
    for side in (True, False):
        fams = [int(f) for f in np.unique(fam[pos == side])]
        for i, f1 in enumerate(fams):
            for f2 in fams[i + 1:]:
                m = (fam == f1) | (fam == f2)
                a, _, _ = weighted_auc(x[m], fam[m] == f1, w[m])
                within.append(abs(a - 0.5) * 2)
    finite = np.isfinite(x)
    return {"feature": name, "source": source, "auc": _r(auc), "separation": _r(abs(auc - 0.5) * 2),
            "w100_side": "above" if auc >= 0.5 else "below", "best_balanced_accuracy": _r(bal), "threshold": _r(thr),
            "within_side_separation_mean": _r(np.mean(within)) if within else float("nan"),
            "within_side_separation_max": _r(np.max(within)) if within else float("nan"),
            "w100_median": _r(np.median(x[pos & finite])) if (pos & finite).any() else float("nan"),
            "w10_median": _r(np.median(x[~pos & finite])) if (~pos & finite).any() else float("nan")}


def screen_section(base: dict, lay: Layout, features: list[str], tcode_full, extra: dict | None = None) -> dict:
    tt = tcode_full[base["sample_ids"]]
    rows = []
    for j, c in enumerate(features):
        row = _screen_row(c, "cleaned file", base["sample"][:, j].astype(np.float64), tt, lay)
        row["max_abs_value"] = _r(base["max_abs"][j])
        row["above_float32_exact_integers"] = bool(base["max_abs"][j] > 2 ** 24)
        rows.append(row)
    for c, (x, t_raw) in (extra or {}).items():
        rows.append(_screen_row(c, "raw CSVs", x, t_raw, lay))
    rows.sort(key=lambda r: 1.0 if np.isnan(r["separation"]) else -r["separation"])
    return {"w100_families": list(W100_FAMILIES),
            "w10_families": [f for f in lay.families if f not in W100_FAMILIES],
            "sample": {"rows_per_type_max": int(np.bincount(tt).max()), "rows": int(len(tt)),
                       "weights": "family-balanced: every family weighs the same inside its side, every attack type "
                                  "the same inside its family"},
            "features": rows}


def per_type_values(v: np.ndarray, lay: Layout) -> dict:
    """Value distribution of one column per attack type and per family (exact, from value counts)."""
    out = {"per_type": {}, "per_family": {}}
    for k, fam in enumerate(lay.families):
        parts = []
        for t, a, b, f in zip(lay.types, lay.start, lay.stop, lay.fam_of_type):
            if f == k:
                parts.append(counts_of(v[a:b]))
                out["per_type"][t] = summary_from_counts(*parts[-1])
        if parts:
            out["per_family"][fam] = summary_from_counts(*merge_counts(parts))
    return out


# ---- stage 5: raw CSVs (Time_To_Live, IGMP, IAT in full precision) and IAT per file ---------------
def _read_raw(path: Path, wanted) -> dict[str, np.ndarray]:
    head = pd.read_csv(path, nrows=0).columns
    cols = [c for c in wanted if c in head]
    if not cols:
        return {}
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    return {c: pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64) for c in cols}


def iat_by_type(per_file: list[dict]) -> dict:
    """Per attack type: distinct IAT values per file, and whether the numbered files follow each other in IAT."""
    out = {}
    for t in dict.fromkeys(p["native_label"] for p in per_file):
        mine = sorted((p for p in per_file if p["native_label"] == t and p.get("rows")), key=lambda p: p["file_seq"])
        if not mine:
            continue
        pairs = list(zip(mine[:-1], mine[1:]))
        out[t] = {"files": len(mine), "rows": sum(p["rows"] for p in mine),
                  "distinct_per_file_median": _r(np.median([p["distinct"] for p in mine])),
                  "distinct_per_file_max": max(p["distinct"] for p in mine),
                  "files_that_never_step_down": sum(p["steps_down_share"] == 0 for p in mine),
                  "consecutive_file_pairs": len(pairs),
                  "consecutive_files_in_order": sum(a["max"] <= b["min"] for a, b in pairs),
                  "consecutive_file_medians_rising": sum(a["median"] < b["median"] for a, b in pairs)}
    return out


def raw_section(raw: Path, report: dict, lay: Layout, per_type: int, log) -> dict:
    found = list_csvs(raw)
    files = {p.name: p for p in found}
    if len(files) != len(found):
        raise SystemExit(f"two raw CSVs under {raw} share a file name; point --raw at one copy of the dataset")
    entries = list(report["files"])
    missing = [e["file"] for e in entries if e["file"] not in files]
    if missing:
        raise SystemExit(f"{len(missing)} raw CSVs named in the cleaning report are not under {raw} (e.g. "
                         f"{missing[:3]}). Run with --raw none to skip the raw part.")
    per_file, ttl_t, ttl_f, igmp_t, sample_x, sample_t, t0 = [], {}, {}, {}, [], [], time.time()
    done = 0
    for k, fam in enumerate(lay.families):                        # family by family: memory stays one family
        fam_ttl = []
        for t in [t for t, f in zip(lay.types, lay.fam_of_type) if f == k]:
            ttl_parts, igmp_parts = [], []
            for e in [e for e in entries if e["native_label"] == t]:
                cols = _read_raw(files[e["file"]], RAW_COLUMNS)
                rec = {"file": e["file"], "native_label": t, "file_seq": e["file_seq"]}
                rec.update(order_summary(cols["IAT"]) if "IAT" in cols else {"rows": 0})
                per_file.append(rec)
                if "Time_To_Live" in cols:
                    ttl_parts.append(cols["Time_To_Live"])
                if "IGMP" in cols:
                    igmp_parts.append(cols["IGMP"])
                done += 1
                log(f"    raw {done}/{len(entries)} {e['file']} ({time.time() - t0:.0f} s)", end="\r")
            if ttl_parts:
                v = np.concatenate(ttl_parts)
                v = v[np.isfinite(v)]
                fam_ttl.append(counts_of(v))
                ttl_t[t] = summary_from_counts(*fam_ttl[-1])
                pick = even_positions(0, len(v), per_type)
                sample_x.append(v[pick])
                sample_t.append(np.full(len(pick), lay.types.index(t), np.int16))
            if igmp_parts:
                v = np.concatenate(igmp_parts)
                igmp_t[t] = {"mean": _r(np.nanmean(v)), "share_nonzero": _share(np.count_nonzero(v > 0), len(v))}
        if fam_ttl:
            ttl_f[fam] = summary_from_counts(*merge_counts(fam_ttl))
    log("")
    out = {"source": str(raw), "files_read": len(entries),
           "columns_found": sorted(c for c in RAW_COLUMNS if (c == "IAT" and any(p.get("rows") for p in per_file))
                                   or (c == "Time_To_Live" and ttl_t) or (c == "IGMP" and igmp_t)),
           "iat_per_file": per_file, "iat_per_type": iat_by_type(per_file)}
    if ttl_t:
        out["time_to_live"] = {"per_type": ttl_t, "per_family": ttl_f}
        out["_ttl_sample"] = (np.concatenate(sample_x), np.concatenate(sample_t))
    if igmp_t:
        out["igmp"] = igmp_t
    return out


def iat_from_cleaned(iat: np.ndarray, report: dict) -> dict:
    per_file = []
    for e in report["files"]:
        if e.get("rows_written", 0) <= 0:
            continue
        rec = {"file": e["file"], "native_label": e["native_label"], "file_seq": e["file_seq"]}
        rec.update(order_summary(iat[e["row_start"]:e["row_stop"]]))
        per_file.append(rec)
    return {"source": "cleaned file (float32: values above 2^24 = 16,777,216 are rounded, so distinct counts there are "
                      "lower bounds of the raw ones; the direction of change is kept)",
            "iat_per_file": per_file, "iat_per_type": iat_by_type(per_file)}


# ---- the whole report ------------------------------------------------------------------------------
def _print(msg="", end="\n"):
    print(msg, end=end, flush=True)


def build(src, report: dict, manifests: dict, *, raw: Path | None = None, feature_sets=tuple(FEATURE_SETS),
          per_type: int = 20_000, dataset: str = DATASET, log=_print) -> dict:
    """The DATA-4 / DATA-1b report as a dict. `src` gives the cleaned rows (ParquetSource or FrameSource),
    `manifests` is {protocol: fold manifest}, `raw` the folder of the per-capture CSVs (None: skipped).
    `log(msg, end=...)` receives progress lines."""
    t_start = time.time()

    def stage(name):
        log(f"[{time.time() - t_start:6.0f} s] {name}")

    lay = Layout(report, dataset)
    features = list(report["features"])
    unknown = [s for s in feature_sets if s not in FEATURE_SETS]
    if unknown:
        raise SystemExit(f"unknown feature set(s) {unknown}; choose from {list(FEATURE_SETS)}")
    feature_sets = [s for s in FEATURE_SETS if s in feature_sets]      # "all" first: its keys are already read
    for name, m in manifests.items():
        if m["n_rows"] != lay.n:
            raise SystemExit(f"manifest {name} covers {m['n_rows']:,} rows, the cleaned file has {lay.n:,}")
    tcode = lay.type_code()
    y8 = lay.fam_of_type[tcode]
    labels = {"c8": (y8, lay.families), "c7m": (lay.c7m_of_type[tcode], lay.families7),
              "binary": ((y8 != 0).astype(np.int8), ["benign", "attack"])}

    stage("rows per family, attack type and fold")
    out = {"task": "DATA-4 and DATA-1b", "version": VERSION, "dataset": dataset, "about": ABOUT,
           "families": lay.families, "families_c7m": lay.families7, "features": features,
           "settings": {"per_type_sample": per_type,
                        "feature_sets": {s: [f for f in features if f not in FEATURE_SETS[s]] for s in feature_sets}},
           "inputs": {"cleaning_report": {"settings": report.get("settings"), "totals": report.get("totals")},
                      "manifests": {n: {"protocol": m["protocol"], "n_folds": m["n_folds"], "gap": m["gap"],
                                        "source": m.get("source")} for n, m in manifests.items()}},
           "rows": rows_section(lay, tcode, y8, manifests)}

    stage("pass over the cleaned file: dup_group, conflict, sample_rate, IAT, Number, screen sample")
    base = base_pass(src, lay, features, per_type, log)
    rate = base.pop("rate")
    all_one = bool(np.all(rate == 1))
    w = None if all_one else (1.0 / rate.astype(np.float64))    # metrics weight a row by 1 / sample_rate
    out["inputs"]["sample_rate_all_one"] = all_one
    del rate
    stage("Number per attack type; IAT per file from the cleaned file")
    keep = base.pop("keep")
    if "Number" in keep:
        out["number"] = per_type_values(keep.pop("Number"), lay)
    if "IAT" in keep:
        out["iat_cleaned"] = iat_from_cleaned(keep.pop("IAT"), report)
    del keep

    raw_out = None
    if raw is not None and Path(raw).is_dir():
        stage(f"raw CSVs under {raw}: Time_To_Live, IGMP, IAT in full precision")
        raw_out = raw_section(Path(raw), report, lay, per_type, log)
    else:
        log(f"           raw CSVs skipped ({'folder not found: ' + str(raw) if raw else 'not asked for'})")
    stage("window-size screen")
    extra = {"Time_To_Live": raw_out.pop("_ttl_sample")} if raw_out and "_ttl_sample" in raw_out else {}
    out["screen"] = screen_section(base, lay, features, tcode, extra)
    del base["sample"], base["sample_ids"]
    if raw_out:
        out["raw"] = raw_out

    out["feature_sets"] = {}
    dup = base.pop("dup") if "all" in feature_sets else None
    base.pop("dup", None)
    for s in feature_sets:
        cols = [f for f in features if f not in FEATURE_SETS[s]]
        stage(f"feature set {s!r} ({len(cols)} features): numbering the groups of identical rows")
        if s == "all":
            gid, n_groups = dense_ids(dup)
            dup = None                                            # free the 64-bit keys before the analysis
        else:
            gid, n_groups = dense_ids(feature_set_keys(src, lay, cols, log))
        stage(f"feature set {s!r}: duplicates, conflicts, twins, ceiling")
        out["feature_sets"][s] = {"features": len(cols), **analyse_set(
            gid, n_groups, lay, tcode, labels, w, manifests, base["conflict"] if s == "all" else None)}
        del gid
    out["seconds"] = round(time.time() - t_start, 1)
    return out


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---- printed summary --------------------------------------------------------------------------------
def _pct(x) -> str:
    return "   n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:5.1f}%"


def summary(q: dict) -> str:
    L, fams = [], q["families"]
    n = sum(q["rows"]["per_family"].values())
    L.append(f"DATA-4 quality report: {n:,} rows, {len(q['rows']['per_type'])} attack types, {len(fams)} families")
    L.append("rows per family: " + ", ".join(f"{f} {q['rows']['per_family'][f]:,}" for f in fams))
    L.append("")
    L.append("feature set         groups      copy   cross-fam  same-fam  ben/att | ceiling macro-F1, whole file "
             "and mean of A test folds: c8 | c7m | binary accuracy")
    for s, v in q["feature_sets"].items():
        c, x = v["conflicts"], v["ceiling"]
        am = x.get("A_test_mean", {})
        L.append(f"{s:17s} {v['groups']:11,}  {_pct(v['duplicates']['share_with_an_exact_copy'])}   "
                 f"{_pct(c['share_cross_family'])}   {_pct(c['share_same_family'])}  {_pct(c['share_benign_vs_attack'])} | "
                 f"c8 {x['whole_file']['c8']['macro_f1']:.3f} / {am.get('c8', {}).get('macro_f1', float('nan')):.3f} | "
                 f"c7m {x['whole_file']['c7m']['macro_f1']:.3f} / {am.get('c7m', {}).get('macro_f1', float('nan')):.3f} | "
                 f"bin {x['whole_file']['binary']['accuracy']:.4f}")
        if c.get("rows_differing_from_stored_conflict"):
            L.append(f"{'':17s} NOTE: recomputed conflict codes differ from the file's on "
                     f"{c['rows_differing_from_stored_conflict']:,} rows")
    L.append("")
    L.append("share of TEST rows with an exact twin among that fold's training rows, folds 0..4")
    for s, v in q["feature_sets"].items():
        L.append(f"  {s:17s} " + "   ".join(f"{p}: " + " ".join(_pct(f['test']['share']) for f in folds)
                                            for p, folds in v["twins"].items()))
    s0 = next(iter(q["feature_sets"]), None)
    folds = q["feature_sets"][s0]["ceiling"].get("A", []) if s0 else []
    if folds:
        L.append(f"ceiling F1 per family, protocol A test folds (mean), set {s0!r}: " + ", ".join(
            f"{f} {np.nanmean([x['test']['c8']['per_class_f1'][f] for x in folds]):.3f}" for f in fams))
        L.append(f"attack rows identical to a benign row in the same A test fold (mean): "
                 f"{_pct(np.mean([x['test']['binary']['attack_rows_identical_to_a_benign_row'] for x in folds]))}")
    L.append("")
    sc = q["screen"]
    L.append(f"window-size screen: {'/'.join(sc['w100_families'])} (100-packet windows) vs the rest, "
             f"{sc['sample']['rows']:,} sample rows, family-balanced. sep = |AUC-0.5|*2; within = same-side families")
    L.append("  feature            AUC    sep    within(mean/max)  bal.acc  W100 median     W10 median")
    for r in sc["features"][:15]:
        L.append(f"  {r['feature'][:18]:18s} {r['auc']:.3f}  {r['separation']:.3f}  "
                 f"{r['within_side_separation_mean']:.3f} / {r['within_side_separation_max']:.3f}    "
                 f"{r['best_balanced_accuracy']:.3f}    {r['w100_median']:<15.6g} {r['w10_median']:<12.6g}"
                 f"{'  (raw CSVs)' if r['source'] != 'cleaned file' else ''}")
    for title, part in (("Number", q.get("number", {}).get("per_family", {})),
                        ("Time_To_Live (raw CSVs)", q.get("raw", {}).get("time_to_live", {}).get("per_family", {}))):
        if part:
            L.append(f"{title} per family: median [p01, p99], distinct values, whole-number share, top values (share)")
            for f, v in part.items():
                if v.get("rows"):
                    tops = ", ".join(f"{a:g} ({100 * b:.0f}%)" for a, b in v["top_values"][:3])
                    L.append(f"  {f:11s} {v['median']:<10g} [{v['p01']:g}, {v['p99']:g}]  {v['distinct']:,} values, "
                             f"whole {_pct(v['integer_share'])}; {tops}")
    for key, title in (("raw", "raw CSVs, float64"), ("iat_cleaned", "cleaned file, float32")):
        part = q.get(key, {})
        dist = [p["distinct"] for p in part.get("iat_per_file", []) if p.get("rows")]
        if not dist:
            continue
        never_down = sum(p.get("steps_down_share") == 0 for p in part["iat_per_file"] if p.get("rows"))
        L.append(f"IAT per file ({title}): {len(dist)} files; distinct values per file median {np.median(dist):,.0f}, "
                 f"min {min(dist):,}, max {max(dist):,}; files where IAT never steps down: {never_down}")
        pairs = sum(v["consecutive_file_pairs"] for v in part["iat_per_type"].values())
        if pairs:
            ordered = sum(v["consecutive_files_in_order"] for v in part["iat_per_type"].values())
            rising = sum(v["consecutive_file_medians_rising"] for v in part["iat_per_type"].values())
            L.append(f"  numbered files of one attack: IAT ranges in file order for {ordered} of {pairs} consecutive "
                     f"pairs; medians rising for {rising}")
    L.append(f"done in {q['seconds'] / 60:.1f} min")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="DATA-4 / DATA-1b data quality report for the cleaned file v1.")
    ap.add_argument("--parquet", default=str(PARQUET))
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--folds", nargs="+", default=[str(p) for p in FOLDS], help="fold manifests (protocols A, B)")
    ap.add_argument("--raw", default=str(RAW), help="folder of the per-capture CSVs; 'none' to skip")
    ap.add_argument("--feature-sets", default=",".join(FEATURE_SETS), help=f"comma list from {list(FEATURE_SETS)}")
    ap.add_argument("--per-type", type=int, default=20_000, help="screen sample: rows per attack type")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    report = json.loads(Path(a.report).read_text())
    manifests = {}
    for p in a.folds:
        m = load_manifest(p)
        manifests[m["protocol"]] = m
    raw = None if a.raw.strip().lower() in ("", "none") else Path(a.raw)
    sets = [s.strip() for s in a.feature_sets.split(",") if s.strip()]
    print(f"DATA-4: reads the cleaned file {1 + sum(s != 'all' for s in sets)} time(s)"
          f"{' and the raw CSVs once' if raw and raw.is_dir() else ''}; needs about 2-3 GB of memory. "
          f"If it stops with 'Killed', Docker ran out of memory: free memory or raise Docker's limit, then rerun",
          flush=True)
    q = build(ParquetSource(a.parquet), report, manifests, raw=raw, feature_sets=sets, per_type=a.per_type)
    q["inputs"].update({"parquet": str(a.parquet), "parquet_bytes": Path(a.parquet).stat().st_size,
                        "report": str(a.report), "folds": list(a.folds), "raw": str(raw) if raw else None,
                        "git_commit": git_commit(), "python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(q, indent=1) + "\n")
    print()
    print(summary(q))
    print(f"wrote {out} ({out.stat().st_size / 1e3:,.0f} kB)")


if __name__ == "__main__":
    main()
