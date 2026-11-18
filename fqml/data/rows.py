"""Read chosen rows of a cleaned dataset without loading the whole file. Owner: Ganna.

The cleaned CICIoT2023 file has 46.8 M rows; one fold's validation block alone is 9.35 M. Experiments
name rows by `row_id` (= position in the file, INTERFACES 1.1), so a reader only has to open the row
groups that contain the wanted rows, read the wanted columns, and keep the wanted positions.

    ids = fold_rows(manifest, fold=0, role="val")             # fqml.data.folds
    df = read_rows(PARQUET, ids, columns=["y", "dup_group", "conflict", *features])
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CICIOT2023_V1 = ROOT / "data/interim/ciciot2023_v1/ciciot2023_clean_v1.parquet"


def read_rows(path, row_ids, columns) -> pd.DataFrame:
    """Rows at positions `row_ids` (sorted, unique, int) with `columns`, in row_id order.

    Opens only the row groups that hold at least one wanted row. The result has a `row_id` column
    (the positions) and a fresh 0..n-1 index."""
    import pyarrow.parquet as pq

    row_ids = np.asarray(row_ids, dtype=np.int64)
    if len(row_ids) and (np.any(np.diff(row_ids) <= 0) or row_ids[0] < 0):
        raise ValueError("row_ids must be sorted, unique and >= 0")
    pf = pq.ParquetFile(path)
    sizes = np.array([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    if len(row_ids) and row_ids[-1] >= starts[-1]:
        raise ValueError(f"row_id {row_ids[-1]} is past the end of the file ({starts[-1]} rows)")
    cols = [c for c in columns if c != "row_id"]
    group_of = np.searchsorted(starts, row_ids, side="right") - 1
    parts = []
    for g in np.unique(group_of):
        local = row_ids[group_of == g] - starts[g]
        table = pf.read_row_group(int(g), columns=cols)
        part = table.take(local).to_pandas()
        part.insert(0, "row_id", row_ids[group_of == g])
        parts.append(part)
    if not parts:
        empty = pf.schema_arrow.empty_table().select(cols).to_pandas()
        empty.insert(0, "row_id", np.zeros(0, np.int64))
        return empty
    return pd.concat(parts, ignore_index=True)


def even_pick(rows: np.ndarray, k: int) -> np.ndarray:
    """k rows spread evenly over `rows` (kept in order); all of them if there are at most k."""
    rows = np.asarray(rows, dtype=np.int64)
    if k <= 0 or len(rows) <= k:
        return rows
    return rows[np.linspace(0, len(rows) - 1, k).round().astype(np.int64)]


def cap_per_label(rows: np.ndarray, label_ranges: dict[str, tuple[int, int]], cap: int,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """At most `cap` of `rows` from every attack type (types are contiguous row_id ranges).

    rng=None: evenly spread over the type's rows in recorded order (time-blocked protocols).
    rng given: a random choice (the random-split reference). Returns sorted row_ids."""
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    out = []
    for start, stop in label_ranges.values():
        mine = rows[np.searchsorted(rows, start):np.searchsorted(rows, stop)]
        if rng is None or len(mine) <= cap:
            out.append(even_pick(mine, cap))
        else:
            out.append(np.sort(rng.choice(mine, cap, replace=False)))
    return np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)
