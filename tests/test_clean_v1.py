"""Cleaner v1 (Phase 0 amendment 1): order kept, duplicates marked not deleted, conflicts marked,
even sampling with a stored rate, row_id ranges per attack type."""
import json

import numpy as np
import pandas as pd
import pytest

from fqml.data.clean_ciciot2023_v1 import (ParquetSink, clean, even_sample, file_seq, label_ranges,
                                           row_hash)
from fqml.data.schema import CICIOT2023_FEATURES

FEATS = CICIOT2023_FEATURES[:6]


class ListSink:
    def __init__(self):
        self.parts = []

    def write(self, df):
        self.parts.append(df)

    def frame(self):
        return pd.concat(self.parts, ignore_index=True)


def _write(folder, name, rows, ts0=0.0):
    df = pd.DataFrame(rows, columns=FEATS)
    df.insert(0, "ts", ts0 + np.arange(len(df)))
    df["not_a_feature"] = 1
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / name, index=False)
    return folder / name


def _rows(seed, n):
    return np.random.default_rng(seed).normal(size=(n, len(FEATS))).round(3).tolist()


@pytest.fixture
def raw(tmp_path):
    """7 small capture files. The SHARED row appears in the benign capture, a DDoS file and a DoS file."""
    shared = [9.0] * len(FEATS)
    files = [
        _write(tmp_path / "Benign_Final", "BenignTraffic.pcap.csv", _rows(1, 30) + [shared]),
        _write(tmp_path / "Benign_Final", "BenignTraffic1.pcap.csv", _rows(2, 20), ts0=100),
        # numbered files must come out as 0, 2, 10 (not 0, 10, 2)
        _write(tmp_path / "DDoS-ICMP_Flood", "DDoS-ICMP_Flood.pcap.csv",
               _rows(3, 40) + [shared, [np.nan] + [1.0] * 5, [np.inf] + [1.0] * 5]),
        _write(tmp_path / "DDoS-ICMP_Flood", "DDoS-ICMP_Flood10.pcap.csv", _rows(5, 40), ts0=900),
        _write(tmp_path / "DDoS-ICMP_Flood", "DDoS-ICMP_Flood2.pcap.csv", _rows(3, 40), ts0=200),  # copy of file 0
        _write(tmp_path / "DDoS-UDP_Flood", "DDoS-UDP_Flood.pcap.csv", _rows(6, 25) + [[7.0] * 6]),
        _write(tmp_path / "DoS-UDP_Flood", "DoS-UDP_Flood.pcap.csv", _rows(7, 25) + [shared, [7.0] * 6]),
    ]
    return files


def test_file_seq_and_hash():
    assert [file_seq(n) for n in ("A.pcap.csv", "A2.pcap.csv", "A10.pcap.csv", "DDoS-HTTP_Flood-.pcap.csv")] == [0, 2, 10, 0]
    a = np.array([[0.0, 1.5, -2.0], [-0.0, 1.5, -2.0], [1.5, 0.0, -2.0], [0.0, 1.5, -2.0000002]], np.float32)
    h = row_hash(a)
    assert h.dtype == np.uint64 and h[0] == h[1]            # -0.0 equals 0.0
    assert len({int(v) for v in h}) == 3                    # column order and tiny differences matter
    big = np.random.default_rng(0).normal(size=(200_000, 5)).astype(np.float32)
    assert len(np.unique(row_hash(big))) == len(np.unique(big, axis=0))


def test_even_sample():
    assert even_sample(10, 0).all() and even_sample(10, 10).all() and even_sample(3, 10).all()
    for n, cap in [(10, 3), (1000, 7), (7_000_003, 1_000_000)]:
        m = even_sample(n, cap)
        assert m.sum() == cap
        gaps = np.diff(np.flatnonzero(m))
        assert gaps.max() - gaps.min() <= 1                 # evenly spaced


def test_clean_keeps_order_and_marks_duplicates(raw):
    sink = ListSink()
    rep = clean(raw, sink, per_label_cap=0, log=lambda m: None)
    df = sink.frame()
    # order: label-map order (benign first), then file number 0, 2, 10; rows as in the file
    assert list(dict.fromkeys(df["file"])) == [
        "BenignTraffic.pcap.csv", "BenignTraffic1.pcap.csv", "DDoS-UDP_Flood.pcap.csv",
        "DDoS-ICMP_Flood.pcap.csv", "DDoS-ICMP_Flood2.pcap.csv", "DDoS-ICMP_Flood10.pcap.csv",
        "DoS-UDP_Flood.pcap.csv"]
    assert (df["row_id"] == np.arange(len(df))).all()
    for _, g in df.groupby("file", sort=False):
        assert (np.diff(g["row_index"]) > 0).all() and (np.diff(g["ts"]) > 0).all()
    # only NaN / inf rows are removed; nothing else is deleted
    t = rep["totals"]
    assert t["rows_raw"] == 31 + 20 + 43 + 40 + 40 + 26 + 27 and t["rows_removed_nan_inf"] == 2
    assert len(df) == t["rows_written"] == t["rows_finite"] == t["rows_raw"] - 2
    icmp0 = df[df["file"] == "DDoS-ICMP_Flood.pcap.csv"]
    assert list(icmp0["row_index"]) == list(range(41))      # rows 41, 42 (NaN, inf) are gone
    # duplicates are marked: file 2 is a copy of file 0 -> same dup_group row by row
    a = icmp0["dup_group"].to_numpy()[:40]
    b = df[df["file"] == "DDoS-ICMP_Flood2.pcap.csv"]["dup_group"].to_numpy()
    assert (a == b).all() and df["dup_group"].dtype == np.int64
    assert t["distinct_feature_rows"] == len(df) - 40 - 2 - 1   # 40 copies, SHARED x3, the 7.0 row x2
    assert t["rows_with_an_exact_copy"] == 80 + 3 + 2
    # conflicts: SHARED spans Benign, DDoS and DoS (2); the 7.0 row spans DDoS and DoS (2)
    shared = df[(df[FEATS] == 9.0).all(axis=1)]
    assert len(shared) == 3 and (shared["conflict"] == 2).all() and shared["dup_group"].nunique() == 1
    assert t["rows_conflict_cross_family"] == 5 and t["conflict_groups_benign_vs_attack"] == 1
    assert (df.loc[df["conflict"] == 0, "dup_group"].isin(shared["dup_group"])).sum() == 0
    # labels, types
    assert set(df["family"]) == {"Benign", "DDoS", "DoS"} and (df.loc[df["family"] == "Benign", "y"] == 0).all()
    assert all(df[c].dtype == np.float32 for c in FEATS) and (df["sample_rate"] == 1).all()
    assert rep["features"] == FEATS and rep["key_columns"] == ["ts"] and rep["extra_columns_ignored"] == ["not_a_feature"]
    assert rep["columns"][:10] == ["row_id", "file", "file_seq", "row_index", "native_label", "family", "y",
                                   "dup_group", "conflict", "sample_rate"]
    # row_id ranges per attack type are contiguous and match the data
    for lab, (start, stop) in label_ranges(rep).items():
        assert (df["native_label"].to_numpy()[start:stop] == lab).all()
        assert (df["native_label"] == lab).sum() == stop - start
    json.dumps(rep)                                          # the report must be plain JSON


def test_same_family_conflict_code(tmp_path):
    v = [[3.0] * len(FEATS)]
    files = [_write(tmp_path / "a", "DDoS-ICMP_Flood.pcap.csv", _rows(1, 5) + v),
             _write(tmp_path / "b", "DDoS-UDP_Flood.pcap.csv", _rows(2, 5) + v)]
    sink = ListSink()
    rep = clean(files, sink, per_label_cap=0, log=lambda m: None)
    df = sink.frame()
    assert sorted(df["conflict"].unique()) == [0, 1] and (df["conflict"] == 1).sum() == 2
    assert rep["totals"]["rows_conflict_same_family"] == 2 and rep["totals"]["rows_conflict_cross_family"] == 0


def test_sampling_is_even_and_stored(raw):
    sink = ListSink()
    rep = clean(raw, sink, per_label_cap=30, log=lambda m: None)
    df = sink.frame()
    counts = df["native_label"].value_counts()
    assert counts["DDoS-ICMP_Flood"] == 30 and counts["BenignTraffic"] == 30
    assert counts["DDoS-UDP_Flood"] == 26 and counts["DoS-UDP_Flood"] == 27       # small types kept whole
    icmp = df[df["native_label"] == "DDoS-ICMP_Flood"]
    assert np.allclose(icmp["sample_rate"], 30 / 121) and icmp["file"].nunique() == 3   # every file contributes
    assert (df.loc[df["native_label"] == "DoS-UDP_Flood", "sample_rate"] == 1).all()
    assert (df["row_id"] == np.arange(len(df))).all()
    assert rep["per_label"]["DDoS-ICMP_Flood"]["rows_written"] == 30
    # 1 / sample_rate weights give back the real class sizes
    w = (1 / df["sample_rate"]).groupby(df["native_label"]).sum()
    assert round(w["DDoS-ICMP_Flood"]) == 121 and round(w["BenignTraffic"]) == 51


def test_drop_cross_family_policy(raw):
    sink = ListSink()
    rep = clean(raw, sink, per_label_cap=0, conflicts="drop-cross-family", log=lambda m: None)
    df = sink.frame()
    assert (df["conflict"] != 2).all() and len(df) == rep["totals"]["rows_finite"] - 5
    with pytest.raises(ValueError):
        clean(raw, ListSink(), per_label_cap=0, conflicts="delete-everything", log=lambda m: None)


def test_parquet_sink_round_trip(raw, tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    out = tmp_path / "out" / "clean.parquet"
    sink, ref = ParquetSink(out), ListSink()
    clean(raw, sink, per_label_cap=0, log=lambda m: None)
    assert not out.exists()                                  # nothing visible until close()
    sink.close()
    clean(raw, ref, per_label_cap=0, log=lambda m: None)
    back = pd.read_parquet(out)
    pd.testing.assert_frame_equal(back, ref.frame(), check_dtype=False)
    assert back["dup_group"].dtype == np.int64 and back[FEATS[0]].dtype == np.float32
    assert pq.ParquetFile(out).metadata.num_row_groups == 7  # one row group per source file
