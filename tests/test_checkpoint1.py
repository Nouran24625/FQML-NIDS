"""Row reader, per-type sampling and the Checkpoint 1 pipeline on a small made-up dataset."""
import json

import numpy as np
import pandas as pd
import pytest

from fqml.data.rows import cap_per_label, even_pick, read_rows
from fqml.data.schema import CICIOT2023_FEATURES

FEATS = CICIOT2023_FEATURES[:6]
TYPES = ["BenignTraffic", "DDoS-ICMP_Flood", "DoS-UDP_Flood", "Recon-PortScan", "XSS",
         "DictionaryBruteForce", "DNS_Spoofing", "Mirai-udpplain"]          # one per family, in family order


def test_even_pick_and_cap():
    assert list(even_pick(np.arange(10), 20)) == list(range(10))
    assert list(even_pick(np.arange(10), 3)) == [0, 4, 9] or list(even_pick(np.arange(10), 3)) == [0, 5, 9]
    ranges = {"a": (0, 100), "b": (100, 105), "c": (105, 300)}
    rows = np.arange(0, 300, 2)
    got = cap_per_label(rows, ranges, 10)
    assert len(got) == 10 + 3 + 10 and set(got) <= set(rows) and list(got) == sorted(got)
    rnd = cap_per_label(rows, ranges, 10, np.random.default_rng(0))
    assert len(rnd) == 23 and len(set(rnd)) == 23 and set(rnd) <= set(rows)


@pytest.fixture
def cleaned(tmp_path):
    pytest.importorskip("pyarrow")
    from fqml.data.clean_ciciot2023_v1 import ParquetSink, clean
    rng = np.random.default_rng(0)
    files = []
    for k, t in enumerate(TYPES):
        for part in ("", "1"):
            X = rng.normal(loc=3.0 * k, scale=1.0, size=(150, len(FEATS))).round(3)
            if t == "DoS-UDP_Flood":                       # half of DoS identical to DDoS rows, as in D8
                X[:75] = rng.normal(loc=3.0, scale=0.4, size=(75, len(FEATS))).round(0)
            if t == "DDoS-ICMP_Flood":
                X[:75] = rng.normal(loc=3.0, scale=0.4, size=(75, len(FEATS))).round(0)
            folder = tmp_path / "raw" / t
            folder.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(X, columns=FEATS).to_csv(folder / f"{t}{part}.pcap.csv", index=False)
            files.append(folder / f"{t}{part}.pcap.csv")
    out = tmp_path / "clean.parquet"
    sink = ParquetSink(out)
    report = clean(files, sink, per_label_cap=0, log=lambda m: None)
    sink.close()
    (tmp_path / "report.json").write_text(json.dumps(report))
    return out, report


def test_read_rows(cleaned):
    out, report = cleaned
    full = pd.read_parquet(out)
    ids = np.array([0, 1, 149, 150, 151, 1000, len(full) - 1])
    got = read_rows(out, ids, ["y", FEATS[0]])
    assert list(got["row_id"]) == list(ids) and list(got.columns) == ["row_id", "y", FEATS[0]]
    assert np.allclose(got[FEATS[0]], full[FEATS[0]].to_numpy()[ids])
    assert (got["y"].to_numpy() == full["y"].to_numpy()[ids]).all()
    assert len(read_rows(out, np.zeros(0, np.int64), ["y"])) == 0
    with pytest.raises(ValueError):
        read_rows(out, [5, 3], ["y"])
    with pytest.raises(ValueError):
        read_rows(out, [len(full)], ["y"])


@pytest.mark.parametrize("split", ["R", "A-gap0", "A-gap10"])
def test_checkpoint1_runs(cleaned, split):
    pytest.importorskip("xgboost")
    from fqml.eval.checkpoint1 import merge_ddos_dos, run_setting
    out, report = cleaned
    r = run_setting(split, "all", data=out, report=report, fold=0, seed=0, train_per_type=100, threads=1,
                    log=lambda m: None)
    assert r["val_rows"] > 0 and r["train_rows"] <= 100 * len(TYPES)
    for k in ("scores_all", "scores_no_twin", "scores_no_conflict"):
        s = r[k]
        assert 0 <= s["c8"]["macro_f1"] <= 1 and len(s["c8"]["per_class_f1"]) == 8
        assert len(s["c7m"]["per_class_f1"]) == 7 and 0 <= s["binary"]["false_positive_rate"] <= 1
    assert r["scores_all"]["binary"]["f1"] > 0.9                # benign is far from every attack here
    assert r["val_conflict_share"] > 0 and r["top_features"] and r["weighting"] == "balanced"
    assert set(r["per_type"]) == set(TYPES) and all(0 <= d["recall"] <= 1 for d in r["per_type"].values())
    assert sum(d["rows"] for d in r["per_type"].values()) == r["val_rows"]
    json.dumps(r)
    y7, fam7 = merge_ddos_dos(np.arange(8), report["families"])
    assert list(y7) == [0, 1, 1, 2, 3, 4, 5, 6] and fam7[1] == "DDoS_DoS" and len(fam7) == 7


def test_class_weights():
    from fqml.eval.checkpoint1 import class_weights
    y = np.array([0] * 90 + [1] * 9 + [2] * 1)
    b = class_weights(y, 3, "balanced")
    assert np.allclose([b[y == k].sum() for k in range(3)], 100 / 3)          # equal total weight per family
    s = class_weights(y, 3, "sqrt")
    assert np.isclose(s.mean(), 1) and np.isclose(s[y == 2][0] / s[y == 0][0], np.sqrt(90))
    assert (class_weights(y, 3, "none") == 1).all()
    with pytest.raises(ValueError):
        class_weights(y, 3, "inverse")
    with pytest.raises(RuntimeError):
        class_weights(np.array([0, 0, 1]), 3, "sqrt")


@pytest.mark.parametrize("weighting", ["sqrt", "none"])
def test_checkpoint1_weightings_run(cleaned, weighting):
    pytest.importorskip("xgboost")
    from fqml.eval.checkpoint1 import run_setting
    out, report = cleaned
    r = run_setting("A-gap10", "all", data=out, report=report, fold=0, seed=0, train_per_type=100, threads=1,
                    weighting=weighting, log=lambda m: None)
    assert r["weighting"] == weighting and r["scores_all"]["binary"]["f1"] > 0.9
