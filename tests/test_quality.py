"""DATA-4 / DATA-1b quality report on a small made-up dataset built with cleaner v1. Every number the report
gives for twins, conflicts and the ceiling is checked against a slow, obvious pandas version."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from fqml.data.clean_ciciot2023_v1 import clean, label_ranges
from fqml.data.folds import build_time_blocked, build_unseen_type, fold_rows, twin_mask
from fqml.data.label_maps import family_index
from fqml.data.quality import (FrameSource, build, counts_of, dense_ids, even_positions, majority_labels,
                               merge_counts, scores_from_confusion, summary, summary_from_counts, weighted_auc)

FEATS = ["Header_Length", "Rate", "Tot sum", "IAT", "Number", "Variance"]      # cleaner order (schema order)
TYPES = ["BenignTraffic", "DDoS-ICMP_Flood", "DDoS-UDP_Flood", "DoS-UDP_Flood", "Recon-PortScan", "Recon-OSScan",
         "XSS", "SqlInjection", "DictionaryBruteForce", "DNS_Spoofing", "MITM-ArpSpoofing", "Mirai-udpplain",
         "Mirai-greip_flood"]
W100 = {"DDoS", "DoS", "Mirai"}
N = 120


def silent(*a, **k):
    pass


def approx(x):
    """The report rounds its numbers to 6 decimals."""
    return pytest.approx(x, abs=2e-6)


class ListSink:
    def __init__(self):
        self.parts = []

    def write(self, df):
        self.parts.append(df)

    def frame(self):
        return pd.concat(self.parts, ignore_index=True)


def _file(t: str, part: str, rng) -> pd.DataFrame:
    k = TYPES.index(t)
    fam = family_index([t], "ciciot2023")[0]
    w100 = ["Benign", "DDoS", "DoS", "Recon", "Web", "BruteForce", "Spoofing", "Mirai"][fam] in W100
    X = np.column_stack([rng.normal(10 * k, 1, N).round(1), rng.normal(5 * k, 1, N).round(1),
                         rng.normal(3 * k, 1, N).round(1),
                         1000 * k + (500 if part else 0) + np.arange(N) // 7,      # IAT: few values, never down
                         np.full(N, 99.5 if w100 else 9.5),                         # Number: a window-size marker
                         rng.normal(k, 1, N).round(1)])
    if t in ("DDoS-ICMP_Flood", "DoS-UDP_Flood", "DDoS-UDP_Flood"):
        m = 60 if t != "DDoS-UDP_Flood" else 20                  # identical rows across DDoS / DoS (as in D8)
        X[:m] = np.column_stack([rng.integers(0, 3, m), rng.integers(0, 3, m), rng.integers(0, 2, m),
                                 np.full(m, 7.0), np.full(m, 99.5), rng.integers(0, 2, m)])
    if t in ("BenignTraffic", "XSS") and not part:
        X[0] = [1, 2, 3, 9, 9.5, 4]                              # one benign row equal to an attack row
    if t == "Mirai-greip_flood":                                 # both files: same rows except IAT
        X[:10, [0, 1, 2, 5]] = np.random.default_rng(42).normal(size=(10, 4)).round(1)
    df = pd.DataFrame(X, columns=FEATS)
    df["Time_To_Live"] = (128.0 if fam == 0 else 64.0) - (np.arange(N) % 3)
    df["IGMP"] = 0.0
    if t == "Recon-OSScan" and part:
        df.loc[5, "Rate"] = np.nan                               # removed by the cleaner, kept in the raw file
    return df


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("q")
    rng = np.random.default_rng(0)
    files = []
    for t in TYPES:
        for part in ("", "1"):
            folder = tmp / "raw" / t
            folder.mkdir(parents=True, exist_ok=True)
            _file(t, part, rng).to_csv(folder / f"{t}{part}.pcap.csv", index=False)
            files.append(folder / f"{t}{part}.pcap.csv")
    sink = ListSink()
    report = clean(files, sink, per_label_cap=0, log=silent)
    report = json.loads(json.dumps(report))                       # as read back from disk
    df = sink.frame()
    ranges = label_ranges(report)
    fam = {lab: report["per_label"][lab]["family"] for lab in ranges}
    manifests = {"A": build_time_blocked(ranges, dataset="ciciot2023", n_folds=5, gap=3),
                 "B": build_unseen_type(ranges, fam, dataset="ciciot2023", n_folds=5, gap=3)}
    q = build(FrameSource(df, batch_rows=97), report, manifests, raw=tmp / "raw", per_type=50, log=silent)
    return {"df": df, "report": report, "manifests": manifests, "q": q, "tmp": tmp, "files": files}


# ---- helpers against which the report is checked ---------------------------------------------------------
def _majority(groups: pd.Series, y: np.ndarray) -> np.ndarray:
    """Slow majority label per group (ties to the smaller label)."""
    t = pd.DataFrame({"g": groups.to_numpy(), "y": y})
    best = (t.groupby(["g", "y"]).size().reset_index(name="n")
            .sort_values(["g", "n", "y"], ascending=[True, False, True]).drop_duplicates("g").set_index("g")["y"])
    return best.loc[t["g"]].to_numpy()


def _macro(y, p):
    return f1_score(y, p, labels=sorted(set(y)), average="macro")


# ---- unit tests ------------------------------------------------------------------------------------------
def test_dense_ids_and_majority():
    keys = np.array([7, 3, 7, 9, 3, 3], np.int64)
    gid, n = dense_ids(keys)
    assert n == 3 and gid.dtype == np.int32 and sorted(set(gid)) == [0, 1, 2]
    assert gid[0] == gid[2] and gid[1] == gid[4] == gid[5] and len({gid[0], gid[1], gid[3]}) == 3
    g = np.array([0, 0, 0, 1, 1, 2, 2])
    y = np.array([2, 1, 2, 0, 1, 3, 3])
    assert list(majority_labels(g, y, 4)) == [2, 2, 2, 0, 0, 3, 3]           # tie in group 1 -> smaller label
    assert list(majority_labels(g, y, 4, w=np.array([1, 5, 1, 1, 1, 1, 1.0]))) == [1, 1, 1, 0, 0, 3, 3]
    assert len(majority_labels(np.zeros(0, int), np.zeros(0, int), 4)) == 0


def test_scores_from_confusion_follows_eval1():
    conf = np.array([[3, 1, 0], [0, 2, 0], [0, 0, 0]], float)               # class 2 has no true rows
    s = scores_from_confusion(conf, ["a", "b", "c"])
    y = [0, 0, 0, 0, 1, 1]
    p = [0, 0, 0, 1, 1, 1]
    assert s["skipped"] == ["c"] and np.isnan(s["per_class_f1"]["c"])
    assert s["macro_f1"] == approx(_macro(y, p)) and s["accuracy"] == approx(5 / 6)


def test_weighted_auc_matches_sklearn():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 20, 3000).astype(float)                              # many ties
    pos = rng.random(3000) < 0.3
    x[pos] += 3
    w = rng.random(3000)
    auc, bal, thr = weighted_auc(x, pos, w)
    assert auc == approx(roc_auc_score(pos, x, sample_weight=w))
    assert weighted_auc(-x, pos, w)[0] == approx(1 - auc)
    pred = x > thr
    tpr, tnr = np.average(pred[pos], weights=w[pos]), np.average(~pred[~pos], weights=w[~pos])
    assert bal == approx((tpr + tnr) / 2)
    assert np.isnan(weighted_auc(x, np.zeros(3000, bool), w)[0])


def test_value_summaries():
    v = np.array([1.0, 2.0, 2.0, 3.5, np.nan, 2.0, 10.0])
    s = summary_from_counts(*counts_of(v))
    fin = v[np.isfinite(v)]
    assert s["rows"] == 6 and s["distinct"] == 4 and s["top_values"][0] == [2.0, 0.5]
    for name, p in (("p01", 0.01), ("median", 0.5), ("p99", 0.99)):
        assert s[name] == np.quantile(fin, p, method="inverted_cdf")
    assert s["integer_share"] == approx(5 / 6)
    merged = summary_from_counts(*merge_counts([counts_of(v[:3]), counts_of(v[3:])]))
    assert merged == s
    assert list(even_positions(10, 20, 3)) == [10, 14, 19] or list(even_positions(10, 20, 3)) == [10, 15, 19]
    assert list(even_positions(5, 8, 10)) == [5, 6, 7]


# ---- the report on the made-up dataset ---------------------------------------------------------------------
def test_rows_per_fold(data):
    q, df, mA = data["q"], data["df"], data["manifests"]["A"]
    assert sum(q["rows"]["per_family"].values()) == len(df)
    for k in range(5):
        for role in ("train", "val", "test"):
            assert q["rows"]["per_fold"]["A"][k][role]["rows"] == len(fold_rows(mA, k, role))
    assert q["rows"]["per_fold"]["B"][0]["test_types"] == data["manifests"]["B"]["folds"][0]["test_types"]


def test_duplicates_and_conflicts_match_the_cleaner(data):
    q, df = data["q"], data["df"]
    s = q["feature_sets"]["all"]
    c = s["conflicts"]
    assert c["rows_differing_from_stored_conflict"] == 0
    assert c["rows_cross_family"] == int((df["conflict"] == 2).sum()) > 0
    assert c["rows_same_family"] == int((df["conflict"] == 1).sum())
    assert c["rows_benign_vs_attack"] == 2
    size = df.groupby("dup_group")["dup_group"].transform("size")
    assert s["duplicates"]["rows_with_an_exact_copy"] == int((size > 1).sum())
    assert s["groups"] == df["dup_group"].nunique()
    ddos = df["family"] == "DDoS"
    assert s["duplicates"]["per_family"]["DDoS"]["distinct"] == df.loc[ddos, "dup_group"].nunique()


def test_no_iat_set_groups_more_rows(data):
    q, df = data["q"], data["df"]
    keys = df[[f for f in FEATS if f != "IAT"]].astype(np.float32).apply(tuple, axis=1)
    assert q["feature_sets"]["no_iat"]["groups"] == keys.nunique() < q["feature_sets"]["all"]["groups"]
    keys35 = df[["Header_Length", "Rate", "Tot sum", "Variance"]].astype(np.float32).apply(tuple, axis=1)
    assert q["feature_sets"]["no_iat_no_number"]["groups"] == keys35.nunique()


@pytest.mark.parametrize("protocol", ["A", "B"])
def test_twins_against_pandas(data, protocol):
    q, df, m = data["q"], data["df"], data["manifests"][protocol]
    keys36 = df[[f for f in FEATS if f != "IAT"]].astype(np.float32).apply(tuple, axis=1)
    for k in range(5):
        tr, te = fold_rows(m, k, "train"), fold_rows(m, k, "test")
        want = twin_mask(df["dup_group"].to_numpy()[te], df["dup_group"].to_numpy()[tr]).mean()
        assert q["feature_sets"]["all"]["twins"][protocol][k]["test"]["share"] == approx(want)
        want36 = keys36.iloc[te].isin(set(keys36.iloc[tr])).mean()
        assert q["feature_sets"]["no_iat"]["twins"][protocol][k]["test"]["share"] == approx(want36)


@pytest.mark.parametrize("protocol, fold", [("A", 0), ("A", 3), ("B", 0), ("B", 1)])
def test_ceiling_against_pandas(data, protocol, fold):
    q, df = data["q"], data["df"]
    te = fold_rows(data["manifests"][protocol], fold, "test")
    part = df.iloc[te]
    y8 = part["y"].to_numpy()
    y7 = family_index(part["native_label"], "ciciot2023", "c7m")
    got = q["feature_sets"]["all"]["ceiling"][protocol][fold]["test"]
    for task, y in (("c8", y8), ("c7m", y7), ("binary", (y8 != 0).astype(int))):
        p = _majority(part["dup_group"], y)
        assert got[task]["macro_f1"] == approx(_macro(y, p)), task
        assert got[task]["accuracy"] == approx((p == y).mean()), task
    keys36 = part[[f for f in FEATS if f != "IAT"]].astype(np.float32).apply(tuple, axis=1)
    p36 = _majority(keys36, y8)
    assert q["feature_sets"]["no_iat"]["ceiling"][protocol][fold]["test"]["c8"]["macro_f1"] == approx(_macro(y8, p36))
    whole = q["feature_sets"]["all"]["ceiling"]["whole_file"]["c8"]
    assert whole["macro_f1"] == approx(_macro(df["y"].to_numpy(), _majority(df["dup_group"], df["y"].to_numpy())))


def test_window_screen_flags_number(data):
    rows = data["q"]["screen"]["features"]
    top = rows[0]
    assert top["feature"] == "Number" and top["separation"] == 1.0 and top["best_balanced_accuracy"] == 1.0
    assert top["within_side_separation_max"] == 0.0                  # same value for every family on one side
    ttl = next(r for r in rows if r["feature"] == "Time_To_Live")
    assert ttl["source"] == "raw CSVs"
    assert data["q"]["number"]["per_family"]["DDoS"]["top_values"][0] == [99.5, 1.0]


def test_raw_time_to_live_and_iat(data):
    raw = data["q"]["raw"]
    assert raw["files_read"] == len(TYPES) * 2 and set(raw["columns_found"]) == {"IAT", "Time_To_Live", "IGMP"}
    assert raw["time_to_live"]["per_family"]["Benign"]["median"] == 127.0
    assert raw["time_to_live"]["per_family"]["Mirai"]["max"] == 64.0
    per_file = {p["file"]: p for p in raw["iat_per_file"]}
    for f in data["files"]:
        want = pd.read_csv(f)["IAT"]
        assert per_file[f.name]["distinct"] == want.nunique() and per_file[f.name]["rows"] == len(want)
    mirai = raw["iat_per_type"]["Mirai-udpplain"]
    assert mirai["files_that_never_step_down"] == 2 and mirai["consecutive_files_in_order"] == 1
    cleaned = data["q"]["iat_cleaned"]["iat_per_type"]["Mirai-udpplain"]
    assert cleaned["consecutive_file_pairs"] == 1 and cleaned["files_that_never_step_down"] == 2


def test_report_is_json_and_summary_prints(data):
    text = json.dumps(data["q"])
    assert "Infinity" not in text and json.loads(text)["task"] == "DATA-4 and DATA-1b"
    s = summary(data["q"])
    assert "window-size screen" in s and "Number" in s and "no_iat" in s


def test_parquet_source_gives_the_same_report(data, tmp_path):
    pytest.importorskip("pyarrow")
    from fqml.data.clean_ciciot2023_v1 import ParquetSink
    from fqml.data.quality import ParquetSource
    out = tmp_path / "clean.parquet"
    sink = ParquetSink(out)
    cuts = np.linspace(0, len(data["df"]), 8).astype(int)              # several row groups, as in the real file
    for a, b in zip(cuts[:-1], cuts[1:]):
        sink.write(data["df"].iloc[a:b])
    sink.close()
    q = build(ParquetSource(out, batch_rows=113), data["report"], data["manifests"], raw=None,
              feature_sets=["all", "no_iat"], per_type=50, log=silent)
    for s in ("all", "no_iat"):
        assert q["feature_sets"][s]["groups"] == data["q"]["feature_sets"][s]["groups"]
        assert q["feature_sets"][s]["ceiling"]["A"] == data["q"]["feature_sets"][s]["ceiling"]["A"]
    assert "raw" not in q and q["screen"]["features"][0]["feature"] == "Number"
