"""Shared metrics module (EVAL-1): the eight tests of docs/specs/EVAL-1_metrics.md section 4, plus the subsets."""
import json
import math

import numpy as np
import pytest

from fqml.data.label_maps import family_index, load_label_map
from fqml.eval import checkpoint1 as cp1
from fqml.eval.metrics import (FPRS, binary, evaluate, fit_thresholds, multiclass, per_type, threshold_at_fpr,
                               to_c7m, unseen)

FAMILIES, MAP = load_label_map("ciciot2023")
FAM7, _ = load_label_map("ciciot2023", "c7m")


def close(a, b):
    """Equal, or both NaN."""
    return (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, abs=1e-12)


# ---- 1. hand-computed cases -----------------------------------------------------------------------
def test_multiclass_by_hand():
    #            true:  B  B  B  A  A  C  C  C
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    p = np.array([0, 1, 0, 1, 1, 2, 0, 2])
    r = multiclass(y, p, ["B", "A", "C"])
    # confusion (rows = true): B -> [2, 1, 0], A -> [0, 2, 0], C -> [1, 0, 2]
    assert r["confusion"] == [[2, 1, 0], [0, 2, 0], [1, 0, 2]]
    assert r["accuracy"] == pytest.approx(6 / 8)
    pc = r["per_class"]
    assert (pc["B"]["precision"], pc["B"]["recall"]) == (pytest.approx(2 / 3), pytest.approx(2 / 3))
    assert (pc["A"]["precision"], pc["A"]["recall"]) == (pytest.approx(2 / 3), pytest.approx(1.0))
    assert (pc["C"]["precision"], pc["C"]["recall"]) == (pytest.approx(1.0), pytest.approx(2 / 3))
    assert pc["B"]["f1"] == pytest.approx(2 / 3) and pc["A"]["f1"] == pytest.approx(0.8)
    assert pc["C"]["f1"] == pytest.approx(0.8)
    assert r["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 0.8) / 3)
    assert [pc[f]["support"] for f in "BAC"] == [3, 2, 3] and r["skipped"] == []


def test_binary_roc_auc_by_hand():
    # the classic example: of the 4 (attack, benign) pairs, 3 are ranked right -> ROC-AUC 0.75
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    r = binary(y, s, {}, p=np.array([0, 0, 0, 1]))
    assert r["roc_auc"] == pytest.approx(0.75)
    assert r["default"]["attack_recall"] == pytest.approx(0.5) and r["default"]["fpr"] == 0.0
    assert r["default"]["precision"] == 1.0 and r["default"]["f1"] == pytest.approx(2 / 3)


# ---- 2. threshold rule ----------------------------------------------------------------------------
def test_threshold_keeps_alarms_within_budget():
    s = np.linspace(0.0, 1.0, 1000)
    y = np.zeros(1000, int)
    for f in (0.01, 0.001):
        t = threshold_at_fpr(s, y, f)
        assert (s > t).sum() <= 1000 * f


def test_tied_scores_raise_no_alarm():
    y = np.array([0] * 50 + [1] * 50)
    s = np.full(100, 0.5)
    t = fit_thresholds(s, y)
    r = binary(y, s, t)
    for name in FPRS:
        assert r[name]["attack_recall"] == 0.0 and r[name]["fpr_reached"] == 0.0
        assert math.isnan(r[name]["precision"])            # no alarm at all: precision is undefined


def test_no_benign_in_validation_gives_nan():
    t = threshold_at_fpr(np.array([0.2, 0.9]), np.array([1, 2]), 0.01)
    assert math.isnan(t)
    r = binary(np.array([0, 1]), np.array([0.1, 0.9]), {"fpr_1pct": t})
    assert math.isnan(r["fpr_1pct"]["attack_recall"])


# ---- 3. thresholds come from validation -----------------------------------------------------------
def test_threshold_from_validation_applied_to_test():
    rng = np.random.default_rng(0)
    y_val = np.zeros(2000, int)
    s_val = rng.uniform(0.0, 0.5, 2000)                    # benign scores on validation
    t = fit_thresholds(s_val, y_val)
    y_test = np.array([0] * 2000 + [1] * 100)
    s_test = np.concatenate([rng.uniform(0.25, 0.75, 2000), rng.uniform(0.6, 1.0, 100)])  # benign drifted up
    r = binary(y_test, s_test, t)
    assert r["fpr_1pct"]["threshold"] == t["fpr_1pct"]
    assert r["fpr_1pct"]["fpr_reached"] > 0.4             # measured on test, not forced back to 1%
    assert r["fpr_1pct"]["attack_recall"] == 1.0


# ---- 4. c7m comes from the label map --------------------------------------------------------------
def test_c7m_matches_label_map():
    rng = np.random.default_rng(1)
    natives = np.array(sorted(MAP))
    true_native = rng.choice(natives, 500)
    pred_native = rng.choice(natives, 500)
    y, p = family_index(true_native, "ciciot2023"), family_index(pred_native, "ciciot2023")
    y7, p7 = family_index(true_native, "ciciot2023", "c7m"), family_index(pred_native, "ciciot2023", "c7m")
    assert (to_c7m(y, FAMILIES) == y7).all() and (to_c7m(p, FAMILIES) == p7).all()
    a = multiclass(to_c7m(y, FAMILIES), to_c7m(p, FAMILIES), FAM7)
    b = multiclass(y7, p7, FAM7)
    assert json.dumps(a) == json.dumps(b)


# ---- 5. row weights -------------------------------------------------------------------------------
def _scores_only(r):
    """The rates of a multiclass or binary result (weighted counts left out)."""
    if "per_class" in r:
        return [r["accuracy"], r["macro_f1"]] + [v for c in r["per_class"].values()
                                                 for k, v in c.items() if k != "support"]
    return [v for k, v in r.items() if k.endswith("auc")] + \
           [v for name in r if isinstance(r[name], dict) for v in r[name].values()]


def test_doubling_every_weight_changes_nothing():
    rng = np.random.default_rng(2)
    y, p = rng.integers(0, 3, 300), rng.integers(0, 3, 300)
    s = rng.random(300)
    t = {"fpr_1pct": 0.7}
    for one, two in ((multiclass(y, p, ["B", "A", "C"]), multiclass(y, p, ["B", "A", "C"], w=np.full(300, 2.0))),
                     (binary(y, s, t, p=p), binary(y, s, t, np.full(300, 2.0), p=p))):
        assert all(close(a, b) for a, b in zip(_scores_only(one), _scores_only(two)))


def test_weight_two_equals_duplicated_rows():
    rng = np.random.default_rng(3)
    y, p = rng.integers(0, 3, 300), rng.integers(0, 3, 300)
    s = rng.random(300)
    w = np.where(y == 2, 2.0, 1.0)                         # class 2 weighs double ...
    dup = np.concatenate([np.arange(300), np.flatnonzero(y == 2)])   # ... or its rows appear twice
    t = {"fpr_1pct": 0.7, "fpr_0.1pct": 0.9}
    a, b = multiclass(y, p, ["B", "A", "C"], w), multiclass(y[dup], p[dup], ["B", "A", "C"])
    assert a["confusion"] == b["confusion"] and _scores_only(a) == pytest.approx(_scores_only(b))
    a, b = binary(y, s, t, w, p=p), binary(y[dup], s[dup], t, p=p[dup])
    assert _scores_only(a) == pytest.approx(_scores_only(b))


# ---- 6. NaN rule ----------------------------------------------------------------------------------
def test_class_with_no_rows_is_nan_and_skipped():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    r = multiclass(y, p, ["B", "A", "C"])
    assert math.isnan(r["per_class"]["C"]["f1"]) and math.isnan(r["per_class"]["C"]["recall"])
    assert r["skipped"] == ["C"]
    assert r["macro_f1"] == pytest.approx((r["per_class"]["B"]["f1"] + r["per_class"]["A"]["f1"]) / 2)


# ---- 7. agreement with Checkpoint 1 ---------------------------------------------------------------
def _example(seed=4):
    """A small made-up test block: two native types per family where the map has them, rows in row_id order,
    predictions mostly right, scores higher for attacks."""
    rng = np.random.default_rng(seed)
    natives = [MAP_BY_FAMILY[f][:2] for f in FAMILIES]
    types = [t for pair in natives for t in pair]
    per_label, start = {}, 0
    for t in types:
        n = int(rng.integers(30, 120))
        per_label[t] = {"row_start": start, "row_stop": start + n, "rows_written": n, "family": MAP[t]}
        start += n
    report = {"per_label": per_label, "families": FAMILIES}
    row_ids = np.arange(start)
    native = np.concatenate([[t] * d["rows_written"] for t, d in per_label.items()])
    y = family_index(native, "ciciot2023")
    p = np.where(rng.random(start) < 0.7, y, rng.integers(0, len(FAMILIES), start))
    s = np.clip(np.where(y != 0, 0.6, 0.2) + rng.normal(0, 0.2, start), 0, 1)
    return report, row_ids, y, p, s


MAP_BY_FAMILY = {f: sorted(n for n, fam in MAP.items() if fam == f) for f in FAMILIES}


def test_agrees_with_checkpoint1():
    report, row_ids, y, p, s = _example()
    ranges = cp1.label_ranges(report)
    unseen_types = ["DDoS-HTTP_Flood", "Recon-OSScan"]
    t = fit_thresholds(s, y)                               # fitted on the same rows, as Checkpoint 1 does
    ev = evaluate(y, p, s, families=FAMILIES, row_ids=row_ids, label_ranges=ranges, thresholds=t,
                  train_dup_groups=None, dup_group=None, conflict=None, unseen_types=unseen_types)
    ref = cp1.score(y, p, FAMILIES, s)
    new = ev["all"]
    assert new["c8"]["skipped"] == [] and new["c7m"]["skipped"] == []     # so the two macro-F1 rules agree
    assert new["c8"]["macro_f1"] == pytest.approx(ref["c8"]["macro_f1"], abs=1e-12)
    assert new["c7m"]["macro_f1"] == pytest.approx(ref["c7m"]["macro_f1"], abs=1e-12)
    assert [new["c8"]["per_class"][f]["f1"] for f in FAMILIES] == pytest.approx(ref["c8"]["per_class_f1"])
    assert new["c8"]["confusion"] == ref["confusion_c8"]
    rb, nb = ref["binary"], new["binary"]
    assert nb["default"]["f1"] == pytest.approx(rb["f1"]) and nb["default"]["fpr"] == pytest.approx(
        rb["false_positive_rate"])
    assert nb["default"]["attack_recall"] == pytest.approx(rb["attack_recall"])
    assert nb["default"]["precision"] == pytest.approx(rb["precision"])
    for old, name in (("1pct", "fpr_1pct"), ("0.1pct", "fpr_0.1pct")):
        assert nb[name]["attack_recall"] == pytest.approx(rb[f"recall_at_fpr_{old}"])
        assert nb[name]["fpr_reached"] == pytest.approx(rb[f"fpr_reached_{old}"])
    for old, name in (("roc_auc", "roc_auc"), ("pr_auc", "attack_pr_auc"), ("benign_pr_auc", "benign_pr_auc")):
        assert nb[name] == pytest.approx(rb[old])
    assert new["dos_predicted_ddos"] == pytest.approx(ref["dos_predicted_ddos"])
    assert new["ddos_predicted_dos"] == pytest.approx(ref["ddos_predicted_dos"])

    ru, nu = cp1.unseen_scores(row_ids, y, p, s, report, unseen_types), ev["unseen"]
    assert nu["rows"] == ru["rows"]
    for old, name in (("detected_at_fpr_1pct", "detected_fpr_1pct"), ("detected_at_fpr_0.1pct", "detected_fpr_0.1pct"),
                      ("detected_at_default", "detected_at_default"), ("family_correct", "family_correct"),
                      ("family_correct_c7m", "family_correct_c7m")):
        assert nu[name] == pytest.approx(ru[old])

    rt, nt = cp1.per_type_recall(row_ids, y, p, report), ev["per_type"]["types"]
    assert set(rt) == set(nt)
    for lab in rt:
        assert nt[lab]["family_recall"] == pytest.approx(rt[lab]["recall"])
        assert nt[lab]["most_confused_with"] == rt[lab]["most_confused_with"]
        assert nt[lab]["share_to_it"] == pytest.approx(rt[lab]["share_to_it"])


# ---- 8. JSON --------------------------------------------------------------------------------------
def test_evaluate_is_json_with_version():
    report, row_ids, y, p, s = _example(seed=5)
    rng = np.random.default_rng(5)
    ev = evaluate(y, p, s, families=FAMILIES, row_ids=row_ids, label_ranges=cp1.label_ranges(report),
                  thresholds=fit_thresholds(s, y), train_dup_groups=rng.integers(0, 50, 100),
                  dup_group=rng.integers(0, 200, len(y)), conflict=rng.integers(0, 2, len(y)),
                  unseen_types=["Backdoor_Malware"])
    text = json.dumps(ev)
    back = json.loads(text)
    assert back["metrics_version"] == 1
    assert {"all", "no_twin", "no_conflict", "per_type", "unseen", "thresholds", "rows"} <= set(back)
    for sub in ("all", "no_twin", "no_conflict"):
        assert {"c8", "c7m", "binary"} <= set(back[sub])


# ---- subsets and per-type averages ----------------------------------------------------------------
def test_subsets_pick_the_right_rows():
    report, row_ids, y, p, s = _example(seed=6)
    n = len(y)
    dup_group = np.arange(n)                               # every row its own group ...
    train_groups = dup_group[: n // 4]                     # ... and the first quarter has a twin in training
    conflict = (np.arange(n) % 10 == 0).astype(int)
    ev = evaluate(y, p, s, families=FAMILIES, row_ids=row_ids, label_ranges=cp1.label_ranges(report),
                  thresholds=fit_thresholds(s, y), train_dup_groups=train_groups, dup_group=dup_group,
                  conflict=conflict)
    assert ev["all"]["rows"] == n
    assert ev["no_twin"]["rows"] == n - n // 4
    assert ev["no_conflict"]["rows"] == int((conflict == 0).sum())
    assert "unseen" not in ev


def test_per_type_counts_every_type_equally():
    # one big type caught at 100%, one small type caught at 0%: over rows ~99%, over types 50%
    ranges = {"BenignTraffic": (0, 100), "DDoS-ICMP_Flood": (100, 10100), "XSS": (10100, 10110)}
    row_ids = np.arange(10110)
    y = np.array([0] * 100 + [1] * 10000 + [4] * 10)
    s = np.array([0.0] * 100 + [1.0] * 10000 + [0.0] * 10)
    r = per_type(row_ids, y, y, s, ranges, {"fpr_1pct": 0.5}, families=FAMILIES)
    assert r["attack_types"] == 2
    assert r["types"]["DDoS-ICMP_Flood"]["detection_fpr_1pct"] == 1.0
    assert r["types"]["XSS"]["detection_fpr_1pct"] == 0.0
    assert r["mean_detection_fpr_1pct"] == pytest.approx(0.5)
    assert r["types"]["BenignTraffic"]["is_attack"] is False


def test_unseen_without_rows_is_nan():
    ranges = {"BenignTraffic": (0, 10), "XSS": (10, 20)}
    y = np.zeros(10, int)
    r = unseen(np.arange(10), y, y, np.zeros(10), ranges, ["XSS"], {"fpr_1pct": 0.5}, FAMILIES)
    assert r["rows"] == 0 and math.isnan(r["detected_fpr_1pct"]) and math.isnan(r["family_correct"])
