"""FED-7: recall on attack types a client never saw."""
import math

import numpy as np

from fqml.fl.unseen import held_types, per_client_unseen, summary, unseen_scores


def test_unseen_scores_by_hand():
    native = np.array(["BenignTraffic", "A", "A", "B", "B", "B", "C"], dtype=object)
    y = np.array([0, 1, 1, 2, 2, 2, 3])
    p = np.array([0, 1, 0, 2, 0, 1, 3])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.95])
    r = unseen_scores(y, p, native, held={"A"}, attack_score=s, threshold=0.5)
    assert r["unseen_types"] == ["B", "C"] and r["rows"] == 4
    assert r["per_type"]["B"]["detection_default"] == 2 / 3          # p != 0 on two of three B rows
    assert r["per_type"]["B"]["family_recall"] == 1 / 3
    assert r["per_type"]["B"]["detection_at_threshold"] == 2 / 3
    assert r["detection_default_rows"] == 3 / 4
    assert r["detection_default_types"] == (2 / 3 + 1) / 2           # each type counts once
    assert r["family_recall_types"] == (1 / 3 + 1) / 2


def test_nothing_unseen_is_nan():
    native = np.array(["BenignTraffic", "A"], dtype=object)
    r = unseen_scores(np.array([0, 1]), np.array([0, 1]), native, held={"A"})
    assert r["rows"] == 0 and math.isnan(r["detection_default_rows"]) and math.isnan(r["family_recall_types"])


def test_per_client_and_summary():
    native_train = np.array(["BenignTraffic", "A", "B", "C"], dtype=object)
    parts = [np.array([0, 1]), np.array([0, 2, 3])]
    held = held_types(native_train, parts)
    assert held == [{"A"}, {"B", "C"}]
    native = np.array(["A", "B", "C"], dtype=object)
    y = np.array([1, 2, 3])
    p_global = np.array([1, 0, 3])
    res = per_client_unseen(y, native, held, [p_global, p_global])
    assert res[0]["unseen_types"] == ["B", "C"] and res[1]["unseen_types"] == ["A"]
    s = summary(res)
    assert s["clients"] == 2 and s["min"] == 0.5 and s["max"] == 1.0
