"""FED-4 trial run: the scoring helpers (fixed-FPR threshold, unseen-type summary) and the model pair."""
import numpy as np

from experiments.fed4_trial import make_model, n_params, scores, unseen

FAMILIES = ["Benign", "DDoS", "Web"]


def _proba(attack_score, pred_class):
    p = np.zeros((len(attack_score), 3))
    p[:, 0] = 1 - attack_score
    p[np.arange(len(p)), pred_class] += attack_score
    return p


def test_threshold_lets_one_percent_of_benign_through():
    rng = np.random.default_rng(0)
    y = np.array([0] * 1000 + [1] * 500)
    att = np.concatenate([rng.random(1000) * 0.5, 0.4 + rng.random(500) * 0.6])
    _, a, thr, s = scores(y, _proba(att, np.where(y == 0, 1, 1)), FAMILIES)
    assert abs((a[y == 0] > thr).mean() - 0.01) <= 0.002
    assert 0 < s["attack_recall_at_1pct_fpr"] <= 1
    assert set(s["recall"]) == set(FAMILIES)


def test_unseen_counts_only_types_a_client_never_held():
    y = np.array([0, 0, 1, 1, 2, 2])
    native = np.array(["BenignTraffic", "BenignTraffic", "DDoS-A", "DDoS-A", "Web-X", "Web-X"], dtype=object)
    pred = np.array([0, 0, 1, 1, 0, 2])
    att = np.array([0.0, 0.1, 0.9, 0.9, 0.2, 0.9])
    held = [{"DDoS-A"}, {"DDoS-A", "Web-X"}]
    out = unseen(y, native, held, [pred, pred], [att, att], [0.5, 0.5])
    # client 0 never held Web-X: 1 of 2 rows detected and 1 of 2 given the right family; client 1 has no unseen type
    assert out["detection_at_1pct_fpr"] == {"clients": 1, "mean": 0.5, "min": 0.5, "max": 0.5}
    assert out["family_recall"]["mean"] == 0.5


def test_matched_mlp_is_close_to_the_vqc_size():
    vqc, mlp = make_model("vqc", 8), make_model("mlp", 8)
    assert abs(n_params(mlp) - n_params(vqc)) <= 0.1 * n_params(vqc)
