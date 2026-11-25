"""FED-4 trial run: the scoring helpers (fixed-FPR threshold, unseen-type summary) and the model pair."""
import numpy as np
import pytest

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


def test_protocol_b_scores_types_no_client_held():
    y = np.array([0, 0, 1, 1, 2, 2])
    native = np.array(["BenignTraffic", "BenignTraffic", "DDoS-A", "DDoS-A", "Web-X", "Web-X"], dtype=object)
    pred = np.array([0, 0, 1, 1, 0, 2])
    att = np.array([0.0, 0.1, 0.9, 0.9, 0.2, 0.9])
    held = [{"DDoS-A"}, set()]                      # client 1 never held DDoS-A; nobody held Web-X
    out = unseen(y, native, held, [pred, pred], [att, att], [0.5, 0.5], everywhere={"DDoS-A"})
    assert out["nowhere_types"] == ["Web-X"]
    assert out["nowhere_detection_at_1pct_fpr"]["mean"] == 0.5 and out["nowhere_family_recall"]["mean"] == 0.5
    assert "nowhere_types" not in unseen(y, native, held, [pred, pred], [att, att], [0.5, 0.5])


def test_size_ladder_mlp():
    assert abs(n_params(make_model("mlp_p3000", 8)) - 3000) <= 300


def test_factory_models_equal_the_first_runs_models():
    """make_model now goes through fqml.models.classical.build (CLS-1); the models of the earlier result files must
    be the same, weight for weight."""
    import torch

    from fqml.models.mlp import MatchedMLP
    from fqml.quantum.torch_sim import TorchVQC
    old = {"vqc": TorchVQC(8, 8, 3, seed=0), "mlp": MatchedMLP(8, 8, target_params=64, seed=0),
           "mlp_p3000": MatchedMLP(8, 8, target_params=3000, seed=0)}
    for kind, ref in old.items():
        new = make_model(kind, 8)
        a, b = dict(ref.named_parameters()), dict(new.named_parameters())
        assert a.keys() == b.keys() and all(torch.equal(a[n], b[n]) for n in a), kind


def test_rff_and_vqc_variants_train_under_fedavg():
    import torch

    from fqml.fl.batched import run_fedavg
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    X = rng.random((300, 8)) * np.pi
    parts = np.array_split(rng.permutation(300), 4)
    for kind in ("rff", "vqc_lin", "logreg"):
        torch.manual_seed(0)
        res = run_fedavg(make_model(kind, 3), X, y, parts, rounds=2, clients_per_round=2, local_steps=2,
                         batch_size=16, seed=0)
        assert all(torch.isfinite(v).all() for v in res.params.values()), kind
    with pytest.raises(SystemExit):
        make_model("xgb", 3)
