"""PRIV-5 federated canary audit (experiments/priv5_fed_audit.py) on synthetic data."""
import numpy as np
import pytest

pytest.importorskip("scipy")
from experiments.priv5_fed_audit import run_fed_audit  # noqa: E402
from fqml.models.mlp import MatchedMLP  # noqa: E402


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    X = (rng.random((n, 8)) * np.pi).astype(np.float32)
    X[:, 0] = (y + rng.random(n)) * np.pi / 4               # learnable
    return X, y, [np.sort(p) for p in np.array_split(rng.permutation(n), 4)]


def test_memorising_model_leaks_and_the_bound_is_valid():
    """A big model, few honest rows, many steps: it memorises the canaries, so both attackers guess well."""
    X, y, parts = _data()
    out = run_fed_audit(MatchedMLP(8, 4, target_params=3000, seed=0), X, y, parts, 4, m=200, rounds=15,
                        clients_per_round=4, local_steps=40, batch_size=16, lr=0.01)
    assert out["canaries"] == 200 and 0 < out["canaries_included"] < 200
    assert out["client_rows"] == len(parts[0]) + out["canaries_included"]
    for who in ("outsider", "curious_server"):
        r = out[who]
        assert r["guesses"] == 2 * out["k"] and 0 <= r["correct"] <= r["guesses"]
        assert r["empirical_epsilon_lower_bound"] >= 0
    assert out["curious_server"]["empirical_epsilon_lower_bound"] > 0      # the update shows the canaries


def test_untrained_model_shows_no_leak():
    X, y, parts = _data(seed=1)
    out = run_fed_audit(MatchedMLP(8, 4, target_params=64, seed=0), X, y, parts, 4, m=200, rounds=1,
                        clients_per_round=1, local_steps=1, batch_size=8, lr=1e-6)
    assert out["outsider"]["empirical_epsilon_lower_bound"] < 1.0
