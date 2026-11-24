"""Q-3 (RFF) and CLS-1 (model factory): the RFF frequencies follow the circuit's spectrum, every E0 model builds by
name with the tuned settings, and the class weights follow D11 part 1."""
import math

import numpy as np
import pytest
import torch

from fqml.models.classical import build, class_weights_np, fit_sklearn, is_torch, load_tuning, n_params, sample_weights
from fqml.models.rff import RFF, degeneracy_weights, rff_params, sample_frequencies

E0_TORCH = ["vqc", "vqc_lin", "vqc_noent", "mlp_p64", "mlp_p120", "mlp_p3000", "mlp_large", "rff", "logreg"]


def test_degeneracy_weights_are_the_circuit_spectrum():
    w = degeneracy_weights(3)
    assert np.allclose(w * 64, [1, 6, 15, 20, 15, 6, 1])


def test_frequencies_whole_bounded_distinct_no_sign_pairs():
    W = sample_frequencies(8, 256, 3, seed=1)
    assert W.shape == (256, 8) and np.all(W == np.round(W)) and np.abs(W).max() <= 3
    keys = {tuple(r) for r in W.astype(int)}
    assert len(keys) == 256 and (0,) * 8 not in keys
    assert not any(tuple(-v for v in k) in keys for k in keys)


def test_frequency_distribution_matches_degeneracy():
    W = sample_frequencies(8, 4000, 3, seed=2).astype(int)
    freq = np.bincount((W + 3).ravel(), minlength=7) / W.size
    assert np.allclose(freq, degeneracy_weights(3), atol=0.02)


def test_rff_shapes_params_seed_and_periodicity():
    m1, m2 = RFF(8, 8, 64, seed=3), RFF(8, 8, 64, seed=3)
    x = torch.rand(5, 8) * math.pi
    assert torch.equal(m1(x), m2(x)) and m1(x).shape == (5, 8)
    assert n_params(m1) == rff_params(8, 64)
    assert torch.allclose(m1(x), m1(x + 2 * math.pi), atol=1e-4)       # whole-number frequencies: period 2 pi
    assert not torch.equal(RFF(8, 8, 64, seed=4)(x), m1(x))


@pytest.mark.parametrize("name", E0_TORCH)
def test_every_torch_model_builds_and_runs(name):
    assert is_torch(name)
    m = build(name, n_in=8, n_classes=8, seed=0)
    out = m(torch.rand(4, 8) * math.pi)
    assert out.shape == (4, 8) and torch.isfinite(out).all()


def test_ladder_rungs_are_near_their_target():
    for p in (64, 120, 128, 256, 512, 1000, 3000):
        got = n_params(build(f"mlp_p{p}", n_in=8, n_classes=8, seed=0))
        assert abs(got - p) <= 9, (p, got)                               # widths move in steps of 17


def test_vqc_names_map_to_the_prereg_models():
    assert n_params(build("vqc", n_in=8, n_classes=8)) == 64
    assert n_params(build("vqc_lin", n_in=8, n_classes=8)) == 120
    noent = build("vqc_noent", n_in=8, n_classes=8)
    assert noent.entangler == "none" and noent.readout == "linear"
    with pytest.raises(ValueError):
        build("vqc", n_in=35, n_classes=8)


def test_same_seed_same_start_and_global_rng_untouched():
    state = torch.get_rng_state()
    a, b = build("mlp_large", n_in=8, n_classes=8, seed=5), build("mlp_large", n_in=8, n_classes=8, seed=5)
    assert torch.equal(torch.get_rng_state(), state)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


@pytest.mark.parametrize("name", ["lr", "rf", "xgb"])
def test_tree_and_linear_models_fit_with_tuned_settings(name):
    trials, fixed = load_tuning(name)
    assert len(trials) == 12
    rng = np.random.default_rng(0)
    X = rng.random((400, 8)).astype(np.float32) * np.pi
    y = (X[:, 0] > 1.5).astype(int) + 2 * (X[:, 1] > 1.5).astype(int)
    fixed = {**fixed, "n_estimators": 20} if name in ("rf", "xgb") else fixed
    m = build(name, n_in=8, n_classes=4, seed=0, **fixed, **trials[0])
    fit_sklearn(m, name, X[:300], y[:300], 4, X[300:], y[300:])
    assert m.predict_proba(X[300:]).shape == (100, 4)


def test_class_weights_follow_d11():
    y = np.array([0] * 100 + [1] * 25 + [2] * 4)
    w = class_weights_np(y, 4)
    assert w[3] == 0 and np.isclose(w[:3].mean(), 1.0)
    assert np.isclose(w[1] / w[0], 2.0) and np.isclose(w[2] / w[0], 5.0)       # 1 / sqrt(count)
    assert np.allclose(sample_weights(y, 4)[:100], w[0])


def test_unknown_model_is_an_error():
    with pytest.raises(ValueError):
        build("svm", n_in=8, n_classes=8)
