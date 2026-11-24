"""PRIV-2: DP-SGD. Per-sample gradients equal a one-at-a-time loop for every model (PRIV-3 check 2), clipping
bounds every sample, no clip and no noise gives the ordinary batch gradient, Poisson batches have the right rate,
and sigma is calibrated so the accountant's epsilon meets the target (D13)."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fqml.models.classical import build
from fqml.privacy.dpsgd import (Accountant, clip_and_sum, dp_fit, noise_multiplier_for, per_sample_grads,
                                per_sample_norms, poisson_batches)

MODELS = ["vqc", "vqc_lin", "vqc_noent", "mlp_p64", "rff", "logreg", "mlp_large"]


def _data(n=12, n_classes=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 8, generator=g) * math.pi, torch.randint(0, n_classes, (n,), generator=g)


@pytest.mark.parametrize("name", MODELS)
def test_per_sample_grads_equal_one_at_a_time(name):
    m = build(name, n_in=8, n_classes=8, seed=1)
    m.eval() if name == "mlp_large" else m.train()                     # dropout 0 anyway; keep deterministic
    x, y = _data()
    cw = torch.rand(8) + 0.5
    ps = per_sample_grads(m, x, y, cw)
    for i in range(len(x)):
        m.zero_grad()
        F.cross_entropy(m(x[i:i + 1]), y[i:i + 1], weight=cw, reduction="sum").backward()
        for k, p in m.named_parameters():
            assert torch.allclose(ps[k][i], p.grad, atol=1e-5), (name, k, i)


def test_clipping_bounds_each_sample_and_keeps_small_ones():
    m = build("mlp_p64", n_in=8, n_classes=8, seed=2)
    x, y = _data(32)
    g = per_sample_grads(m, x, y)
    norms = per_sample_norms(g)
    C = float(norms.median())
    sums, before = clip_and_sum(g, C)
    assert torch.allclose(before, norms)
    factor = (C / (norms + 1e-6)).clamp(max=1.0)
    clipped = {k: v * factor.view(-1, *[1] * (v.dim() - 1)) for k, v in g.items()}
    assert torch.all(per_sample_norms(clipped) <= C + 1e-4)
    small = norms < C
    assert torch.allclose(per_sample_norms(clipped)[small], norms[small], atol=1e-6)
    for k in sums:
        assert torch.allclose(sums[k], clipped[k].sum(0), atol=1e-5)


def test_no_clip_no_noise_is_the_ordinary_gradient():
    m = build("vqc", n_in=8, n_classes=8, seed=3)
    x, y = _data(16)
    sums, _ = clip_and_sum(per_sample_grads(m, x, y), float("inf"))
    m.zero_grad()
    F.cross_entropy(m(x), y, reduction="sum").backward()
    for k, p in m.named_parameters():
        assert torch.allclose(sums[k], p.grad, atol=1e-5)


def test_poisson_batches_have_the_sampling_rate():
    g = torch.Generator().manual_seed(4)
    sizes = [len(b) for b in poisson_batches(10_000, 0.02, 200, g)]
    assert abs(np.mean(sizes) - 200) < 5                                 # expected 200, sd of mean ~1
    assert len(set(sizes)) > 1                                           # sizes vary: Poisson, not fixed


def test_sigma_calibration_meets_the_target_epsilon():
    q, steps, delta = 256 / 150_000, 5_000, 1e-6
    for eps in (8.0, 4.0, 1.0):
        sigma = noise_multiplier_for(eps, delta, q, steps)
        acc = Accountant()
        for _ in range(steps):
            acc.step(sigma, q)
        got = acc.epsilon(delta)
        assert got <= eps * 1.02 and got >= eps * 0.8, (eps, sigma, got)
    assert noise_multiplier_for(8.0, delta, q, steps) < noise_multiplier_for(1.0, delta, q, steps)
    assert noise_multiplier_for(float("inf"), delta, q, steps) == 0.0


@pytest.mark.parametrize("name", ["vqc_lin", "mlp_p64"])
def test_dp_fit_runs_reports_epsilon_and_restores_best(name):
    rng = np.random.default_rng(5)
    X = (rng.random((600, 8)) * np.pi).astype(np.float32)
    y = (X[:, 0] > X[:, 1]).astype(np.int64) * 3
    m = build(name, n_in=8, n_classes=8, seed=5)
    res = dp_fit(m, X[:500], y[:500], 8, target_epsilon=4.0, delta=1e-5, clip=1.0, batch_size=50, max_epochs=3,
                 lr=0.05, seed=5, X_val=X[500:], y_val=y[500:], patience=5)
    assert 0 < res.epsilon <= 4.0 * 1.02 and res.noise_multiplier > 0 and res.steps == 30
    assert res.best_epoch >= 0 and len(res.history) == 3 and 0 <= res.clipped_share <= 1


def test_dp_fit_is_deterministic_and_infinite_epsilon_adds_no_noise():
    rng = np.random.default_rng(6)
    X = (rng.random((200, 8)) * np.pi).astype(np.float32)
    y = rng.integers(0, 8, 200)
    runs = []
    for _ in range(2):
        m = build("mlp_p64", n_in=8, n_classes=8, seed=6)
        dp_fit(m, X, y, 8, target_epsilon=2.0, delta=1e-5, batch_size=40, max_epochs=2, lr=0.01, seed=6)
        runs.append(torch.cat([p.detach().flatten() for p in m.parameters()]))
    assert torch.equal(runs[0], runs[1])
    m = build("mlp_p64", n_in=8, n_classes=8, seed=6)
    res = dp_fit(m, X, y, 8, target_epsilon=float("inf"), batch_size=40, max_epochs=1, lr=0.01, seed=6)
    assert res.noise_multiplier == 0.0 and math.isinf(res.epsilon) and math.isinf(res.clip)
