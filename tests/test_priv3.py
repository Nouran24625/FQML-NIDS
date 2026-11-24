"""PRIV-3 checks 1 and 3. Check 1: our DP-SGD step equals Opacus's on the same MLP and batch (clipping, summing,
scaling), and our accountant is at least as tight as Opacus's default. Check 3: the canary-audit bound is correct and a
private run passes its audit. (Check 2, VQC per-sample gradients = one-at-a-time loop, is in tests/test_dpsgd.py.)"""
import copy
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fqml.models.classical import build
from fqml.privacy.audit import canary_scores, epsilon_lower_bound, guess_and_count, make_canaries, run_audit
from fqml.privacy.dpsgd import Accountant, clip_and_sum, noise_multiplier_for, per_sample_grads


@pytest.mark.parametrize("clip", [0.05, 1.0, 100.0])
def test_dp_step_equals_opacus(clip):
    from opacus import GradSampleModule
    from opacus.optimizers import DPOptimizer

    m1 = build("mlp_p256", n_in=8, n_classes=8, seed=7)
    m2 = copy.deepcopy(m1)
    g = torch.Generator().manual_seed(7)
    x, y = torch.rand(64, 8, generator=g) * math.pi, torch.randint(0, 8, (64,), generator=g)
    sums, _ = clip_and_sum(per_sample_grads(m1, x, y), clip)
    ours = [sums[k] / 64 for k, _ in m1.named_parameters()]
    gsm = GradSampleModule(m2)
    opt = DPOptimizer(torch.optim.SGD(gsm.parameters(), lr=1.0), noise_multiplier=0.0, max_grad_norm=clip,
                      expected_batch_size=64)
    opt.zero_grad()
    F.cross_entropy(gsm(x), y).backward()
    opt.pre_step()
    for a, p in zip(ours, gsm.parameters()):
        assert torch.allclose(a, p.grad, atol=1e-6)


def test_accountant_is_at_least_as_tight_as_opacus_default():
    from opacus.accountants import RDPAccountant
    from opacus.accountants.utils import get_noise_multiplier

    q, steps, delta = 256 / 150_000, 3_000, 1e-6
    ours, theirs = Accountant(), RDPAccountant()
    for _ in range(steps):
        ours.step(1.1, q)
        theirs.step(noise_multiplier=1.1, sample_rate=q)
    assert ours.epsilon(delta) <= theirs.get_epsilon(delta) + 1e-9
    assert ours.epsilon(delta) >= 0.8 * theirs.get_epsilon(delta)
    sigma_ours = noise_multiplier_for(4.0, delta, q, steps)
    sigma_opacus = get_noise_multiplier(target_epsilon=4.0, target_delta=delta, sample_rate=q, steps=steps,
                                        accountant="rdp")
    assert sigma_ours <= sigma_opacus + 1e-6 and sigma_ours >= 0.9 * sigma_opacus


def test_epsilon_lower_bound():
    assert epsilon_lower_bound(100, 200) == 0.0                    # coin flipping proves nothing
    assert epsilon_lower_bound(0, 0) == 0.0
    vals = [epsilon_lower_bound(c, 200) for c in (130, 160, 190, 200)]
    assert vals == sorted(vals) and vals[0] > 0 and 3.5 < vals[-1] < 6
    assert epsilon_lower_bound(190, 200, beta=0.01) < epsilon_lower_bound(190, 200, beta=0.05)
    assert epsilon_lower_bound(190, 200, delta=1e-6, m=1000) <= epsilon_lower_bound(190, 200)


def test_canaries_and_guessing():
    X, y, inc = make_canaries(2000, 8, 8, seed=1)
    assert X.shape == (2000, 8) and X.min() >= 0 and X.max() <= math.pi and abs(inc.mean() - 0.5) < 0.04
    perfect = inc.astype(float) + np.random.default_rng(0).random(2000) * 0.1
    assert guess_and_count(perfect, inc, 200) == (400, 400)


def test_scores_prefer_fitted_canaries():
    m = build("mlp_p256", n_in=8, n_classes=8, seed=2)
    X, y, _ = make_canaries(50, 8, 8, seed=2)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    Xt, yt = torch.as_tensor(X[:25]), torch.as_tensor(y[:25])
    for _ in range(300):
        opt.zero_grad()
        F.cross_entropy(m(Xt), yt).backward()
        opt.step()
    s = canary_scores(m, X, y)
    assert s[:25].mean() > s[25:].mean()


def test_private_run_passes_its_audit():
    rng = np.random.default_rng(3)
    X = (rng.random((400, 8)) * np.pi).astype(np.float32)
    y = rng.integers(0, 8, 400)
    r = run_audit(lambda s: build("mlp_p256", n_in=8, n_classes=8, seed=s), X, y, 8, m=200, target_epsilon=2.0,
                  delta=1e-5, seed=3, clip=1.0, batch_size=50, max_epochs=3, lr=0.01)
    assert r["passes"] and 0 <= r["empirical_epsilon_lower_bound"] <= r["accountant_epsilon"]
    assert r["guesses"] == 40 and r["canaries"] == 200
