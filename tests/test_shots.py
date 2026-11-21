"""Finite-shot mode of the PyTorch VQC (task Q-1): the parameter-shift rule must reproduce backprop
exactly, shot estimates must be unbiased with the binomial variance, and shot gradients must be
unbiased and per-sample-consistent. Statistical tolerances are 5 standard errors (false failure
probability < 1e-6 per check); every random draw uses a fixed seed."""
import pytest
import torch

from fqml.quantum.shots import sample_expz, shift_jacobian
from fqml.quantum.torch_sim import TorchVQC

VARIANTS = [  # (entangler, version, readout)
    ("chain", 2, "local"), ("chain", 2, "linear"), ("none", 2, "linear"), ("ring", 1, "local")]


def _model(n_classes, entangler="chain", version=2, readout="local", seed=0):
    return TorchVQC(8, n_classes, 3, seed=seed, entangler=entangler, version=version,
                    readout=readout).double()


def _x(b, seed=0):
    return torch.rand(b, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)) * torch.pi


@pytest.mark.parametrize("entangler,version,readout", VARIANTS)
@pytest.mark.parametrize("n_classes", [2, 8])
def test_shift_rule_equals_backprop(n_classes, entangler, version, readout):
    """With exact (noise-free) evaluations, parameter shift must equal the autograd Jacobian."""
    m = _model(n_classes, entangler, version, readout)
    x = _x(5)
    jac = shift_jacobian(m, x, m.theta)                                  # [B, r, L, n, 2]
    th = m.theta.detach().expand(5, *m.theta.shape).clone().requires_grad_(True)
    z = m.expz(x, th)
    for r in range(z.shape[1]):
        g, = torch.autograd.grad(z[:, r].sum(), th, retain_graph=True)
        assert torch.allclose(jac[:, r], g, atol=1e-10), f"readout {r}"


def test_sampled_expz_unbiased_with_binomial_variance():
    m = _model(8, readout="linear")
    R, S = 4000, 200
    x = _x(1).expand(R, 8)
    p = m.probs(x)
    exact = (p @ m.zsign.double())[0]
    z = sample_expz(p, m.zsign, S, torch.Generator().manual_seed(1))    # R independent S-shot estimates
    var = (1 - exact ** 2) / S                                           # variance of a mean of S +/-1 values
    assert torch.all((z.mean(0) - exact).abs() <= 5 * (var / R).sqrt() + 1e-12)
    ok = var > 1e-3                                                      # skip qubits that are ~deterministic
    ratio = z.var(0)[ok] / var[ok]
    assert torch.all((ratio - 1).abs() < 5 * (2 / (R - 1)) ** 0.5), ratio


def test_all_readout_qubits_come_from_the_same_shots():
    """One measurement gives every qubit at once: with 1 shot every <Z> is exactly +1 or -1 and the
    readout bits form one bitstring that had non-zero probability."""
    m = _model(8, readout="linear")
    x = _x(64)
    p = m.probs(x)
    z = sample_expz(p, m.zsign, 1, torch.Generator().manual_seed(2))
    assert torch.all(z.abs() == 1)
    bits = ((1 - z) / 2).long()                                          # qubit 0 = most significant bit
    index = (bits * (2 ** torch.arange(7, -1, -1))).sum(1)
    assert torch.all(p[torch.arange(64), index] > 0)


@pytest.mark.parametrize("grad", ["shift", "exact"])
def test_shot_gradient_is_unbiased(grad):
    """Mean of many shot gradients = exact gradient (linear loss, so the expectation is exact)."""
    m = _model(8, readout="linear")
    m.shot_grad, m.shot_generator = grad, torch.Generator().manual_seed(3)
    x = _x(4)
    w = torch.randn(4, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    (w * m.expz(x)).sum().backward()
    exact = m.theta.grad.clone()
    draws = []
    for _ in range(400):
        m.zero_grad()
        (w * m.expz(x, shots=256)).sum().backward()
        draws.append(m.theta.grad.clone())
    g = torch.stack(draws)
    se = g.std(0) / len(draws) ** 0.5
    assert torch.all((g.mean(0) - exact).abs() <= 5 * se + 1e-9)
    if grad == "exact":
        assert torch.allclose(g[0], exact, atol=1e-12)                   # linear loss: no noise reaches it


def test_shift_gradient_noise_falls_with_shots():
    m = _model(8, readout="linear")
    m.shot_generator = torch.Generator().manual_seed(5)
    x = _x(4)
    (m.expz(x).sum()).backward()
    exact = m.theta.grad.clone()

    def mse(shots):
        err = []
        for _ in range(60):
            m.zero_grad()
            m.expz(x, shots=shots).sum().backward()
            err.append(((m.theta.grad - exact) ** 2).sum())
        return torch.stack(err).mean()

    ratio = mse(64) / mse(1024)                                          # expected 16
    assert 10 < ratio < 25, ratio


def test_per_sample_shot_gradients_sum_to_the_batch_gradient():
    """Same seed -> same shots; per-sample weights must give per-sample gradients that add up."""
    m = _model(8)
    x = _x(6)
    y = torch.randint(0, 8, (6,), generator=torch.Generator().manual_seed(6))
    loss = torch.nn.CrossEntropyLoss(reduction="sum")
    m.shot_generator = torch.Generator().manual_seed(7)
    loss(m.logits(x, shots=128), y).backward()
    shared = m.theta.grad.clone()
    m.shot_generator = torch.Generator().manual_seed(7)
    th = m.theta.detach().expand(6, *m.theta.shape).clone().requires_grad_(True)
    loss(m.logits(x, th, shots=128), y).backward()
    assert torch.allclose(th.grad.sum(0), shared, atol=1e-10)


def test_model_level_training_step_with_shots():
    m = TorchVQC(8, 8, 3, seed=8, readout="linear", shots=128)
    m.shot_generator = torch.Generator().manual_seed(9)
    x = torch.rand(16, 8) * torch.pi
    y = torch.randint(0, 8, (16,))
    torch.nn.CrossEntropyLoss()(m(x), y).backward()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    m.shots = None                                                        # back to exact: unchanged model
    assert torch.allclose(m(x), m.logits(x, shots=None))


def test_inputs_get_no_gradient_in_shot_mode():
    m = TorchVQC(8, 2, 3, seed=0, shots=64)
    with pytest.raises(ValueError):
        m(torch.rand(2, 8, requires_grad=True))
