"""The PyTorch simulator must give the same results as the PennyLane VQC (decision D1),
for every circuit version and entangler (D6, D7)."""
import pytest
import torch
from torch import nn

from fqml.models.vqc import VQC, n_params
from fqml.quantum.torch_sim import TorchVQC

VARIANTS = [("ring", 1), ("ring", 2), ("chain", 2)]


@pytest.mark.parametrize("n_classes", [2, 8])
def test_same_starting_weights(n_classes):
    a, b = VQC(8, n_classes, 3, seed=1), TorchVQC(8, n_classes, 3, seed=1)
    assert n_params(a) == n_params(b)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and torch.equal(pa, pb)


@pytest.mark.parametrize("entangler,version", VARIANTS)
@pytest.mark.parametrize("n_classes", [2, 8])
def test_outputs_and_gradients_match_pennylane(n_classes, entangler, version):
    kw = dict(entangler=entangler, version=version)
    a, b = VQC(8, n_classes, 3, seed=2, **kw), TorchVQC(8, n_classes, 3, seed=2, **kw).double()
    x = torch.rand(6, 8, dtype=torch.float64) * torch.pi
    th = a.theta.detach().double().requires_grad_(True)
    za = torch.stack(list(a.circuit(x, th)), dim=-1)
    zb = b.expz(x)
    assert torch.allclose(za, zb, atol=1e-10)
    za.sum().backward()
    zb.sum().backward()
    assert torch.allclose(th.grad, b.theta.grad, atol=1e-10)


@pytest.mark.parametrize("n_classes", [2, 8])
def test_training_step_matches(n_classes):
    a, b = VQC(8, n_classes, 3, seed=3), TorchVQC(8, n_classes, 3, seed=3)
    x = torch.rand(16, 8) * torch.pi
    y = torch.randint(0, n_classes, (16,))
    la, lb = nn.CrossEntropyLoss()(a(x), y), nn.CrossEntropyLoss()(b(x), y)
    assert abs(la.item() - lb.item()) < 1e-5
    la.backward()
    lb.backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-4)


@pytest.mark.parametrize("entangler,version", VARIANTS)
def test_per_sample_weights_equal_shared_weights(entangler, version):
    m = TorchVQC(8, 8, 3, seed=4, entangler=entangler, version=version)
    x = torch.rand(5, 8) * torch.pi
    th = m.theta.detach().expand(5, *m.theta.shape)
    assert torch.allclose(m.expz(x), m.expz(x, th), atol=1e-6)


@pytest.mark.parametrize("version", [1, 2])
def test_per_sample_gradients_in_one_pass(version):
    """DP-SGD trick: give every sample its own copy of the weights -> one backward = per-sample grads."""
    m = TorchVQC(8, 8, 3, seed=5, version=version)
    x = torch.rand(4, 8) * torch.pi
    y = torch.randint(0, 8, (4,))
    ref = []
    for i in range(4):
        m.zero_grad()
        nn.CrossEntropyLoss()(m(x[i:i + 1]), y[i:i + 1]).backward()
        ref.append(m.theta.grad.clone())
    th = m.theta.detach().expand(4, *m.theta.shape).clone().requires_grad_(True)
    nn.CrossEntropyLoss(reduction="sum")(m.logits(x, th), y).backward()
    for i in range(4):
        assert torch.allclose(th.grad[i], ref[i], atol=1e-5)


def _dead_angles(n_classes, entangler, version):
    """Number of theta entries whose gradient is ~0 for every one of 16 random inputs (D6)."""
    m = TorchVQC(8, n_classes, 3, seed=7, entangler=entangler, version=version).double()
    x = torch.rand(16, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(7)) * torch.pi
    th = m.theta.detach().expand(16, *m.theta.shape).clone().requires_grad_(True)
    m.expz(x, th).sum().backward()                       # per-sample gradients
    return int((th.grad.abs().amax(dim=0) < 1e-9).sum())


@pytest.mark.parametrize("n_classes,entangler,version,dead", [
    (8, "ring", 1, 8),     # D6: the last layer's 8 RZ angles never reach the output
    (2, "ring", 1, 9),     # binary: + the last RY on the readout qubit
    (2, "ring", 2, 0),     # version 2 fixes it ...
    (8, "ring", 2, 0),
    (2, "chain", 2, 0),    # ... for the chain too (reading the last qubit)
    (8, "chain", 2, 0),
])
def test_dead_parameters(n_classes, entangler, version, dead):
    assert _dead_angles(n_classes, entangler, version) == dead


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU on this machine")
def test_gpu_matches_cpu():
    m = TorchVQC(8, 8, 3, seed=6)
    x = torch.rand(32, 8) * torch.pi
    assert torch.allclose(m(x), m.to("cuda")(x.to("cuda")).cpu(), atol=1e-5)
