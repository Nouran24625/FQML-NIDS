"""The PyTorch simulator must give the same results as the PennyLane VQC (decision D1),
for every circuit version and entangler (D6, D7)."""
import pytest
import torch
from torch import nn

from fqml.models.vqc import VQC, n_params
from fqml.quantum.torch_sim import TorchVQC

VARIANTS = [("ring", 1), ("ring", 2), ("chain", 2), ("none", 2)]


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


def _dead_angles(n_classes, entangler, version, readout="local"):
    """Number of theta entries whose gradient is ~0 for every one of 16 random inputs (D6).
    For the linear readout the outputs are mixed with random weights first, so no angle looks alive or
    dead only because of the starting output weights."""
    m = TorchVQC(8, n_classes, 3, seed=7, entangler=entangler, version=version, readout=readout).double()
    x = torch.rand(16, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(7)) * torch.pi
    th = m.theta.detach().expand(16, *m.theta.shape).clone().requires_grad_(True)
    z = m.expz(x, th)                                    # per-sample gradients
    if readout == "linear":
        w = torch.randn(m.n_out, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
        z = z @ w.T
    z.sum().backward()
    return int((th.grad.abs().amax(dim=0) < 1e-9).sum())


@pytest.mark.parametrize("n_classes,entangler,version,dead,readout", [
    (8, "ring", 1, 8, "local"),      # D6: the last layer's 8 RZ angles never reach the output
    (2, "ring", 1, 9, "local"),      # binary: + the last RY on the readout qubit
    (2, "ring", 2, 0, "local"),      # version 2 fixes it ...
    (8, "ring", 2, 0, "local"),
    (2, "chain", 2, 0, "local"),     # ... for the chain too (reading the last qubit)
    (8, "chain", 2, 0, "local"),
    (7, "chain", 2, 0, "local"),     # 7 classes (DDoS and DoS merged, label-map variant c7m)
    (8, "none", 2, 0, "local"),      # no CNOTs, every qubit read - but each output sees ONE feature
    (2, "none", 2, 42, "local"),     # no CNOTs, only the last qubit read: 7 qubits never reach the output
    (7, "none", 2, 6, "local"),      # qubit 0 is not read
    (2, "none", 2, 0, "linear"),     # the fix: every output reads every qubit
    (8, "none", 2, 0, "linear"),
    (7, "none", 2, 0, "linear"),
    (2, "chain", 2, 0, "linear"),    # the VQC the control is compared with
    (8, "chain", 2, 0, "linear"),
])
def test_dead_parameters(n_classes, entangler, version, dead, readout):
    assert _dead_angles(n_classes, entangler, version, readout) == dead


def test_no_entangler_outputs_use_one_feature_each():
    """entangler="none": <Z_q> depends only on feature q, so changing another feature changes nothing."""
    m = TorchVQC(8, 8, 3, seed=8, entangler="none").double()
    x = torch.rand(4, 8, dtype=torch.float64) * torch.pi
    x2 = x.clone()
    x2[:, 3] = torch.rand(4, dtype=torch.float64) * torch.pi
    z, z2 = m.expz(x), m.expz(x2)
    keep = [q for q in range(8) if q != 3]
    assert torch.allclose(z[:, keep], z2[:, keep], atol=1e-12) and not torch.allclose(z[:, 3], z2[:, 3])
    assert n_params(m) == n_params(TorchVQC(8, 8, 3))        # same parameter count as the chain circuit
    with pytest.raises(ValueError):
        TorchVQC(8, 8, 3, entangler="star")


@pytest.mark.parametrize("n_classes", [2, 8])
def test_no_entangler_linear_readout_uses_every_feature(n_classes):
    """The entanglement-free control (entangler="none", readout="linear") sees all 8 features in every
    output, like the entangled VQC; only the CNOT gates differ between the two."""
    m = TorchVQC(8, n_classes, 3, seed=9, entangler="none", readout="linear").double()
    with torch.no_grad():
        m.scale.copy_(torch.randn(m.scale.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(9)))
    x = torch.rand(4, 8, dtype=torch.float64) * torch.pi
    out = m(x)
    for q in range(8):
        x2 = x.clone()
        x2[:, q] = (x2[:, q] + 1.0) % torch.pi
        moved = (m(x2) - out).abs().amax(dim=0)
        assert (moved[1:] if n_classes == 2 else moved).min() > 1e-6, (q, moved)


@pytest.mark.parametrize("n_classes,local,linear", [(2, 50, 57), (8, 64, 120), (7, 62, 111)])
def test_linear_readout_counts_and_start(n_classes, local, linear):
    """Linear readout: (n + 1) weights per output. It starts as the local readout, same seed same model."""
    for entangler in ("chain", "none"):
        a = TorchVQC(8, n_classes, 3, seed=1, entangler=entangler)
        b = TorchVQC(8, n_classes, 3, seed=1, entangler=entangler, readout="linear")
        assert n_params(a) == local and n_params(b) == linear
        assert b.scale.shape == (b.n_out, 8) and torch.equal(a.theta, b.theta)
        x = torch.rand(5, 8) * torch.pi
        assert torch.allclose(a(x), b(x), atol=1e-6)
    with pytest.raises(ValueError):
        TorchVQC(8, 2, 3, readout="global")


@pytest.mark.parametrize("entangler", ["chain", "none"])
@pytest.mark.parametrize("n_classes", [2, 8])
def test_linear_readout_matches_pennylane(n_classes, entangler):
    kw = dict(entangler=entangler, readout="linear")
    a, b = VQC(8, n_classes, 3, seed=2, **kw), TorchVQC(8, n_classes, 3, seed=2, **kw)
    w = torch.linspace(-1.5, 1.5, a.scale.numel()).reshape(a.scale.shape)
    with torch.no_grad():                                # move off the start so every output weight matters
        a.scale.copy_(w)
        b.scale.copy_(w)
    assert n_params(a) == n_params(b)
    x = torch.rand(6, 8) * torch.pi
    y = torch.randint(0, n_classes, (6,))
    la, lb = a(x), b(x)
    assert la.shape == lb.shape == (6, n_classes) and torch.allclose(la, lb, atol=1e-5)
    nn.CrossEntropyLoss()(la, y).backward()
    nn.CrossEntropyLoss()(lb, y).backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-4)


def test_linear_readout_per_sample_weights_and_gradients():
    """DP-SGD (D4) with the linear readout: per-sample theta AND per-sample output weights [B, n_out, n]."""
    m = TorchVQC(8, 8, 3, seed=5, entangler="none", readout="linear")
    x = torch.rand(4, 8) * torch.pi
    y = torch.randint(0, 8, (4,))
    th = m.theta.detach().expand(4, *m.theta.shape)
    sc = m.scale.detach().expand(4, *m.scale.shape)
    bi = m.bias.detach().expand(4, *m.bias.shape)
    assert torch.allclose(m.logits(x), m.logits(x, th, sc, bi), atol=1e-6)
    ref = []
    for i in range(4):
        m.zero_grad()
        nn.CrossEntropyLoss()(m(x[i:i + 1]), y[i:i + 1]).backward()
        ref.append((m.theta.grad.clone(), m.scale.grad.clone()))
    th = th.clone().requires_grad_(True)
    sc = sc.clone().requires_grad_(True)
    nn.CrossEntropyLoss(reduction="sum")(m.logits(x, th, sc), y).backward()
    for i in range(4):
        assert torch.allclose(th.grad[i], ref[i][0], atol=1e-5)
        assert torch.allclose(sc.grad[i], ref[i][1], atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU on this machine")
def test_gpu_matches_cpu():
    m = TorchVQC(8, 8, 3, seed=6)
    x = torch.rand(32, 8) * torch.pi
    assert torch.allclose(m(x), m.to("cuda")(x.to("cuda")).cpu(), atol=1e-5)
