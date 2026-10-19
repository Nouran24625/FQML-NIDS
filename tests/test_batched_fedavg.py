"""Batched FedAvg (D3) must equal training clients one by one, and partitions must be reproducible."""
import numpy as np
import pytest
import torch
from torch.func import functional_call

from fqml.fl.batched import run_fedavg
from fqml.fl.partition import dirichlet_partition, iid_partition
from fqml.models.mlp import MatchedMLP
from fqml.quantum.torch_sim import TorchVQC


def _data(n=600, classes=8):
    g = torch.Generator().manual_seed(0)
    X = torch.rand(n, 8, generator=g) * torch.pi
    y = torch.randint(0, classes, (n,), generator=g)
    return X, y


@pytest.mark.parametrize("kind", ["vqc", "vqc_binary", "mlp"])
def test_batched_equals_sequential(kind):
    """Batched FedAvg (D3) must give the same global model as training clients one by one.

    Runs in float64. In float32, rounding noise (~1e-7) is about as large as Adam's eps (1e-8), so for
    weights whose true gradient is ~0 (e.g. the VQC's last-layer RZ angles, see D6) Adam blows the noise
    up into full-size steps and the two loops drift apart on those weights. In float64 the noise (~1e-16)
    is far below eps, so every weight must match to 1e-7, with no exceptions.
    """
    torch.manual_seed(0)                       # MatchedMLP initialises from the global RNG
    classes = 2 if kind == "vqc_binary" else 8
    X, y = _data(classes=classes)
    X = X.double()
    parts = dirichlet_partition(y.numpy(), n_clients=6, alpha=0.5, seed=0, min_size=5)
    if kind.startswith("vqc"):
        model = TorchVQC(8, classes, 3, seed=0)
    else:
        model = MatchedMLP(8, classes, target_params=64)
    counts = torch.bincount(y, minlength=classes).double()
    w = (1 / counts.sqrt()) / (1 / counts.sqrt()).mean()
    kw = dict(rounds=2, clients_per_round=3, local_steps=3, batch_size=8, lr=0.05, class_weights=w,
              seed=1, dtype=torch.float64)
    a = run_fedavg(model, X, y, parts, mode="batched", **kw)
    b = run_fedavg(model, X, y, parts, mode="sequential", **kw)
    assert [h["clients"] for h in a.history] == [h["clients"] for h in b.history]

    model = model.double()
    with torch.no_grad():
        la = functional_call(model, a.params, (X,))
        lb = functional_call(model, b.params, (X,))
    assert torch.allclose(la, lb, atol=1e-7), (la - lb).abs().max()          # same predictions
    for n in a.params:                                                       # same weights, all of them
        assert torch.allclose(a.params[n], b.params[n], atol=1e-7), (n, (a.params[n] - b.params[n]).abs().max())
    start = {n: p.detach() for n, p in model.named_parameters()}
    assert any(not torch.allclose(a.params[n], start[n]) for n in a.params)  # training actually moved them

def test_dirichlet_partition_reproducible_and_complete():
    y = np.random.default_rng(0).integers(0, 8, 5000)
    p1 = dirichlet_partition(y, 50, 0.5, seed=3)
    p2 = dirichlet_partition(y, 50, 0.5, seed=3)
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2))
    allrows = np.concatenate(p1)
    assert len(allrows) == len(y) and len(np.unique(allrows)) == len(y)
    assert min(len(p) for p in p1) >= 10


def test_smaller_alpha_is_more_skewed():
    y = np.random.default_rng(0).integers(0, 8, 20000)

    def mean_classes_present(alpha):
        parts = dirichlet_partition(y, 50, alpha, seed=0, min_size=1)
        return np.mean([np.mean(np.bincount(y[p], minlength=8) > 0.02 * len(p)) for p in parts])

    assert mean_classes_present(0.1) < mean_classes_present(1.0)


def test_iid_partition_sizes():
    parts = iid_partition(1003, 10, seed=0)
    assert sum(len(p) for p in parts) == 1003
    assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1
