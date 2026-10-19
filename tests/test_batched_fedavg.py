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
    classes = 2 if kind == "vqc_binary" else 8
    X, y = _data(classes=classes)
    parts = dirichlet_partition(y.numpy(), n_clients=6, alpha=0.5, seed=0, min_size=5)
    if kind.startswith("vqc"):
        model = TorchVQC(8, classes, 3, seed=0)
    else:
        model = MatchedMLP(8, classes, target_params=64)
    counts = torch.bincount(y, minlength=classes).float()
    w = (1 / counts.sqrt()) / (1 / counts.sqrt()).mean()
    kw = dict(rounds=2, clients_per_round=3, local_steps=3, batch_size=8, lr=0.05, class_weights=w, seed=1)
    a = run_fedavg(model, X, y, parts, mode="batched", **kw)
    b = run_fedavg(model, X, y, parts, mode="sequential", **kw)
    assert [h["clients"] for h in a.history] == [h["clients"] for h in b.history]

    # 1) The two final global models must make the same predictions.
    with torch.no_grad():
        la = functional_call(model, a.params, (X,))
        lb = functional_call(model, b.params, (X,))
    assert torch.allclose(la, lb, atol=1e-5), (la - lb).abs().max()

    # 2) Every weight that affects the output must match too.
    #    Exception: the last layer's RZ angles of the VQC (theta[-1, :, 1]). After them come only CNOTs and
    #    a Z measurement, so they never change the output and their true gradient is exactly 0. Float32
    #    round-off leaves ~1e-7 noise there, and Adam rescales any gradient to a step of size ~lr, so the
    #    two loops drift on these 8 angles by different noise. That is harmless (check 1 proves it).
    for n in a.params:
        pa, pb = a.params[n], b.params[n]
        if kind.startswith("vqc") and n == "theta":
            pa, pb = pa[:-1], pb[:-1]                                          # all layers but the last
            assert torch.allclose(a.params[n][-1, :, 0], b.params[n][-1, :, 0], atol=1e-5)  # last-layer RY
        assert torch.allclose(pa, pb, atol=1e-5), n

    # and training actually moved the weights
    start = dict(model.named_parameters())
    assert any(not torch.allclose(a.params[n], start[n]) for n in a.params)


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
