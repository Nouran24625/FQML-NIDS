"""Checks the VQC and its classical twin behave as the plan specifies."""
import torch

from fqml.models.mlp import MatchedMLP
from fqml.models.sizes import size_ladder
from fqml.models.vqc import VQC, n_params


def test_vqc_param_counts_match_plan():
    # Plan section 7: 2*n*L rotation angles + 2 per output
    assert n_params(VQC(n_qubits=8, n_classes=2, n_layers=3)) == 48 + 2   # binary
    assert n_params(VQC(n_qubits=8, n_classes=8, n_layers=3)) == 48 + 16  # 8 families


def test_vqc_output_shape_and_gradients():
    model = VQC(n_qubits=8, n_classes=8, n_layers=3)
    x = torch.rand(5, 8) * torch.pi
    logits = model(x)
    assert logits.shape == (5, 8)
    loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, 8, (5,)))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_vqc_is_reproducible_with_seed():
    x = torch.rand(3, 8) * torch.pi
    a = VQC(seed=1)(x)
    b = VQC(seed=1)(x)
    assert torch.allclose(a, b)


def test_mlp_is_parameter_matched():
    target = n_params(VQC(n_qubits=8, n_classes=2, n_layers=3))  # 50
    mlp = MatchedMLP(8, 2, target_params=target)
    assert abs(n_params(mlp) - target) <= 11   # an 8->h->2 MLP can only move in steps of 11

def test_mlp_size_ladder_models():
    """Every rung builds, has exactly the logged parameter count, and keeps the common model contract."""
    for n_classes in (2, 8):
        vqc = n_params(VQC(n_qubits=8, n_classes=n_classes, n_layers=3))
        for rung in size_ladder(8, n_classes, vqc):
            mlp = MatchedMLP(8, n_classes, target_params=rung["target"], seed=0)
            assert mlp.hidden == rung["hidden"] and n_params(mlp) == rung["params"]
            assert mlp(torch.rand(5, 8) * torch.pi).shape == (5, n_classes)


def test_mlp_seed_fixes_the_starting_weights_and_leaves_the_global_rng_alone():
    torch.manual_seed(123)
    expected_next = torch.rand(3)
    torch.manual_seed(123)
    a = MatchedMLP(8, 8, target_params=256, seed=7)
    assert torch.equal(torch.rand(3), expected_next)          # global RNG untouched by a seeded model
    b = MatchedMLP(8, 8, target_params=256, seed=7)
    c = MatchedMLP(8, 8, target_params=256, seed=8)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert any(not torch.equal(p, q) for p, q in zip(a.parameters(), c.parameters()))
