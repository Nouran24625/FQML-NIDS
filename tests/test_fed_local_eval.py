"""FED-7 local-only mode and FED-1 federated evaluation (fqml/fl/batched.py)."""
import numpy as np
import pytest
import torch
from torch.func import functional_call

from fqml.fl.batched import client_batches, evaluate, predict_proba, run_fedavg
from fqml.fl.partition import dirichlet_partition
from fqml.models.mlp import MatchedMLP
from fqml.quantum.torch_sim import TorchVQC

FAMILIES = ["Benign", "DDoS", "DoS", "Recon", "Web", "BruteForce", "Spoofing", "Mirai"]


def _data(n=600, classes=8):
    g = torch.Generator().manual_seed(0)
    return torch.rand(n, 8, generator=g, dtype=torch.float64) * torch.pi, torch.randint(0, classes, (n,), generator=g)


@pytest.mark.parametrize("kind", ["vqc", "mlp"])
def test_local_mode_equals_each_client_alone(kind):
    """mode="local" must give, for every client, the model that client gets training alone: one Adam, its own
    rows only, rounds x local_steps steps, the same batch order."""
    torch.manual_seed(0)
    X, y = _data()
    parts = dirichlet_partition(y.numpy(), n_clients=4, alpha=0.5, seed=0, min_size=5)
    model = TorchVQC(8, 8, 3, seed=0) if kind == "vqc" else MatchedMLP(8, 8, target_params=64)
    model = model.double()
    counts = torch.bincount(y, minlength=8).double()
    w = (1 / counts.sqrt()) / (1 / counts.sqrt()).mean()
    rounds, steps, bs, lr, seed = 3, 2, 8, 0.05, 1
    res = run_fedavg(model, X, y, parts, rounds=rounds, clients_per_round=2, local_steps=steps, batch_size=bs,
                     lr=lr, class_weights=w, seed=seed, mode="local", dtype=torch.float64)
    assert res.params is None and len(res.history) == rounds
    assert all(h["clients"] == list(range(4)) for h in res.history)

    rng = np.random.default_rng(seed)                       # the same batch stream, round by round
    plan = [[client_batches(parts[k], steps, bs, rng) for k in range(4)] for _ in range(rounds)]
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    for k in range(4):
        m = TorchVQC(8, 8, 3, seed=0).double() if kind == "vqc" else None
        if m is None:
            m = MatchedMLP(8, 8, target_params=64).double()
        with torch.no_grad():
            for n, p in m.named_parameters():
                p.copy_(dict(model.named_parameters())[n])
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        for r in range(rounds):
            for s in range(steps):
                idx = torch.as_tensor(plan[r][k][s])
                loss = lossf(m(X[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        for n, p in m.named_parameters():
            assert torch.allclose(res.client_params[n][k], p.detach(), atol=1e-7), (k, n)


def test_local_mode_clients_differ_and_validation_is_per_client():
    X, y = _data()
    parts = dirichlet_partition(y.numpy(), n_clients=3, alpha=0.1, seed=0, min_size=5)
    res = run_fedavg(TorchVQC(8, 8, 3, seed=0), X.float(), y, parts, rounds=2, clients_per_round=1, local_steps=3,
                     batch_size=8, lr=0.05, mode="local", X_val=X[:100].float(), y_val=y[:100])
    th = res.client_params["theta"]
    assert th.shape[0] == 3 and not torch.allclose(th[0], th[1])
    assert len(res.final_val) == 3 and "val_macro_f1_mean" in res.history[-1]


def test_predict_proba_and_old_keys():
    X, y = _data()
    model = TorchVQC(8, 8, 3, seed=0).double()
    params = {n: p.detach() for n, p in model.named_parameters()}
    proba = predict_proba(model, params, X)
    assert proba.shape == (len(X), 8) and np.allclose(proba.sum(1), 1, atol=1e-5)
    with torch.no_grad():
        assert np.array_equal(proba.argmax(1), functional_call(model, params, (X,)).argmax(1).numpy())
    out = evaluate(model, params, X, y)
    assert set(out) == {"macro_f1", "accuracy"}


def test_fedavg_history_keys_unchanged():
    X, y = _data()
    parts = dirichlet_partition(y.numpy(), n_clients=4, alpha=0.5, seed=0, min_size=5)
    res = run_fedavg(MatchedMLP(8, 8, target_params=64, seed=0), X.float(), y, parts, rounds=1, clients_per_round=2,
                     local_steps=1, batch_size=8, X_val=X[:50].float(), y_val=y[:50])
    assert {"val_macro_f1", "val_accuracy", "round", "clients", "seconds"} <= set(res.history[0])
    assert res.final_val is None and res.client_params is None


def test_evaluate_with_metrics_module():
    pytest.importorskip("fqml.eval.metrics")               # EVAL-1 (Sheghaf, due Wed 25 Nov)
    X, y = _data()
    model = TorchVQC(8, 8, 3, seed=0).double()
    params = {n: p.detach() for n, p in model.named_parameters()}
    out = evaluate(model, params, X, y, families=FAMILIES)
    assert {"macro_f1", "accuracy", "multiclass", "binary"} <= set(out)
    import json
    json.dumps(out["multiclass"]) and json.dumps(out["binary"])


def test_unknown_mode():
    X, y = _data()
    with pytest.raises(ValueError):
        run_fedavg(MatchedMLP(8, 8, target_params=64, seed=0), X, y, [np.arange(10)], rounds=1, clients_per_round=1,
                   mode="flower")
