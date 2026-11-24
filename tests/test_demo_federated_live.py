"""Live federation demo: the parts that run without data or a network (models, splits, label attacks)."""
import numpy as np
import pytest

pytest.importorskip("torch")
from demo.federated_live import label_attack, make_model, make_split  # noqa: E402

FAMS = ["Benign", "DDoS", "DoS", "Recon", "Web", "BruteForce", "Spoofing", "Mirai"]


@pytest.mark.parametrize("kind,size", [("mlp_p512", 512), ("mlp_p3000", 3000)])
def test_size_ladder_models(kind, size):
    m = make_model(kind, 8)
    assert abs(sum(p.numel() for p in m.parameters()) - size) <= 0.1 * size


@pytest.mark.parametrize("kind", ["vqc", "vqc_lin", "vqc_noent", "mlp"])
def test_project_models(kind):
    assert sum(p.numel() for p in make_model(kind, 8).parameters()) > 0


def test_splits_give_every_row_to_one_site():
    rng = np.random.default_rng(0)
    types = [f"DDoS-{i}" for i in range(12)] + [f"Web-{i}" for i in range(6)] + ["BenignTraffic"]
    native = np.array(rng.choice(types, 3000), dtype=object)
    y = np.array([0 if t == "BenignTraffic" else (1 if t.startswith("DDoS") else 4) for t in native])
    for name in ("iid", "dir0.5", "sites"):
        parts = make_split(name, native, y, 3, 0)
        allrows = np.concatenate(parts)
        assert len(parts) == 3 and len(np.unique(allrows)) == len(allrows)


def test_label_attacks():
    y = np.array([0, 1, 4, 4, 7])
    same, n = label_attack(y.copy(), "none", FAMS, "Web")
    assert n == 0 and (same == y).all()
    sup, n = label_attack(y.copy(), "class_suppression", FAMS, "Web")
    assert n == 2 and list(sup) == [0, 1, 0, 0, 7]
    flip, _ = label_attack(y.copy(), "label_flip", FAMS, "Web")
    assert list(flip) == [1, 2, 5, 5, 0]
    with pytest.raises(SystemExit):
        label_attack(y, "backdoor", FAMS, "Web")
