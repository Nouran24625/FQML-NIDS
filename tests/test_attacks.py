"""FED-2: label flipping and class suppression change exactly the rows docs/ATTACKS.md says they change."""
import numpy as np
import pytest

from fqml.attacks.poison import (check_disjoint, choose_malicious, class_suppression, flip_map, holds,
                                 label_flip, poison_labels)
from fqml.fl.partition import dirichlet_partition, iid_partition


def _setup(n=2000, classes=8, clients=10):
    y = np.random.default_rng(0).integers(0, classes, n)
    return y, iid_partition(n, clients, seed=0)


def test_flip_map_moves_every_class():
    for c in (2, 7, 8):
        m = flip_map(c)
        assert sorted(m) == list(range(c)) and all(m[i] != i for i in range(c))
    assert list(flip_map(2)) == [1, 0]


def test_choose_malicious_count_and_seed():
    a = choose_malicious(50, 0.2, seed=3)
    assert a.sum() == 10 and np.array_equal(a, choose_malicious(50, 0.2, seed=3))
    assert not np.array_equal(a, choose_malicious(50, 0.2, seed=4))
    elig = np.zeros(50, bool); elig[:4] = True
    b = choose_malicious(50, 0.2, seed=0, eligible=elig)          # fewer eligible than asked: all of them
    assert b.sum() == 4 and b[:4].all()
    assert choose_malicious(50, 0.0, seed=0).sum() == 0


def test_label_flip_changes_only_malicious_rows():
    y, parts = _setup()
    mal = choose_malicious(len(parts), 0.2, seed=1)
    out = label_flip(y, parts, mal, 8)
    bad = np.concatenate([parts[k] for k in np.flatnonzero(mal)])
    good = np.setdiff1d(np.arange(len(y)), bad)
    assert np.array_equal(out[good], y[good])
    assert np.array_equal(out[bad], (y[bad] + 1) % 8)
    assert (out[bad] != y[bad]).all()


def test_class_suppression_relabels_target_only():
    y, parts = _setup()
    target = y == 4                                               # Web
    mal = choose_malicious(len(parts), 0.2, seed=1)
    out = class_suppression(y, parts, mal, target)
    bad = np.concatenate([parts[k] for k in np.flatnonzero(mal)])
    changed = np.flatnonzero(out != y)
    assert set(changed) <= set(bad[target[bad]])
    assert (out[bad[target[bad]]] == 0).all()
    assert np.array_equal(out[~np.isin(np.arange(len(y)), bad)], y[~np.isin(np.arange(len(y)), bad)])


def test_class_suppression_on_binary_task_uses_family_mask():
    fam = np.random.default_rng(1).integers(0, 8, 1000)
    yb = (fam != 0).astype(np.int64)
    parts = iid_partition(1000, 5, seed=0)
    out, mal, info = poison_labels(yb, parts, "class_suppression", n_classes=2, frac=0.4, seed=0,
                                   target_mask=fam == 4)
    flipped = np.flatnonzero(out != yb)
    assert (fam[flipped] == 4).all() and (out[flipped] == 0).all() and info["rows_changed"] == len(flipped)


def test_suppression_attackers_hold_the_target():
    """Under label skew most clients may hold no Web rows; attackers are drawn among those that do."""
    y = np.random.default_rng(2).integers(0, 8, 5000)
    parts = dirichlet_partition(y, 20, alpha=0.1, seed=0)
    target = y == 4
    _, mal, info = poison_labels(y, parts, "class_suppression", n_classes=8, frac=0.2, seed=0, target_mask=target)
    assert holds(parts, target)[mal].all()
    assert info["eligible_clients"] == int(holds(parts, target).sum())
    assert info["frac_reached"] <= 0.2


def test_none_is_identity_and_info_is_json():
    import json
    y, parts = _setup()
    out, mal, info = poison_labels(y, parts, "none", n_classes=8)
    assert np.array_equal(out, y) and not mal.any() and info["rows_changed"] == 0
    json.dumps(info)


def test_overlapping_partition_rejected():
    with pytest.raises(ValueError):
        check_disjoint([np.array([0, 1]), np.array([1, 2])])
    with pytest.raises(ValueError):
        poison_labels(np.zeros(3, int), [np.array([0, 5])], "label_flip", n_classes=2)
