"""FED-2 attack grid: who is malicious and which labels change."""
import numpy as np

from experiments.fed2_attacks import choose_malicious, make_model, poison


def _setup():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 8, 2000)
    parts = [np.sort(p) for p in np.array_split(rng.permutation(2000), 10)]
    return y, parts


def test_twenty_percent_malicious_and_fixed_by_seed():
    y, parts = _setup()
    a, b = choose_malicious(parts, 0.2, 7), choose_malicious(parts, 0.2, 7)
    assert a.sum() == 2 and (a == b).all()


def test_label_flip_changes_only_malicious_rows_to_the_next_class():
    y, parts = _setup()
    out, mal, n = poison(y, parts, "label_flip", None, 8, 0.2, 1)
    bad = np.concatenate([parts[k] for k in np.flatnonzero(mal)])
    good = np.setdiff1d(np.arange(len(y)), bad)
    assert (out[good] == y[good]).all()
    assert (out[bad] == (y[bad] + 1) % 8).all() and n == len(bad)


def test_class_suppression_turns_only_the_target_into_benign_on_holders():
    y, parts = _setup()
    y[parts[0]] = np.where(y[parts[0]] == 4, 1, y[parts[0]])          # client 0 holds no Web (4)
    out, mal, n = poison(y, parts, "class_suppression", 4, 8, 0.2, 1)
    assert not mal[0]                                                  # can't suppress what it doesn't hold
    changed = np.flatnonzero(out != y)
    assert (y[changed] == 4).all() and (out[changed] == 0).all() and n == len(changed)
    bad = np.concatenate([parts[k] for k in np.flatnonzero(mal)])
    assert (out[bad][y[bad] == 4] == 0).all()


def test_models():
    for kind, lo, hi in (("mlp_p3000", 2700, 3300), ("mlp", 50, 70), ("vqc", 60, 70)):
        n = sum(p.numel() for p in make_model(kind, 8).parameters())
        assert lo <= n <= hi, (kind, n)
