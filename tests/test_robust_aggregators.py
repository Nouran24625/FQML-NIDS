"""FED-6: robust aggregators and the attack hook in run_fedavg."""
import numpy as np
import pytest
import torch

from fqml.fl.batched import fedavg, run_fedavg
from fqml.fl.robust import coordinate_median, get_aggregator, trimmed_mean
from fqml.models.mlp import MatchedMLP


def _stacked(K=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(K, 3, 4, generator=g, dtype=torch.float64),
            "b": torch.randn(K, 4, generator=g, dtype=torch.float64)}


def test_median_matches_numpy_odd_and_even():
    for K in (7, 8):
        s = _stacked(K)
        got = coordinate_median(s, torch.ones(K))
        for n, v in s.items():
            assert got[n].shape == v.shape[1:]
            assert torch.allclose(got[n], torch.as_tensor(np.median(v.numpy(), axis=0)))


def test_trimmed_mean_matches_definition():
    s = _stacked(10)
    got = trimmed_mean(s, torch.ones(10), beta=0.2)          # drop 2 at each end, average 6
    for n, v in s.items():
        a = np.sort(v.numpy(), axis=0)[2:8].mean(axis=0)
        assert torch.allclose(got[n], torch.as_tensor(a))


def test_beta_zero_is_plain_mean_and_equals_fedavg_with_equal_sizes():
    s = _stacked(5)
    tm = trimmed_mean(s, torch.ones(5), beta=0.0)
    fa = fedavg(s, torch.ones(5, dtype=torch.float64))
    for n in s:
        assert torch.allclose(tm[n], s[n].mean(dim=0)) and torch.allclose(tm[n], fa[n])


def test_robust_aggregators_ignore_row_counts():
    s = _stacked(6)
    for agg in (coordinate_median, get_aggregator("trimmed_mean")):
        a, b = agg(s, torch.ones(6)), agg(s, torch.tensor([1.0, 1, 1, 1, 1, 1e6]))
        assert all(torch.equal(a[n], b[n]) for n in s)


def test_one_huge_outlier_moves_fedavg_but_not_the_robust_ones():
    s = {n: v.clone() for n, v in _stacked(10).items()}
    clean = {n: v.clone() for n, v in s.items()}
    for v in s.values():
        v[3] = 1e6
    ones = torch.ones(10, dtype=torch.float64)
    assert (fedavg(s, ones)["w"] - fedavg(clean, ones)["w"]).abs().max() > 1e4
    for agg in (coordinate_median, get_aggregator("trimmed_mean")):
        gap = (agg(s, ones)["w"] - agg(clean, ones)["w"]).abs().max()
        assert gap < 5, (agg, gap)                         # bounded by the spread of the honest values


def test_identical_clients_give_the_same_weights():
    v = torch.randn(3, 4, dtype=torch.float64)
    s = {"w": v.unsqueeze(0).repeat(6, 1, 1)}
    for name in ("fedavg", "median", "trimmed_mean"):
        assert torch.allclose(get_aggregator(name)(s, torch.ones(6, dtype=torch.float64))["w"], v)


def test_bad_settings_are_refused():
    with pytest.raises(ValueError):
        trimmed_mean(_stacked(4), beta=0.5)
    with pytest.raises(ValueError):
        trimmed_mean(_stacked(4), beta=-0.1)
    with pytest.raises(ValueError):
        get_aggregator("krum")
    assert get_aggregator("trimmed_mean0.1").keywords["beta"] == 0.1


# --- run_fedavg hooks -----------------------------------------------------------------------------------------

def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    X = rng.random((n, 8)) * np.pi
    X[:, 0] += y                                            # learnable
    parts = [np.sort(p) for p in np.array_split(rng.permutation(n), 10)]
    return X, y, parts


def _run(**kw):
    X, y, parts = _data()
    torch.manual_seed(0)
    model = MatchedMLP(8, 4, target_params=64, seed=0)
    return run_fedavg(model, X, y, parts, rounds=3, clients_per_round=5, local_steps=3, batch_size=16,
                      seed=1, dtype=torch.float64, **kw)


def test_default_aggregator_is_unchanged():
    a, b = _run(), _run(aggregator=fedavg)
    assert all(torch.equal(a.params[n], b.params[n]) for n in a.params)
    assert "malicious" not in a.history[0]


@pytest.mark.parametrize("name", ["median", "trimmed_mean"])
def test_robust_aggregator_runs_and_differs_from_fedavg(name):
    a, b = _run(), _run(aggregator=get_aggregator(name))
    assert all(torch.isfinite(v).all() for v in b.params.values())
    assert any(not torch.allclose(a.params[n], b.params[n]) for n in a.params)
    assert [h["clients"] for h in a.history] == [h["clients"] for h in b.history]   # same client sampling


def test_attack_hook_sees_the_round_clients_and_their_malicious_flags():
    malicious = np.zeros(10, dtype=bool)
    malicious[[2, 7]] = True
    seen = []

    def attack(stacked, chosen, mal):
        seen.append((np.asarray(chosen).copy(), np.asarray(mal).copy()))
        assert all(v.shape[0] == len(chosen) for v in stacked.values())
        return {n: torch.where(torch.as_tensor(mal).view(-1, *([1] * (v.dim() - 1))), v * 0 + 1e3, v)
                for n, v in stacked.items()}

    res = _run(attack=attack, malicious=malicious, aggregator=get_aggregator("median"))
    assert len(seen) == 3
    for (chosen, mal), h in zip(seen, res.history):
        assert list(chosen) == h["clients"] and list(mal) == [c in (2, 7) for c in chosen]
        assert h["malicious"] == int(mal.sum())


def test_attack_changes_fedavg_and_no_op_attack_changes_nothing():
    malicious = np.zeros(10, dtype=bool)
    malicious[:5] = True
    base = _run(malicious=malicious)
    noop = _run(malicious=malicious, attack=lambda s, c, m: s)
    assert all(torch.equal(base.params[n], noop.params[n]) for n in base.params)
    boom = _run(malicious=malicious, attack=lambda s, c, m: {n: v * 100 for n, v in s.items()})
    assert any(not torch.allclose(base.params[n], boom.params[n]) for n in base.params)


def test_hooks_are_refused_where_they_do_not_apply():
    with pytest.raises(ValueError):
        _run(mode="local", aggregator=get_aggregator("median"))
    with pytest.raises(ValueError):
        _run(attack=lambda s, c, m: s)                      # no malicious mask
    with pytest.raises(ValueError):
        _run(malicious=np.zeros(3, dtype=bool))             # wrong length


# --- input features (D9 and its Number addition) ----------------------------------------------------------------

def test_input_features_drop_iat_and_number_by_default():
    from experiments.fed6_zero_attacker import exclude_tag, input_features, parse_exclude
    report = {"features": ["Header_Length", "IAT", "Number", "Tot sum"]}
    assert input_features(report) == ["Header_Length", "Tot sum"]
    assert input_features(report, parse_exclude("IAT")) == ["Header_Length", "Number", "Tot sum"]
    assert input_features(report, ()) == ["Header_Length", "Number", "Tot sum"]      # IAT is always out
    assert exclude_tag(("IAT",)) == "" and exclude_tag(("IAT", "Number")) == "_xNumber"
    with pytest.raises(SystemExit):
        input_features(report, ("Nmber",))
