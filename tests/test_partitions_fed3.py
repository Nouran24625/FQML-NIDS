"""FED-3: one-type-per-client and sites partitions (INTERFACES 3.1)."""
import numpy as np
import pytest

from fqml.fl.partition import (load_partition, save_partition, site_assignment, sites_partition,
                               type_partition)

TYPES = [f"T{i:02d}" for i in range(12)]
FAMILY = {t: f"F{i % 4}" for i, t in enumerate(TYPES)}


def _native(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array(["BenignTraffic"] + TYPES, dtype=object)
    return labels[rng.integers(0, len(labels), n)]


def _is_exact_cover(parts, n):
    allrows = np.concatenate(parts)
    return len(allrows) == n and np.array_equal(np.sort(allrows), np.arange(n))


def test_type_partition():
    native = _native()
    parts = type_partition(native, seed=0)
    assert len(parts) == len(TYPES) and _is_exact_cover(parts, len(native))
    for t, p in zip(sorted(TYPES), parts):
        held = set(native[p]) - {"BenignTraffic"}
        assert held == {t}
        assert (native[p] == t).sum() == (native == t).sum()          # all rows of its type
        assert (native[p] == "BenignTraffic").sum() > 0
    assert all(np.array_equal(a, b) for a, b in zip(parts, type_partition(native, seed=0)))


@pytest.mark.parametrize("n_sites,k", [(5, 4), (6, 2), (12, 1), (3, 12)])
def test_site_assignment_covers_every_type(n_sites, k):
    held = site_assignment(TYPES, n_sites, k, seed=0)
    assert all(len(h) == k == len(set(h)) for h in held)
    assert set().union(*held) == set(TYPES)
    assert held == site_assignment(TYPES, n_sites, k, seed=0)


def test_site_assignment_impossible():
    with pytest.raises(ValueError):
        site_assignment(TYPES, 2, 5, seed=0)                         # 2 x 5 < 12
    with pytest.raises(ValueError):
        site_assignment(TYPES, 4, 13, seed=0)


def test_sites_partition_rows_follow_held_types():
    native = _native()
    parts, site_types = sites_partition(native, n_sites=5, types_per_site=4, seed=0)
    assert _is_exact_cover(parts, len(native))
    for p, held in zip(parts, site_types):
        assert set(native[p]) - {"BenignTraffic"} == set(held)
        assert (native[p] == "BenignTraffic").sum() > 0
    for t in TYPES:                                                  # a type's rows are split evenly
        counts = [int((native[p] == t).sum()) for p, h in zip(parts, site_types) if t in h]
        assert sum(counts) == (native == t).sum() and max(counts) - min(counts) <= 1
    again, again_types = sites_partition(native, 5, 4, seed=0)
    assert again_types == site_types and all(np.array_equal(a, b) for a, b in zip(parts, again))


def test_sites_partition_by_family():
    native = _native()
    parts, site_types = sites_partition(native, n_sites=4, types_per_site=2, seed=1, groups=FAMILY)
    assert _is_exact_cover(parts, len(native))
    for held in site_types:
        fams = {FAMILY[t] for t in held}
        assert len(fams) == 2                                        # two whole families
        assert set(held) == {t for t in TYPES if FAMILY[t] in fams}
    with pytest.raises(ValueError):
        sites_partition(native, 4, 2, seed=0, groups={"T00": "F0"})


def test_save_load_roundtrip(tmp_path):
    parts, _ = sites_partition(_native(), 5, 4, seed=0)
    path = tmp_path / "p.npz"
    save_partition(parts, path)
    assert all(np.array_equal(a, b) for a, b in zip(parts, load_partition(path)))
