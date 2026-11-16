"""Fold manifests (INTERFACES 1.2): protocol A builder and the checks every manifest must pass."""
import copy

import numpy as np
import pytest

from fqml.data.folds import (build_time_blocked, check_manifest, fold_ranges, fold_rows, load_manifest,
                             merge_ranges, ranges_len, ranges_to_rows, save_manifest, twin_mask)

RANGES = {"BenignTraffic": (0, 1000), "DDoS-ICMP_Flood": (1000, 6003), "XSS": (6003, 6060)}


def _label_of(rows):
    out = np.empty(len(rows), dtype=object)
    for lab, (a, b) in RANGES.items():
        out[(rows >= a) & (rows < b)] = lab
    return out


def test_ranges():
    assert merge_ranges([[5, 9], [0, 3], [3, 5], [20, 20], [8, 12]]) == [[0, 12]]
    assert merge_ranges([[4, 6], [0, 2]]) == [[0, 2], [4, 6]]
    assert ranges_len([[0, 2], [4, 6]]) == 4 and list(ranges_to_rows([[4, 6], [0, 2]])) == [0, 1, 4, 5]


@pytest.mark.parametrize("gap", [0, 7, 50])
def test_protocol_a(gap):
    m = build_time_blocked(RANGES, dataset="toy", n_folds=5, gap=gap)
    assert m["protocol"] == "A" and m["n_rows"] == 6060 and len(m["folds"]) == 5
    all_rows = np.arange(6060)
    tested = np.zeros(6060, int)
    for k in range(5):
        tr, va, te = (fold_rows(m, k, r) for r in ("train", "val", "test"))
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        tested[te] += 1
        for lab, (a, b) in RANGES.items():
            g = m["labels"][lab]["gap"]
            assert g == min(gap, int(0.1 * ((b - a) // 5)))            # reduced for the small type
            block = m["labels"][lab]["blocks"]
            assert block[0][0] == a and block[-1][1] == b and all(block[i][1] == block[i + 1][0] for i in range(4))
            # the test set is exactly block k, the validation set is inside block k+1
            mine = te[(te >= a) & (te < b)]
            assert mine[0] == block[k][0] and mine[-1] == block[k][1] - 1 and len(mine) == block[k][1] - block[k][0]
            v = va[(va >= a) & (va < b)]
            nb = block[(k + 1) % 5]
            assert v[0] >= nb[0] and v[-1] < nb[1]
            # rows of different roles are at least g rows apart inside the attack type
            t = tr[(tr >= a) & (tr < b)]
            for x, y in ((t, mine), (v, mine), (t, v)):
                d = np.abs(x[:, None] - y[None, :]).min() - 1 if len(x) < 3000 and len(y) < 3000 else None
                if d is not None:
                    assert d >= g
        # every attack type is in train, validation and test of every fold (real mix in the test block)
        for rows in (tr, va, te):
            assert set(_label_of(rows)) == set(RANGES)
        # dropped rows = one gap per boundary between blocks of different roles:
        # folds 1, 2 have three such boundaries (train|test, test|val, val|train), folds 0, 3, 4 two
        boundaries = 3 if k in (1, 2) else 2
        assert 6060 - len(tr) - len(va) - len(te) == boundaries * sum(v["gap"] for v in m["labels"].values())
    assert (tested == 1).all() and len(all_rows) == tested.sum()       # each row is tested exactly once


def test_checks_catch_bad_manifests(tmp_path):
    good = build_time_blocked(RANGES, dataset="toy", gap=10)
    save_manifest(good, tmp_path / "toy_A.json")
    assert load_manifest(tmp_path / "toy_A.json") == good

    def broken(edit):
        m = copy.deepcopy(good)
        edit(m)
        with pytest.raises(ValueError):
            check_manifest(m)

    broken(lambda m: m.update(format="something-else"))
    broken(lambda m: m["folds"][0]["train"].append(m["folds"][0]["test"][0]))          # leak: train = test
    broken(lambda m: m["folds"][0].update(val=merge_ranges(m["folds"][0]["val"] + [[195, 205]])))  # val overlaps test
    broken(lambda m: m["folds"][1].update(test=m["folds"][0]["test"]))                 # block tested twice
    broken(lambda m: m["folds"][2].update(test=[[0, 7000]]))                           # outside the data
    broken(lambda m: m["folds"].pop())                                                 # a fold missing

    def close_the_gap(m):                                                              # train touches test
        a, b = m["labels"]["BenignTraffic"]["blocks"][1]
        m["folds"][1]["train"] = merge_ranges(m["folds"][1]["train"] + [[a - 10, a]])
    broken(close_the_gap)
    with pytest.raises(ValueError):
        build_time_blocked({"tiny": (0, 4)}, dataset="toy")
    assert fold_ranges(good, 0, "test")[0] == [0, 200]
    with pytest.raises(ValueError):
        fold_ranges(good, 0, "holdout")


def test_twin_mask():
    train = np.array([5, 5, 9, -3], dtype=np.int64)
    test = np.array([9, 1, -3, 2, 5], dtype=np.int64)
    assert list(twin_mask(test, train)) == [True, False, True, False, True]
