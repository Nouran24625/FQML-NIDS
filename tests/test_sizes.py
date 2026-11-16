"""MLP size ladder (Phase 0 amendment 6). No torch needed."""
import pytest

from fqml.models.sizes import LADDER_TARGETS, matched_hidden, mlp_params, size_ladder


def test_matched_hidden_is_the_closest_width():
    for n_in, n_out in [(8, 2), (8, 8), (8, 7), (37, 8), (46, 2)]:
        for target in list(range(1, 400)) + [512, 1000, 3000, 20000]:
            h = matched_hidden(n_in, n_out, target)
            best = min(abs(mlp_params(n_in, k, n_out) - target) for k in range(1, 3000))
            assert h >= 1 and abs(mlp_params(n_in, h, n_out) - target) == best


@pytest.mark.parametrize("n_classes,vqc", [(2, 50), (8, 64), (7, 62)])
def test_size_ladder(n_classes, vqc):
    rungs = size_ladder(8, n_classes, vqc)
    assert [r["target"] for r in rungs] == [vqc, *LADDER_TARGETS] == [vqc, 128, 256, 512, 1000, 3000]
    assert [r["token"] for r in rungs] == [f"mlp_p{t}" for t in (vqc, 128, 256, 512, 1000, 3000)]
    step = 8 + n_classes + 1
    for r in rungs:
        assert r["params"] == mlp_params(8, r["hidden"], n_classes)
        assert abs(r["params"] - r["target"]) <= step // 2 + 1          # as close as one hidden layer allows
    sizes = [r["params"] for r in rungs]
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
