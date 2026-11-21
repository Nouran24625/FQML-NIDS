"""D11 part 2: the committed tuning trials are exactly what the generator produces (nobody hand-picked a
setting), every model has the same budget, and every value lies inside its search space."""
import math
from pathlib import Path

import pytest

from experiments.make_tuning_configs import MODELS, N_TRIALS, OUT, draw, render, seed_for


@pytest.mark.parametrize("name", list(MODELS))
def test_committed_file_matches_generator(name):
    path = Path(OUT) / f"{name}.yaml"
    assert path.exists(), f"{path} missing: run python -m experiments.make_tuning_configs"
    assert path.read_text() == render(name), f"{path} was edited or is stale"


@pytest.mark.parametrize("name", list(MODELS))
def test_equal_budget_and_values_in_range(name):
    _, _, space = MODELS[name]
    trials = draw(space, seed_for(name))
    assert len(trials) == N_TRIALS
    for t in trials:
        assert set(t) == set(space)
        for k, spec in space.items():
            v = t[k]
            if spec[0] == "choice":
                assert v in spec[1]
            else:
                lo, hi = spec[1], spec[2]
                assert lo * (1 - 1e-3) <= v <= hi * (1 + 1e-3), (k, v)
                assert math.isfinite(v)


def test_discrete_grid_has_no_repeated_trial():
    trials = draw(MODELS["rf"][2], seed_for("rf"))
    assert len({tuple(t.values()) for t in trials}) == N_TRIALS


def test_every_ladder_rung_and_e0_model_has_a_file():
    expected = {"vqc", "vqc_lin", "vqc_noent", "rff", "mlp_large", "lr", "xgb", "rf",
                *(f"mlp_p{p}" for p in (64, 120, 128, 256, 512, 1000, 3000))}
    assert set(MODELS) == expected
