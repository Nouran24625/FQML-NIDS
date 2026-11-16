"""Run IDs carry the protocol and the fold (INTERFACES 5; Phase 0 amendment 4)."""
import pytest

from fqml.utils.run_id import SEEDS, RunID, make_run_id, parse_run_id, setting_runs


def test_round_trip():
    rid = make_run_id(0, "ciciot2023", "mlp_p128", "c8.top8", "A", 3, 2)
    assert rid == "E0-ciciot2023-mlp_p128-c8.top8-A-f3-s2"
    assert parse_run_id(rid) == RunID(0, "ciciot2023", "mlp_p128", "c8.top8", "A", 3, 2)
    assert str(parse_run_id("E12-ciciomt2024-vqc_noent-c2.top8.eps8-C-f0-s1")) == "E12-ciciomt2024-vqc_noent-c2.top8.eps8-C-f0-s1"
    assert parse_run_id(rid).asdict()["fold"] == 3


@pytest.mark.parametrize("bad", [
    "E1-ciciot2023-vqc-dir0.5-s0",            # the Phase 0 format: no protocol, no fold
    "E0-ciciot2023-vqc-c8-D-f0-s0",           # unknown protocol
    "E0-ciciot2023-vqc-c8-A-f0",              # no seed
    "E0-cic-iot-vqc-c8-A-f0-s0",              # '-' inside a part
    "e0-ciciot2023-vqc-c8-A-f0-s0",
])
def test_rejects(bad):
    with pytest.raises(ValueError):
        parse_run_id(bad)


@pytest.mark.parametrize("kw", [dict(dataset="cic-iot"), dict(model="mlp p128"), dict(variant=""),
                                dict(protocol="a"), dict(fold=-1), dict(seed=1.0), dict(seed=True)])
def test_make_rejects(kw):
    args = dict(experiment=0, dataset="ciciot2023", model="vqc", variant="c8.top8", protocol="A", fold=0, seed=0)
    args.update(kw)
    with pytest.raises(ValueError):
        make_run_id(**args)


def test_setting_runs_is_5_folds_x_3_seeds():
    runs = setting_runs(0, "ciciot2023", "vqc", "c8.top8", "A")
    assert len(runs) == 15 == len(set(runs)) and SEEDS == (0, 1, 2)
    assert runs[0].endswith("-A-f0-s0")                      # fold 0, seed 0 is launched first
    assert [parse_run_id(r).seed for r in runs[:5]] == [0] * 5
    assert len(setting_runs(0, "ciciot2023", "vqc", "c8.top8", "C")) == 3
    assert len(setting_runs(0, "ciciot2023", "vqc", "c8.top8", "B", n_folds=4)) == 12
    with pytest.raises(ValueError):
        setting_runs(0, "ciciot2023", "vqc", "c8.top8", "B")
