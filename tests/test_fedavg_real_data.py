"""FedAvg equivalence beyond random numbers (Phase 0 amendment 8).

1. Always (CI): data shaped like real traffic - one class dominating, many identical rows, clients so
   small that their rows repeat inside a round, clients holding a single class.
2. When the cleaned CICIoT2023 file is on the machine: the same check on real rows. Skipped otherwise
   (CI has no data); task FED-4 runs it inside the federated trial run.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fqml.fl.equivalence import (compare_modes, passed, real_data_cases, sample_parquet, sqrt_class_weights,
                                 to_model_inputs)
from fqml.models.mlp import MatchedMLP
from fqml.quantum.torch_sim import TorchVQC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/interim/ciciot2023_v1/ciciot2023_clean_v1.parquet"
REPORT = ROOT / "results/data_reports/ciciot2023_clean_v1.json"


def _traffic_like(n=900, classes=8, seed=0):
    """85% of rows in two classes, a quarter of the rows exact copies of other rows."""
    rng = np.random.default_rng(seed)
    p = np.array([0.45, 0.40] + [0.15 / (classes - 2)] * (classes - 2))
    y = rng.choice(classes, n, p=p)
    X = rng.random((n, 8)) * np.pi
    copies = rng.choice(n, n // 4, replace=False)
    X[copies] = X[rng.choice(n, n // 4)]                     # exact duplicate feature rows
    return X, y


def _skewed_parts(y, seed=0):
    """Up to 4 clients hold one class only, 3 have fewer rows than one batch, 5 share what is left."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    parts = [np.sort(np.flatnonzero(y == c)[:40]) for c in np.unique(y)[:4]]
    used = set(np.concatenate(parts).tolist())
    rest = np.array([i for i in order if i not in used])
    parts += [np.sort(rest[i * 5:(i + 1) * 5]) for i in range(3)]          # 5 rows < batch of 8
    parts += [np.sort(c) for c in np.array_split(rest[15:], 5)]
    return parts


@pytest.mark.parametrize("kind", ["vqc", "vqc_binary", "vqc_lin", "vqc_noent", "vqc_noent_binary", "mlp", "mlp_p512"])
def test_batched_equals_sequential_on_traffic_like_data(kind):
    X, y = _traffic_like()
    classes = 2 if kind.endswith("binary") else 8
    if classes == 2:
        y = (y != 0).astype(np.int64)
    parts = _skewed_parts(y)
    model = {"vqc": lambda: TorchVQC(8, 8, 3, seed=0),
             "vqc_binary": lambda: TorchVQC(8, 2, 3, seed=0),
             "vqc_lin": lambda: TorchVQC(8, 8, 3, seed=0, readout="linear"),
             "vqc_noent": lambda: TorchVQC(8, 8, 3, seed=0, entangler="none", readout="linear"),
             "vqc_noent_binary": lambda: TorchVQC(8, 2, 3, seed=0, entangler="none", readout="linear"),
             "mlp": lambda: MatchedMLP(8, 8, target_params=64, seed=0),
             "mlp_p512": lambda: MatchedMLP(8, 8, target_params=512, seed=0)}[kind]()
    res = compare_modes(model, X, y, parts, class_weights=sqrt_class_weights(y, classes),
                        rounds=3, clients_per_round=6, local_steps=4, batch_size=8, seed=2)
    assert res["same_clients"] and res["weights_moved"]
    assert res["max_weight_diff"] <= 1e-7 and res["max_logit_diff"] <= 1e-7, res
    assert passed(res)


def test_model_inputs_are_in_range():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    F = rng.normal(size=(2000, 12)) * 50
    F[:, 3] = y * 10 + rng.normal(size=2000)                 # informative
    F[:, 5] = 4.0                                            # constant: never chosen
    X, top = to_model_inputs(F, y, n_features=8)
    assert X.shape == (2000, 8) and X.min() >= 0 and X.max() <= np.pi + 1e-12
    assert 3 in top and 5 not in top and list(top) == sorted(top)


def test_sample_parquet_is_even(tmp_path):
    pytest.importorskip("pyarrow")
    import pandas as pd
    df = pd.DataFrame({"a": np.arange(10_000, dtype=np.float32), "y": np.arange(10_000) % 3, "skip": 0})
    df.to_parquet(tmp_path / "d.parquet", row_group_size=3000)
    got = sample_parquet(tmp_path / "d.parquet", ["a", "y"], 500)
    assert list(got.columns) == ["a", "y"] and len(got) == 500
    assert (np.diff(got["a"].to_numpy()) == 20).all()


@pytest.mark.skipif(not (DATA.exists() and REPORT.exists()), reason="cleaned CICIoT2023 (v1) is not on this machine")
def test_batched_equals_sequential_on_real_ciciot2023():
    report = json.loads(REPORT.read_text())
    feats = [c for c in report["features"] if c != "IAT"]
    df = sample_parquet(DATA, feats + ["y"], 20_000)
    y = df["y"].to_numpy(np.int64)
    X, _ = to_model_inputs(df[feats].to_numpy(np.float64), y)
    torch.set_num_threads(1)
    for case in real_data_cases(X, y, len(report["families"])):
        assert case["passed"], case
