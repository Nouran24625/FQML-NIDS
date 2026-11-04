"""Preprocessing + saved demo model ("bundle") used by train.py and detect.py. Owner: Nouran.

Bundle directory:
    bundle.json   classes, chosen features, preprocessing settings, window, model type, test report
    model.pt      torch state_dict + constructor args (model type vqc or mlp)
    model.pkl     pickled scikit-learn model (model type rf)

Preprocessing (same rules as INTERFACES.md 1.3, fitted on TRAINING rows only):
    log1p on count/size/time features -> clip to the train 1st-99th percentile -> min-max to [0, pi]
"""
from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fqml.live.features import RATIO_FEATURES


@dataclass
class Preprocess:
    features: list[str]
    lo: list[float]
    hi: list[float]
    log: list[bool]

    @classmethod
    def fit(cls, X: np.ndarray, features: list[str]) -> "Preprocess":
        log = [f not in RATIO_FEATURES for f in features]
        Z = cls._log(X, log)
        lo = np.percentile(Z, 1, axis=0)
        hi = np.percentile(Z, 99, axis=0)
        return cls(features, lo.tolist(), hi.tolist(), log)

    @staticmethod
    def _log(X, log):
        Z = np.asarray(X, dtype=np.float64).copy()
        for j, use in enumerate(log):
            if use:
                Z[:, j] = np.log1p(np.maximum(Z[:, j], 0))
        return Z

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self._log(X, self.log)
        lo, hi = np.array(self.lo), np.array(self.hi)
        span = np.where(hi > lo, hi - lo, 1.0)
        Z = (np.clip(Z, lo, hi) - lo) / span
        Z[:, hi <= lo] = 0.0                                  # constant on train -> carries no signal
        return (Z * math.pi).astype(np.float32)


def build_torch_model(kind: str, n_features: int, n_classes: int, seed: int = 0, target_params: int | None = None):
    import torch  # noqa: F401  (imported lazily so the rf path works without torch)

    if kind == "vqc":
        from fqml.quantum.torch_sim import TorchVQC
        return TorchVQC(n_qubits=n_features, n_classes=n_classes, n_layers=3, seed=seed,
                        entangler="chain", version=2)
    if kind == "mlp":
        from fqml.models.mlp import MatchedMLP
        return MatchedMLP(n_features, n_classes, target_params=target_params)
    raise ValueError(kind)


class Classifier:
    """Loaded bundle: rows (dicts or DataFrame with the raw feature columns) -> class probabilities."""

    def __init__(self, bundle_dir: str | Path):
        d = Path(bundle_dir)
        self.meta = json.loads((d / "bundle.json").read_text())
        self.classes: list[str] = self.meta["classes"]
        self.pre = Preprocess(**self.meta["preprocess"])
        self.kind = self.meta["model"]["kind"]
        if self.kind == "rf":
            with open(d / "model.pkl", "rb") as f:
                self.model = pickle.load(f)
        else:
            import torch
            ckpt = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
            m = build_torch_model(self.kind, len(self.pre.features), len(self.classes), **ckpt["init"])
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            self.model = m

    def predict_proba(self, rows) -> np.ndarray:
        if len(rows) == 0:
            return np.zeros((0, len(self.classes)))
        if isinstance(rows, list):
            X = np.array([[r[f] for f in self.pre.features] for r in rows], dtype=np.float64)
        else:
            X = rows[self.pre.features].to_numpy(dtype=np.float64)
        Z = self.pre.transform(X)
        if self.kind == "rf":
            P = np.zeros((len(Z), len(self.classes)))
            P[:, self.model.classes_] = self.model.predict_proba(Z)
            return P
        import torch
        with torch.no_grad():
            return torch.softmax(self.model(torch.from_numpy(Z)), dim=1).numpy()


def save_bundle(out: str | Path, *, classes, pre: Preprocess, kind: str, model, init: dict, extra: dict):
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    if kind == "rf":
        with open(d / "model.pkl", "wb") as f:
            pickle.dump(model, f)
    else:
        import torch
        torch.save({"state_dict": model.state_dict(), "init": init}, d / "model.pt")
    meta = {"classes": classes, "preprocess": asdict(pre), "model": {"kind": kind, "init": init}, **extra}
    (d / "bundle.json").write_text(json.dumps(meta, indent=2))
