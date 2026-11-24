"""Every E0 model by name, built the same way for the runner (task CLS-1). Owner: Nouran.

One entry point, so the runner (EXP-0) never special-cases a model:

    from fqml.models.classical import build, load_tuning, is_torch, sample_weights
    trials, fixed = load_tuning("xgb")                         # configs/tuning/xgb.yaml (D11 part 2)
    model = build("xgb", n_in=35, n_classes=8, seed=0, **fixed, **trials[3])

Names (PREREG_E0 section 6, D11 part 2):
  PyTorch (trained by the shared loop, H-10): vqc, vqc_lin, vqc_noent, mlp_p<params> (MatchedMLP at that size,
      e.g. mlp_p64, mlp_p120, mlp_p3000), mlp_large (2 x 256), rff, and logreg (a single linear layer: the
      logistic regression of the DP pilot PRIV-4, which needs per-sample gradients).
  scikit-learn / XGBoost (trained by their own fit, with sample weights): lr, rf, xgb.
Every model gets the same inputs (DATA-6, float32 in [0, pi]) and the same class weighting (D11 part 1):
PyTorch models through the loop's weighted loss, the others through `sample_weights`.
Settings that belong to the training loop (lr, weight_decay, batch_size, max_epochs, patience) are accepted and
ignored by `build`; the runner passes them to the loop.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TUNING_DIR = ROOT / "configs/tuning"
LOOP_SETTINGS = {"lr", "weight_decay", "batch_size", "max_epochs", "patience"}
TORCH_MODELS = ("vqc", "vqc_lin", "vqc_noent", "mlp_large", "rff", "logreg")
VQC_LAYOUT = {"vqc": ("chain", "local"), "vqc_lin": ("chain", "linear"), "vqc_noent": ("none", "linear")}


def is_torch(name: str) -> bool:
    return name in TORCH_MODELS or re.fullmatch(r"mlp_p\d+", name) is not None


def load_tuning(name: str, tuning_dir: Path | str = TUNING_DIR) -> tuple[list[dict], dict]:
    """(trials, fixed) of configs/tuning/<name>.yaml; each trial without its `id`."""
    import yaml

    cfg = yaml.safe_load((Path(tuning_dir) / f"{name}.yaml").read_text())
    trials = [{k: v for k, v in t.items() if k != "id"} for t in cfg["trials"]]
    return trials, dict(cfg.get("fixed") or {})


def class_weights_np(y: np.ndarray, n_classes: int) -> np.ndarray:
    """D11 part 1: weight per class proportional to 1 / sqrt(count), mean 1 over classes; absent classes weight 0."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.zeros(n_classes)
    present = counts > 0
    w[present] = 1.0 / np.sqrt(counts[present])
    return w / w[present].mean()


def sample_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-row weights for scikit-learn and XGBoost: the D11 class weight of each row's class."""
    return class_weights_np(np.asarray(y), n_classes)[np.asarray(y)].astype(np.float64)


def _mlp_large(n_in, n_classes, hidden=(256, 256), dropout=0.0, seed=0):
    import torch
    from torch import nn

    state = torch.get_rng_state()
    torch.manual_seed(seed)
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    net = nn.Sequential(*layers)
    torch.set_rng_state(state)
    return net


def build(name: str, *, n_in: int, n_classes: int, seed: int = 0, n_jobs: int = 1, **settings):
    """The model `name` with the given fixed and trial settings; see the module docstring."""
    s = {k: v for k, v in settings.items() if k not in LOOP_SETTINGS}
    if name in VQC_LAYOUT:
        from fqml.quantum.torch_sim import TorchVQC

        if n_in != 8:
            raise ValueError("the VQC takes the top-8 features (one per qubit)")
        entangler, readout = VQC_LAYOUT[name]
        shot_kw = {k: s[k] for k in ("shots", "shot_grad") if s.get(k) is not None}   # Q-1 (INTERFACES 2.4.0)
        return TorchVQC(8, n_classes, s.get("n_layers", 3), seed=seed, entangler=entangler, readout=readout,
                        **shot_kw)
    m = re.fullmatch(r"mlp_p(\d+)", name)
    if m:
        from fqml.models.mlp import MatchedMLP

        return MatchedMLP(n_in, n_classes, int(s.get("target_params", m.group(1))), seed=seed)
    if name == "mlp_large":
        return _mlp_large(n_in, n_classes, tuple(s.get("hidden", (256, 256))), float(s.get("dropout", 0.0)), seed)
    if name == "logreg":
        import torch
        from torch import nn

        state = torch.get_rng_state()
        torch.manual_seed(seed)
        net = nn.Linear(n_in, n_classes)
        torch.set_rng_state(state)
        return net
    if name == "rff":
        from fqml.models.rff import DEFAULT_FEATURES, RFF

        return RFF(n_in, n_classes, int(s.get("n_features", DEFAULT_FEATURES)), int(s.get("max_frequency", 3)), seed)
    if name == "lr":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=float(s.get("C", 1.0)), max_iter=int(s.get("max_iter", 1000)), random_state=seed)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=int(s.get("n_estimators", 300)), max_depth=s.get("max_depth"),
                                      max_features=s.get("max_features", "sqrt"),
                                      min_samples_leaf=int(s.get("min_samples_leaf", 1)), n_jobs=n_jobs,
                                      random_state=seed)
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=int(s.get("n_estimators", 1000)),
                             early_stopping_rounds=s.get("early_stopping_rounds", 50),
                             tree_method=s.get("tree_method", "hist"), max_depth=int(s.get("max_depth", 6)),
                             learning_rate=float(s.get("learning_rate", 0.1)),
                             subsample=float(s.get("subsample", 1.0)),
                             colsample_bytree=float(s.get("colsample_bytree", 1.0)),
                             min_child_weight=float(s.get("min_child_weight", 1.0)),
                             objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                             eval_metric="mlogloss" if n_classes > 2 else "logloss",
                             n_jobs=n_jobs, random_state=seed)
    raise ValueError(f"unknown model {name!r}")


def fit_sklearn(model, name: str, X, y, n_classes: int, X_val=None, y_val=None):
    """Fit an lr / rf / xgb model with the D11 class weights. XGBoost early-stops on the validation block."""
    w = sample_weights(y, n_classes)
    if name == "xgb":
        if X_val is None:
            raise ValueError("XGBoost needs the validation block for early stopping")
        model.fit(X, y, sample_weight=w, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X, y, sample_weight=w)
    return model


def n_params(model) -> int | None:
    """Trainable parameters of a PyTorch model; None for tree and linear scikit-learn models."""
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except AttributeError:
        return None
