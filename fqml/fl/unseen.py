"""Recall on attack types a client never saw (FED-7). Owner: Kareem.

The metric that gives federation a measured reason to exist (docs/THREAT_MODEL.md §4): score a model on the
test rows whose attack type the client did NOT hold in training. Applied to
  - each local-only model (mode="local"): can a site that trained alone catch what it never saw?
  - the FedAvg global model and the centralised model, on every client's unseen types: same rows, so the three
    are compared pairwise per client.
Pure numpy: the caller passes predictions (p = argmax of probabilities, attack_score = 1 - P(benign)).
Thresholds come from the validation block (EVAL-1 rule 1); `threshold=None` reports only the default decision.
"""
from __future__ import annotations

import numpy as np

BENIGN = "BenignTraffic"


def held_types(native_train, parts, benign: str = BENIGN) -> list[set[str]]:
    """Attack types present in each client's training rows."""
    native_train = np.asarray(native_train, dtype=object)
    return [set(native_train[p].tolist()) - {benign} for p in parts]


def unseen_scores(y, p, native, held: set[str], *, attack_score=None, threshold: float | None = None,
                  benign: str = BENIGN, benign_class: int = 0) -> dict:
    """Scores on the rows (of a test or validation block) whose attack type is not in `held`.

    detection_default: share with p != benign (the model's own decision); detection_at_threshold: share with
    attack_score > threshold; family_recall: share with p == y. Each is given over rows and averaged over types
    (every type counts once, EVAL-1 2.4). Values are NaN when there is no unseen row."""
    y, p = np.asarray(y), np.asarray(p)
    native = np.asarray(native, dtype=object)
    types = sorted(set(native.tolist()) - {benign} - set(held))
    mask = np.isin(native, types) if types else np.zeros(len(native), bool)
    out = {"unseen_types": types, "rows": int(mask.sum())}
    keys = ["detection_default", "family_recall"]
    hit = {"detection_default": p != benign_class, "family_recall": p == y}
    if threshold is not None:
        if attack_score is None:
            raise ValueError("threshold given without attack_score")
        hit["detection_at_threshold"] = np.asarray(attack_score) > threshold
        keys.append("detection_at_threshold")
        out["threshold"] = float(threshold)
    per_type = {}
    for t in types:
        m = native == t
        per_type[t] = {"rows": int(m.sum()), **{k: float(hit[k][m].mean()) for k in keys}}
    out["per_type"] = per_type
    for k in keys:
        out[f"{k}_rows"] = float(hit[k][mask].mean()) if mask.any() else float("nan")
        out[f"{k}_types"] = float(np.mean([v[k] for v in per_type.values()])) if per_type else float("nan")
    return out


def per_client_unseen(y, native, held: list[set[str]], preds, *, scores=None, thresholds=None,
                      benign: str = BENIGN) -> list[dict]:
    """unseen_scores for every client. `preds[k]` (and `scores[k]`, `thresholds[k]`) are client k's model's
    outputs (local-only), or the same global model's outputs repeated for every k (FedAvg, centralised)."""
    out = []
    for k, h in enumerate(held):
        out.append(unseen_scores(y, preds[k], native, h,
                                 attack_score=None if scores is None else scores[k],
                                 threshold=None if thresholds is None else thresholds[k], benign=benign))
    return out


def summary(per_client: list[dict], key: str = "detection_default_types") -> dict:
    """Mean, min and max over clients of one score (clients with no unseen rows skipped)."""
    v = np.array([c[key] for c in per_client], dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return {"clients": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"clients": int(len(v)), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
