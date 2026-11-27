"""Shared metrics module (EVAL-1). Builds it: Sheghaf. Spec and review: Nouran (docs/specs/EVAL-1_metrics.md).

Turns a model's outputs into the scores every E0 table uses. Inputs are plain arrays; no function here sees a model:

    y             true family of each row (int, 0 = Benign)
    p             predicted family of each row (the argmax of the model's probabilities)
    attack_score  1 - P(Benign) for each row, a float in [0, 1]: the higher, the surer the model is of "attack"
    w             optional row weights (1 / sample_rate); by default every row weighs 1

Rules (spec section 1):
    - Alarm thresholds are chosen on the VALIDATION block (`threshold_at_fpr`) and applied to the TEST block.
    - Undefined values are NaN, never 0. A class or attack type with no rows is NaN and left out of averages.
    - Every result is plain JSON with "metrics_version": 1.

The runner (EXP-0) makes two calls:

    t = fit_thresholds(attack_score_val, y_val)                # {"fpr_1pct": ..., "fpr_0.1pct": ...}
    result = evaluate(y_test, p_test, attack_score_test, families=..., row_ids=..., label_ranges=...,
                      thresholds=t, train_dup_groups=..., dup_group=..., conflict=..., unseen_types=...)

Differences from fqml/eval/checkpoint1.py, the reference implementation:
    - macro-F1 skips classes with no true rows (Checkpoint 1 counted them as 0) and lists them in "skipped".
    - the 7-class task (c7m) comes from the label-map variant, not from a hand-written merge.
    - thresholds are passed in (from validation) instead of being fitted on the rows being scored.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from fqml.data.folds import twin_mask
from fqml.data.label_maps import family_index, load_label_map

METRICS_VERSION = 1
FPRS = {"fpr_1pct": 0.01, "fpr_0.1pct": 0.001}    # the two false-alarm budgets of D11: 1% and 0.1%
NAN = float("nan")


# ---- small helpers -------------------------------------------------------------------------------
def _weights(w, n: int) -> np.ndarray:
    if w is None:
        return np.ones(n)
    w = np.asarray(w, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"w has shape {w.shape}, expected ({n},)")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("row weights must be finite and >= 0")
    return w


def _ratio(num, den) -> float:
    """num / den, or NaN when den is 0 (rule 6: undefined is NaN, never 0)."""
    return float(num) / float(den) if den > 0 else NAN


def _mean(values) -> float:
    """Mean of the values that are not NaN; NaN if none are left."""
    kept = [v for v in values if not math.isnan(v)]
    return float(np.mean(kept)) if kept else NAN


def _number(x):
    """A weighted count as JSON: int when it is a whole number (always, without weights), else float."""
    x = float(x)
    return int(x) if x.is_integer() else x


def _same_length(**arrays) -> int:
    lengths = {k: len(v) for k, v in arrays.items() if v is not None}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"arrays of different lengths: {lengths}")
    return next(iter(lengths.values()), 0)


# ---- the 7-class task (c7m) from the label map ---------------------------------------------------
def c7m_index(families, dataset: str = "ciciot2023") -> tuple[np.ndarray, list[str]]:
    """(m, c7m families) with m[i] = the c7m class of 8-class family i, read from the label map (rule 4)."""
    fam8, map8 = load_label_map(dataset)
    if list(families) != list(fam8):
        raise ValueError(f"families {list(families)} differ from the {dataset} label map {fam8}")
    fam7, _ = load_label_map(dataset, "c7m")
    natives = sorted(map8)
    m = np.full(len(fam8), -1, dtype=np.int64)
    for a, b in zip(family_index(natives, dataset), family_index(natives, dataset, "c7m")):
        if m[a] not in (-1, b):
            raise ValueError(f"label map: family {fam8[a]} maps to two c7m classes")
        m[a] = b
    if (m < 0).any():
        raise ValueError("label map: a family has no native label")
    return m, list(fam7)


def to_c7m(labels, families, dataset: str = "ciciot2023") -> np.ndarray:
    """8-class family indices -> c7m class indices."""
    m, _ = c7m_index(families, dataset)
    return m[np.asarray(labels, dtype=np.int64)]


# ---- 2.1 multi-class ------------------------------------------------------------------------------
def multiclass(y, p, families, w=None) -> dict:
    """Accuracy, macro-F1, per-class precision / recall / F1 / support, confusion matrix (rows = true class).

    Macro-F1 averages over the classes that have true rows; the others are listed in "skipped". Support and the
    confusion matrix are weighted counts (plain counts without weights)."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.int64)
    n = _same_length(y=y, p=p)
    k = len(families)
    if n and (min(y.min(), p.min()) < 0 or max(y.max(), p.max()) >= k):
        raise ValueError(f"class indices must be in 0..{k - 1}")
    w = _weights(w, n)
    cm = np.bincount(y * k + p, weights=w, minlength=k * k).reshape(k, k)   # fast on millions of test rows
    support, predicted, tp = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    per_class = {}
    for i, fam in enumerate(families):
        per_class[fam] = {
            "precision": _ratio(tp[i], predicted[i]),
            "recall": _ratio(tp[i], support[i]),
            "f1": _ratio(2 * tp[i], support[i] + predicted[i]) if support[i] > 0 else NAN,
            "support": _number(support[i]),
        }
    present = [fam for i, fam in enumerate(families) if support[i] > 0]
    return {
        "rows": n,
        "accuracy": _ratio(tp.sum(), cm.sum()),
        "macro_f1": _mean([per_class[f]["f1"] for f in present]),
        "skipped": [f for f in families if f not in present],
        "per_class": per_class,
        "confusion": [[_number(v) for v in row] for row in cm],
        "families": list(families),
    }


# ---- 2.2 thresholds -------------------------------------------------------------------------------
def threshold_at_fpr(attack_score_val, y_val, fpr: float) -> float:
    """The alarm threshold at which at most `fpr` of the validation BENIGN rows score above it.

    Same rule as Checkpoint 1: np.quantile(benign scores, 1 - fpr, method="higher"); an alarm is score > threshold.
    NaN when the validation rows hold no benign row (then every score that uses it is NaN too)."""
    if not 0 < fpr < 1:
        raise ValueError(f"fpr must be between 0 and 1, got {fpr}")
    s = np.asarray(attack_score_val, dtype=float)
    y = np.asarray(y_val)
    _same_length(attack_score_val=s, y_val=y)
    benign = s[y == 0]
    if not len(benign):
        return NAN
    return float(np.quantile(benign, 1.0 - fpr, method="higher"))


def fit_thresholds(attack_score_val, y_val, fprs: dict | None = None) -> dict:
    """{"fpr_1pct": t, "fpr_0.1pct": t} from the validation block, ready for `binary`, `per_type` and `evaluate`."""
    return {name: threshold_at_fpr(attack_score_val, y_val, f) for name, f in (fprs or FPRS).items()}


# ---- 2.3 binary (attack vs benign) ----------------------------------------------------------------
def binary(y, attack_score, thresholds, w=None, *, p=None) -> dict:
    """Attack = y != 0.

    For each named threshold: attack recall, the false-positive rate actually reached, precision. Plus ROC-AUC,
    benign PR-AUC (benign as the positive class), attack PR-AUC and, when `p` is given, F1 / recall / precision /
    FPR at the default decision p != 0 (reported, never used to judge: D11)."""
    y = np.asarray(y)
    s = np.asarray(attack_score, dtype=float)
    n = _same_length(y=y, attack_score=s, p=p)
    w = _weights(w, n)
    att = y != 0
    ben = ~att
    w_att, w_ben = w[att].sum(), w[ben].sum()
    both = w_att > 0 and w_ben > 0
    out = {"rows": n, "attack_rows": int(att.sum()), "benign_rows": int(ben.sum()),
           "roc_auc": float(roc_auc_score(att, s, sample_weight=w)) if both else NAN,
           "benign_pr_auc": float(average_precision_score(ben, -s, sample_weight=w)) if both else NAN,
           "attack_pr_auc": float(average_precision_score(att, s, sample_weight=w)) if both else NAN}
    for name, t in thresholds.items():
        t = float(t)
        if math.isnan(t):
            out[name] = {"threshold": NAN, "attack_recall": NAN, "fpr_reached": NAN, "precision": NAN}
            continue
        alarm = s > t
        caught = w[att & alarm].sum()
        out[name] = {"threshold": t,
                     "attack_recall": _ratio(caught, w_att),
                     "fpr_reached": _ratio(w[ben & alarm].sum(), w_ben),
                     "precision": _ratio(caught, w[alarm].sum())}
    if p is not None:
        said_attack = np.asarray(p) != 0
        tp, fp, fn = w[att & said_attack].sum(), w[ben & said_attack].sum(), w[att & ~said_attack].sum()
        out["default"] = {"f1": _ratio(2 * tp, 2 * tp + fp + fn), "attack_recall": _ratio(tp, w_att),
                          "precision": _ratio(tp, tp + fp), "fpr": _ratio(fp, w_ben)}
    return out


# ---- 2.4 per attack type --------------------------------------------------------------------------
def _type_masks(row_ids, label_ranges):
    """(native label, mask of its rows) for every type with rows, in row order. Types are row_id ranges."""
    for lab, (a, b) in sorted(label_ranges.items(), key=lambda kv: kv[1][0]):
        m = (row_ids >= a) & (row_ids < b)
        if m.any():
            yield lab, m


def per_type(row_ids, y, p, attack_score, label_ranges, thresholds, *, families=None, w=None) -> dict:
    """For every native type present: rows, its family, family recall, the family it is most often mistaken for
    (and that share of its rows), and detection (share with score > threshold) at each threshold.

    Plus macro averages over ATTACK types (benign left out), so a 225-row Web type counts as much as a
    7-million-row flood: mean family recall, and mean detection at each threshold."""
    row_ids = np.asarray(row_ids)
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.int64)
    s = np.asarray(attack_score, dtype=float)
    n = _same_length(row_ids=row_ids, y=y, p=p, attack_score=s)
    w = _weights(w, n)
    k = len(families) if families is not None else int(max(y.max(initial=0), p.max(initial=0))) + 1
    name = (lambda i: families[i]) if families is not None else int
    types, covered = {}, np.zeros(n, bool)
    for lab, m in _type_masks(row_ids, label_ranges):
        covered |= m
        yt, pt, st, wt = y[m], p[m], s[m], w[m]
        fam = int(np.bincount(yt).argmax())               # every row of one type has the same family
        total = wt.sum()
        right = pt == yt
        wrong = np.bincount(pt[~right], weights=wt[~right], minlength=k)
        entry = {"rows": int(m.sum()), "family": name(fam), "is_attack": fam != 0,
                 "family_recall": _ratio(wt[right].sum(), total),
                 "most_confused_with": name(int(wrong.argmax())) if (~right).any() else None,
                 "share_to_it": _ratio(wrong.max(), total) if (~right).any() else 0.0}
        for tname, t in thresholds.items():
            entry[f"detection_{tname}"] = NAN if math.isnan(float(t)) else _ratio(wt[st > float(t)].sum(), total)
        types[lab] = entry
    attack = [e for e in types.values() if e["is_attack"]]
    out = {"types": types, "attack_types": len(attack), "rows_without_type": int((~covered).sum()),
           "mean_family_recall": _mean([e["family_recall"] for e in attack])}
    for tname in thresholds:
        out[f"mean_detection_{tname}"] = _mean([e[f"detection_{tname}"] for e in attack])
    return out


# ---- 2.5 protocol B: attack types the model never saw ---------------------------------------------
def unseen(row_ids, y, p, attack_score, label_ranges, unseen_types, thresholds, families, *, w=None,
           dataset: str = "ciciot2023") -> dict:
    """On the types the model never saw in training: rows, detection at each threshold (over rows and averaged
    over types), detection at the default decision, and the right family (8-class and c7m)."""
    row_ids = np.asarray(row_ids)
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.int64)
    s = np.asarray(attack_score, dtype=float)
    n = _same_length(row_ids=row_ids, y=y, p=p, attack_score=s)
    w = _weights(w, n)
    unseen_types = list(unseen_types)
    missing = sorted(set(unseen_types) - set(label_ranges))
    if missing:
        raise ValueError(f"unseen types not in label_ranges: {missing}")
    ranges = {lab: label_ranges[lab] for lab in unseen_types}
    mask = np.zeros(n, bool)
    for _, m in _type_masks(row_ids, ranges):
        mask |= m
    m7, _ = c7m_index(families, dataset)
    total = w[mask].sum()
    out = {"types": unseen_types, "rows": int(mask.sum())}
    for tname, t in thresholds.items():
        out[f"detected_{tname}"] = NAN if math.isnan(float(t)) else _ratio(w[mask & (s > float(t))].sum(), total)
    out["detected_at_default"] = _ratio(w[mask & (p != 0)].sum(), total)
    out["family_correct"] = _ratio(w[mask & (p == y)].sum(), total)
    out["family_correct_c7m"] = _ratio(w[mask & (m7[p] == m7[y])].sum(), total)
    pt = per_type(row_ids, y, p, s, ranges, thresholds, families=families, w=w)
    for tname in thresholds:
        out[f"mean_detected_{tname}"] = pt[f"mean_detection_{tname}"]
    out["mean_family_correct"] = pt["mean_family_recall"]
    c7m_by_type = []
    for _, m in _type_masks(row_ids, ranges):
        c7m_by_type.append(_ratio(w[m & (m7[p] == m7[y])].sum(), w[m].sum()))
    out["mean_family_correct_c7m"] = _mean(c7m_by_type)
    out["per_type"] = pt["types"]
    return out


# ---- 2.6 the one call the runner makes ------------------------------------------------------------
def _subset_scores(y, p, s, w, families, m7, fam7, thresholds) -> dict:
    c8 = multiclass(y, p, families, w)
    out = {"rows": int(len(y)),
           "c8": c8,
           "c7m": multiclass(m7[y], m7[p], fam7, w),
           "binary": binary(y, s, thresholds, w, p=p)}
    if "DDoS" in families and "DoS" in families:          # the confusion D17 is about
        i_ddos, i_dos = families.index("DDoS"), families.index("DoS")
        cm = np.asarray(c8["confusion"], dtype=float)
        out["dos_predicted_ddos"] = _ratio(cm[i_dos, i_ddos], cm[i_dos].sum())
        out["ddos_predicted_dos"] = _ratio(cm[i_ddos, i_dos], cm[i_ddos].sum())
    return out


def evaluate(y, p, attack_score, *, families, row_ids, label_ranges, thresholds, train_dup_groups, dup_group,
             conflict, unseen_types=None, w=None, dataset: str = "ciciot2023") -> dict:
    """All E0 scores for one model on one test block.

    thresholds        from `fit_thresholds` on the VALIDATION block
    train_dup_groups  `dup_group` of the rows the model trained on; `dup_group` and `conflict` are the test rows'
                      (D8). Pass None for both dup-group arguments to leave out the "no_twin" subset, and None for
                      `conflict` to leave out "no_conflict" (datasets without these columns).
    unseen_types      protocol B only: the attack types held out of training

    Returns {"metrics_version": 1, "rows", "families", "thresholds", "all", "no_twin", "no_conflict",
    "per_type", "unseen" (protocol B only)}; each subset holds 8-class and c7m `multiclass` and `binary`."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.int64)
    s = np.asarray(attack_score, dtype=float)
    row_ids = np.asarray(row_ids)
    n = _same_length(y=y, p=p, attack_score=s, row_ids=row_ids, dup_group=dup_group, conflict=conflict)
    w = _weights(w, n)
    families = list(families)
    m7, fam7 = c7m_index(families, dataset)
    subsets = {"all": np.ones(n, bool)}
    if dup_group is not None and train_dup_groups is not None:
        subsets["no_twin"] = ~twin_mask(dup_group, train_dup_groups)
    if conflict is not None:
        subsets["no_conflict"] = np.asarray(conflict) == 0
    out = {"metrics_version": METRICS_VERSION, "rows": n, "families": families,
           "thresholds": {k: float(v) for k, v in thresholds.items()}}
    for name, m in subsets.items():
        out[name] = _subset_scores(y[m], p[m], s[m], w[m], families, m7, fam7, thresholds)
    out["per_type"] = per_type(row_ids, y, p, s, label_ranges, thresholds, families=families, w=w)
    if unseen_types:
        out["unseen"] = unseen(row_ids, y, p, s, label_ranges, unseen_types, thresholds, families, w=w,
                               dataset=dataset)
    return out
