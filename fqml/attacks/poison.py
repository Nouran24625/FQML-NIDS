"""Data-poisoning attacks on federated learning (FED-2, docs/ATTACKS.md). Owner: Kareem.

Both Phase 1 attacks are label attacks: a malicious client trains honestly, but on labels it has changed.
  label_flip         untargeted: every row of a malicious client gets the label of the next class
                     (c -> (c + 1) mod C). Aim: lower the global model's quality everywhere.
  class_suppression  targeted label flipping for intrusion detection: a malicious client relabels the rows of
                     ONE attack family (the target) as Benign. Aim: the global model stops flagging that family
                     while macro-F1 and the other classes barely move.

Where they plug in: every partitioner gives each training row to exactly one client (INTERFACES 3.1), so a label
attack is a change of the training-label vector on the malicious clients' rows. `poison_labels` returns that
vector; it is passed to `run_fedavg` in place of `y` (INTERFACES 3.2). The loop itself is unchanged, validation
and test labels are never touched, and class weights stay those of the clean labels (the attacker cannot change
the training configuration).

Malicious clients are fixed for the whole run (a persistent attacker) and drawn with their own seed, independent
of which clients a round samples. For class suppression they are drawn among the clients that hold at least one
target row, since a client with no target rows cannot suppress it.
"""
from __future__ import annotations

import numpy as np

ATTACKS = ("none", "label_flip", "class_suppression")


def check_disjoint(parts: list[np.ndarray], n_rows: int | None = None) -> None:
    """Label attacks need every row to belong to at most one client."""
    allrows = np.concatenate([np.asarray(p, dtype=np.int64) for p in parts]) if parts else np.zeros(0, np.int64)
    if len(np.unique(allrows)) != len(allrows):
        raise ValueError("partition gives a row to two clients; label attacks need disjoint clients")
    if n_rows is not None and len(allrows) and (allrows.min() < 0 or allrows.max() >= n_rows):
        raise ValueError("partition row positions outside the training set")


def holds(parts: list[np.ndarray], row_mask: np.ndarray) -> np.ndarray:
    """[n_clients] bool: client k holds at least one row where `row_mask` is True."""
    row_mask = np.asarray(row_mask, dtype=bool)
    return np.array([bool(row_mask[p].any()) for p in parts])


def choose_malicious(n_clients: int, frac: float, seed: int, eligible: np.ndarray | None = None) -> np.ndarray:
    """[n_clients] bool. round(frac * n_clients) clients, drawn without replacement from the eligible ones
    (all clients when eligible is None). If fewer are eligible, all eligible clients are malicious; the caller
    logs the fraction actually reached (`poison_labels` does)."""
    if not 0 <= frac <= 1:
        raise ValueError("frac must be in [0, 1]")
    pool = np.arange(n_clients) if eligible is None else np.flatnonzero(np.asarray(eligible, dtype=bool))
    k = min(int(round(frac * n_clients)), len(pool))
    mask = np.zeros(n_clients, dtype=bool)
    if k:
        mask[np.random.default_rng(seed).choice(pool, k, replace=False)] = True
    return mask


def flip_map(n_classes: int) -> np.ndarray:
    """c -> (c + 1) mod C: every class goes to a different one (binary: attack <-> benign)."""
    if n_classes < 2:
        raise ValueError("label flipping needs at least 2 classes")
    return (np.arange(n_classes) + 1) % n_classes


def _rows_of(parts: list[np.ndarray], malicious: np.ndarray) -> np.ndarray:
    picked = [np.asarray(parts[k], dtype=np.int64) for k in np.flatnonzero(malicious)]
    return np.concatenate(picked) if picked else np.zeros(0, np.int64)


def label_flip(y, parts, malicious, n_classes: int) -> np.ndarray:
    """Training labels with every row of the malicious clients sent to the next class."""
    y = np.asarray(y)
    out = y.copy()
    rows = _rows_of(parts, malicious)
    out[rows] = flip_map(n_classes)[y[rows]]
    return out


def class_suppression(y, parts, malicious, target_mask, benign: int = 0) -> np.ndarray:
    """Training labels with the target rows of the malicious clients relabelled `benign`.

    `target_mask` [n_rows] marks the target family's rows. 8-class task: `y == family index`; binary task:
    the family's rows from the native label (the binary label alone cannot tell which attack a row is)."""
    y = np.asarray(y)
    target_mask = np.asarray(target_mask, dtype=bool)
    if target_mask.shape != y.shape:
        raise ValueError("target_mask must have one entry per training row")
    out = y.copy()
    rows = _rows_of(parts, malicious)
    out[rows[target_mask[rows]]] = benign
    return out


def poison_labels(y, parts, attack: str, *, n_classes: int, frac: float = 0.2, seed: int = 0,
                  target_mask=None, benign: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """(poisoned training labels, malicious-client mask, info for the run log).

    attack: "none" | "label_flip" | "class_suppression" (the last needs `target_mask`).
    info: attack, frac asked, frac reached, malicious client ids, rows changed, eligible clients."""
    if attack not in ATTACKS:
        raise ValueError(f"unknown attack {attack!r}; one of {ATTACKS}")
    y = np.asarray(y)
    check_disjoint(parts, len(y))
    n = len(parts)
    eligible = None
    if attack == "none":
        mal = np.zeros(n, dtype=bool)
        out = y.copy()
    elif attack == "label_flip":
        mal = choose_malicious(n, frac, seed)
        out = label_flip(y, parts, mal, n_classes)
    else:
        if target_mask is None:
            raise ValueError("class_suppression needs target_mask")
        eligible = holds(parts, target_mask)
        mal = choose_malicious(n, frac, seed, eligible=eligible)
        out = class_suppression(y, parts, mal, target_mask, benign=benign)
    info = {"attack": attack, "frac": float(frac), "frac_reached": float(mal.mean()) if n else 0.0,
            "malicious": np.flatnonzero(mal).tolist(), "rows_changed": int((out != y).sum()),
            "eligible_clients": None if eligible is None else int(eligible.sum()), "seed": int(seed)}
    return out, mal, info
