"""Client partitioners (Plan §6, 'Partition'). Owner: Kareem.

Splits the row indices of a training set among N clients:
  iid_partition        every client gets a random, equal-size share (same class mix everywhere)
  dirichlet_partition  label skew: for each class, its rows are divided among clients in proportions
                       drawn from Dirichlet(alpha). Small alpha (0.1) = very uneven, 1.0 = mild.
Both are fully determined by the seed, so every model and every teammate gets the same partition.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def iid_partition(n_rows: int, n_clients: int, seed: int) -> list[np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n_rows)
    return [np.sort(p) for p in np.array_split(perm, n_clients)]


def dirichlet_partition(y, n_clients: int, alpha: float, seed: int, min_size: int = 10,
                        max_tries: int = 100) -> list[np.ndarray]:
    """Label-skew partition. Retries (with the same rng stream) until every client has >= min_size rows."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    for _ in range(max_tries):
        parts: list[list[int]] = [[] for _ in range(n_clients)]
        for c in classes:
            ids = np.flatnonzero(y == c)
            rng.shuffle(ids)
            share = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(share) * len(ids)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(ids, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.asarray(p, dtype=np.int64)) for p in parts]
    raise RuntimeError(f"could not give every client >= {min_size} rows; lower min_size or raise alpha")


def save_partition(parts: list[np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"client_{k:03d}": p for k, p in enumerate(parts)})


def load_partition(path: str | Path) -> list[np.ndarray]:
    with np.load(path) as z:
        return [z[k] for k in sorted(z.files)]
