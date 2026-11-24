"""Client partitioners (Plan §6, 'Partition'). Owner: Kareem.

Splits the row indices of a training set among N clients:
  iid_partition        every client gets a random, equal-size share (same class mix everywhere)
  dirichlet_partition  label skew: for each class, its rows are divided among clients in proportions
                       drawn from Dirichlet(alpha). Small alpha (0.1) = very uneven, 1.0 = mild.
  type_partition       one attack type per client, benign rows spread over all clients: the most extreme
                       skew. Not a realistic setting; it bounds how bad label skew can get (FED-3).
  sites_partition      the sites split (FED-3): each client is a site holding benign traffic plus a subset of
                       the attack types; every type is held by at least one site. A site never sees the other
                       types, which is what the local-only baseline (FED-7) and "recall on attack types a
                       client never saw" measure.
All are fully determined by the seed, so every model and every teammate gets the same partition. Every
partitioner gives each row to exactly one client (label attacks rely on this, fqml.attacks.poison).
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


BENIGN = "BenignTraffic"


def _attack_types(native: np.ndarray, benign: str) -> list[str]:
    types = sorted(set(native.tolist()) - {benign})
    if not types:
        raise ValueError("no attack types in native_label")
    return types


def _split_even(rows: np.ndarray, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    return np.array_split(rng.permutation(rows), n)


def type_partition(native_label, seed: int, *, benign: str = BENIGN) -> list[np.ndarray]:
    """One client per attack type (sorted by name): client i holds every row of type i plus 1/T of the benign
    rows, T = number of attack types. INTERFACES 3.1 (FED-3)."""
    native = np.asarray(native_label, dtype=object)
    types = _attack_types(native, benign)
    rng = np.random.default_rng(seed)
    ben = _split_even(np.flatnonzero(native == benign), len(types), rng)
    return [np.sort(np.concatenate([np.flatnonzero(native == t), ben[i]])).astype(np.int64)
            for i, t in enumerate(types)]


def site_assignment(units: list[str], n_sites: int, per_site: int, seed: int) -> list[list[str]]:
    """Which units (attack types or groups of them) each site holds: exactly `per_site` each, every unit at least
    once. Units are first dealt round-robin in a seeded random order (so every unit is covered), then each site
    is filled with units it lacks, drawn at random."""
    n = len(units)
    if not 1 <= per_site <= n:
        raise ValueError(f"per_site must be in 1..{n}")
    if -(-n // n_sites) > per_site:
        raise ValueError(f"{n_sites} sites x {per_site} cannot cover {n} units; raise n_sites or per_site")
    rng = np.random.default_rng(seed)
    order = [units[i] for i in rng.permutation(n)]
    held: list[list[str]] = [[] for _ in range(n_sites)]
    for i, u in enumerate(order):
        held[i % n_sites].append(u)
    for s in range(n_sites):
        missing = [u for u in units if u not in held[s]]
        held[s] += [missing[i] for i in rng.choice(len(missing), per_site - len(held[s]), replace=False)]
    return [sorted(h) for h in held]


def sites_partition(native_label, n_sites: int, types_per_site: int, seed: int, *, benign: str = BENIGN,
                    groups: dict[str, str] | None = None) -> tuple[list[np.ndarray], list[list[str]]]:
    """The sites split (INTERFACES 3.1, FED-3). Returns (partition, attack types held by each site).

    Every site gets an equal random share of the benign rows. Each attack type's rows are shuffled and split
    evenly among the sites that hold it.
    `groups` (optional, {attack type: group}, e.g. the family of each type): sites then hold whole groups and
    `types_per_site` counts groups. Holding whole families makes a site miss entire families, the case where a
    lone site is expected to fail (Checkpoint 1: types unseen inside a known family are still detected)."""
    native = np.asarray(native_label, dtype=object)
    types = _attack_types(native, benign)
    if groups is None:
        unit_types = {t: [t] for t in types}
    else:
        missing = sorted(set(types) - set(groups))
        if missing:
            raise ValueError(f"groups has no entry for {missing[:5]}")
        unit_types = {}
        for t in types:
            unit_types.setdefault(groups[t], []).append(t)
    site_units = site_assignment(sorted(unit_types), n_sites, types_per_site, seed)
    site_types = [sorted(t for u in units for t in unit_types[u]) for units in site_units]
    rng = np.random.default_rng([seed, 1])
    parts: list[list[np.ndarray]] = [[b] for b in _split_even(np.flatnonzero(native == benign), n_sites, rng)]
    for t in types:
        holders = [s for s in range(n_sites) if t in site_types[s]]
        for s, chunk in zip(holders, _split_even(np.flatnonzero(native == t), len(holders), rng)):
            parts[s].append(chunk)
    return [np.sort(np.concatenate(p)).astype(np.int64) for p in parts], site_types


def save_partition(parts: list[np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"client_{k:03d}": p for k, p in enumerate(parts)})


def load_partition(path: str | Path) -> list[np.ndarray]:
    with np.load(path) as z:
        return [z[k] for k in sorted(z.files)]
