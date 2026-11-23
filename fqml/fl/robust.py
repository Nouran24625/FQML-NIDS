"""Robust aggregators (FED-6, docs/ATTACKS.md section 5). Owner: Kareem.

Every aggregator has the signature of `fqml.fl.batched.fedavg`:

    aggregate(stacked, n_rows) -> global params
    stacked: dict[name, tensor[K, ...]]   one row per client sampled this round
    n_rows:  tensor[K]                    each client's number of training rows

and is passed to `run_fedavg(..., aggregator=...)`. All of them work coordinate by coordinate:

  median        coordinate-wise median over the K clients (Yin et al., 2018). With an even K it is the mean of
                the two middle values, so it does not depend on the order of the clients.
  trimmed mean  coordinate-wise: drop the floor(beta * K) largest and smallest values, average the rest
                (Yin et al., 2018). beta = 0.2 per side is the project default (docs/ATTACKS.md).

Both ignore `n_rows`: they are defined on the updates alone, so a client cannot gain weight by claiming many rows.
That also means that, with no attacker, a large honest client counts as much as a small one. This is one of
the costs the zero-attacker control measures (experiments/fed6_zero_attacker.py).
"""
from __future__ import annotations

from functools import partial

import torch

from fqml.fl.batched import fedavg

DEFAULT_BETA = 0.2


def coordinate_median(stacked: dict[str, torch.Tensor], n_rows: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Coordinate-wise median over the client dimension (dim 0); `n_rows` is ignored."""
    out = {}
    for n, v in stacked.items():
        K = v.shape[0]
        s = v.sort(dim=0).values
        out[n] = s[K // 2] if K % 2 else (s[K // 2 - 1] + s[K // 2]) / 2
    return out


def trimmed_mean(stacked: dict[str, torch.Tensor], n_rows: torch.Tensor | None = None, *,
                 beta: float = DEFAULT_BETA) -> dict[str, torch.Tensor]:
    """Coordinate-wise trimmed mean: drop floor(beta * K) values at each end, average the rest. `n_rows` is
    ignored. beta = 0 is the plain (unweighted) mean."""
    if not 0 <= beta < 0.5:
        raise ValueError("beta must be in [0, 0.5)")
    out = {}
    for n, v in stacked.items():
        K = v.shape[0]
        t = int(beta * K)                                   # beta < 0.5, so at least one client is left
        out[n] = v.sort(dim=0).values[t:K - t].mean(dim=0)
    return out


AGGREGATORS = {
    "fedavg": fedavg,
    "median": coordinate_median,
    "trimmed_mean": partial(trimmed_mean, beta=DEFAULT_BETA),
}


def get_aggregator(name: str):
    """'fedavg', 'median' or 'trimmed_mean' (beta = 0.2); 'trimmed_mean0.1' sets beta explicitly."""
    if name in AGGREGATORS:
        return AGGREGATORS[name]
    if name.startswith("trimmed_mean"):
        return partial(trimmed_mean, beta=float(name[len("trimmed_mean"):]))
    raise ValueError(f"unknown aggregator {name!r}; known: {sorted(AGGREGATORS)}")
