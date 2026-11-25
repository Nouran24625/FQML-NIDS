"""PRIV-5, federated side: the one-run canary audit for a FedAvg detector. Owner: Kareem. Reviewer: Nouran.

docs/PREREG_MIA.md §5 (amended 25 Nov) uses the PRIV-3 audit (fqml.privacy.audit, Nouran) for every arm. That
module trains centrally (dp_fit); this wrapper trains with fqml.fl.batched.run_fedavg instead and reuses the
audit's canaries, scores, guessing rule and bound unchanged, so central and federated arms are audited the same way.

  - m canaries (fqml.privacy.audit.make_canaries), each included with probability 1/2, all placed on ONE client
    (`canary_client`): in a federation a record lives at one site.
  - Two attackers (docs/THREAT_MODEL.md §2, PREREG_MIA §2):
      outsider        scores the canaries with the final global model;
      curious server  scores them with the canary client's next update: one more round of local training from the
                      final global model, exactly what that client sends the server (fqml.fl.batched._local_sequential).
  - Guesses: top k called "in", bottom k "out", k = m / 10; the bound is fqml.privacy.audit.epsilon_lower_bound at
    95% confidence. Without DP this is the only privacy number there is (PREREG_MIA §5); a federated DP arm comes
    with client-level DP, after Gate 1.

The pilot runs after Gate 1 (PREREG_MIA). This module is the code it needs, tested on synthetic data now.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from fqml.fl.batched import _local_sequential, client_batches, run_fedavg
from fqml.privacy.audit import canary_scores, epsilon_lower_bound, guess_and_count, make_canaries


def _with_params(model, params):
    m = copy.deepcopy(model)
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(params[n])
    return m


def run_fed_audit(model, X, y, parts, n_classes: int, *, canary_client: int = 0, m: int = 1000,
                  k: int | None = None, beta: float = 0.05, seed: int = 0, rounds: int, clients_per_round: int,
                  local_steps: int = 50, batch_size: int = 64, lr: float = 0.01, class_weights=None, **fed_kwargs):
    """Train FedAvg with m canaries (half of them included) on client `canary_client`, then audit the outsider and
    the curious server. Returns a plain dict (JSON-ready)."""
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.int64)
    Xc, yc, inc = make_canaries(m, X.shape[1], n_classes, seed)
    n0 = len(X)
    Xt = np.concatenate([X, Xc[inc]])
    yt = np.concatenate([y, yc[inc]])
    parts = [np.asarray(p) for p in parts]
    parts[canary_client] = np.concatenate([parts[canary_client], n0 + np.arange(int(inc.sum()))])

    res = run_fedavg(model, Xt, yt, parts, rounds=rounds, clients_per_round=clients_per_round,
                     local_steps=local_steps, batch_size=batch_size, lr=lr, class_weights=class_weights, seed=seed,
                     **fed_kwargs)
    k = k or max(1, m // 10)
    out = {"canaries": m, "canaries_included": int(inc.sum()), "canary_client": canary_client,
           "client_rows": int(len(parts[canary_client])), "k": k, "beta": beta, "rounds": rounds, "lr": lr}

    # outsider: the released global model
    glob = _with_params(model, res.params)
    correct, guesses = guess_and_count(canary_scores(glob, Xc, yc), inc, k)
    out["outsider"] = {"correct": correct, "guesses": guesses,
                       "empirical_epsilon_lower_bound": epsilon_lower_bound(correct, guesses, beta=beta)}

    # curious server: the canary client's update in the next round (same local step as every round)
    rng = np.random.default_rng(seed + 10_000)
    batches = client_batches(parts[canary_client], local_steps, batch_size, rng)[None]
    Xt_t, yt_t = torch.as_tensor(Xt), torch.as_tensor(yt)
    w = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
    upd = _local_sequential(model, res.params, Xt_t, yt_t, torch.as_tensor(batches), lr, w)
    local = _with_params(model, {n: v[0] for n, v in upd.items()})
    correct, guesses = guess_and_count(canary_scores(local, Xc, yc), inc, k)
    out["curious_server"] = {"correct": correct, "guesses": guesses,
                             "empirical_epsilon_lower_bound": epsilon_lower_bound(correct, guesses, beta=beta)}
    return out
