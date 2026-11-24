"""Batched FedAvg simulation (decision D3, proposed). Owner: Kareem.

Idea: instead of training the K sampled clients of a round one after another (as Flower does), keep
their K copies of the model weights STACKED in one tensor per parameter ([K, ...]) and train them all
at once. torch.func.vmap runs the model once per client copy in a single batched call, so one
local step processes K x 64 samples together - this is what makes the GPU fast (D1 measurements).

It is mathematically the same as training each client separately:
  - the loss is the SUM of each client's own loss, so client k's weights only get client k's gradient;
  - Adam works element by element, so one Adam over the stacked tensors = K independent Adams.
tests/test_batched_fedavg.py checks this against the one-by-one reference (mode="sequential").

Each round: pick clients -> each client trains `local_steps` mini-batches from its own rows
(fresh Adam each round) -> FedAvg = average of client weights, weighted by client data size.
Works for any nn.Module whose forward is vmap-safe (TorchVQC, MatchedMLP).

mode="local" (FED-7) is the local-only baseline: every client trains alone on its own rows, all clients at once in
the same batched call, with no aggregation. Each client takes `local_steps` steps in every round, so over the run
it gets rounds x local_steps steps, as many as the global model gets in FedAvg (an average FedAvg client trains
only in the rounds it is sampled). This gives the baseline every chance; the comparison metric is each client's
recall on the attack types it never held (fqml.fl.unseen).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.func import functional_call, vmap


@dataclass
class FedResult:
    params: dict[str, torch.Tensor] | None             # final global weights (None in mode="local")
    history: list[dict] = field(default_factory=list)  # one entry per round
    client_params: dict[str, torch.Tensor] | None = None  # mode="local": one model per client, [N, ...] per name
    final_val: dict | list | None = None   # full validation scores at the end (FED-1); a list (one per client) in "local"


def client_batches(rows: np.ndarray, steps: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices for one client's local work in one round: [steps, batch_size].
    Shuffled; if the client has fewer than steps*batch_size rows its shuffled rows repeat in order."""
    need = steps * batch_size
    perm = rng.permutation(rows)
    return np.resize(perm, need).reshape(steps, batch_size)


def client_losses(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    """Per-client cross-entropy, [K]. With class weights w it equals nn.CrossEntropyLoss(weight=w)."""
    K, B, C = logits.shape
    loss = nn.functional.cross_entropy(logits.reshape(K * B, C), y.reshape(-1), reduction="none").reshape(K, B)
    if w is None:
        return loss.mean(dim=1)
    wy = w[y]
    return (wy * loss).sum(dim=1) / wy.sum(dim=1)


def _local_batched(model, gparams, X, y, batches, lr, w):
    K = batches.shape[0]
    stacked = {n: v.detach().unsqueeze(0).repeat(K, *([1] * v.dim())).clone().requires_grad_(True)
               for n, v in gparams.items()}
    opt = torch.optim.Adam(list(stacked.values()), lr=lr)
    forward = vmap(lambda p, xb: functional_call(model, p, (xb,)))
    for s in range(batches.shape[1]):
        idx = batches[:, s]                                  # [K, B]
        loss = client_losses(forward(stacked, X[idx]), y[idx], w).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {n: v.detach() for n, v in stacked.items()}


def _local_sequential(model, gparams, X, y, batches, lr, w):
    """Reference: the same local training, one client at a time (what Flower would do)."""
    outs = []
    lossf = nn.CrossEntropyLoss(weight=w)
    for k in range(batches.shape[0]):
        m = copy.deepcopy(model)
        with torch.no_grad():
            for n, p in m.named_parameters():
                p.copy_(gparams[n])
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        for s in range(batches.shape[1]):
            idx = batches[k, s]
            loss = lossf(m(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        outs.append({n: p.detach().clone() for n, p in m.named_parameters()})
    return {n: torch.stack([o[n] for o in outs]) for n in outs[0]}


def fedavg(stacked: dict[str, torch.Tensor], n_rows: torch.Tensor) -> dict[str, torch.Tensor]:
    """Weighted average over the client dimension (dim 0)."""
    wts = n_rows / n_rows.sum()
    return {n: (v * wts.view(-1, *([1] * (v.dim() - 1)))).sum(dim=0) for n, v in stacked.items()}


@torch.no_grad()
def predict_proba(model, params, X, batch_size: int = 8192) -> np.ndarray:
    """[n, n_classes] softmax probabilities of the model with weights `params` (rows sum to 1, float32)."""
    was_training = model.training
    model.eval()
    out = [torch.softmax(functional_call(model, params, (X[i:i + batch_size],)), dim=1)
           for i in range(0, len(X), batch_size)]
    model.train(was_training)
    return torch.cat(out).float().cpu().numpy()


@torch.no_grad()
def evaluate(model, params, X, y, batch_size: int = 8192, *, families: list[str] | None = None) -> dict:
    """macro_f1 and accuracy (unchanged keys). With `families` (FED-1) it adds the shared metrics module's scores
    (fqml.eval.metrics, EVAL-1): per-class precision / recall / F1 / support and the confusion matrix
    (`multiclass`), and the threshold-free binary scores (`binary` with no thresholds: ROC-AUC, benign and attack
    PR-AUC, F1 and false-positive rate at the default decision). Thresholds at a fixed FPR are set on the
    validation block by the runner (EXP-0), not here."""
    proba = predict_proba(model, params, X, batch_size)
    p = proba.argmax(axis=1)
    t = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    out = {"macro_f1": float(f1_score(t, p, average="macro")), "accuracy": float((p == t).mean())}
    if families is not None:
        from fqml.eval import metrics                     # EVAL-1 (Sheghaf)
        out["multiclass"] = metrics.multiclass(t, p, families)
        out["binary"] = metrics.binary(t, 1.0 - proba[:, 0], {})
    return out


def _local_only(model, gparams, X, y, parts, rounds, local_steps, batch_size, lr, w, rng, device,
                X_val, y_val, on_round, families, result):
    """mode="local": all N clients train alone, stacked in one vmap call; one Adam per client for the whole run
    (a site training alone has no reason to reset it)."""
    N = len(parts)
    stacked = {n: v.detach().unsqueeze(0).repeat(N, *([1] * v.dim())).clone().requires_grad_(True)
               for n, v in gparams.items()}
    opt = torch.optim.Adam(list(stacked.values()), lr=lr)
    forward = vmap(lambda p, xb: functional_call(model, p, (xb,)))
    clients = list(range(N))
    for r in range(rounds):
        t0 = time.perf_counter()
        batches = torch.as_tensor(np.stack([client_batches(parts[k], local_steps, batch_size, rng) for k in clients]),
                                  device=device)
        for s in range(local_steps):
            idx = batches[:, s]
            loss = client_losses(forward(stacked, X[idx]), y[idx], w).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        entry = {"round": r, "train_seconds": time.perf_counter() - t0, "clients": clients}
        entry["seconds"] = time.perf_counter() - t0
        result.history.append(entry)
        if on_round:
            on_round(entry)
    client_params = {n: v.detach() for n, v in stacked.items()}
    result.params, result.client_params = None, client_params
    if X_val is not None:
        scores = [evaluate(model, {n: v[k] for n, v in client_params.items()}, X_val, y_val, families=families)
                  for k in clients]
        result.final_val = scores
        f1s = [sc["macro_f1"] for sc in scores]
        result.history[-1].update({"val_macro_f1_mean": float(np.mean(f1s)), "val_macro_f1_min": float(np.min(f1s)),
                                   "val_macro_f1_max": float(np.max(f1s))})
    return result


def run_fedavg(model: nn.Module, X, y, parts: list[np.ndarray], *, rounds: int, clients_per_round: int,
               local_steps: int = 50, batch_size: int = 64, lr: float = 0.01,
               class_weights: torch.Tensor | None = None, seed: int = 0, device: str = "cpu",
               mode: str = "batched", X_val=None, y_val=None, on_round=None,
               dtype: torch.dtype = torch.float32, families: list[str] | None = None,
               aggregator=None, attack=None, malicious: np.ndarray | None = None) -> FedResult:
    """Simulate FedAvg. mode = "batched" (D3), "sequential" (reference, one client at a time) or "local"
    (FED-7: local-only, no aggregation; `clients_per_round` is ignored, every client trains every round).
    With X_val and `families`, `final_val` holds the full validation scores at the end (FED-1).

    FED-6 hooks (INTERFACES 3.2), applied each round between local training and aggregation:
      attack(stacked, chosen, malicious) -> stacked   weight attack; `chosen` = the round's client ids [K],
                                                      `malicious` = [K] bool, which of them are malicious
      aggregator(stacked, n_rows) -> params           default `fedavg`; robust ones in fqml.fl.robust
    `malicious` is the [n_clients] bool mask (fqml.attacks.poison.choose_malicious); needed with `attack`.
    Label attacks need no hook: pass the poisoned labels as `y` (fqml.attacks.poison.poison_labels)."""
    if mode not in ("batched", "sequential", "local"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "local" and (aggregator is not None or attack is not None):
        raise ValueError('mode="local" has no aggregation: aggregator and attack do not apply')
    if attack is not None and malicious is None:
        raise ValueError("attack needs the malicious client mask")
    if malicious is not None:
        malicious = np.asarray(malicious, dtype=bool)
        if malicious.shape != (len(parts),):
            raise ValueError("malicious must be a bool mask with one entry per client")
    aggregate = fedavg if aggregator is None else aggregator
    model = copy.deepcopy(model).to(device=device, dtype=dtype)
    X = torch.as_tensor(X, dtype=dtype, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    if X_val is not None:
        X_val = torch.as_tensor(X_val, dtype=dtype, device=device)
        y_val = torch.as_tensor(y_val, dtype=torch.long, device=device)
    w = None if class_weights is None else class_weights.to(device=device, dtype=dtype)
    sizes = np.array([len(p) for p in parts])
    rng = np.random.default_rng(seed)
    gparams = {n: p.detach().clone() for n, p in model.named_parameters()}
    result = FedResult(params=gparams)
    if mode == "local":
        return _local_only(model, gparams, X, y, parts, rounds, local_steps, batch_size, lr, w, rng, device,
                           X_val, y_val, on_round, families, result)
    local = {"batched": _local_batched, "sequential": _local_sequential}[mode]
    for r in range(rounds):
        t0 = time.perf_counter()
        chosen = np.sort(rng.choice(len(parts), clients_per_round, replace=False))
        batches = np.stack([client_batches(parts[k], local_steps, batch_size, rng) for k in chosen])
        stacked = local(model, gparams, X, y, torch.as_tensor(batches, device=device), lr, w)
        if attack is not None:
            stacked = attack(stacked, chosen, malicious[chosen])
        gparams = aggregate(stacked, torch.as_tensor(sizes[chosen], dtype=dtype, device=device))
        if device == "cuda":
            torch.cuda.synchronize()
        entry = {"round": r, "train_seconds": time.perf_counter() - t0, "clients": chosen.tolist()}
        if malicious is not None:
            entry["malicious"] = int(malicious[chosen].sum())
        if X_val is not None:
            entry.update({f"val_{k}": v for k, v in evaluate(model, gparams, X_val, y_val).items()})
        entry["seconds"] = time.perf_counter() - t0
        result.history.append(entry)
        if on_round:
            on_round(entry)
    result.params = gparams
    if X_val is not None and families is not None:
        result.final_val = evaluate(model, gparams, X_val, y_val, families=families)
    return result
