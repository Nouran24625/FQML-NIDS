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
    params: dict[str, torch.Tensor]                 # final global weights
    history: list[dict] = field(default_factory=list)  # one entry per round


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
def evaluate(model, params, X, y, batch_size: int = 8192) -> dict:
    preds = []
    for i in range(0, len(X), batch_size):
        preds.append(functional_call(model, params, (X[i:i + batch_size],)).argmax(dim=1))
    p = torch.cat(preds).cpu().numpy()
    t = y.cpu().numpy()
    return {"macro_f1": float(f1_score(t, p, average="macro")), "accuracy": float((p == t).mean())}


def run_fedavg(model: nn.Module, X, y, parts: list[np.ndarray], *, rounds: int, clients_per_round: int,
               local_steps: int = 50, batch_size: int = 64, lr: float = 0.01,
               class_weights: torch.Tensor | None = None, seed: int = 0, device: str = "cpu",
               mode: str = "batched", X_val=None, y_val=None, on_round=None,
               dtype: torch.dtype = torch.float32) -> FedResult:
    """Simulate FedAvg. mode = "batched" (D3) or "sequential" (reference, one client at a time)."""
    local = {"batched": _local_batched, "sequential": _local_sequential}[mode]
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
    for r in range(rounds):
        t0 = time.perf_counter()
        chosen = np.sort(rng.choice(len(parts), clients_per_round, replace=False))
        batches = np.stack([client_batches(parts[k], local_steps, batch_size, rng) for k in chosen])
        stacked = local(model, gparams, X, y, torch.as_tensor(batches, device=device), lr, w)
        gparams = fedavg(stacked, torch.as_tensor(sizes[chosen], dtype=dtype, device=device))
        if device == "cuda":
            torch.cuda.synchronize()
        entry = {"round": r, "train_seconds": time.perf_counter() - t0, "clients": chosen.tolist()}
        if X_val is not None:
            entry.update({f"val_{k}": v for k, v in evaluate(model, gparams, X_val, y_val).items()})
        entry["seconds"] = time.perf_counter() - t0
        result.history.append(entry)
        if on_round:
            on_round(entry)
    result.params = gparams
    return result
