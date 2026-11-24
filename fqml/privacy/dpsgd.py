"""DP-SGD for every E0 model, quantum or classical (task PRIV-2; decisions D4 and D13). Owner: Nouran.

One training step, as in Abadi et al. (2016):
  1. Poisson sampling: every training row joins the batch independently with probability q = B / n (D13).
  2. Per-sample gradients of the weighted cross-entropy (D11 class weights):
       - TorchVQC: per-sample weight copies, one backward pass (D4, INTERFACES 4);
       - every other PyTorch model: torch.func (vmap over grad of a functional call).
  3. Clip each sample's whole gradient (all parameters together) to norm C.
  4. Sum, add Gaussian noise N(0, sigma^2 C^2 I), divide by the expected batch size B.
  5. Optimizer step (Adam; post-processing, so it costs no privacy).
Accounting (D13): Opacus's RDP accountant, the PRV accountant beside it; sigma is calibrated for the target
epsilon over the run's maximum number of steps; a run that stops early reports the epsilon of the steps taken.

Privacy unit (D13): one row (one flow window) in centralised Phase 1 runs; every epsilon reported here is
example-level. Class weights come from the training label counts, which are treated as public.

    from fqml.privacy.dpsgd import dp_fit
    res = dp_fit(model, X, y, n_classes=8, target_epsilon=4.0, delta=1e-6, clip=1.0,
                 batch_size=256, max_epochs=100, lr=0.01, seed=0, X_val=Xv, y_val=yv)
    res.epsilon, res.noise_multiplier, res.steps, res.best_epoch

target_epsilon=float("inf") is the non-private reference of the pilot: same Poisson batches, no clipping, no noise.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ----------------------------------------------------------------------------------------------- per-sample grads
def _is_vqc(model) -> bool:
    return hasattr(model, "theta") and hasattr(model, "logits") and hasattr(model, "expz")


def per_sample_grads(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                     class_weight: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Gradient of each sample's weighted loss  w_{y_i} * CE(model(x_i), y_i)  for every trainable parameter.

    Returns {parameter name: [B, *parameter shape]}. Keys are the model's `named_parameters()` names."""
    if _is_vqc(model):
        B = x.shape[0]
        th = model.theta.detach().expand(B, *model.theta.shape).clone().requires_grad_(True)
        sc = model.scale.detach().expand(B, *model.scale.shape).clone().requires_grad_(True)
        bi = model.bias.detach().expand(B, *model.bias.shape).clone().requires_grad_(True)
        logits = model.logits(x, th, sc, bi)
        loss = F.cross_entropy(logits, y, weight=class_weight, reduction="sum")
        g_th, g_sc, g_bi = torch.autograd.grad(loss, [th, sc, bi])
        return {"theta": g_th, "scale": g_sc, "bias": g_bi}

    from torch.func import functional_call, grad, vmap

    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_one(p, xb, yb):
        out = functional_call(model, (p, buffers), (xb.unsqueeze(0),))
        return F.cross_entropy(out, yb.unsqueeze(0), weight=class_weight, reduction="sum")

    return vmap(grad(loss_one), in_dims=(None, 0, 0), randomness="different")(params, x, y)


def per_sample_norms(grads: dict[str, torch.Tensor]) -> torch.Tensor:
    """[B] L2 norm of each sample's whole gradient (all parameters together)."""
    return torch.sqrt(sum(g.reshape(g.shape[0], -1).pow(2).sum(1) for g in grads.values()))


def clip_and_sum(grads: dict[str, torch.Tensor], clip: float) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Scale each sample's gradient to norm <= clip, then sum over the batch. Returns (sums, norms before clip)."""
    norms = per_sample_norms(grads)
    if math.isinf(clip):
        factor = torch.ones_like(norms)
    else:
        factor = (clip / (norms + 1e-6)).clamp(max=1.0)
    sums = {k: torch.einsum("b,b...->...", factor.to(g.dtype), g) for k, g in grads.items()}
    return sums, norms


# ------------------------------------------------------------------------------------------------------ sampling
def poisson_batches(n: int, sample_rate: float, steps: int, generator: torch.Generator, device="cpu"):
    """Yield `steps` index tensors; every row is in a batch independently with probability `sample_rate`.
    A batch can be empty (the step then adds noise only)."""
    for _ in range(steps):
        mask = torch.rand(n, generator=generator, device=device) < sample_rate
        yield torch.nonzero(mask, as_tuple=False).flatten()


# ---------------------------------------------------------------------------------------------------- accounting
def noise_multiplier_for(target_epsilon: float, delta: float, sample_rate: float, steps: int) -> float:
    """sigma such that the RDP accountant gives epsilon <= target after `steps` steps at `sample_rate`."""
    if math.isinf(target_epsilon):
        return 0.0
    from opacus.accountants.utils import get_noise_multiplier

    return float(get_noise_multiplier(target_epsilon=target_epsilon, target_delta=delta, sample_rate=sample_rate,
                                      steps=steps, accountant="rdp"))


class Accountant:
    """RDP accountant (reported epsilon) with the PRV accountant kept beside it as a cross-check."""

    def __init__(self):
        from opacus.accountants import RDPAccountant

        self.rdp = RDPAccountant()
        self._history: list[tuple[float, float, int]] = []          # (sigma, q, steps), compressed

    def step(self, noise_multiplier: float, sample_rate: float):
        if noise_multiplier <= 0:
            return
        self.rdp.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        if self._history and self._history[-1][:2] == (noise_multiplier, sample_rate):
            s, q, k = self._history[-1]
            self._history[-1] = (s, q, k + 1)
        else:
            self._history.append((noise_multiplier, sample_rate, 1))

    @property
    def steps(self) -> int:
        return sum(k for *_, k in self._history)

    def epsilon(self, delta: float) -> float:
        return float(self.rdp.get_epsilon(delta=delta)) if self._history else float("inf")

    def epsilon_prv(self, delta: float) -> float | None:
        """PRV estimate of the same history; None if Opacus cannot compute it for these settings."""
        if not self._history:
            return None
        try:
            from opacus.accountants import PRVAccountant

            prv = PRVAccountant()
            prv.history = list(self._history)
            return float(prv.get_epsilon(delta=delta))
        except Exception:                                            # noqa: BLE001 - cross-check only
            return None


# -------------------------------------------------------------------------------------------------------- training
@dataclass
class DPResult:
    target_epsilon: float
    epsilon: float                 # RDP accountant, after the steps actually taken
    epsilon_prv: float | None
    delta: float
    noise_multiplier: float
    clip: float
    sample_rate: float
    steps: int
    best_epoch: int
    best_score: float
    stopped_because: str           # "patience" | "max_epochs" | "max_seconds"
    history: list[dict] = field(default_factory=list)
    clipped_share: float = 0.0     # share of sampled rows whose gradient was clipped


def _class_weight_tensor(y: np.ndarray, n_classes: int, device) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.zeros(n_classes)
    present = counts > 0
    w[present] = 1.0 / np.sqrt(counts[present])
    w = w / w[present].mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def _macro_f1(model, X, y, n_classes, batch=8192) -> float:
    from sklearn.metrics import f1_score

    model.eval()
    preds = [model(X[i:i + batch]).argmax(1).cpu() for i in range(0, len(X), batch)]
    model.train()
    return float(f1_score(y, torch.cat(preds).numpy(), average="macro", labels=list(range(n_classes)),
                          zero_division=0))


def dp_fit(model: nn.Module, X, y, n_classes: int, *, target_epsilon: float, delta: float = 1e-6,
           clip: float = 1.0, batch_size: int = 256, max_epochs: int = 100, lr: float = 0.01,
           weight_decay: float = 0.0, seed: int = 0, X_val=None, y_val=None, patience: int = 10,
           eval_every: int = 1, max_seconds: float | None = None, on_epoch=None) -> DPResult:
    """Train `model` with DP-SGD (D13). The model ends with the weights of the best validation epoch.

    X: float32 [n, d] in [0, pi]; y: int64 family index. Validation rows are not training rows, so early stopping
    on them costs no privacy of the training rows."""
    device = next(model.parameters()).device
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
    n = len(Xt)
    q = batch_size / n
    steps_per_epoch = max(1, round(n / batch_size))
    max_steps = steps_per_epoch * max_epochs
    private = not math.isinf(target_epsilon)
    sigma = noise_multiplier_for(target_epsilon, delta, q, max_steps) if private else 0.0
    C = clip if private else float("inf")
    cw = _class_weight_tensor(np.asarray(y), n_classes, device)
    Xv = None if X_val is None else torch.as_tensor(np.asarray(X_val), dtype=torch.float32, device=device)

    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    acc = Accountant()
    names = [k for k, p in model.named_parameters() if p.requires_grad]
    params = dict(model.named_parameters())
    best, best_score, best_epoch, since, stopped = None, -1.0, -1, 0, "max_epochs"
    history, n_sampled, n_clipped, t0 = [], 0, 0, time.time()
    model.train()

    for epoch in range(max_epochs):
        total = 0.0
        for idx in poisson_batches(n, q, steps_per_epoch, gen, device):
            if len(idx):
                grads = per_sample_grads(model, Xt[idx], yt[idx], cw)
                sums, norms = clip_and_sum(grads, C)
                n_sampled += len(idx)
                if private:
                    n_clipped += int((norms > C).sum())
                total += float(norms.mean()) * len(idx)
            else:
                sums = {k: torch.zeros_like(params[k]) for k in names}
            for k in names:
                g = sums[k]
                if private:
                    g = g + torch.normal(0.0, sigma * C, size=g.shape, generator=gen, device=device)
                params[k].grad = (g / batch_size).to(params[k].dtype)
            opt.step()
            opt.zero_grad(set_to_none=True)
            acc.step(sigma, q)

        rec = {"epoch": epoch, "mean_grad_norm": total / max(n_sampled, 1), "seconds": time.time() - t0,
               "epsilon": acc.epsilon(delta) if private else float("inf")}
        if Xv is not None and (epoch + 1) % eval_every == 0:
            score = _macro_f1(model, Xv, np.asarray(y_val), n_classes)
            rec["val_macro_f1"] = score
            if score > best_score:
                best_score, best_epoch, since = score, epoch, 0
                best = copy.deepcopy(model.state_dict())
            else:
                since += 1
        history.append(rec)
        if on_epoch:
            on_epoch(rec)
        if Xv is not None and since >= patience:
            stopped = "patience"
            break
        if max_seconds is not None and time.time() - t0 > max_seconds:
            stopped = "max_seconds"
            break

    if best is not None:
        model.load_state_dict(best)
    return DPResult(target_epsilon=target_epsilon, epsilon=acc.epsilon(delta) if private else float("inf"),
                    epsilon_prv=acc.epsilon_prv(delta) if private else None, delta=delta, noise_multiplier=sigma,
                    clip=C, sample_rate=q, steps=acc.steps if private else (len(history) * steps_per_epoch),
                    best_epoch=best_epoch, best_score=best_score, stopped_because=stopped, history=history,
                    clipped_share=n_clipped / max(n_sampled, 1))
