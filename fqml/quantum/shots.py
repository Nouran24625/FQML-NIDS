"""Finite-shot measurement for the PyTorch VQC simulator (task Q-1). Owner: Nouran.

On hardware, <Z_i> is never known exactly: the circuit is run `shots` times, every run returns one
bitstring for all qubits, and <Z_i> is the average of +1 / -1 over the runs. Gradients cannot be taken
by backpropagation either; they come from the parameter-shift rule.

What this module emulates, exactly as a device would produce it:
  - Forward: each sample's circuit is run `shots` times. One multinomial draw over the 2^n bitstrings per
    sample, so all readout qubits come from the SAME shots (their errors are correlated, as on hardware).
  - Backward (grad="shift"): for every trainable angle, the circuit is run again at angle + pi/2 and
    angle - pi/2, each with its own `shots` runs:  d<Z>/d theta = (<Z>(theta + pi/2) - <Z>(theta - pi/2)) / 2.
    This is exact (not an approximation) for RY and RZ gates, whose generators are Pauli / 2. Every angle
    in fqml/models/vqc.py appears in exactly one such gate, for both circuit versions.
    Cost per sample: 1 + 2 * n_angles circuit runs (n=8, L=3: 1 + 96 = 97).
  - Backward (grad="exact"): the forward value is noisy but the gradient is the exact one. Not physical;
    it isolates the effect of noise in the outputs from the effect of noise in the gradient.

The output layer (scale, bias) is classical: its gradient is computed by autograd from the sampled <Z>,
as a device user would do. Inputs x never receive a gradient (they are data).

Per-sample weights (theta of shape [B, L, n, 2]) are supported, so per-sample gradients for DP-SGD come
out of one backward pass with shots too.
"""
import math

import torch

GRAD_MODES = ("shift", "exact")


def sample_expz(probs: torch.Tensor, zsign: torch.Tensor, shots: int, generator=None) -> torch.Tensor:
    """<Z> estimated from `shots` measurements of every sample.

    probs: [B, 2^n] bitstring probabilities. zsign: [2^n, n_read], +1 / -1 per bitstring and qubit.
    Returns [B, n_read]. One multinomial draw per sample, shared by all readout qubits."""
    if shots < 1:
        raise ValueError("shots must be >= 1")
    p = probs.detach().clamp_min(0)
    p = p / p.sum(-1, keepdim=True)                        # remove ~1e-7 rounding drift in float32
    idx = torch.multinomial(p, shots, replacement=True, generator=generator)        # [B, shots]
    counts = torch.zeros_like(p).scatter_add_(1, idx, torch.ones_like(idx, dtype=p.dtype))
    return (counts / shots) @ zsign.to(p.dtype)


def shift_jacobian(sim, x, theta, shots=None, generator=None, max_rows=None):
    """d<Z_read>/d theta for every sample by the parameter-shift rule.

    shots=None evaluates the shifted circuits exactly (used by the tests to check the rule itself).
    Returns [B, n_read, L, n, 2]. `max_rows` caps how many shifted circuits are simulated at once
    (memory: one row = 2^n complex numbers); default: all at once."""
    B = x.shape[0]
    L, n = sim.n_layers, sim.n_qubits
    P = L * n * 2
    th = theta.detach()
    th = th.expand(B, *th.shape) if th.dim() == 3 else th                  # [B, L, n, 2]
    shift = torch.eye(P, dtype=th.dtype, device=th.device).reshape(P, L, n, 2) * (math.pi / 2)
    th_all = torch.cat([th[:, None] + shift, th[:, None] - shift], dim=1)  # [B, 2P, L, n, 2]
    th_all = th_all.reshape(B * 2 * P, L, n, 2)
    x_all = x.detach()[:, None, :].expand(B, 2 * P, n).reshape(B * 2 * P, n)
    rows = x_all.shape[0]
    step = rows if max_rows is None else max_rows
    z = []
    with torch.no_grad():
        for i in range(0, rows, step):
            p = sim.probs(x_all[i:i + step], th_all[i:i + step])
            z.append(p @ sim.zsign.to(p.dtype) if shots is None
                     else sample_expz(p, sim.zsign, shots, generator))
    z = torch.cat(z).reshape(B, 2, P, -1).to(th.dtype)                    # [B, +/-, P, n_read]
    jac = (z[:, 0] - z[:, 1]) / 2                                          # [B, P, n_read]
    return jac.transpose(1, 2).reshape(B, -1, L, n, 2)


class _ShotExpZ(torch.autograd.Function):
    """<Z_read> from `shots` measurements, with a hardware-style gradient for theta."""

    @staticmethod
    def forward(ctx, x, theta, sim, shots, grad, generator, max_rows):
        with torch.no_grad():
            z = sample_expz(sim.probs(x, theta), sim.zsign, shots, generator).to(theta.dtype)
        ctx.save_for_backward(x, theta)
        ctx.sim, ctx.shots, ctx.grad, ctx.gen, ctx.max_rows = sim, shots, grad, generator, max_rows
        return z

    @staticmethod
    def backward(ctx, gz):
        x, theta = ctx.saved_tensors
        if not ctx.needs_input_grad[1]:
            return (None,) * 7
        if ctx.grad == "shift":
            jac = shift_jacobian(ctx.sim, x, theta, ctx.shots, ctx.gen, ctx.max_rows)
        else:                                                              # "exact"
            with torch.enable_grad():
                th = theta.detach().requires_grad_(True)
                th_b = th.expand(x.shape[0], *th.shape) if th.dim() == 3 else th
                z = ctx.sim.probs(x, th_b) @ ctx.sim.zsign.to(x.dtype)
                g, = torch.autograd.grad(z, th, gz.to(z.dtype))
            return None, g, None, None, None, None, None
        g = torch.einsum("br,br...->b...", gz.to(jac.dtype), jac)          # per-sample [B, L, n, 2]
        if theta.dim() == 3:
            g = g.sum(0)
        return None, g, None, None, None, None, None


def shot_expz(sim, x, theta, shots, grad="shift", generator=None, max_rows=None):
    """Public entry point; see the module docstring."""
    if grad not in GRAD_MODES:
        raise ValueError(f"unknown grad mode {grad!r} (one of {GRAD_MODES})")
    if x.requires_grad:
        raise ValueError("finite-shot mode gives no gradient for the inputs x")
    return _ShotExpZ.apply(x, theta, sim, int(shots), grad, generator, max_rows)
