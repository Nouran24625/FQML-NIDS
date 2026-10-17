"""PyTorch state-vector simulator of the VQC (decision D1). Owner: Sheghaf.

Simulates EXACTLY the circuit in fqml/models/vqc.py (same gates, same order, same starting
weights for the same seed). tests/test_torch_sim.py checks it against PennyLane on every CI run.

Why we have it (measured 17 Oct 2026, Kaggle):
  - same results as PennyLane (max difference ~1e-15)
  - ~2.5x faster on one CPU thread, ~50-80x faster on a T4 GPU when many samples run together
  - accepts PER-SAMPLE weights (theta of shape [batch, L, n, 2]), which gives us
      * batched federated clients: every client's samples carry that client's weights (D3)
      * per-sample gradients for DP-SGD in ONE backward pass (Spike S3, method D)

How it works: the 8-qubit state is a tensor of 2^8 = 256 complex numbers per sample.
  - RY(x) then RY(t0) on a qubit = RY(x + t0), followed by RZ(t1): one 2x2 gate per qubit per layer
  - the CNOT ring only reorders the 256 numbers -> one fixed index permutation per layer
  - <Z_i> = sum of probabilities with sign +1 (qubit i = 0) or -1 (qubit i = 1)
Qubit 0 is the most significant bit, as in PennyLane.
"""
import torch
from torch import nn


def _cnot_ring_permutation(n: int) -> torch.Tensor:
    """Index permutation equal to CNOT(0,1), CNOT(1,2), ..., CNOT(n-1,0) applied in that order."""
    idx = torch.arange(2 ** n)
    perm = idx.clone()
    for q in range(n):
        control_bit = (idx >> (n - 1 - q)) & 1
        perm = perm[idx ^ (control_bit << (n - 1 - (q + 1) % n))]
    return perm


class TorchVQC(nn.Module):
    """Drop-in replacement for fqml.models.vqc.VQC (same constructor, parameters and outputs)."""

    def __init__(self, n_qubits=8, n_classes=2, n_layers=3, seed=0):
        super().__init__()
        self.n_qubits, self.n_layers, self.n_classes = n_qubits, n_layers, n_classes
        self.n_out = 1 if n_classes == 2 else n_classes
        assert self.n_out <= n_qubits, "more classes than qubits"
        g = torch.Generator().manual_seed(seed)              # same order as VQC -> same weights
        self.theta = nn.Parameter(torch.rand(n_layers, n_qubits, 2, generator=g) * 2 * torch.pi)
        self.scale = nn.Parameter(torch.full((self.n_out,), 2.0))
        self.bias = nn.Parameter(torch.zeros(self.n_out))
        idx = torch.arange(2 ** n_qubits)
        bits = (idx[:, None] >> (n_qubits - 1 - torch.arange(self.n_out))) & 1
        self.register_buffer("perm", _cnot_ring_permutation(n_qubits), persistent=False)
        self.register_buffer("zsign", (1 - 2 * bits).float(), persistent=False)   # [256, n_out]

    def expz(self, x, theta=None):
        """<Z_i> for the readout qubits. x: [B, n]. theta: [L, n, 2] (shared) or [B, L, n, 2]."""
        theta = self.theta if theta is None else theta
        B, n = x.shape[0], self.n_qubits
        cdt = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        e0 = torch.zeros(2 ** n, dtype=cdt, device=x.device)
        e0[0] = 1                                              # start in |00000000>
        psi = e0.expand(B, -1)
        ones = [1] * (n - 1)
        for layer in range(self.n_layers):
            a = (x + theta[..., layer, :, 0]) / 2              # RY(t0) RY(x) = RY(x + t0)
            c, s = torch.cos(a), torch.sin(a)                  # [B, n]
            em = torch.exp(-0.5j * theta[..., layer, :, 1])    # RZ phases: [n] or [B, n]
            ep = torch.exp(0.5j * theta[..., layer, :, 1])
            psi = psi.reshape(B, *([2] * n))
            for q in range(n):
                p = psi.movedim(q + 1, -1)
                p0, p1 = p[..., 0], p[..., 1]
                cq, sq = c[:, q].reshape(B, *ones), s[:, q].reshape(B, *ones)
                mq, pq = em[..., q], ep[..., q]
                if mq.dim():                                   # per-sample weights
                    mq, pq = mq.reshape(B, *ones), pq.reshape(B, *ones)
                psi = torch.stack([mq * (cq * p0 - sq * p1),     # RZ . RY on qubit q
                                   pq * (sq * p0 + cq * p1)], dim=-1).movedim(-1, q + 1)
            psi = psi.reshape(B, -1)[:, self.perm]             # CNOT ring
        probs = psi.real ** 2 + psi.imag ** 2
        return probs @ self.zsign.to(probs.dtype)

    def logits(self, x, theta=None, scale=None, bias=None):
        """Same output as VQC.forward; scale/bias may also be per-sample ([B, n_out])."""
        scale = self.scale if scale is None else scale
        bias = self.bias if bias is None else bias
        out = scale * self.expz(x, theta) + bias
        if self.n_out == 1:   # binary: two logits, as in VQC
            return torch.cat([torch.zeros_like(out), out], dim=1)
        return out

    def forward(self, x):
        return self.logits(x)
