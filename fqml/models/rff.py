"""Random Fourier feature model of the VQC's function class (task Q-3, PREREG_E0 comparisons 3-4). Owner: Nouran.

Why this model exists. With angle encoding RY(x_q) repeated in L = 3 layers, every output of the VQC is a
trigonometric polynomial in the inputs: a sum of cos(w . x) and sin(w . x) terms whose frequency vectors w have
whole-number entries with |w_q| <= L = 3 (Schuld, Sweke and Meyer 2021). The circuit chooses the coefficients
through its trainable angles. A classical model that builds such terms directly and fits a linear layer on top
can represent the same kind of function; if the VQC cannot beat it, the quantum circuit adds nothing that this
cheap classical model cannot do (Landman et al. 2022, "Classically approximating variational quantum machine
learning with random Fourier features").

Frequencies. The full set {-3..3}^8 has 7^8 = 5.8 million vectors, so D of them are sampled. Each entry is drawn
from the circuit's own frequency degeneracy: one RY(x) encoding contributes a frequency difference of -1, 0 or +1
with multiplicities 1, 2, 1, so three encodings give k in {-3..3} with weights (1, 6, 15, 20, 15, 6, 1) / 64 =
binomial(6, k + 3) / 64. Frequency vectors are drawn with this weight, w = 0 is excluded (the bias covers it), and
w and -w are not both kept (cos and sin already cover the sign).

Model: phi(x) = [cos(x W^T), sin(x W^T)] in R^{2D}, logits = Linear(2D -> n_classes)(phi(x)).
Trainable parameters: 2 D n_classes + n_classes. Frequencies are fixed (a buffer), drawn from `seed`.
Inputs: float32 in [0, pi] (INTERFACES 1.3), the same inputs as the VQC.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

DEFAULT_FEATURES = 256          # D; fixed for E0 (D11 part 2 tunes only lr and weight decay)


def degeneracy_weights(max_frequency: int) -> np.ndarray:
    """P(k) for k = -L..L: binomial(2L, k + L) / 4^L, the frequency multiplicities of L RY encodings."""
    L = max_frequency
    w = np.array([math.comb(2 * L, k + L) for k in range(-L, L + 1)], dtype=np.float64)
    return w / w.sum()


def sample_frequencies(n_in: int, n_features: int, max_frequency: int = 3, seed: int = 0) -> np.ndarray:
    """[n_features, n_in] distinct whole-number frequency vectors, none zero, no pair w / -w."""
    rng = np.random.default_rng(seed)
    p = degeneracy_weights(max_frequency)
    values = np.arange(-max_frequency, max_frequency + 1)
    seen, out = set(), []
    while len(out) < n_features:
        w = rng.choice(values, size=n_in, p=p)
        if not w.any():
            continue
        key, neg = tuple(int(v) for v in w), tuple(int(-v) for v in w)
        if key in seen or neg in seen:
            continue
        seen.add(key)
        out.append(w)
    return np.stack(out).astype(np.float32)


class RFF(nn.Module):
    def __init__(self, n_in: int = 8, n_classes: int = 8, n_features: int = DEFAULT_FEATURES,
                 max_frequency: int = 3, seed: int = 0):
        super().__init__()
        self.n_in, self.n_classes, self.n_features, self.max_frequency = n_in, n_classes, n_features, max_frequency
        omega = torch.from_numpy(sample_frequencies(n_in, n_features, max_frequency, seed))
        self.register_buffer("omega", omega)                       # [D, n_in], fixed
        state = torch.get_rng_state()                              # starting weights depend only on seed
        torch.manual_seed(seed)
        self.linear = nn.Linear(2 * n_features, n_classes)
        torch.set_rng_state(state)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = x.to(self.omega.dtype) @ self.omega.T                  # [B, D]
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1) / math.sqrt(self.n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.features(x))


def rff_params(n_classes: int, n_features: int = DEFAULT_FEATURES) -> int:
    return 2 * n_features * n_classes + n_classes
