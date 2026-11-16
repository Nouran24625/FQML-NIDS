"""Classical MLP at a chosen parameter budget (INTERFACES 2.3). Owner: Nouran.

`MatchedMLP(n_in, n_classes, target_params)` is one hidden layer (ReLU) whose width gives the parameter
count closest to `target_params`. With the VQC's own count as the target it is the same-size twin of
the VQC; with the other rungs of `fqml.models.sizes.size_ladder` it is the MLP size ladder
(Phase 0 amendment 6): same features, same training loop, only the width changes.
"""
import torch
from torch import nn

from fqml.models.sizes import LADDER_TARGETS, matched_hidden, mlp_params, size_ladder  # noqa: F401 (re-exported)


class MatchedMLP(nn.Module):
    def __init__(self, n_in, n_classes, target_params, seed=None):
        """seed=None keeps the old behaviour (weights from the global torch RNG). With a seed the
        starting weights depend only on the seed, and the global RNG is left exactly as it was."""
        super().__init__()
        self.target_params = target_params
        self.hidden = matched_hidden(n_in, n_classes, target_params)
        state = None
        if seed is not None:
            state = torch.get_rng_state()
            torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(n_in, self.hidden), nn.ReLU(),
                                 nn.Linear(self.hidden, n_classes))
        if state is not None:
            torch.set_rng_state(state)

    def forward(self, x):
        return self.net(x)
