"""Classical twin of the VQC with (almost) the same number of parameters. Owner: Ganna.
This pairing is what makes RQ1 fair: same features, same loop, same parameter budget."""
from torch import nn


def mlp_params(n_in, hidden, n_out):
    return n_in * hidden + hidden + hidden * n_out + n_out


def matched_hidden(n_in, n_out, target):
    """Hidden width whose parameter count is closest to `target`."""
    return min(range(1, 512), key=lambda h: abs(mlp_params(n_in, h, n_out) - target))


class MatchedMLP(nn.Module):
    def __init__(self, n_in, n_classes, target_params):
        super().__init__()
        self.hidden = matched_hidden(n_in, n_classes, target_params)
        self.net = nn.Sequential(nn.Linear(n_in, self.hidden), nn.ReLU(),
                                 nn.Linear(self.hidden, n_classes))

    def forward(self, x):
        return self.net(x)