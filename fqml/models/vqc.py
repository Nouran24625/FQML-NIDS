"""Variational Quantum Classifier (Plan section 7). Owner: Sheghaf.

Each of L layers, on n qubits:
  1. RY(x_i) on every qubit        -> encodes the data (re-uploaded every layer)
  2. RY(theta) and RZ(theta) on every qubit -> trainable rotations
  3. CNOT ring 0->1->...->n-1->0   -> entangles the qubits
Readout: <Z> on the first qubits, then a trainable scale and bias per output.
Trainable parameters = 2*n*L + 2*outputs  (n=8, L=3, binary -> 48 + 2 = 50)
Inputs must already be scaled to [0, pi].
"""
import pennylane as qml
import torch
from torch import nn


class VQC(nn.Module):
    def __init__(self, n_qubits=8, n_classes=2, n_layers=3, seed=0):
        super().__init__()
        self.n_qubits, self.n_layers, self.n_classes = n_qubits, n_layers, n_classes
        # binary -> 1 readout qubit; multiclass -> one readout qubit per class
        self.n_out = 1 if n_classes == 2 else n_classes
        assert self.n_out <= n_qubits, "more classes than qubits"

        g = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(torch.rand(n_layers, n_qubits, 2, generator=g) * 2 * torch.pi)
        self.scale = nn.Parameter(torch.full((self.n_out,), 2.0))
        self.bias = nn.Parameter(torch.zeros(self.n_out))

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, theta):
            for layer in range(n_layers):
                for q in range(n_qubits):
                    qml.RY(x[..., q], wires=q)               # data re-uploading
                for q in range(n_qubits):
                    qml.RY(theta[layer, q, 0], wires=q)      # trainable
                    qml.RZ(theta[layer, q, 1], wires=q)
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])  # entangling ring
            return [qml.expval(qml.PauliZ(q)) for q in range(self.n_out)]

        self.circuit = circuit

    def forward(self, x):
        z = torch.stack(list(self.circuit(x, self.theta)), dim=-1).float()  # [batch, n_out]
        logits = self.scale * z + self.bias
        if self.n_out == 1:  # binary: two logits so we can use the same loss as multiclass
            return torch.cat([torch.zeros_like(logits), logits], dim=1)
        return logits


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)