"""Variational Quantum Classifier (Plan section 7). Owner: Sheghaf.

Each of L layers, on n qubits:
  version 1 (original):   RY(x_q), RY(theta0), RZ(theta1) on every qubit, then CNOTs
  version 2 (D6 fix):     RY(x_q), RZ(theta1), RY(theta0) on every qubit, then CNOTs
Entangler: "ring" (CNOT q -> (q+1) % n) or "chain" (CNOT q -> q+1, no wrap-around; fits IBM chips
with no extra 2-qubit gates, Spike S4).
DEFAULT (decisions D6 + D7, 20 Oct 2026): entangler="chain", version=2. Version 1 / ring are kept
only to reproduce older results.
Readout: <Z> on the readout qubits, then a trainable output layer.
  readout="local" (default, D6): output k reads one qubit; version 1 the FIRST n_out qubits, version 2
  the LAST n_out; logits = scale * z + bias (scale [n_out]).
  readout="linear" (Phase 0 amendments): every output is a weighted sum over ALL qubits;
  logits = z @ scale.T + bias (scale [n_out, n]). Starts as the local circuit (initial_scale). This is
  the readout of the entanglement-free control (entangler="none") and of the VQC it is compared with.

Why version 2 (decision D6): in version 1 the last layer's RZ angles are followed only by CNOTs and
the Z readout, so they can never change the output (8 dead parameters; 9 for binary). With the chain,
qubit 0 is only ever a CNOT control, so reading it would leave 37 of 48 angles dead. Version 2 puts RZ
between the two RYs and reads the last qubit(s): 0 dead parameters for ring and chain.
Parameter count is the same in both versions.

Trainable parameters, local readout  = 2*n*L + 2*outputs      (n=8, L=3: binary 50, 8 classes 64)
                     linear readout = 2*n*L + (n+1)*outputs  (n=8, L=3: binary 57, 8 classes 120)
Inputs must already be scaled to [0, pi].
"""
import pennylane as qml
import torch
from torch import nn

from fqml.quantum.layout import entangler_pairs, initial_scale, readout_qubits, readout_wires  # noqa: F401 (re-exported)


class VQC(nn.Module):
    def __init__(self, n_qubits=8, n_classes=2, n_layers=3, seed=0, entangler="chain", version=2,
                 readout="local"):
        super().__init__()
        self.n_qubits, self.n_layers, self.n_classes = n_qubits, n_layers, n_classes
        self.entangler, self.version, self.readout = entangler, version, readout
        # binary -> 1 readout qubit; multiclass -> one readout qubit per class
        self.n_out = 1 if n_classes == 2 else n_classes
        assert self.n_out <= n_qubits, "more classes than qubits"
        assert version in (1, 2), "circuit version must be 1 or 2"

        g = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(torch.rand(n_layers, n_qubits, 2, generator=g) * 2 * torch.pi)
        self.scale = nn.Parameter(torch.tensor(initial_scale(n_qubits, self.n_out, version, readout)))
        self.bias = nn.Parameter(torch.zeros(self.n_out))

        pairs = entangler_pairs(n_qubits, entangler)
        wires = readout_wires(n_qubits, self.n_out, version, readout)
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, theta):
            for layer in range(n_layers):
                for q in range(n_qubits):
                    qml.RY(x[..., q], wires=q)               # data re-uploading
                for q in range(n_qubits):
                    if version == 1:
                        qml.RY(theta[layer, q, 0], wires=q)  # trainable
                        qml.RZ(theta[layer, q, 1], wires=q)
                    else:
                        qml.RZ(theta[layer, q, 1], wires=q)  # D6: RZ between the two RYs
                        qml.RY(theta[layer, q, 0], wires=q)
                for c, t in pairs:
                    qml.CNOT(wires=[c, t])                   # entangling ring or chain
            return [qml.expval(qml.PauliZ(q)) for q in wires]

        self.circuit = circuit

    def forward(self, x):
        z = torch.stack(list(self.circuit(x, self.theta)), dim=-1).float()  # [batch, n_read]
        if self.readout == "linear":
            logits = z @ self.scale.T + self.bias
        else:
            logits = self.scale * z + self.bias
        if self.n_out == 1:  # binary: two logits so we can use the same loss as multiclass
            return torch.cat([torch.zeros_like(logits), logits], dim=1)
        return logits


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
