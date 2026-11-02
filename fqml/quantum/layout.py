"""Circuit layout shared by the PennyLane VQC, the PyTorch simulator and the Qiskit circuit.
No heavy imports here, so every implementation can use it. Decisions D6 (version) and D7 (entangler).
"""


def entangler_pairs(n_qubits, entangler="ring"):
    """CNOT (control, target) pairs, applied in this order at the end of every layer."""
    if entangler == "ring":
        return [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    if entangler == "chain":
        return [(q, q + 1) for q in range(n_qubits - 1)]
    raise ValueError(f"unknown entangler {entangler!r}")


def readout_qubits(n_qubits, n_out, version=1):
    """Qubits whose <Z> is read: version 1 = the first n_out, version 2 = the last n_out.
    (With the chain, qubit 0 is only ever a CNOT control, so reading it would leave most angles dead.)"""
    return list(range(n_out)) if version == 1 else list(range(n_qubits - n_out, n_qubits))
