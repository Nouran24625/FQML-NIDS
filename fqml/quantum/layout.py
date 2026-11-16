"""Circuit layout shared by the PennyLane VQC, the PyTorch simulator and the Qiskit circuit.
No heavy imports here, so every implementation can use it. Decisions D6 (version) and D7 (entangler).
"""


ENTANGLERS = ("chain", "ring", "none")


def entangler_pairs(n_qubits, entangler="ring"):
    """CNOT (control, target) pairs, applied in this order at the end of every layer.

    "none" (INTERFACES 2.0.0) returns no pairs: the entanglement-free control. Without CNOTs every
    qubit evolves alone, so <Z_q> depends only on feature q and qubit q's own 6 angles. An output can
    then only use the qubits that are read: with the binary readout (last qubit only) 42 of the 48
    angles cannot affect the output (tests/test_torch_sim.py::test_dead_parameters)."""
    if entangler == "none":
        return []
    if entangler == "ring":
        return [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    if entangler == "chain":
        return [(q, q + 1) for q in range(n_qubits - 1)]
    raise ValueError(f"unknown entangler {entangler!r}")


def readout_qubits(n_qubits, n_out, version=1):
    """Qubits whose <Z> is read: version 1 = the first n_out, version 2 = the last n_out.
    (With the chain, qubit 0 is only ever a CNOT control, so reading it would leave most angles dead.)"""
    return list(range(n_out)) if version == 1 else list(range(n_qubits - n_out, n_qubits))
