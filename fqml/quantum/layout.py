"""Circuit layout shared by the PennyLane VQC, the PyTorch simulator and the Qiskit circuit.
No heavy imports here, so every implementation can use it. Decisions D6 (version) and D7 (entangler).
"""


ENTANGLERS = ("chain", "ring", "none")
READOUTS = ("local", "linear")


def entangler_pairs(n_qubits, entangler="ring"):
    """CNOT (control, target) pairs, applied in this order at the end of every layer.

    "none" (INTERFACES 2.0.0) returns no pairs: the entanglement-free control. Without CNOTs every
    qubit evolves alone, so <Z_q> depends only on feature q and qubit q's own 6 angles. With the
    "local" readout an output therefore sees ONE feature (binary: 42 of the 48 angles cannot affect it);
    the control is only meaningful with readout="linear" (see readout_wires)."""
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


def readout_wires(n_qubits, n_out, version=2, readout="local"):
    """Qubits whose <Z> feeds the output layer.

    readout="local" (D6, the Phase 0 circuit): output k reads ONE qubit, the k-th of readout_qubits(...);
        logits = scale * z + bias, scale [n_out].
    readout="linear" (Phase 0 amendments): every output is a trained weighted sum over ALL n qubits;
        logits = z @ scale.T + bias, scale [n_out, n_qubits]. Needed for a fair entanglement-free control,
        and used by the VQC it is compared with, so the two differ only by the CNOT gates.
    """
    if readout == "local":
        return readout_qubits(n_qubits, n_out, version)
    if readout == "linear":
        return list(range(n_qubits))
    raise ValueError(f"unknown readout {readout!r} (one of {READOUTS})")


def initial_scale(n_qubits, n_out, version=2, readout="local", value=2.0):
    """Starting output weights as nested lists. "linear" starts as the "local" circuit: weight `value` on
    the qubit "local" would read for each output, 0 elsewhere, so both readouts begin as the same model."""
    if readout == "local":
        return [value] * n_out
    if readout not in READOUTS:
        raise ValueError(f"unknown readout {readout!r} (one of {READOUTS})")
    w = [[0.0] * n_qubits for _ in range(n_out)]
    for k, q in enumerate(readout_qubits(n_qubits, n_out, version)):
        w[k][q] = value
    return w
