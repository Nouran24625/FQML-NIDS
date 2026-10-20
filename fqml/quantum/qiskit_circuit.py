"""The VQC as a Qiskit circuit, for hardware work (Spike S4, experiment E6). Owner: Sheghaf.

Same gates and order as fqml/models/vqc.py and fqml/quantum/torch_sim.py:
  per layer: RY(x_q) on every qubit, then RY(theta[l,q,0]) RZ(theta[l,q,1]) on every qubit, then CNOTs.
entangler = "ring"  -> CNOT q -> (q+1) % n   (what we train today)
entangler = "chain" -> CNOT q -> q+1 for q < n-1 (no wrap-around; fits real chips with fewer SWAPs)

Qiskit numbers qubits little-endian (qubit 0 = rightmost bit) while PennyLane / TorchVQC treat qubit 0
as the leftmost. Gates act on the same qubit indices either way, so per-qubit <Z_q> values agree;
only raw statevector indices differ. Always compare <Z_q>, never raw amplitudes.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def build_vqc(n_qubits: int = 8, n_layers: int = 3, entangler: str = "ring"):
    """Returns (circuit, x_params, theta_params). theta is flat, ordered [layer, qubit, {RY, RZ}]."""
    x = ParameterVector("x", n_qubits)
    th = ParameterVector("theta", n_layers * n_qubits * 2)
    qc = QuantumCircuit(n_qubits)
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(x[q], q)
        for q in range(n_qubits):
            base = (layer * n_qubits + q) * 2
            qc.ry(th[base], q)
            qc.rz(th[base + 1], q)
        pairs = [(q, (q + 1) % n_qubits) for q in range(n_qubits)] if entangler == "ring" \
            else [(q, q + 1) for q in range(n_qubits - 1)]
        for c, t in pairs:
            qc.cx(c, t)
    return qc, x, th


def bind(qc, x_params, theta_params, x_values, theta_values):
    """theta_values may be a [L, n, 2] array (as in TorchVQC.theta) or flat."""
    theta_values = np.asarray(theta_values, dtype=float).reshape(-1)
    values = dict(zip(x_params, np.asarray(x_values, dtype=float)))
    values.update(zip(theta_params, theta_values))
    return qc.assign_parameters(values)


def expz_statevector(bound_circuit, n_out: int) -> np.ndarray:
    """Exact <Z_q> for q < n_out (no noise, no shots)."""
    from qiskit.quantum_info import SparsePauliOp, Statevector
    sv = Statevector.from_instruction(bound_circuit)
    n = bound_circuit.num_qubits
    return np.array([sv.expectation_value(SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], num_qubits=n)).real
                     for q in range(n_out)])
