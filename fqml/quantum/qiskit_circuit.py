"""The VQC as a Qiskit circuit, for hardware work (Spike S4, experiment E6). Owner: Sheghaf.

Same gates and order as fqml/models/vqc.py and fqml/quantum/torch_sim.py:
  per layer: RY(x_q) on every qubit, then the trainable rotations, then CNOTs.
  version 1: RY(theta[l,q,0]) RZ(theta[l,q,1])      version 2 (D6): RZ(theta[l,q,1]) RY(theta[l,q,0])
entangler = "ring"  -> CNOT q -> (q+1) % n
entangler = "chain" -> CNOT q -> q+1 for q < n-1 (no wrap-around; fits IBM chips with no SWAPs, S4)
Readout qubits: fqml.models.vqc.readout_qubits (version 1 = first n_out, version 2 = last n_out).

Qiskit numbers qubits little-endian (qubit 0 = rightmost bit) while PennyLane / TorchVQC treat qubit 0
as the leftmost. Gates act on the same qubit indices either way, so per-qubit <Z_q> values agree;
only raw statevector indices differ. Always compare <Z_q>, never raw amplitudes.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from fqml.quantum.layout import entangler_pairs


def build_vqc(n_qubits: int = 8, n_layers: int = 3, entangler: str = "chain", version: int = 2):
    """Returns (circuit, x_params, theta_params). theta is flat, ordered [layer, qubit, {RY, RZ}]."""
    x = ParameterVector("x", n_qubits)
    th = ParameterVector("theta", n_layers * n_qubits * 2)
    qc = QuantumCircuit(n_qubits)
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(x[q], q)
        for q in range(n_qubits):
            base = (layer * n_qubits + q) * 2
            if version == 1:
                qc.ry(th[base], q)
                qc.rz(th[base + 1], q)
            else:
                qc.rz(th[base + 1], q)
                qc.ry(th[base], q)
        for c, t in entangler_pairs(n_qubits, entangler):
            qc.cx(c, t)
    return qc, x, th


def bind(qc, x_params, theta_params, x_values, theta_values):
    """theta_values may be a [L, n, 2] array (as in TorchVQC.theta) or flat."""
    theta_values = np.asarray(theta_values, dtype=float).reshape(-1)
    values = dict(zip(x_params, np.asarray(x_values, dtype=float)))
    values.update(zip(theta_params, theta_values))
    return qc.assign_parameters(values)


def expz_statevector(bound_circuit, n_out: int | None = None, qubits: list[int] | None = None) -> np.ndarray:
    """Exact <Z_q> (no noise, no shots) for the given qubits, or for q < n_out."""
    from qiskit.quantum_info import SparsePauliOp, Statevector
    qubits = list(range(n_out)) if qubits is None else qubits
    sv = Statevector.from_instruction(bound_circuit)
    n = bound_circuit.num_qubits
    return np.array([sv.expectation_value(SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], num_qubits=n)).real
                     for q in qubits])
