"""The Qiskit circuit must compute the same <Z_q> as TorchVQC (and so as the PennyLane VQC)."""
import numpy as np
import pytest
import torch

pytest.importorskip("qiskit")

from fqml.quantum.qiskit_circuit import bind, build_vqc, expz_statevector  # noqa: E402
from fqml.quantum.torch_sim import TorchVQC  # noqa: E402


@pytest.mark.parametrize("seed", [0, 1])
def test_qiskit_matches_torch_sim(seed):
    model = TorchVQC(8, 8, 3, seed=seed).double()
    x = torch.rand(8, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)) * torch.pi
    qc, xp, tp = build_vqc(8, 3, "ring")
    z_qiskit = expz_statevector(bind(qc, xp, tp, x.numpy(), model.theta.detach().numpy()), n_out=8)
    z_torch = model.expz(x.unsqueeze(0))[0].detach().numpy()
    assert np.allclose(z_qiskit, z_torch, atol=1e-8)


def test_chain_has_fewer_cnots():
    ring, _, _ = build_vqc(8, 3, "ring")
    chain, _, _ = build_vqc(8, 3, "chain")
    assert ring.count_ops()["cx"] == 24 and chain.count_ops()["cx"] == 21
