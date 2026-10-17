"""SPIKE S1 - Which quantum simulator trains our VQC fastest? (owner: Sheghaf)

Question : samples/second for one training step (forward + backward + update) of the
           8-qubit, 3-layer, 8-class VQC, on each simulator option:
             A) default.qubit + backprop  (whole batch at once = "broadcasting")
             B) lightning.qubit + adjoint (C++ simulator, one sample at a time)
           plus the parameter-matched MLP as the classical reference.
Pass rule: >= 500 samples/s  ->  one federated client round (50 steps x 64 samples)
           takes <= ~6.4 s, which keeps the plan's ~850 runs inside the compute budget.
Output   : a table on screen, results/spikes/S1_backend_throughput.json, and an MLflow run.

Run:  docker compose -f docker/compose.yml exec dev python -m spikes.s1_backend_throughput
"""
import json
import platform
import statistics
import time
from pathlib import Path

import pennylane as qml
import torch
from torch import nn

from fqml.models.mlp import MatchedMLP
from fqml.models.vqc import VQC, n_params

N_QUBITS, N_LAYERS, N_CLASSES = 8, 3, 8
BATCH, REPS = 64, 5
ROUND_SAMPLES = 50 * BATCH          # one federated client round in the plan


class LoopVQC(nn.Module):
    """Same circuit as fqml.models.vqc.VQC, but evaluated one sample at a time.
    Needed for simulators/gradient methods that do not take a whole batch at once."""

    def __init__(self, device_name, diff_method, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(torch.rand(N_LAYERS, N_QUBITS, 2, generator=g) * 2 * torch.pi)
        self.scale = nn.Parameter(torch.full((N_CLASSES,), 2.0))
        self.bias = nn.Parameter(torch.zeros(N_CLASSES))
        dev = qml.device(device_name, wires=N_QUBITS)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(x, theta):
            for layer in range(N_LAYERS):
                for q in range(N_QUBITS):
                    qml.RY(x[q], wires=q)
                for q in range(N_QUBITS):
                    qml.RY(theta[layer, q, 0], wires=q)
                    qml.RZ(theta[layer, q, 1], wires=q)
                for q in range(N_QUBITS):
                    qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
            return [qml.expval(qml.PauliZ(q)) for q in range(N_CLASSES)]

        self.circuit = circuit

    def forward(self, x):
        rows = [torch.stack(list(self.circuit(xi, self.theta))) for xi in x]
        return self.scale * torch.stack(rows).float() + self.bias


def bench(model, x, y):
    """Median seconds for one training step, after one warm-up step."""
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    def step():
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    step()
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        step()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    torch.manual_seed(0)
    x = torch.rand(BATCH, N_QUBITS) * torch.pi
    y = torch.randint(0, N_CLASSES, (BATCH,))
    vqc_params = n_params(VQC(n_qubits=N_QUBITS, n_classes=N_CLASSES, n_layers=N_LAYERS))

    candidates = [
        ("A  VQC default.qubit + backprop (batched)",
         lambda: VQC(n_qubits=N_QUBITS, n_classes=N_CLASSES, n_layers=N_LAYERS)),
        ("B  VQC lightning.qubit + adjoint (per sample)",
         lambda: LoopVQC("lightning.qubit", "adjoint")),
        ("   MLP-matched (classical reference)",
         lambda: MatchedMLP(N_QUBITS, N_CLASSES, target_params=vqc_params)),
    ]

    rows = []
    print(f"torch {torch.__version__} | pennylane {qml.__version__} | "
          f"CPU threads {torch.get_num_threads()} | batch {BATCH}\n")
    for name, build in candidates:
        try:
            model = build()
            sec = bench(model, x, y)
            sps = BATCH / sec
            row = {"option": name.strip(), "params": n_params(model), "s_per_step": round(sec, 4),
                   "samples_per_s": round(sps, 1), "s_per_client_round": round(ROUND_SAMPLES / sps, 2)}
        except Exception as e:  # keep going so one failure does not hide the others
            row = {"option": name.strip(), "error": f"{type(e).__name__}: {str(e)[:150]}"}
        rows.append(row)
        if "error" in row:
            print(f"{name:48s}  ERROR {row['error']}")
        else:
            print(f"{name:48s}  {row['samples_per_s']:>9,.1f} samples/s   "
                  f"{row['s_per_client_round']:>7.2f} s per client round")

    quantum = [r for r in rows if r["option"].startswith(("A", "B")) and "error" not in r]
    best = max(quantum, key=lambda r: r["samples_per_s"]) if quantum else None
    verdict = {
        "chosen": best and best["option"],
        "samples_per_s": best and best["samples_per_s"],
        "s_per_client_round": best and best["s_per_client_round"],
        "pass_500_samples_per_s": bool(best and best["samples_per_s"] >= 500),
    }
    print("\nVERDICT:", verdict)

    out = Path("results/spikes/S1_backend_throughput.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"machine": platform.platform(), "torch": torch.__version__,
                               "pennylane": qml.__version__, "threads": torch.get_num_threads(),
                               "batch": BATCH, "rows": rows, "verdict": verdict}, indent=2))
    print(f"saved {out}")

    try:
        import mlflow
        mlflow.set_experiment("phase0-spikes")
        with mlflow.start_run(run_name=f"S1-{platform.node()}"):
            mlflow.log_params({"machine": platform.platform()[:200], "batch": BATCH,
                               "chosen": verdict["chosen"]})
            for r in rows:
                if "error" not in r:
                    mlflow.log_metric(f"sps_{r['option'][:1] if r['option'][0] in 'AB' else 'MLP'}",
                                      r["samples_per_s"])
            mlflow.log_artifact(str(out))
        print("logged to MLflow experiment 'phase0-spikes'")
    except Exception as e:
        print(f"(MLflow logging skipped: {e})")


if __name__ == "__main__":
    main()
