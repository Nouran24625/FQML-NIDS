"""SPIKE S2 - Seconds per federated round, VQC vs MLP (owner: Kareem). Gate 0 item.

Question : how long does one FedAvg round take with the plan's defaults, and how many hours does
           the Run Plan need on this machine?
Setup    : synthetic data (round time does not depend on the values), 50 clients, Dirichlet 0.5,
           20 clients per round, 50 local steps x batch 64 (= 3,200 samples per client), 8 features,
           8 classes, validation on 5,000 rows each round.
Compares : TorchVQC (D1) and the matched MLP, each with the batched loop (D3) and, on CPU, the
           one-client-at-a-time loop (what Flower would do) as a reference.
Output   : table, results/spikes/S2_round_time.json, MLflow run in 'phase0-spikes'.

Run (laptop / Docker):  docker compose -f docker/compose.yml exec dev python -m spikes.s2_round_time
Run (Kaggle GPU)     :  python -m spikes.s2_round_time --device cuda
"""
import argparse
import json
import platform
import statistics
from pathlib import Path

import numpy as np
import torch

from fqml.fl.batched import run_fedavg
from fqml.fl.partition import dirichlet_partition
from fqml.models.mlp import MatchedMLP
from fqml.models.vqc import n_params
from fqml.quantum.torch_sim import TorchVQC

QUANTUM_FED_RUNS_PLAN_B = 400   # approx. quantum federated runs in Run Plan version B (results/run_matrix.csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rounds", type=int, default=3, help="timed rounds (+1 warm-up)")
    ap.add_argument("--clients", type=int, default=50)
    ap.add_argument("--per-round", type=int, default=20)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--skip-sequential", action="store_true")
    a = ap.parse_args()
    if a.device == "cpu":
        torch.set_num_threads(1)                               # decision D2
    rng = np.random.default_rng(0)
    n = a.clients * 4000
    X = (rng.random((n, 8)) * np.pi).astype(np.float32)
    y = rng.integers(0, 8, n)
    Xv, yv = X[:5000], y[:5000]
    parts = dirichlet_partition(y, a.clients, 0.5, seed=0)
    vqc = TorchVQC(8, 8, 3, seed=0)
    models = {"vqc_torchsim": vqc, "mlp_matched": MatchedMLP(8, 8, target_params=n_params(vqc))}
    modes = ["batched"] if (a.skip_sequential or a.device != "cpu") else ["batched", "sequential"]

    rows = []
    print(f"device {a.device} | torch {torch.__version__} | {a.per_round} clients/round x {a.steps} steps x "
          f"batch {a.batch}\n")
    print(f"{'model':13s} {'mode':11s} {'s / round':>10s} {'h / 100-round run':>18s} {'samples/s':>11s}")
    for mname, model in models.items():
        for mode in modes:
            res = run_fedavg(model, X, y, parts, rounds=a.rounds + 1, clients_per_round=a.per_round,
                             local_steps=a.steps, batch_size=a.batch, seed=0, device=a.device, mode=mode,
                             X_val=Xv, y_val=yv)
            secs = statistics.median(h["seconds"] for h in res.history[1:])     # skip warm-up round
            sps = a.per_round * a.steps * a.batch / statistics.median(h["train_seconds"] for h in res.history[1:])
            run_h = 100 * secs / 3600
            rows.append({"model": mname, "mode": mode, "s_per_round": round(secs, 3),
                         "h_per_100_round_run": round(run_h, 3), "train_samples_per_s": round(sps)})
            print(f"{mname:13s} {mode:11s} {secs:10.2f} {run_h:18.2f} {sps:11,.0f}")

    best = min((r for r in rows if r["model"] == "vqc_torchsim"), key=lambda r: r["s_per_round"])
    verdict = {
        "vqc_s_per_round": best["s_per_round"], "vqc_mode": best["mode"],
        "plan_B_quantum_hours_one_run_at_a_time": round(QUANTUM_FED_RUNS_PLAN_B * best["h_per_100_round_run"], 1),
    }
    print("\nVERDICT:", verdict)
    out = Path("results/spikes/S2_round_time.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"machine": platform.platform(), "device": a.device, "torch": torch.__version__,
                               "args": vars(a), "rows": rows, "verdict": verdict}, indent=2))
    print(f"saved {out}")
    try:
        import mlflow
        mlflow.set_experiment("phase0-spikes")
        with mlflow.start_run(run_name=f"S2-{a.device}-{platform.node()}"):
            mlflow.log_params({k: v for k, v in vars(a).items()})
            for r in rows:
                mlflow.log_metric(f"s_per_round_{r['model']}_{r['mode']}", r["s_per_round"])
            mlflow.log_artifact(str(out))
        print("logged to MLflow experiment 'phase0-spikes'")
    except Exception as e:
        print(f"(MLflow logging skipped: {e})")


if __name__ == "__main__":
    main()
