"""SPIKE S3 - Can we get PER-SAMPLE gradients of the VQC for DP-SGD? (owner: Nouran)

Why    : DP-SGD clips the gradient of EACH training sample before adding noise, so we need
         one gradient per sample, not one per batch. Libraries (e.g. Opacus) do this for normal
         neural networks, but not for quantum circuits. Plan risk table: Medium / High.
Models : vqc (PennyLane), vqc_torchsim (our PyTorch simulator, decision D1), mlp_matched
Methods: A  micro-batches of 1   - one backward pass per sample (slow, always correct = reference)
         B  retain-graph         - one batched forward, then one backward per sample loss
         C  torch.func vmap      - PyTorch's automatic per-sample gradients (fastest IF supported)
         D  expanded weights     - every sample gets its own copy of the weights, one backward
                                   pass gives all per-sample gradients (torchsim only)
Check  : B and C must equal A (max absolute difference < 1e-5), otherwise they are wrong.
Pass   : at least one method is correct; we pick the fastest correct one for the VQC.
Output : table, results/spikes/S3_per_sample_grads.json, MLflow run in 'phase0-spikes'.

Run:  docker compose -f docker/compose.yml exec dev python -m spikes.s3_per_sample_grads
"""
import json
import platform
import time
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call, grad, vmap

from fqml.models.mlp import MatchedMLP
from fqml.models.vqc import VQC, n_params
from fqml.quantum.torch_sim import TorchVQC

N_QUBITS, N_LAYERS, N_CLASSES, BATCH = 8, 3, 8, 64
STEPS_PER_ROUND = 50                     # local DP-SGD steps per client round (plan)
loss_fn = nn.CrossEntropyLoss()


def flat(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def method_a_microbatch(model, x, y):
    out = []
    for i in range(len(x)):
        model.zero_grad()
        loss_fn(model(x[i:i + 1]), y[i:i + 1]).backward()
        out.append(flat([p.grad.detach().clone() for p in model.parameters()]))
    return torch.stack(out)


def method_b_retain_graph(model, x, y):
    per_sample = nn.CrossEntropyLoss(reduction="none")(model(x), y)
    params = list(model.parameters())
    return torch.stack([flat(torch.autograd.grad(per_sample[i], params, retain_graph=True))
                        for i in range(len(x))])


def method_c_vmap(model, x, y):
    params = {k: v.detach() for k, v in model.named_parameters()}

    def loss_one(p, xi, yi):
        return loss_fn(functional_call(model, p, (xi.unsqueeze(0),)), yi.unsqueeze(0))

    g = vmap(grad(loss_one), in_dims=(None, 0, 0))(params, x, y)
    return torch.stack([flat([g[k][i] for k in params]) for i in range(len(x))])


def method_d_expanded(model, x, y):
    if not hasattr(model, "logits"):
        raise NotImplementedError("only for the PyTorch simulator")
    B = len(x)
    th = model.theta.detach().expand(B, *model.theta.shape).clone().requires_grad_(True)
    sc = model.scale.detach().expand(B, -1).clone().requires_grad_(True)
    bi = model.bias.detach().expand(B, -1).clone().requires_grad_(True)
    nn.CrossEntropyLoss(reduction="sum")(model.logits(x, th, sc, bi), y).backward()
    return torch.cat([th.grad.reshape(B, -1), sc.grad, bi.grad], dim=1)   # same order as parameters()


def normal_batch_step(model, x, y):
    model.zero_grad()
    loss_fn(model(x), y).backward()


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def main():
    torch.set_num_threads(1)                 # decision D2: 1 thread per run
    torch.manual_seed(0)
    x = torch.rand(BATCH, N_QUBITS) * torch.pi
    y = torch.randint(0, N_CLASSES, (BATCH,))
    vqc = VQC(n_qubits=N_QUBITS, n_classes=N_CLASSES, n_layers=N_LAYERS)
    models = {"vqc": vqc,
              "vqc_torchsim": TorchVQC(n_qubits=N_QUBITS, n_classes=N_CLASSES, n_layers=N_LAYERS),
              "mlp_matched": MatchedMLP(N_QUBITS, N_CLASSES, target_params=n_params(vqc))}
    methods = {"A_microbatch": method_a_microbatch, "B_retain_graph": method_b_retain_graph,
               "C_vmap": method_c_vmap, "D_expanded": method_d_expanded}

    rows = []
    print(f"torch {torch.__version__} | batch {BATCH} | {N_QUBITS} qubits, {N_LAYERS} layers, "
          f"{N_CLASSES} classes\n")
    print(f"{'model':12s} {'method':16s} {'correct?':10s} {'max diff vs A':>14s} "
          f"{'s / batch':>10s} {'s / DP round':>13s}")
    for mname, model in models.items():
        _, t_normal = timed(normal_batch_step, model, x, y)
        reference = None
        for meth, fn in methods.items():
            row = {"model": mname, "method": meth, "normal_batch_step_s": round(t_normal, 4)}
            try:
                G, sec = timed(fn, model, x, y)
                if reference is None:
                    reference = G
                diff = float((G - reference).abs().max())
                row.update(correct=diff < 1e-5, max_abs_diff_vs_A=diff, s_per_batch=round(sec, 4),
                           s_per_dp_round=round(STEPS_PER_ROUND * sec, 2),
                           slowdown_vs_normal=round(sec / t_normal, 1),
                           grad_shape=list(G.shape))
                print(f"{mname:12s} {meth:16s} {'yes' if row['correct'] else 'NO':10s} "
                      f"{diff:14.2e} {sec:10.3f} {row['s_per_dp_round']:13.1f}")
            except NotImplementedError:
                continue
            except Exception as e:
                row.update(correct=False, error=f"{type(e).__name__}: {str(e)[:160]}")
                print(f"{mname:12s} {meth:16s} {'FAILED':10s}  {row['error'][:70]}")
            rows.append(row)

    verdict = {}
    for q in ("vqc", "vqc_torchsim"):
        ok = [r for r in rows if r["model"] == q and r.get("correct")]
        best = min(ok, key=lambda r: r["s_per_batch"]) if ok else None
        verdict[q] = {"method": best and best["method"],
                      "s_per_batch": best and best["s_per_batch"],
                      "s_per_dp_client_round": best and best["s_per_dp_round"],
                      "slowdown_vs_normal_training": best and best["slowdown_vs_normal"]}
    verdict["risk_closed"] = any(v["method"] for k, v in verdict.items() if k != "risk_closed")
    print("\nVERDICT:", verdict)

    out = Path("results/spikes/S3_per_sample_grads.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"machine": platform.platform(), "torch": torch.__version__,
                               "batch": BATCH, "rows": rows, "verdict": verdict}, indent=2))
    print(f"saved {out}")
    try:
        import mlflow
        mlflow.set_experiment("phase0-spikes")
        with mlflow.start_run(run_name=f"S3-{platform.node()}"):
            mlflow.log_params({"batch": BATCH, "vqc_method": verdict["vqc"]["method"],
                               "torchsim_method": verdict["vqc_torchsim"]["method"]})
            for r in rows:
                if r.get("correct"):
                    mlflow.log_metric(f"s_per_batch_{r['model']}_{r['method']}", r["s_per_batch"])
            mlflow.log_artifact(str(out))
        print("logged to MLflow experiment 'phase0-spikes'")
    except Exception as e:
        print(f"(MLflow logging skipped: {e})")


if __name__ == "__main__":
    main()
