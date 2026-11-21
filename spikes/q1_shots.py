"""Q-1 - Cost and noise of training the VQC with finite shots (owner: Nouran). Input to Q-1b and E5.

Question : how long does one training step (batch 64, forward + backward) take when every <Z> is
           estimated from S shots and gradients come from the parameter-shift rule, as on hardware,
           compared with exact simulation + backprop? How noisy are those gradients?
Setup    : synthetic inputs uniform in [0, pi] and random labels. Step time does not depend on the
           values; the gradient norms do, so the norm figures here are provisional and are re-measured
           on real rows once DATA-6 fixes the top-8 features (Q-1b).
Modes    : exact          - exact <Z>, backprop (how E0 trains now)
           shots_shift    - S-shot <Z>, parameter-shift gradients with S shots per shifted circuit
                            (what training on hardware would be; 97 circuits per sample)
           shots_exactgrad- S-shot <Z>, exact gradient (isolates noise in the outputs)
           infer          - S-shot forward only, no gradient (test-time shots)
Measures : seconds per step (median and 10-90% range over repeated steps); relative bias and relative
           RMS error of the shot gradient against the exact one; per-sample gradient norms (exact) and
           per-sample shot-gradient noise (inputs to Q-1b: shot noise vs DP noise).
Output   : table, results/spikes/Q1_shots.json

Run (Docker, CPU):   docker compose -f docker/compose.yml exec dev python -m spikes.q1_shots
Run (Colab/Kaggle T4): python -m spikes.q1_shots --device cuda
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from fqml.models.vqc import n_params
from fqml.quantum.torch_sim import TorchVQC

MODELS = {  # name: (n_classes, entangler, readout) - the E0 quantum models (PREREG_E0 §6)
    "vqc": (8, "chain", "local"),
    "vqc_lin": (8, "chain", "linear"),
    "vqc_noent": (8, "none", "linear"),
    "vqc_binary": (2, "chain", "local"),
}


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _time(fn, repeats, device, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        _sync(device)
        t = time.perf_counter()
        fn()
        _sync(device)
        ts.append(time.perf_counter() - t)
    ts.sort()
    return {"median_s": statistics.median(ts), "p10_s": ts[len(ts) // 10], "p90_s": ts[(9 * len(ts)) // 10]}


def _flat_grad(m):
    return torch.cat([p.grad.reshape(-1) for p in m.parameters()])


def _per_sample_grads(m, x, y, shots):
    """[B, n_params] per-sample gradients in one backward (per-sample copies of every parameter)."""
    B = x.shape[0]
    th = m.theta.detach().expand(B, *m.theta.shape).clone().requires_grad_(True)
    sc = m.scale.detach().expand(B, *m.scale.shape).clone().requires_grad_(True)
    bi = m.bias.detach().expand(B, *m.bias.shape).clone().requires_grad_(True)
    loss = nn.CrossEntropyLoss(reduction="sum")(m.logits(x, th, sc, bi, shots=shots), y)
    loss.backward()
    return torch.cat([th.grad.reshape(B, -1), sc.grad.reshape(B, -1), bi.grad.reshape(B, -1)], 1)


def _quantiles(v):
    q = torch.quantile(v.double().cpu(), torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64))
    return {"p50": q[0].item(), "p90": q[1].item(), "p99": q[2].item(), "max": v.max().item()}


def run_model(name, cfg, a, device):
    n_classes, entangler, readout = cfg
    m = TorchVQC(8, n_classes, 3, seed=0, entangler=entangler, readout=readout).to(device)
    m.shot_generator = torch.Generator(device=device).manual_seed(0)
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(a.batch, 8, generator=g) * torch.pi).to(device)
    y = torch.randint(0, n_classes, (a.batch,), generator=g).to(device)
    loss_fn = nn.CrossEntropyLoss()

    def step(shots, grad="shift"):
        m.shot_grad = grad

        def fn():
            m.zero_grad(set_to_none=True)
            loss_fn(m.logits(x, shots=shots), y).backward()
        return fn

    def infer(shots):
        def fn():
            with torch.no_grad():
                m.logits(x, shots=shots)
        return fn

    out = {"model": name, "n_classes": n_classes, "entangler": entangler, "readout": readout,
           "n_params": n_params(m), "n_angles": m.theta.numel(), "batch": a.batch, "modes": []}
    t_exact = _time(step(None), a.repeats, device)
    out["modes"].append({"mode": "exact", "shots": None, **t_exact, "x_exact": 1.0})

    step(None)()                                                        # exact gradient of the batch loss
    g_exact = _flat_grad(m).detach()
    ps_exact = _per_sample_grads(m, x, y, shots=None).detach()
    out["grad_norm_exact"] = g_exact.norm().item()
    out["per_sample_grad_norm_exact"] = _quantiles(ps_exact.norm(dim=1))

    for S in a.shots:
        for mode, grad in (("shots_shift", "shift"), ("shots_exactgrad", "exact")):
            t = _time(step(S, grad), a.repeats, device)
            draws = []
            for _ in range(a.grad_repeats):
                step(S, grad)()
                draws.append(_flat_grad(m).detach())
            d = torch.stack(draws)
            row = {"mode": mode, "shots": S, **t, "x_exact": t["median_s"] / t_exact["median_s"],
                   "grad_rel_bias": ((d.mean(0) - g_exact).norm() / g_exact.norm()).item(),
                   "grad_rel_rmse": (((d - g_exact) ** 2).sum(1).mean().sqrt() / g_exact.norm()).item()}
            if mode == "shots_shift":
                m.shot_grad = "shift"
                noise = torch.stack([(_per_sample_grads(m, x, y, shots=S).detach() - ps_exact).pow(2).sum(1)
                                     for _ in range(a.ps_repeats)])          # [R, B] squared error norms
                row["per_sample_shot_noise_sq_mean"] = noise.mean().item()
                row["per_sample_shot_noise_per_coord_var"] = noise.mean().item() / n_params(m)
            out["modes"].append(row)
        t = _time(infer(S), a.repeats, device)
        out["modes"].append({"mode": "infer", "shots": S, **t, "x_exact": t["median_s"] / t_exact["median_s"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shots", type=int, nargs="+", default=[128, 1024])
    ap.add_argument("--repeats", type=int, default=20, help="timed steps per setting (+3 warm-up)")
    ap.add_argument("--grad-repeats", type=int, default=30, help="shot gradients drawn per setting")
    ap.add_argument("--ps-repeats", type=int, default=10, help="per-sample shot gradients drawn per setting")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", default="results/spikes/Q1_shots.json")
    a = ap.parse_args()
    if a.device == "cpu":
        torch.set_num_threads(1)                                            # decision D2
    dev_name = torch.cuda.get_device_name(0) if a.device.startswith("cuda") else platform.processor()
    print(f"device {a.device} ({dev_name}) | torch {torch.__version__} | batch {a.batch} | shots {a.shots}\n")
    rows = [run_model(name, MODELS[name], a, a.device) for name in a.models]

    print(f"{'model':11s} {'mode':16s} {'shots':>6s} {'s/step':>9s} {'x exact':>8s} "
          f"{'grad bias':>10s} {'grad rmse':>10s}")
    for r in rows:
        for md in r["modes"]:
            print(f"{r['model']:11s} {md['mode']:16s} {str(md['shots'] or '-'):>6s} {md['median_s']:9.4f} "
                  f"{md['x_exact']:8.1f} {md.get('grad_rel_bias', float('nan')):10.3f} "
                  f"{md.get('grad_rel_rmse', float('nan')):10.3f}")
        q = r["per_sample_grad_norm_exact"]
        print(f"{'':11s} per-sample exact grad norm: p50 {q['p50']:.3f}  p90 {q['p90']:.3f}  "
              f"p99 {q['p99']:.3f}  max {q['max']:.3f}\n")

    result = {"spike": "Q1_shots", "machine": platform.platform(), "device": a.device, "device_name": dev_name,
              "torch": torch.__version__, "args": vars(a), "data": "synthetic: x ~ U[0, pi], random labels",
              "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
