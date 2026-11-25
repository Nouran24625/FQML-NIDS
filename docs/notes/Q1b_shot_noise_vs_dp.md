# Q-1b — Shot noise against DP noise in the VQC's training gradient

**Author:** Nouran · **Reviewer:** Kareem · **Date:** 24 Nov 2026 · **Input to:** E5, D13, PRIV-4
**Data:** `results/spikes/Q1_shots.json` (Q-1: Tesla T4, batch 64, 128 and 1,024 shots, synthetic inputs in [0, π]).

## 1. The question
A VQC trained on hardware never sees an exact gradient: every `<Z>` comes from S measurements (shots), so the
gradient carries **shot noise**. DP-SGD adds **Gaussian noise** on purpose. If the two are of the same size,
DP noise on a quantum model partly "comes for free"; if shot noise is much smaller, it can be ignored when
reasoning about privacy-trained VQCs. This note puts both on one scale.

## 2. The two noises, per coordinate of the batch gradient
**DP-SGD (D13).** Each per-sample gradient g_i is clipped to norm C, the B clipped gradients are summed, and
Gaussian noise N(0, σ²C² I) is added before dividing by B. Per coordinate of the averaged gradient:

  Var_DP = σ² C² / B²

**Shots (Q-1).** With parameter shift, each derivative of `<Z>` is (`<Z>`(θ+π/2) − `<Z>`(θ−π/2)) / 2, both
terms estimated from S shots. The per-sample gradient error is unbiased with per-coordinate variance κ / S,
where κ depends on the model's readout and loss. Summed over B independent samples and averaged:

  Var_shot = κ / (B S)

(For one readout with |∂L/∂`<Z>`| ≤ 1, each derivative has variance ≤ (1/S + 1/S)/4 = 1/(2S), so κ ≤ 1/2: the
design-note bound.)

**They are equal at**

  **S\* = κ B / (σ² C²)**

Below S\* shots, shot noise is larger than DP noise; above it, DP noise dominates.

## 3. κ measured (Q-1)
κ = (measured per-sample shot-noise variance per coordinate) × S. It is the same at 128 and 1,024 shots, which
confirms the 1/S law.

| Model | κ at 128 shots | κ at 1,024 shots | κ used | Per-sample gradient norm (exact), p50 |
|---|---|---|---|---|
| `vqc` (local readout, 64 params) | 1.313 | 1.302 | **1.31** | 1.34 |
| `vqc_lin` (linear readout, 120) | 0.750 | 0.746 | **0.75** | 1.47 |
| `vqc_noent` (no CNOTs, 120) | 0.652 | 0.668 | **0.66** | 3.10 |
| `vqc_binary` (50) | 0.507 | 0.494 | **0.50** | 0.56 |

The binary model sits exactly on the design bound (κ = 1/2). The 8-class `vqc` is above it, most likely
because its local readout multiplies each `<Z>` by an output weight starting at 2 and the loss gradient spreads
over eight correlated outputs; the bound assumed one output with |∂L/∂`<Z>`| ≤ 1. The measured κ, not the
bound, is what the tables below use.

## 4. Crossover S\* (no clipping: upper bounds)

**Batch 256 (D11 part 2):**

| Model | σ = 0.5, C = 1 | σ = 1, C = 1 | σ = 2, C = 1 | σ = 1, C = 0.1 |
|---|---|---|---|---|
| `vqc` | 1,339 | 335 | 84 | 33,479 |
| `vqc_lin` | 766 | 192 | 48 | 19,154 |
| `vqc_noent` | 676 | 169 | 42 | 16,901 |
| `vqc_binary` | 513 | 128 | 32 | 12,813 |

At batch 64 every value is 4× smaller (`vqc`, σ = 1, C = 1: 84 shots).

These are **upper bounds**: clipping shrinks a per-sample gradient, and its shot noise with it, whenever its
norm exceeds C. With C = 1 that is about half of `vqc`'s samples and almost all of `vqc_noent`'s (p50 norm 3.1),
so the true crossover is lower.

## 5. What it means
1. **At the D13 settings (B = 256, C = 1) and 1,024 shots, DP noise dominates.** Shot variance is 0.33 of the DP
   variance for `vqc` at σ = 1 (total noise standard deviation +15%), 0.19 for `vqc_lin` (+9%), 0.17 for
   `vqc_noent` (+8%), 0.13 for the binary model (+6%). Lower still after clipping.
2. **The clip norm decides.** S\* grows as 1/C²: at C = 0.1 shot noise dominates up to tens of thousands of
   shots. A privacy-trained VQC on hardware at a small clip norm would be limited by shots, not by DP.
3. **Shot noise is not a privacy mechanism here.** It is not calibrated to a sensitivity and no accountant covers
   it; it only adds to the noise the optimiser sees. It is never subtracted from the DP budget.
4. **Without DP, shot noise is the only noise.** At 1,024 shots the batch-gradient RMS error is 21% of the
   gradient for `vqc` (Q-1). Whether that costs accuracy is what E5's shot-trained subset measures.

## 6. Limits and the re-measurement
- κ and the gradient norms were measured on synthetic inputs and random labels. Both depend on the data, so
  this note is re-run on real rows once DATA-6 fixes the per-fold features:
  `python -m spikes.q1_shots --device cuda --batch 256` with real fold-0 training rows (an added `--parquet`
  option), and the tables above are updated in place.
- Read-out noise and gate errors of real hardware add to shot noise; they are not modelled (no noisy simulation
  in Phase 1).
