# Thesis chapter outline — Quantum and classical models for leakage-aware intrusion detection (WRITE)

**Author:** Nouran · **Status:** outline, Phase 1 (24 Nov 2026) · Each section names the repository file or task its
content comes from, and whether that result exists yet. The chapter answers the Phase 1 question: on CICIoT2023,
tested without leakage, does a small variational quantum circuit beat classical models of the same size, and how far
is it from the best classical models?

## 1. Why this comparison is hard to do fairly
- Published quantum IDS results mostly use random row splits, include capture artifacts as features and compare the
  quantum model against weak or differently sized classical models (Q-5 table of published CICIoT2023 results,
  after 9 Dec).
- Three ways a comparison goes wrong, each with our measurement:
  - **Leakage from the split:** random rows inflate 8-class macro-F1 by about 5 points over protocol A
    (Checkpoint 1, `results/checkpoint1.json`); 67–73% of protocol A test rows have an exact twin in training
    (DATA-4).
  - **Shortcut features:** `IAT` (D9) and `Number`, the packet-window size; without `Number`, unseen attack types
    caught at 1% false alarms fall from 98% to 52% under protocol B (`results/checkpoint1_no_number.json`).
  - **Unequal rivals:** the size ladder and the RFF control (sections 4, 5).
- The pre-registration that fixes every rule before any result (`docs/PREREG_E0.md`, EXP-1).

## 2. The quantum model
- Circuit v2: angle encoding `RY(x_q)` re-uploaded in 3 layers, trainable `RZ`/`RY`, CNOT chain; local and linear
  readout; 64 / 120 parameters for 8 classes; no dead parameters (D6, D7, Phase 0 amendments).
- The entanglement-free control and why the local readout made it unfair (Q-2, `fqml/quantum/layout.py`).
- What the circuit can represent: a trigonometric polynomial of its inputs with whole-number frequencies up to 3 per
  feature, and the frequency multiplicities (1, 6, 15, 20, 15, 6, 1)/64 (Schuld et al. 2021; `fqml/models/rff.py`).
- Simulation: the PyTorch state-vector simulator and its match to PennyLane and Qiskit (D1, `tests/test_torch_sim.py`).

## 3. Shots: what training on hardware would cost
- Finite-shot measurement and parameter-shift gradients (Q-1, `fqml/quantum/shots.py`, `tests/test_shots.py`).
- Cost on a T4: same time as exact training at batch 64, about 3× at batch 256; noise falls as 1/√S
  (`results/spikes/Q1_shots.json`, `Q1_shots_b256.json`).
- Shot noise against DP noise on one scale: S* = κB/(σ²C²), κ measured per model (Q-1b,
  `docs/notes/Q1b_shot_noise_vs_dp.md`); re-measured on real rows after DATA-6.
- Does shot training cost accuracy? The shot-trained subset of E5 (`vqc`, `vqc_lin`, 5 folds) — after Gate 1 runs.

## 4. Classical rivals
- MatchedMLP at the VQC's parameter count, and the size ladder 64 → 3,000 parameters (`fqml/models/sizes.py`,
  CLS-1, `fqml/models/classical.py`).
- The RFF model of the circuit's own function class (Q-3); why beating it, not only an MLP, is the test of the
  quantum part (Landman et al. 2022).
- Reference models: logistic regression, MLP-large, XGBoost, Random Forest, at 5,000 and 50,000 rows per type.

## 5. Experimental protocol (E0)
- Data: 35 features, per-fold preprocessing to [0, π], top 8 by mutual information with a redundancy rule (DATA-6
  spec, `docs/specs/DATA-6_preprocessing.md`).
- Protocols A (primary), B (unseen types), C (CICIoMT2024) (D8a, DATA-8b).
- Training rules identical for every model: per-type sample, 1/√ class weights, 12 pre-committed tuning trials per
  fold, early stopping, batch 256 (D11 parts 1 and 2, `configs/tuning/`).
- Metrics: 8- and 7-class macro-F1 decide; recall at 1% and 0.1% FPR, per-type detection, no-twin scores and the
  conflict ceiling reported (D17, PREREG §2, EVAL-1).
- Statistics: corrected resampled t-test over folds, Holm over six comparisons (EVAL-2 spec,
  `docs/specs/EVAL-2_stats.md`).

## 6. Results (after Gate 1, 18 Dec)
- The six comparisons under A, with B and C beside them; outcome per PREREG §5.
- Classical-equivalent size of each VQC.
- Gap to XGBoost; which attack types fail for every model; no-twin scores.
- Entanglement: `vqc_lin` vs `vqc_noent`.
- Shot evaluation of every VQC (1,024 shots) and the shot-trained subset.

## 7. Small models under privacy (DP pilot)
- DP-SGD for every model, quantum included (PRIV-2, `fqml/privacy/dpsgd.py`), settings and privacy unit (D13).
- Correctness: equal to Opacus on the same MLP, VQC per-sample gradients equal a one-at-a-time loop, canary audit
  below the accountant (PRIV-3, `experiments/priv3_checks.py`, `tests/test_priv3.py`).
- The pilot: accuracy lost at ε ∈ {∞, 8, 4, 1} against model size, VQC vs size ladder vs RFF (PRIV-4, after 4 Dec).

## 8. Limits
- Simulation only (hardware inference is E6, later); one dataset until CICIoMT2024 is in; example-level privacy in
  Phase 1; no noise model of real hardware; 8 qubits = 8 features.
