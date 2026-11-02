# FQML-NIDS — Frozen interfaces (Gate 0)

**Version:** 1.0.0 · **Frozen on:** 2 Nov 2026 · **Applies to:** `main` after merging `track-a/a3-d6-d7-circuit`
and `track-b/datasets-doc`.

Every track codes against what is written here. **After the freeze, any change to an item in this file
needs a pull request that (1) says "Interface change" in its title, (2) updates this file and its change
log, and (3) is approved by the owner of that section plus one other member.**

| Section | Owner |
|---|---|
| 1. Data files and splits | Ganna |
| 2. Models (quantum and classical) | Sheghaf (quantum), Ganna (classical) |
| 3. Federated loop and partitions | Kareem |
| 4. Privacy and post-quantum hooks | Nouran |
| 5. Results logging | Ganna |

---

## 1. Data files and splits (owner: Ganna)

### 1.1 Cleaned dataset file
- Path: `data/interim/<dataset>_clean.parquet` (CICIoT2023 today: `data/interim/ciciot2023_pcapcsv_clean.parquet`).
- Never committed. Its SHA-256 is recorded in `data/checksums.json`; check with
  `python -m fqml.data.checksums verify interim`.
- Columns:

| Column | Type | Meaning |
|---|---|---|
| feature columns | float | The dataset's standard features present in every source file (CICIoT2023: the names in `fqml.data.schema.CICIOT2023_FEATURES` that exist in all 310 CSVs, 37 today) |
| `native_label` | str | The dataset's own label (CICIoT2023: 34 labels) |
| `family` | str | Attack family from `fqml/data/label_maps/<dataset>.yaml` |
| `y` | int64 | Family index; **0 is always Benign** |
| `capture` | str | **Added in Phase 1 (council plan, item 1):** source capture file name. Used only for splitting and as a federated client key, **never as a feature** |

- Families (CICIoT2023, in index order): Benign, DDoS, DoS, Recon, Web, BruteForce, Spoofing, Mirai.
  For the VQC, family index = readout output index.
- Binary task: `y_binary = (y != 0)`.

### 1.2 Splits
- **Default from Phase 1: capture-disjoint split** (`fqml/data/split.py`, Phase 1). Whole capture files are
  held out for test, for every family with at least 2 captures. Families with 1 capture (CICIoT2023:
  BruteForce) are reported as "cannot be tested on an unseen capture".
- Order is fixed: **split first, then cap/balance the training side only.** The test set keeps the real
  class mix.
- The old random row split is kept only to measure leakage (Checkpoint 1, 9 Nov).

### 1.3 Preprocessing (fit on train only)
- Feature selection, clipping (1st/99th percentile) and min-max scaling are fitted on the training split
  only, then applied to validation and test.
- **Model inputs are float32 in [0, π].** The VQC needs this range; classical models use the same inputs
  so comparisons stay fair.
- Default feature counts: all features and top-8 by mutual information (8 = one per qubit).

---

## 2. Models (owners: Sheghaf quantum, Ganna classical)

### 2.1 Common contract
Every model is a `torch.nn.Module` with `forward(x) -> logits`:
- `x`: float tensor `[batch, n_features]`, values in [0, π].
- `logits`: `[batch, n_classes]`. **Binary models also return 2 logits**, so one cross-entropy path works
  for every model and task.
- The parameter count is `sum(p.numel() for p in model.parameters() if p.requires_grad)`
  (`fqml.models.vqc.n_params`).
- `forward` must be `torch.func.vmap`-safe (no Python branching on tensor values, no in-place ops on
  inputs), because the federated loop trains clients in one batched call (section 3).

### 2.2 VQC: circuit v2 (decisions D6 + D7)
- Classes: `fqml.models.vqc.VQC` (PennyLane, the reference), `fqml.quantum.torch_sim.TorchVQC` (training,
  D1) and `fqml.quantum.qiskit_circuit.build_vqc` (hardware, E6). **All three build the same circuit;**
  tests check PennyLane = TorchVQC and Qiskit = TorchVQC on every CI run.
- Constructor (same for `VQC` and `TorchVQC`):
  `(n_qubits=8, n_classes=2, n_layers=3, seed=0, entangler="chain", version=2)`.
- Each layer: `RY(x_q)` on every qubit, then `RZ(θ[l,q,1])`, `RY(θ[l,q,0])`, then CNOT chain
  `q → q+1` (no wrap-around). Layout lives in `fqml/quantum/layout.py`.
- Readout: `<Z>` of the **last** `n_out` qubits; `logits = scale · z + bias`.
  `n_out = 1` for binary (output `[0, logit]`), else `n_classes`. Requires `n_out ≤ n_qubits`.
- Weights: `theta [L, n, 2]` (index 0 = RY angle, 1 = RZ angle), `scale [n_out]`, `bias [n_out]`;
  same values for the same seed in every implementation.
- Parameter counts (n = 8, L = 3): **binary 50, 8 classes 64. Every parameter can affect the output**
  (0 dead parameters, D6).
- `version=1` and `entangler="ring"` stay available **only** to reproduce results from before 20 Oct.
- Per-sample weights: `TorchVQC.logits(x, theta, scale, bias)` accepts `theta [B, L, n, 2]`,
  `scale [B, n_out]` and `bias [B, n_out]`. This is what DP-SGD per-sample gradients use (D4).

### 2.3 Classical models
- `fqml.models.mlp.MatchedMLP(n_in, n_classes, target_params)`: one hidden layer whose width gives the
  parameter count closest to `target_params` (the VQC's count for the same task).
- Phase 1 adds (same contract where they are PyTorch models; scikit-learn / XGBoost models use the same
  data, split and seeds): logistic regression, large MLP, XGBoost, random forest, and a random-Fourier-feature
  model (council plan).

### 2.4 Centralised training
`fqml/train/loop.py`: `train(model, X, y, n_classes, epochs, batch_size, lr, seed)` with class-weighted
cross-entropy (weights ∝ 1/√frequency, mean 1) and Adam. The same loop trains every PyTorch model.

---

## 3. Federated loop and partitions (owner: Kareem)

### 3.1 Partitions (`fqml/fl/partition.py`)
- `iid_partition(n_rows, n_clients, seed)` and `dirichlet_partition(y, n_clients, alpha, seed, min_size=10)`
  return `list[np.ndarray]`: client k → sorted row indices into the **training** split.
- Fully determined by the seed. Saved with `save_partition(parts, path)` as `.npz` with keys
  `client_000`, `client_001`, …; loaded with `load_partition(path)`.
- Path: `data/partitions/<dataset>_<scheme>_n<N>_s<seed>.npz`, for example
  `data/partitions/ciciot2023_dir0.5_n50_s0.npz`. Not committed (derived from data).
- Phase 1 adds "captures as clients" (one client per capture file or group of captures).

### 3.2 Batched FedAvg (`fqml/fl/batched.py`, decision D3)
```python
run_fedavg(model, X, y, parts, *, rounds, clients_per_round, local_steps=50, batch_size=64,
           lr=0.01, class_weights=None, seed=0, device="cpu", mode="batched",
           X_val=None, y_val=None, on_round=None, dtype=torch.float32) -> FedResult
```
- Each round: sample `clients_per_round` clients without replacement → each trains `local_steps`
  mini-batches of `batch_size` from its own rows, with a **fresh Adam every round** → FedAvg weighted by
  client row counts.
- `mode="sequential"` is the one-client-at-a-time reference; `tests/test_batched_fedavg.py` checks both give
  the same global weights.
- `FedResult.params`: final global weights (`dict[name, tensor]`). `FedResult.history`: one dict per round
  with `round`, `clients`, `train_seconds`, `seconds`, and `val_macro_f1`, `val_accuracy` when validation
  data is passed.
- Defaults (Plan §8.1): N = 50 clients, 20 per round, 100 rounds, 50 local steps of 64, Dirichlet α = 0.5.
- **Where robust aggregators and attacks plug in (Phase 1–3):** they act on the stacked client weights
  (`dict[name, tensor[K, …]]`, flattened to a `[K, P]` matrix when needed) between local training and
  aggregation. An aggregator is a function `stacked, n_rows -> global params`, with the same signature as
  `fedavg`. An attack modifies the rows of malicious clients before aggregation.
- `evaluate()` currently returns macro-F1 and accuracy. **Phase 1 extends it** (council plan) to per-class
  F1, recall, false-positive rate and PR-AUC; the existing keys stay.

---

## 4. Privacy and post-quantum hooks (owner: Nouran)
- DP-SGD on the VQC uses per-sample weights (D4): each sample gets its own weight copy via
  `TorchVQC.logits(x, theta[B, …], …)`, one backward pass gives all per-sample gradients, then
  clip → sum per client → add noise. It runs inside the client's local step of the batched loop.
- Privacy accounting reports (ε, δ) per run; Nouran sets δ before the first DP run.
- Post-quantum channel (`fqml/pq`): ML-KEM-768 key exchange and ML-DSA-65 model signing (liboqs 0.16.0).
  Used in E7 with Flower. `python -m fqml.pq.smoke` must pass on every machine.

---

## 5. Results logging (owner: Ganna)
- **Run ID:** `E{n}-{dataset}-{model}-{variant}-s{seed}`, for example `E1-ciciot2023-vqc-dir0.5-s0`.
- **MLflow:** tracking URI `sqlite:////workspace/mlruns/mlflow.db`; one experiment per experiment number
  (`E0`, `E1`, …); spikes in `phase0-spikes`. UI on host port 5001.
- **Params logged for every run:** run ID, dataset, split (`capture` or `row`), features, model, entangler,
  circuit version, n_clients, clients_per_round, partition scheme and α, rounds, local_steps, batch_size,
  lr, seed, device, git commit hash.
- **Metrics:** per round `val_macro_f1`, `val_accuracy`, `round_seconds`; at the end test macro-F1,
  accuracy, per-class F1, recall, false-positive rate (benign flagged as attack), PR-AUC, total seconds.
- **Artifacts:** final weights (`.pt`), the config, the confusion matrix.
- One row per finished run is appended to `results/results_log.csv`. Figures and tables are made from that
  file by script, never by hand.
- Seeds: 0, 1, 2 for every reported run (5 seeds for E0 per the council plan). Run seed 0 first; seeds 1–2
  only after seed 0 looks right.

---

## Change log
| Version | Date | Change | Approved by |
|---|---|---|---|
| 1.0.0 | 2 Nov 2026 | First freeze: circuit v2 + chain (D6, D7), batched FedAvg (D3), capture column and capture-disjoint split announced for Phase 1 | Kareem, Sheghaf, Nouran, Ganna |
