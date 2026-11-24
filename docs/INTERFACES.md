# FQML-NIDS — Frozen interfaces

**Version:** 2.2.0 (see change log) · **First frozen:** 2 Nov 2026 (Gate 0) · **Amended:** 23 Nov 2026
(federated section 3, FED-1 / FED-3 / FED-7; earlier amendments in the change log and `docs/DECISIONS.md`)

Every track codes against what is written here. **Any change to an item in this file needs a pull request
that (1) says "Interface change" in its title, (2) updates this file and its change log, and (3) is approved
by the owner of that section plus one other member.**

| Section | Owner |
|---|---|
| 1. Data files, folds and preprocessing | Ganna |
| 2. Models (quantum and classical) | Nouran |
| 3. Federated loop and partitions | Kareem |
| 4. Privacy and post-quantum hooks | Nouran (privacy), Sheghaf (post-quantum channel) |
| 5. Results logging | Sheghaf |

Items marked **(Phase 1: TASK)** are agreed signatures whose code is written in that Phase 1 task. Everything
else is implemented and tested on `main`.

---

## 1. Data files, folds and preprocessing (owner: Ganna)

### 1.1 Cleaned dataset file (cleaner v1)
- Path: `data/interim/<dataset>_v1/<dataset>_clean_v1.parquet` (CICIoT2023:
  `data/interim/ciciot2023_v1/ciciot2023_clean_v1.parquet`), written by `fqml.data.clean_ciciot2023_v1`.
  One row group per source file. Never committed; its SHA-256 is recorded with
  `python -m fqml.data.checksums record interim/<dataset>_v1`.
- Cleaning report (committed): `results/data_reports/<dataset>_clean_v1.json`: counts after every step,
  duplicate and conflict rates, the settings used, and the `row_id` range of every attack type and file.
- **Row order is part of the interface:** native labels in label-map order, then files by number, then rows
  as they are in the file. Nothing is shuffled. Every attack type's rows are one contiguous `row_id` range.
- Columns:

| Column | Type | Meaning |
|---|---|---|
| `row_id` | int64 | Position of the row in the cleaned file, 0 … N−1. Fold manifests name rows by it |
| `file` | str | Source capture file name (e.g. `DDoS-ICMP_Flood3.pcap.csv`). **Never a feature** |
| `file_seq` | int32 | The file's number inside its attack type (no number = 0) |
| `row_index` | int64 | 0-based row of the source CSV (before any row was removed) |
| `native_label` | str | The dataset's own label (CICIoT2023: 34 labels) |
| `family` | str | Attack family from `fqml/data/label_maps/<dataset>.yaml` (primary map) |
| `y` | int64 | Family index; **0 is always Benign** |
| `dup_group` | int64 | Rows with identical feature values share one id (over all files and labels). Duplicates are **marked, not deleted** |
| `conflict` | int8 | 0 = all rows with these feature values have one native label; 1 = several attack types of one family; 2 = several families |
| `sample_rate` | float32 | Share of this attack type's rows that were kept (1.0 = all). Metrics weight a row by `1 / sample_rate` |
| `ts` | float64 | Only if every source CSV has it: the capture timestamp, an order key. **Never a feature** |
| feature columns | float32 | The dataset's standard features present in every source file (`fqml.data.schema.CICIOT2023_FEATURES`) |

- Rows with NaN or ±inf are removed (counted per file). Nothing else is removed unless decision D8 chooses
  `--conflicts drop-cross-family`. The sampling cap (`--per-label-cap`) is also decision D8.
- Families (CICIoT2023, in index order): Benign, DDoS, DoS, Recon, Web, BruteForce, Spoofing, Mirai.
  For the VQC, family index = readout output index. Binary task: `y_binary = (y != 0)`.
- **Label-map variants.** The 8 families are the primary task. `load_label_map(name, variant)` and
  `family_index(native_label, name, variant)` give a variant's classes from `native_label`; the file itself
  always stores the primary `family` and `y`. Variant `c7m` (CICIoT2023): DDoS and DoS merged, 7 classes
  (Benign, DDoS_DoS, Recon, Web, BruteForce, Spoofing, Mirai). Whether it is reported is decision D17.
- The Phase 0 file `data/interim/ciciot2023_pcapcsv_clean.parquet` (v0) is **retired**: it may not be used
  for any result (see `docs/DECISIONS.md`, Phase 0 amendments).

### 1.2 Evaluation protocols and fold manifests
- Three protocols (decision D8a sets their details):
  - **A, time-blocked (primary):** each attack type's rows, in recorded order, cut into 5 blocks; fold k
    tests on block k, validates on block k+1 (wrapping), trains on the rest, with a gap of dropped rows at
    every boundary between blocks of different roles. The benign capture is handled the same way.
  - **B, unseen attack type:** whole attack types held out inside a family.
  - **C, independent dataset:** train on CICIoT2023, test on CICIoMT2024 (shared classes; needs D16).
  - `R`, the old random row split, exists only to measure leakage (Checkpoint 1). Never a reported result.
- **Fold manifest:** `data/folds/<dataset>_<protocol>.json` (committed), format `fqml-folds/1`
  (`fqml/data/folds.py`). For every fold it lists the train, validation and test rows as `row_id` ranges
  `[start, stop)`; it also records the gap used for every attack type and the cleaning report it was built
  from. `load_manifest(path)`, `fold_rows(manifest, fold, role)` → sorted `row_id` array.
- **Every manifest must pass `check_manifest`:** no row has two roles in a fold; rows of different roles
  inside an attack type are at least the gap apart; no row is a test row in two folds; for protocol A every
  row is a test row exactly once.
- Protocol A is built with `python -m fqml.data.folds build-a --report <cleaning report> --gap <rows>`.
  Builders for B and C: **(Phase 1: DATA-3, DATA-8b)**.
- Order is fixed: **split first, then balance or sub-sample the training side only.** Test blocks keep the
  class mix of the cleaned file.
- **Twins.** A test row whose `dup_group` also occurs in that fold's training rows is a twin
  (`fqml.data.folds.twin_mask`). Scores are reported on all test rows and again without twins.

### 1.3 Preprocessing (fitted per fold, on that fold's training rows only)
- Feature selection, clipping (1st/99th percentile) and min-max scaling are fitted on the training rows of
  one fold, then applied to that fold's validation and test rows; the fitted settings are saved per fold
  **(Phase 1: DATA-6)**.
- **Model inputs are float32 in [0, π].** The VQC needs this range; classical models use the same inputs
  so comparisons stay fair.
- Default feature counts: all features and top-8 by mutual information (8 = one per qubit).
- `file`, `file_seq`, `row_index`, `row_id`, `dup_group`, `conflict`, `sample_rate` and `ts` are never model
  inputs. Whether `IAT` is excluded is decision D9.

---

## 2. Models (owner: Nouran)

### 2.1 Common contract
Every model is a `torch.nn.Module` with `forward(x) -> logits`:
- `x`: float tensor `[batch, n_features]`, values in [0, π].
- `logits`: `[batch, n_classes]`. **Binary models also return 2 logits**, so one cross-entropy path works
  for every model and task.
- The parameter count is `sum(p.numel() for p in model.parameters() if p.requires_grad)`
  (`fqml.models.vqc.n_params`). **Results log this actual count**, never a target size.
- `forward` must be `torch.func.vmap`-safe (no Python branching on tensor values, no in-place ops on
  inputs), because the federated loop trains clients in one batched call (section 3).

### 2.2 VQC: circuit v2 (decisions D6 + D7)
- Classes: `fqml.models.vqc.VQC` (PennyLane, the reference), `fqml.quantum.torch_sim.TorchVQC` (training,
  D1) and `fqml.quantum.qiskit_circuit.build_vqc` (hardware, E6). **All three build the same circuit;**
  tests check PennyLane = TorchVQC and Qiskit = TorchVQC on every CI run.
- Constructor (same for `VQC` and `TorchVQC`):
  `(n_qubits=8, n_classes=2, n_layers=3, seed=0, entangler="chain", version=2, readout="local")`.
- Each layer: `RY(x_q)` on every qubit, then `RZ(θ[l,q,1])`, `RY(θ[l,q,0])`, then CNOT chain
  `q → q+1` (no wrap-around). Layout lives in `fqml/quantum/layout.py`.
- `n_out = 1` for binary (output `[0, logit]`), else `n_classes`. Requires `n_out ≤ n_qubits`.
- **Readout** (`readout`):
  - `"local"` (default, D6): output k reads ONE qubit, the k-th of the **last** `n_out` qubits;
    `logits = scale · z + bias`, `scale [n_out]`.
  - `"linear"`: every output is a trained weighted sum over **all** n qubits;
    `logits = z · scaleᵀ + bias`, `scale [n_out, n]`. It starts as the local circuit (weight 2.0 on the
    qubit the local readout would read, 0 elsewhere), so with the same seed both begin as the same model.
- Weights: `theta [L, n, 2]` (index 0 = RY angle, 1 = RZ angle), `scale` as above, `bias [n_out]`;
  same values for the same seed in every implementation.
- Parameter counts (n = 8, L = 3). **Every parameter can affect the output** (0 dead parameters, D6):

| | binary | 8 classes | 7 classes (`c7m`) |
|---|---|---|---|
| local readout | 50 | 64 | 62 |
| linear readout | 57 | 120 | 111 |

- `entangler` values: `"chain"` (default), `"ring"`, `"none"`.
  - `version=1` and `entangler="ring"` stay available **only** to reproduce results from before 20 Oct.
  - **`entangler="none"`** removes every CNOT: the entanglement-free control. Each qubit then evolves
    alone, so `<Z_q>` depends only on feature q. **With the local readout this control is invalid:** each
    output sees one feature (binary: 42 of the 48 angles never reach the output). The control is therefore
    always `entangler="none", readout="linear"` (model name `vqc_noent`), and it is compared with
    `entangler="chain", readout="linear"` (model name `vqc_lin`): the two differ only by the CNOT gates.
- Per-sample weights: `TorchVQC.logits(x, theta, scale, bias)` accepts `theta [B, L, n, 2]`, `bias [B, n_out]`,
  and `scale [B, n_out]` (local) or `[B, n_out, n]` (linear). This is what DP-SGD per-sample gradients use (D4).
- **Finite shots** (`TorchVQC` only, Q-1, `fqml/quantum/shots.py`): extra constructor arguments
  `shots=None, shot_grad="shift"`; `TorchVQC.logits(..., shots=...)` and `TorchVQC.expz(x, theta, shots=...)`
  override the model setting for one call. `shots=None` (default) is the exact simulation with backprop,
  unchanged. `shots=S`: every sample's circuit is measured S times (one bitstring per shot for all qubits)
  and `<Z>` is the average. Gradients for `theta`: `shot_grad="shift"` uses the parameter-shift rule with
  S shots for each of the 2 × 48 shifted circuits (what hardware training does); `shot_grad="exact"` keeps
  the exact gradient (noise in the outputs only). `scale` and `bias` get autograd gradients from the
  sampled `<Z>`; inputs `x` get no gradient. Per-sample weights work with shots. Optional attributes:
  `shot_generator` (a `torch.Generator` on the model's device) and `shot_max_rows` (memory cap).
  `TorchVQC.probs(x, theta)` returns the `[B, 2^n]` bitstring probabilities.

### 2.3 Classical models and the model size setting
- `fqml.models.mlp.MatchedMLP(n_in, n_classes, target_params, seed=None)`: one hidden layer (ReLU) whose
  width gives the parameter count closest to `target_params`. `target_params` is **the model size setting**.
- **MLP size ladder** (`fqml.models.sizes.size_ladder(n_in, n_classes, vqc_params)`): the VQC's own count
  (50 / 64), then 128, 256, 512, 1,000 and 3,000 parameters. Same model family at every rung; only the width
  changes. With 8 inputs the sizes reached are 46, 123, 255, 508, 1,003, 3,005 (binary) and 59, 127, 263,
  518, 994, 3,000 (8 classes).
- Model names in run IDs: `vqc` (chain, local readout: the main model), `vqc_lin` (chain, linear readout),
  `vqc_noent` (no CNOTs, linear readout: the entanglement-free control), `mlp_p<target>` (e.g. `mlp_p50`,
  `mlp_p128`), `rff`, `lr`, `xgb`, `rf`, `mlp_large`.
- **(Phase 1: CLS-1, Q-3)** logistic regression, MLP-large (2 × 256), XGBoost, random forest and the
  random-Fourier-feature model. PyTorch models follow 2.1; scikit-learn / XGBoost models use the same
  data, folds and seeds.

### 2.4 Centralised training
`fqml/train/loop.py`: `train(model, X, y, n_classes, epochs, batch_size, lr, seed)` with class-weighted
cross-entropy (weights ∝ 1/√frequency, mean 1) and Adam. The same loop trains every PyTorch model.
Validation data and early stopping: **(Phase 1: H-10)**.

---

## 3. Federated loop and partitions (owner: Kareem)

### 3.1 Partitions (`fqml/fl/partition.py`)
- Every partitioner returns `list[np.ndarray]`: client k → sorted row positions into the **training rows of
  one fold**. Fully determined by its seed.
- `iid_partition(n_rows, n_clients, seed)` and `dirichlet_partition(y, n_clients, alpha, seed, min_size=10)`.
- `type_partition(native_label, seed, *, benign="BenignTraffic")` (FED-3): one attack type per client (sorted
  by name; the most extreme skew), benign rows spread evenly over all clients.
- `sites_partition(native_label, n_sites, types_per_site, seed, *, benign="BenignTraffic", groups=None)`
  (FED-3): the **sites split**. Each client is a site that holds an equal share of the benign rows plus exactly
  `types_per_site` attack types; every attack type is held by at least one site; a type's rows are split evenly
  among the sites holding it. Returns `(partition, site_types)`, `site_types[k]` = sorted attack types of site k.
  With `groups` ({attack type: group}, e.g. its family) sites hold whole groups and `types_per_site` counts groups.
- Every partitioner gives each row to exactly one client (label attacks rely on this, `docs/ATTACKS.md`).
- Charts and per-client counts for one fold: `python -m experiments.fed3_partitions --fold 0`
  (`results/partitions/fed3_<protocol>_f<fold>.json`, figures beside it).
- Saved with `save_partition(parts, path)` as `.npz` with keys `client_000`, `client_001`, …; loaded with
  `load_partition(path)`. Path: `data/partitions/<dataset>_<protocol>_f<fold>_<scheme>_n<N>_s<seed>.npz`,
  for example `data/partitions/ciciot2023_A_f0_dir0.5_n50_s0.npz`. Not committed (derived from data).

### 3.2 Batched FedAvg (`fqml/fl/batched.py`, decision D3)
```python
run_fedavg(model, X, y, parts, *, rounds, clients_per_round, local_steps=50, batch_size=64,
           lr=0.01, class_weights=None, seed=0, device="cpu", mode="batched",
           X_val=None, y_val=None, on_round=None, dtype=torch.float32) -> FedResult
```
- Each round: sample `clients_per_round` clients without replacement → each trains `local_steps`
  mini-batches of `batch_size` from its own rows, with a **fresh Adam every round** → FedAvg weighted by
  client row counts.
- `mode="sequential"` is the one-client-at-a-time reference. `tests/test_batched_fedavg.py` (random data)
  and `tests/test_fedavg_real_data.py` (traffic-like data in CI; real CICIoT2023 rows when the cleaned file
  is on the machine) check both give the same global weights. `fqml.fl.equivalence.compare_modes` is the
  same check as a function; **the federated trial run calls it on its real data before round 1**
  (Phase 1: FED-4).
- `mode="local"` (FED-7): **local-only training.** Every client trains alone on its own rows, `local_steps`
  steps in every round (rounds × local_steps in total, as many as the global model gets), one Adam per client for
  the whole run, no aggregation. `clients_per_round` is ignored. `FedResult.params` is `None`;
  `FedResult.client_params` holds one model per client (`dict[name, tensor[N, …]]`). Reported with each client's
  recall on attack types it never held: `fqml.fl.unseen` (`held_types`, `unseen_scores`, `per_client_unseen`).
- `FedResult.params`: final global weights (`dict[name, tensor]`). `FedResult.history`: one dict per round
  with `round`, `clients`, `train_seconds`, `seconds`, and `val_macro_f1`, `val_accuracy` when validation
  data is passed. `FedResult.final_val`: with `X_val` and `families=` (FED-1), the full validation scores at
  the end (one dict per client in `mode="local"`).
- Defaults (Plan §8.1): N = 50 clients, 20 per round, 100 rounds, 50 local steps of 64, Dirichlet α = 0.5.
- **Where robust aggregators and attacks plug in (Phase 1–3):** they act on the stacked client weights
  (`dict[name, tensor[K, …]]`, flattened to a `[K, P]` matrix when needed) between local training and
  aggregation. An aggregator is a function `stacked, n_rows -> global params`, with the same signature as
  `fedavg`. An attack modifies the rows of malicious clients before aggregation. Every robust aggregator is
  measured with zero attackers first.
- `evaluate(model, params, X, y, batch_size=8192, *, families=None)` returns `macro_f1` and `accuracy` as
  before; with `families` (FED-1) it adds `multiclass` and `binary` from the shared metrics module
  (`fqml.eval.metrics`, EVAL-1: per-class precision / recall / F1, confusion matrix, ROC-AUC, PR-AUC, F1 and
  false-positive rate at the default decision). Fixed-FPR thresholds are set on validation by the runner.
  `predict_proba(model, params, X)` gives the probabilities the runner passes to `fqml.eval.metrics.evaluate`.
- **Attacks** (FED-2, `fqml/attacks/poison.py`, `docs/ATTACKS.md`): label flipping and class suppression are
  label attacks; `poison_labels(y, parts, attack, ...)` returns the training labels passed to `run_fedavg` in
  place of `y`.

---

## 4. Privacy and post-quantum hooks (owners: Nouran privacy, Sheghaf post-quantum)
- DP-SGD on the VQC uses per-sample weights (D4): each sample gets its own weight copy via
  `TorchVQC.logits(x, theta[B, …], …)`, one backward pass gives all per-sample gradients, then
  clip → sum per client → add noise. It runs inside the client's local step of the batched loop.
- Privacy accounting reports (ε, δ) per run; Nouran sets δ before the first DP run (decision D13, which
  also states the privacy unit: one flow record in centralised runs; a device or a site in federated runs).
- Post-quantum channel (`fqml/pq`, D15): ML-KEM-768 key exchange and ML-DSA-65 model signing (liboqs 0.16.0),
  hybrid with X25519 by default. Used in E7. `python -m fqml.pq.smoke` must pass on every machine.
  Guide: `docs/PQ_CHANNEL.md`.
  - **Settings:** `configs/pq/<name>.yaml`, loaded with `fqml.pq.config.load_config(name)`: `default` (above),
    `cnsa2` (ML-KEM-1024 + ML-DSA-87, SHA-384), and the E7 baselines `classical` (X25519 + Ed25519) and `none`.
    Server and sites must load the same file; the handshake refuses different settings.
  - **Keys:** `python -m fqml.pq.keys gen server site-01 ... --config <name>` writes `keys/<name>/<party>.key`
    (secret; `keys/` and `*.key` are git-ignored) and `<party>.pub` (pinned by the other side).
    Each settings file has its own key folder (an ML-DSA-65 key cannot sign for `cnsa2`).
  - **API for E7:** server `Listener(cfg, me, trust, host, port).accept() -> Link`; site
    `connect(cfg, me, server_pub, host, port) -> Link`; `load_server(cfg, dir)` / `load_site(cfg, dir, name)`
    read the key folder. `Link.send_model(weights, round=r, purpose="global" | "update" | "final", meta={...})`
    and `Link.recv_model(round=r, purpose=...) -> ModelMessage(weights, round, purpose, signer, meta)`.
    `weights` is a dict of named tensors or arrays (like `FedResult.params`) or a list of arrays (Flower);
    numpy arrays come back (`fqml.pq.serialize.to_torch` converts). `Link.stats`: handshake time and bytes
    on the wire, for E7.
  - **Errors:** every failed check raises a `fqml.pq.primitives.SecurityError` (`HandshakeError`,
    `IntegrityError`, `SignatureError`) and closes the link. Never catch one to carry on with the data.

---

## 5. Results logging (owner: Sheghaf)
- **Run ID:** `E{n}-{dataset}-{model}-{variant}-{protocol}-f{fold}-s{seed}`, for example
  `E0-ciciot2023-vqc-c8.top8-A-f0-s0`. Built and parsed only with `fqml.utils.run_id`
  (`make_run_id`, `parse_run_id`, `setting_runs`). No part contains `-`.
  - `variant`: the rest of the setting, joined with dots: task (`c2` binary, `c8` eight families, `c7m`
    DDoS and DoS merged), features (`top8`, `all`), federated split (`iid`, `dir0.5`, `sites`), privacy
    (`eps8`), …
  - `protocol`: `A`, `B`, `C`, or `R` (old random split, leakage reference only).
  - `fold`: 0–4 for A; the index of the held-out group for B; 0 for C and R.
- **Folds and seeds: every reported setting runs on all of its folds × seeds 0, 1, 2** (protocol A:
  5 × 3 = 15 runs). The fold is the unit of comparison: seeds are averaged inside a fold first, and models
  are compared pairwise across folds. Fold 0, seed 0 runs first; the rest only after it looks right.
- **MLflow:** tracking URI `sqlite:////workspace/mlruns/mlflow.db`; one experiment per experiment number
  (`E0`, `E1`, …); spikes in `phase0-spikes`. UI on host port 5001.
- **Params logged for every run:** run ID, dataset, cleaned-file fingerprint, protocol, fold, features,
  task and label-map variant, model, `target_params` and the actual parameter count, entangler, circuit
  version, n_clients, clients_per_round, partition scheme and α, rounds, local_steps, batch_size, lr, seed,
  device, git commit hash.
- **Metrics:** per round `val_macro_f1`, `val_accuracy`, `round_seconds`; at the end test macro-F1,
  accuracy, per-class F1, recall, false-positive rate (benign flagged as attack), PR-AUC, total seconds;
  the same scores on test rows with no twin in training (1.2). Rows are weighted by `1 / sample_rate`.
- **Artifacts:** final weights (`.pt`), the config, the confusion matrix.
- One result file per finished run; `results/results_log.csv` is built from them by script **(Phase 1:
  EXP-0, EVAL-3)**. Figures and tables are made from that file by script, never by hand.

---

## Change log
| Version | Date | Change | Approved by |
|---|---|---|---|
| 1.0.0 | 2 Nov 2026 | First freeze: circuit v2 + chain (D6, D7), batched FedAvg (D3), capture column and capture-disjoint split announced for Phase 1 | Kareem, Sheghaf, Nouran, Ganna |
| 1.1.0 | 5 Nov 2026 | Section 4: post-quantum channel API (`fqml/pq`, D15): settings files, key files, `Link` API for E7, error types. The frozen choice (ML-KEM-768, ML-DSA-65, liboqs 0.16.0) is unchanged and is the default | Nouran, Sheghaf (PR #12) |
| 2.0.0 | 16 Nov 2026 | Phase 0 amendments. Section owners follow the new areas. **1:** cleaner v1 file (`row_id`, `file`, `file_seq`, `row_index`, `dup_group`, `conflict`, `sample_rate`, optional `ts`; fixed row order; duplicates marked, not deleted), v0 file retired, label-map variants (`c7m`), protocols A / B / C and fold manifests (`fqml-folds/1`) replace the capture-disjoint split, preprocessing fitted per fold. **2:** `entangler="none"`, `readout="linear"` (the control is always `none` + `linear`, compared with `chain` + `linear`), model size setting and MLP size ladder. **3:** sites split, one-type-per-client split and local-only mode (signatures), partition path carries protocol and fold, real-data FedAvg equivalence check. **5:** run ID carries protocol and fold; 5 folds × 3 seeds replaces "seeds 0, 1, 2 on one split" | Pending: Ganna, Kareem, Sheghaf, Nouran (this PR) |
| 2.2.0 | 23 Nov 2026 | Section 3 (Kareem): `type_partition`, `sites_partition` (+ optional `groups`) implemented (FED-3); `mode="local"`, `FedResult.client_params`, `fqml.fl.unseen` (FED-7); `evaluate(..., families=)`, `predict_proba`, `FedResult.final_val`, `run_fedavg(..., families=)` (FED-1); label attacks (FED-2). Existing calls unchanged | Pending: Ganna, Kareem, Sheghaf, Nouran (this PR) |
