# FQML-NIDS — Decision Log

One entry per decision that changes how the project is built or run.
Format: what was decided, why (with the measurement), who decided, and what would make us revisit it.

Numbering: D1–D7 are Phase 0. D8–D14 are reserved for the Phase 1 decisions listed under "Pending
decisions". D15 is the post-quantum channel (5 Nov). D16 and D17 are Phase 1 decisions added on 16 Nov.
A number is never reused.

---

## D1 — Quantum simulator for training: our PyTorch state-vector simulator

**Date:** 17 Oct 2026 · **Owner:** Sheghaf (quantum) · **Status:** accepted — matched to `fqml/models/vqc.py`, tests pass in CI (run #4)

**Decision.** Train the VQC with our own PyTorch simulator (`fqml/quantum/torch_sim.py`).
PennyLane stays as the reference implementation (a unit test checks both give the same outputs and
gradients) and for the shots / noisy / IBM-hardware paths.

**Evidence (Spike S1 + Kaggle tests, batch = training samples per step, 8 qubits, L = 3, 8 classes):**

| Backend | Machine | samples/s |
|---|---|---|
| PennyLane default.qubit + backprop, 8 threads | Nouran's Mac | 870 |
| PennyLane lightning.qubit + adjoint | Nouran's Mac | 118 |
| PennyLane default.qubit, 1 thread | Kaggle CPU | 760–1,150 |
| Our PyTorch simulator, 1 thread | Kaggle CPU | 2,150–2,580 |
| Our PyTorch simulator, batch 1,280 | Kaggle T4 GPU | 48,755 |
| Our PyTorch simulator, batch 4,096 | Kaggle T4 GPU | 71,646 |

Correctness check vs PennyLane: max output difference 4.0e-16, max gradient difference 1.1e-15.

**Revisit if:** the circuit changes in a way the simulator does not support (then add it, with a test).

---

## D2 — CPU runs: 1 thread per run, several runs in parallel

**Date:** 17 Oct 2026 · **Owner:** all · **Status:** accepted

**Decision.** On CPU machines, each run uses 1 thread (`OMP_NUM_THREADS=1`), and several runs share
the machine: **6 on the Mac, 8 on the Windows laptop, 4 on a Kaggle CPU session** (measured 17 Oct).

**Evidence (S1, PennyLane default.qubit):**
- Mac, 1 run × 8 threads: 870 samples/s total.
- Mac, 8 runs × 1 thread: 1,400–1,790 samples/s each; 6 finished, 2 ran out of Docker memory.
  Total ≈ 10,000 samples/s (≈ 11× more work).
- Kaggle CPU, 1 run × 2 threads: 547 samples/s. 4 runs × 1 thread: ~400 each, total ≈ 1,600 (≈ 3×).
- Mac with Docker memory raised to 8 GB: 8 copies all finish, but the total stays ~9,570–9,900 samples/s,
  the same as 6 copies, so the Mac is CPU-bound and 6 is enough.
- Windows laptop: 1 run 991 samples/s; 6 copies ~3,010 total; 8 copies ~3,640 total, so 8 is best.
  The Mac does about 2.7x the laptop's work.

**Why:** the 8-qubit circuit is too small to split across threads; threads spend more time coordinating
than computing.

**Revisit if:** a long (overnight) run shows the laptop slowing down from heat.

---

## D3 — Federated simulation for experiment sweeps: batched FedAvg loop (ACCEPTED)

**Date:** 17 Oct 2026 · **Owner:** Kareem (federation) · **Status:** accepted 2 Nov 2026 (all four members). Flower stays for E7 and the demo.

**Proposal.** For E1–E5 and E8, simulate federated rounds with our own loop that trains all sampled
clients of a round together in one GPU batch (each client keeps its own weights). Keep Flower for
E7 (post-quantum channel, real messages) and the defense demo.

**Why:** the GPU speed in D1 only appears when many samples run together (batch ≥ 1,280 =
20 clients × 64). Flower trains clients one by one, which would lose most of that gain.

**Effect on compute (plan version B: full rounds and clients, E5 shots only at evaluation):**
~950 h (Mac, old way) → ~87 h (Mac, D2) → ~12–17 h (one Kaggle T4 with D1 + D3).
*Note, 16 Nov 2026:* these hours were computed for the old run matrix (`results/run_matrix.csv`, never
committed), which is replaced by the E0 version 2 matrix (Phase 0 amendments, item 2). The speed
measurements below stand; the totals are to be recomputed for the new matrix.

**Evidence (Spike S2, 20 clients x 50 steps x batch 64 per round):** equivalence test batched =
sequential passes (6 of 6, PR #1). Laptop CPU, 1 thread: VQC 16.9 s per round batched vs 25.4 s
sequential; MLP 0.06 s. Kaggle T4 GPU: VQC 1.8 s per round.

**Open questions:** how Byzantine attacks, robust aggregators and DP-SGD plug into the batched loop;
check results match a Flower run on a small case.

*Amendment, 16 Nov 2026 (item 8):* the equivalence test used random numbers only. It now also runs on
traffic-like data in CI and on real CICIoT2023 rows wherever the cleaned file exists
(`fqml/fl/equivalence.py`, `tests/test_fedavg_real_data.py`); the federated trial run (FED-4) calls
`compare_modes` on its own data before round 1.

*Checked on real data, 18 Nov 2026:* `python -m fqml.fl.equivalence` on 20,007 rows of the cleaned
CICIoT2023 (top-8 features, IAT excluded; 3,293 exact duplicate rows in the sample; smallest client 10 rows
under Dirichlet 0.1): **8 of 8 cases pass** (VQC 8-class and binary, entanglement-free VQC, MLP; IID and
Dirichlet 0.1). Largest difference between batched and one-client-at-a-time training 1.1e-10 in the weights,
4.6e-11 in the logits (tolerance 1e-7). Record: `results/checks/fedavg_equivalence.json`. Phase 0
amendment 8 is closed.

---

## D4 — DP-SGD per-sample gradients: expanded weights on the PyTorch simulator

**Date:** 17 Oct 2026 · **Owner:** Nouran (privacy) · **Status:** accepted

**Decision.** For record-level DP-SGD on the VQC, compute per-sample gradients with method D:
every sample in the batch gets its own copy of the weights, and one backward pass returns all
per-sample gradients. Implemented in `TorchVQC.logits(x, theta, scale, bias)`; tested in
`tests/test_torch_sim.py::test_per_sample_gradients_in_one_pass`.

**Evidence (Spike S3, Nouran's Mac, 1 thread, batch 64, 8 classes; all methods match reference A):**

| Model | Method | s per batch | s per DP client round (50 steps) |
|---|---|---|---|
| PennyLane VQC | A micro-batches of 1 | 0.681 | 34.0 |
| PennyLane VQC | B retain-graph | 0.672 | 33.6 |
| PennyLane VQC | C vmap | 2.122 | 106.1 |
| PyTorch simulator | A micro-batches of 1 | 0.200 | 10.0 |
| PyTorch simulator | C vmap | 0.023 | 1.1 |
| **PyTorch simulator** | **D expanded weights** | **0.010** | **0.5** |
| MLP-matched | B retain-graph | 0.003 | 0.2 |

**Effect:** DP-SGD costs about the same as normal training (the earlier ×3 planning assumption is
dropped). Closes the Plan §14 risk "per-sample gradients for quantum models" (Medium / High).
Fallback if ever needed: method C (vmap), also correct.

---

## D5 — Library versions are frozen in a lock file

**Date:** 17 Oct 2026 · **Owner:** Nouran · **Status:** accepted

**Decision.** Every machine installs exact versions: `requirements/lock.txt` (139 libraries, from
`pip freeze` of the working image) plus `requirements/torch-version.txt` (torch 2.14.1). The Dockerfile
installs only from these and ends with `pip check`. `requirements.txt` stays as the short human list.
On Kaggle: `pip install -c requirements/lock.txt <packages>` keeps Kaggle's GPU torch and pins the rest.

**Why:** a fresh install picks the newest versions of the day; Kaggle already had torch 2.11 while
Docker had 2.14. Identical versions are a Gate 0 requirement.

**Evidence:** rebuilt image: 14 tests pass, 1 skipped; PQ smoke test pass; CI green (run #6); same image
builds on the Mac and the Windows laptop.

**How to change a version or add a library (always in a pull request):**
1. Add it to `requirements.txt`.
2. `docker compose -f docker/compose.yml exec dev pip install <library>`
3. Re-freeze: `pip freeze --exclude-editable | grep -viE '^(torch|liboqs-python|nvidia-|triton)' | sort`
   inside the container, written to `requirements/lock.txt`.
4. Rebuild the image, run `pytest`, open the pull request; CI must be green.

---

## D6 — VQC circuit version 2: no dead parameters

**Date:** 20 Oct 2026 · **Owners:** Sheghaf (quantum) + Kareem · **Status:** accepted 2 Nov 2026 (Sheghaf approved the PR)

**Problem.** In the original circuit (version 1: RY(x), RY(θ0), RZ(θ1), then CNOTs) the last layer's RZ
angles are followed only by CNOTs and the Z readout, so they can never change the output. Measured
(gradient = 0 for all inputs): 8 of 48 angles dead for 8 classes; 9 for binary (also the last RY on the
readout qubit). So the "parameter-matched" MLP was matched against parameters that never train.

**Decision.** Circuit version 2: per layer RY(x), **RZ(θ1), RY(θ0)** (RZ between the two RYs), then
CNOTs; the readout uses the **last** n_out qubits (binary: qubit 7). Same parameter count (50 binary,
64 for 8 classes), so the matched MLP and every interface stay the same. Implemented in `vqc.py`,
`torch_sim.py`, `qiskit_circuit.py` (shared layout in `fqml/quantum/layout.py`); version 1 kept only to
reproduce old runs.

**Evidence.** `tests/test_torch_sim.py::test_dead_parameters`: version 1 = 8 (8-class) / 9 (binary)
dead angles; version 2 = **0** for ring and chain. PennyLane = PyTorch simulator = Qiskit for every
variant (tests). Accuracy: see D7 (version 2 is equal or better). Speed: ~2,730 vs ~2,310 samples/s
on 1 CPU thread (Spike S5).

---

## D7 — Entangler: CNOT chain instead of ring; transpile at optimization level 3

**Date:** 20 Oct 2026 · **Owners:** Sheghaf + Kareem · **Status:** accepted 2 Nov 2026 (Sheghaf approved the PR)

**Decision.** Use a CNOT **chain** (q → q+1, no 7 → 0 link) instead of the ring. For hardware (E6),
transpile with Qiskit **optimization level 3** and prefer IBM Heron backends (Marrakesh, Kingston, Fez).

**Evidence (Spike S4, IBM fake backends, 3 layers, best of 3 transpiler seeds):**

| | Ring | Chain |
|---|---|---|
| 2-qubit gates on chip (opt 1–3) | 60 (24 + 36 added for routing) | **21 (0 added)** |
| Depth (opt 2–3) | ~151 | **~43** |
| Est. success, Heron (Fez / Marrakesh / Kingston), opt 3 | 0.77 / 0.82 / 0.82 | **0.89 / 0.92 / 0.88** |
| Est. success, Eagle (Brisbane / Sherbrooke), opt 3 | 0.55 / 0.57 | **0.78 / 0.79** |

IBM heavy-hex chips have no 8-qubit loop, so the ring's closing CNOT needs SWAPs; the chain maps
directly.

**Tuna-17 (Quantum Inspire, QuTech; wiring read from the SDK on 20 Oct 2026: 17 qubits, 24 couplers,
native CZ + RX/RY/RZ; saved in `results/spikes/tuna17_target.json`):** both layouts fit with **0 extra
2-qubit gates** at optimization level 1-3 (Tuna-17's wiring contains 8-qubit loops, unlike IBM's).
Chain: 21 CZ, depth 30-35. Ring: 24 CZ, depth 56-62. No calibration data is published, so no success
estimate. The chain is therefore the better choice on both providers. Optimization levels 0–1 are worse (Sherbrooke levels 0–1 picked a qubit marked broken in the
calibration: est. success 0.000), levels 2 and 3 give the same result.

**Evidence (Spike S5, accuracy, centralised CICIoT2023, 100k train / 20k test, 8 MI-selected features,
1,000 steps × 64, 3 seeds, macro-F1):**

| Task | Ring v1 (old) | Ring v2 | **Chain v2 (new)** |
|---|---|---|---|
| 8-class | 0.429 ± 0.005 | 0.443 ± 0.010 | **0.465 ± 0.013** |
| Binary | 0.758 ± 0.012 | 0.759 ± 0.006 | **0.762 ± 0.018** |

The new circuit is not worse on either task (8-class: +3.6 points). The chain must read the last qubit:
with qubit 0 as readout, 37 of 48 angles would be dead (qubit 0 is only ever a CNOT control).

> **S5 scores are NOT VALID as model performance (16 Nov 2026, Phase 0 amendments, item 2).** They may not
> be quoted in the thesis, the paper or any comparison. What S5 still supports is only the *relative*
> statement above (same data and split for all three circuits), and Phase 1 re-measures it under
> protocol A. Reasons, all visible in this repository: (1) train and test were a random row sample of
> one file (`spikes/s5_circuit_variants.py`, `df.sample`), so neighbouring packet windows of the same
> attack run sit on both sides; (2) that file (cleaned v0) was capped to 1,000,000 rows and de-duplicated
> **before** the split, so the class mix is not the real one; (3) the 8-class runs used `IAT` as one of
> the 8 features (`results/spikes/S5_circuit_variants.json`), a column a public audit ties to the capture
> setup rather than the traffic; (4) 3 seeds on one split measure training noise, not data noise.
> D6 (dead parameters) and the S4 hardware evidence do not depend on S5 and stand.

**Accounts (Gate 0):** IBM Open plan active with backends ibm_fez, ibm_kingston, ibm_marrakesh (all
Heron); Quantum Inspire account active with Tuna-17. Details in `docs/ACCOUNTS.md`.

**Revisit if:** Phase 1 tuning shows a clear accuracy gap at full scale, or IBM's available backends change.

---

## D15 — Post-quantum channel for E7: hybrid ML-KEM-768 + X25519, ML-DSA-65 signatures, pinned keys

**Date:** 5 Nov 2026 · **Owner:** Sheghaf (post-quantum channel) · **Status:** accepted 5 Nov 2026 (Nouran,
owner of INTERFACES section 4, merged PR #12; Sheghaf approved it in PR #12). Interface change: INTERFACES 1.1.0

**Proposal.** `fqml/pq` carries the federated messages of E7 (server ↔ sites) over a channel built from the
NIST post-quantum standards, with the classical and no-protection baselines chosen by one settings file
(`configs/pq/*.yaml`, read by `fqml/pq/config.py`):

| Settings | Session keys | Signatures | Hash |
|---|---|---|---|
| `default` | ML-KEM-768 + X25519 (hybrid) | ML-DSA-65 | SHA-256 |
| `cnsa2` | ML-KEM-1024 | ML-DSA-87 | SHA-384 |
| `classical` (E7 baseline) | X25519 | Ed25519 | SHA-256 |
| `none` (E7 baseline) | — | — | — |

- **Handshake** (`channel.py`): 3 messages; server and site each sign the transcript with a pinned identity
  key (`keys.py`, like SSH known_hosts, no certificate authority); fresh ML-KEM and X25519 keys for every
  connection (forward secrecy); session keys = HKDF(ML-KEM secret ‖ X25519 secret, salt = transcript hash);
  key confirmation both ways. Both sides compare a fingerprint of the settings, so nobody can talk a site
  down to a weaker mode.
- **Messages:** AES-256-GCM with one key per direction and the sequence number inside the authenticated
  data: tampered, replayed, reordered or dropped messages are rejected.
- **Models:** each model travels as a signed envelope (weights + round + purpose + signer + meta such as
  `n_rows`), so an old round's model cannot be replayed and a stored model can be checked later.
- **Transport:** `link.py` (TCP, clear refusal reasons on both sides, byte and time counters for E7).
  Received bytes are never unpickled (`serialize.py`).

**Why hybrid by default:** ML-KEM is young; if a flaw is ever found, X25519 still protects the session
keys, and a recorded session ("harvest now, decrypt later") stays safe unless ML-KEM is broken too. It costs
94 bytes and under 0.5 ms per handshake (S6). `cnsa2` is the all-post-quantum, higher-security option.

**Evidence.**
- `tests/test_pq_channel.py` (42 tests, real liboqs 0.16.0): every mode end to end, in memory and over TCP.
  Rejected: tampered, replayed, reordered or relabelled messages; unknown sites; impostor server and site
  keys; every downgrade pair; a man in the middle editing any handshake field; stale or re-signed models;
  malformed input. liboqs keys, ciphertexts and signatures interoperate with OpenSSL's independent FIPS
  203/204 implementation in both directions. Mutation check: 12 deliberate bugs (e.g. skipping a signature
  check, leaving the sequence number out of the authenticated data) each turned the suite red.
- Spike S6, **preliminary**: cloud Linux VM, 1 thread, Python 3.13. Byte counts are exact on every machine;
  the times must be re-run on the Mac in Docker (`results/spikes/S6_pq_overhead.json`).

| Settings | Handshake CPU (both sides) | Handshake bytes | VQC model message on the wire | 1M-parameter model, send / receive |
|---|---|---|---|---|
| none | 0.05 ms | 308 | 407 B | 4.4 / 1.1 ms |
| classical | 0.7 ms | 638 | 524 B | 15.9 / 9.4 ms |
| pq-768, no hybrid | 0.5 ms | 9,330 | 3,771 B | 11.0 / 10.4 ms |
| **default** | **0.8 ms** | **9,424** | **3,771 B** | **10.9 / 10.9 ms** |
| cnsa2 | 0.7 ms | 12,862 | 5,089 B | 11.0 / 10.7 ms |

**Effect.** Time is not the cost: post-quantum handshakes and model messages take under a millisecond, like
the classical ones. **Bytes are:** +8.8 KB per connection, and every signed model carries the 3,309-byte
ML-DSA-65 signature. For the 64-parameter VQC (256 B of weights) a model message grows from 524 B
(classical) to 3,771 B; for a 1M-parameter model the overhead is under 0.1%. E7 should report bytes per
round, not only time.

**Open questions (E7 design, with Nouran):** (1) sign the sites' updates as well, or only the server's
global and final models? The channel already authenticates every message; signatures add provenance, and
dropping them for updates halves the signature bytes. (2) How Flower carries the channel: `Link` payloads
inside Flower messages, or our own server loop over `link.py`. (3) One connection per run, or a reconnect
every round (+9.4 KB each time).

**Revisit if:** NIST or liboqs change ML-KEM / ML-DSA, a flaw is published in either, or E7 shows the bytes
matter (then ML-DSA-44, or signing global models only).

---

## Phase 0 amendments — 16 Nov 2026

**Owners:** all four · **Status:** proposed in the pull request "Interface change: Phase 0 amendments
(INTERFACES 2.0.0)"; accepted when Ganna, Kareem, Sheghaf and Nouran have approved it.

**What stands.** Nothing in the Phase 0 engineering core is redone: the environment and lock file (D5),
CI, the simulator (D1), threads (D2), batched FedAvg (D3), per-sample gradients (D4), circuit version 2
(D6), the CNOT chain (D7) and the post-quantum channel (D15).

**Why anything changes.** Three facts about CICIoT2023 that Phase 0 did not use:
- The dataset paper (Neto et al., Sensors 2023, https://www.mdpi.com/1424-8220/23/13/5941): each attack
  was run as one experiment and all traffic captured during it carries that attack's label; benign
  traffic is one 16-hour capture; features are averages over windows of 10 packets (100 for DDoS, DoS
  and Mirai). So the numbered files of one attack are pieces of **one** run, and a split by file still
  trains and tests on the same run.
- A public audit of the dataset
  (https://github.com/Mohammed-Baqir/Deep-Network-Intrusion-Detection-through-Data-Engineering) reports
  that `IAT` takes values tied to a per-campaign constant and alone separates DDoS from DoS (AUC 0.998),
  and that 5-class macro-F1 falls from about 0.93 to about 0.33 on an independent capture campaign
  (CICIoMT2024). Task DATA-1 replicates the `IAT` finding on our files before D9.
- Our own Phase 0 cleaning (v0) capped, de-duplicated and shuffled the rows before any split and kept
  neither file name nor row order, so no leakage-free split could be built from it.

| # | Phase 0 item | Amendment | In this repository |
|---|---|---|---|
| 1 | Cleaned file v0 (`ciciot2023_pcapcsv_clean.parquet`) | **Retired.** Replaced by cleaner v1: row order kept, `file` / `row_index` / `row_id`, duplicates marked (`dup_group`) not deleted, conflicting labels marked, even sampling with a stored `sample_rate` | `fqml/data/clean_ciciot2023_v1.py`, `tests/test_clean_v1.py`; `interim` fingerprint marked retired in `data/checksums.json` |
| 2 | Spike S5 scores (0.465 / 0.762) and the old run matrix | **Not valid** as results (note under D7); the run matrix is replaced by the E0 version 2 matrix (protocols A / B / C, 5 folds × 3 seeds) | Notes under D3 and D7; `results/spikes/README.md` |
| 3 | `INTERFACES.md` 1.1.0 | **2.0.0:** new data columns, fold manifests, run ID with protocol and fold, model size setting, `entangler="none"` with `readout="linear"`, sites split and local-only mode (signatures) | `docs/INTERFACES.md`, `fqml/data/folds.py`, `fqml/quantum/layout.py`, tests |
| 4 | "Seeds 0, 1, 2 on one split" | **5 folds × 3 seeds**, the fold is the unit of comparison; run ID `E{n}-{dataset}-{model}-{variant}-{protocol}-f{fold}-s{seed}` | `fqml/utils/run_id.py`, `tests/test_run_id.py` |
| 5 | Label map (8 families) | Stays primary. A 7-class variant with DDoS and DoS merged (`c7m`) is available; whether it is reported is D17 | `fqml/data/label_maps/` |
| 6 | MatchedMLP at the VQC's parameter count only | **MLP size ladder:** the VQC's count, then 128, 256, 512, 1,000, 3,000 parameters; result reported as the VQC's classical-equivalent size | `fqml/models/sizes.py`, `fqml/models/mlp.py` |
| 7 | Dataset list | CICIoMT2024 added as the independent test set; order CICIoT2023 → CICIoMT2024 → CICIDS2017; Edge-IIoTset and UNSW-NB15 stay fingerprinted and unused | `docs/DATASETS.md` |
| 8 | FedAvg equivalence test on random numbers only | Also on traffic-like data (CI) and on real rows (when the cleaned file is present) | `fqml/fl/equivalence.py`, `tests/test_fedavg_real_data.py` |

**Found while making the amendments.**
- **The entanglement-free control needed its own readout.** With the D6 readout (output k reads one
  qubit) and no CNOTs, every output sees exactly one feature: for 8 classes, class k sees only feature k;
  for the binary task 42 of the 48 angles and 7 of the 8 features never reach the output. Such a control
  would lose for lack of inputs, not for lack of entanglement, so "the VQC beats it" would prove nothing.
  Fix: `readout="linear"` (every output a trained weighted sum over all 8 qubits). The control is
  `entangler="none", readout="linear"` (`vqc_noent`), compared with `entangler="chain", readout="linear"`
  (`vqc_lin`, 57 / 120 parameters): only the CNOTs differ. Checked: with the linear readout both have
  0 dead angles and all 8 features reach every output (`tests/test_torch_sim.py`). The main model `vqc`
  keeps the D6 readout (50 / 64 parameters). E0 core comparison grows from four models to five.
- The number D15 was planned for the merged DDoS + DoS decision but was already taken by the post-quantum
  channel. That decision is **D17**.
- `experiments/run_matrix.py`, named in the plan as the home of the old run matrix, was never committed;
  only `spikes/s2_round_time.py` refers to its output.

**Revisit if:** the file inventory (task DATA-1) shows the numbered files of an attack are separate
recordings and not pieces of one run. Protocol A still holds then, and a split by file becomes a fourth
check.

---

## D8 — CICIoT2023 cleaned with cleaner v1: every row kept, duplicates and conflicts marked

**Date:** 17 Nov 2026 · **Owner:** Ganna (data) · **Status:** run by Nouran on 17 Nov with the settings below;
Ganna confirms or changes them (re-running costs about 25 minutes).

**Decision.** `python -m fqml.data.clean_ciciot2023_v1 --per-label-cap 0 --conflicts keep` on the 309
per-capture CSV files (`raw/ciciot2023` fingerprint checked: 310/310 files match).
- `--per-label-cap 0`: no attack type is thinned. Thinning before the split would thin the test blocks too;
  sub-sampling is done per fold, on the training side only (INTERFACES 1.2). Every `sample_rate` is 1.0.
- `--conflicts keep`: identical feature rows with different labels stay, marked in `conflict`. Every test
  score is also reported on rows with `conflict == 0` (see below).

**Result** (`results/data_reports/ciciot2023_clean_v1.json`; output fingerprinted as `interim/ciciot2023_v1`):

| | Rows |
|---|---|
| Raw | 46,776,700 |
| Removed (NaN or ±inf) | 1,040 |
| Written | 46,775,660 |
| Distinct feature rows | 20,358,149 (**56.5% of rows have an exact copy**) |
| In cross-family conflicts (`conflict == 2`) | 14,217,879 (30.4%) |
| … of which benign vs attack | 30,988 (0.07%) |
| In same-family conflicts (`conflict == 1`) | 557,569 (1.2%) |

- 37 of the 46 standard features are present in every file. The per-capture CSVs have **no timestamp
  column**, so protocol A assumes the row order inside a file is the recorded order (the dataset paper
  extracts features window by window from each capture). `IGMP` and `Time_To_Live` are present but not
  used; `Time_To_Live` is very likely the merged CSV's `Duration` (TTL). Whether it is added goes with D9.
- **The cross-family conflicts are almost entirely DDoS against DoS.** Rows in such conflicts: DDoS
  10.2 M of 34.0 M (30%), DoS 3.95 M of 7.8 M (**50%**); every other family under 1%, except Spoofing
  (5%, MITM-ArpSpoofing 7%). The affected types are the matching floods: DDoS-SYN/UDP/TCP/SynonymousIP
  (54–65%) and DoS-SYN/UDP/TCP (40–63%). In these 37 features half of all DoS rows are byte-identical to
  DDoS rows, so no model can separate them. This is the evidence for D17 (7-class variant `c7m`).
- **Benign against attack is clean:** only 30,988 rows (0.5% of benign) share their values with an attack
  row, so the binary task is not affected.
- **Duplicates sit in the big floods** (DDoS 56–74%, DoS 42–57%); benign, the small attack types and
  Mirai have 0–11%.

**Consequences.**
- Every test score is reported three ways: all test rows; rows without an exact twin in training
  (INTERFACES 1.2); rows with `conflict == 0`. For the 8-class task the data quality report (DATA-4) adds
  the **best score any model could reach** given the conflicts (each conflict group predicted as its most
  common label).
- D17 is taken on 23 Nov with this evidence and Checkpoint 1.

**Revisit if:** a timestamp source for the rows is found (then protocol A checks the order instead of
assuming it), or D9 adds `Time_To_Live` (then the cleaner is re-run with it and this table is redone).

---

## Checkpoint 1 — what it measured (17–18 Nov 2026)

XGBoost with fixed settings (`fqml/eval/checkpoint1.py`, task DATA-5), trained on at most 50,000 rows per attack
type from the training side, scored on **validation blocks only** (test blocks unused), fold 0, seed 0.
Results: `results/checkpoint1.json` (8 settings) and `results/checkpoint1_fpr.json` (3 weightings with
fixed-alarm-rate scores). One fold and one seed: differences under about 0.005 are noise; the decisions below
rest on larger ones. D8a, D9, D11 (part 1) and D17 are taken from it.

| Split / features (balanced weighting) | 8-class macro-F1 | 7-class (`c7m`) | Binary F1 |
|---|---|---|---|
| Random row split, all features | 0.628 | 0.667 | 0.996 |
| Random row split, no IAT | 0.626 | 0.665 | 0.996 |
| Protocol A, gap 0, all / no IAT | 0.571 / 0.574 | 0.613 / 0.616 | 0.995 |
| Protocol A, gap 100, all / no IAT | 0.575 / 0.573 | 0.615 / 0.615 | 0.995 |
| Protocol A, gap 1000, all / no IAT | 0.576 / 0.573 | 0.617 / 0.615 | 0.995 |

**Protocol B (unseen attack type), all 5 folds** (`results/checkpoint1_protocol_b.json`; sqrt weighting,
no IAT; manifest `data/folds/ciciot2023_B.json`). Scored on each fold's validation types, which the model
never saw in training:

| Fold | Unseen types in validation | Detected at 1% FPR | at 0.1% FPR | Right family, 8-class | Right family, 7-class |
|---|---|---|---|---|---|
| 0 | 8 (incl. DDoS-UDP, DoS-TCP, DNS_Spoofing) | 98.0% | 97.8% | 60.2% | 98.1% |
| 1 | 6 (incl. DDoS-TCP, DoS-SYN, SqlInjection) | 99.2% | 99.0% | 73.5% | 99.4% |
| 2 | 5 (incl. DDoS-PSHACK, DoS-HTTP, XSS) | 99.0% | 98.8% | 90.5% | 98.1% |
| 3 | 4 (incl. DDoS-SYN, DDoS-ACK_Fragmentation) | 99.9% | 99.9% | 30.9% | 93.6% |
| 4 | 9 (incl. DDoS-ICMP, DoS-UDP, MITM) | 97.2% | 97.0% | 57.0% | 98.4% |
| **Mean** | | **98.7%** | **98.5%** | **62.4%** | **97.5%** |

- **Unseen attacks are detected:** 98.7% raise an alarm at 1% false positives, about the same as seen attacks
  under protocol A (98.3%).
- **Almost every family error on an unseen type is DDoS vs DoS:** merged, 97.5% get the right family. Every
  held-out DoS type is called DDoS (DoS-TCP, DoS-SYN, DoS-HTTP, DoS-UDP: 99.2–99.8% to DDoS). This is more
  evidence for D17.
- **Other failures:** DDoS-ACK_Fragmentation 97% to Mirai; DNS_Spoofing 0% right family in fold 0, where all of
  Spoofing is held out (expected); the small Web and BruteForce types 13–59%.
- **Limit:** detection is counted over rows, so the large floods dominate it. Detection per attack type,
  averaged over types, is required in the metrics module (EVAL-1) so the small types count.

Scores without twin rows move by at most 0.003; scores without conflict rows are 0.008–0.019 higher in every
setting alike. Neither changes a conclusion.

---

## D8a — Evaluation protocol A: 5 time-blocked folds, gap 100 rows

**Date:** 18 Nov 2026 · **Owners:** Ganna + Nouran; all four approve · **Status:** proposed; the manifest is in
the pull request `data/folds-a`.

**Decision.** `data/folds/ciciot2023_A.json`, built by `python -m fqml.data.folds build-a --gap 100`: every attack
type (and the benign capture) in recorded order cut into 5 blocks; fold k tests on block k, validates on block
k+1, trains on the rest; 100 rows dropped at every boundary between blocks of different roles (reduced to at
most 10% of a block for Recon-PingSweep 45, Backdoor_Malware 64, XSS 76, Uploading_Attack 25). Per fold about
28.06 M training, 9.35 M validation and 9.36 M test rows; every row is a test row exactly once.

**Why, with the measurement.**
- **The gap is large enough:** 8-class macro-F1 0.571 / 0.575 / 0.576 with gap 0 / 100 / 1,000. The score does
  not fall as the gap grows, so neighbouring rows across a boundary do not leak.
- **Protocol A is needed:** the random row split scores 0.628, protocol A 0.575 (−0.053, about 9% relative);
  7-class 0.667 → 0.615. Most of the drop is a real change over time that a random split hides: under
  protocol A **32% of DoS-UDP_Flood validation rows are predicted Mirai** (196,717 rows; 258 under the random
  split), and Mirai F1 falls from 0.998 to 0.840.
- The binary task hardly moves (F1 0.996 → 0.995); see D11 for how it is judged.

**Protocol B** is built too (`data/folds/ciciot2023_B.json`, `python -m fqml.data.folds build-b --gap 100`):
inside every family with 2+ types, the types are dealt into 5 groups; fold k tests on group k and validates on
group k+1, both never seen in training; benign and DictionaryBruteForce are split in time as in protocol A.
**Still open:** protocol C (CICIoMT2024), task DATA-8b.

**Revisit if:** another fold or seed shows the score still falling between gap 100 and 1,000.

---

## D9 — IAT is dropped from every feature set

**Date:** 18 Nov 2026 · **Owners:** Ganna + Nouran · **Status:** proposed from Checkpoint 1.

**Decision.** `IAT` is never a model input. 36 features remain (37 standard features of cleaner v1 minus IAT).

**Why, with the measurement.** With and without IAT, 8-class macro-F1 differs by at most 0.003 in every split
(random 0.628 / 0.626; protocol A gap 0 0.571 / 0.574, gap 100 0.575 / 0.573, gap 1,000 0.576 / 0.573), in both
directions, and binary F1 is the same. Dropping it removes a feature a public audit ties to the capture setup,
at no measured cost.

**Flag for DATA-4:** XGBoost puts 40–47% of its importance on `Number` in every run. In CICIoT2023 this column
probably reflects the packet-window size (100 packets for DDoS, DoS and Mirai, 10 for the rest), which would
make it a shortcut like IAT. The window-size screen in DATA-4 checks it before the top-8 features are chosen.
`Time_To_Live` stays unused (D8).

**Revisit if:** the window-size screen finds that `Number` (or another feature) is a shortcut; then this entry
gains that feature and Checkpoint 1 is re-run without it.

---

## D11 (part 1) — Class weighting and training sample, the same for every model

**Date:** 18 Nov 2026 · **Owner:** Nouran · **Status:** proposed from Checkpoint 1. Part 2 (tuning per fold, VQC
depth, MLP size ladder) is still due Mon 23 Nov.

**Decision.**
- **Training sample:** at most N rows per native attack type, spread evenly over time inside that fold's training
  blocks (`fqml.data.rows.cap_per_label`). N is set per model in part 2 (Checkpoint 1 used 50,000).
- **Weighting:** each row weighs ∝ 1/√(count of its family in the training sample), mean 1: the rule of the
  shared training loop (`fqml.train.loop.class_weights`). The same for XGBoost, the MLPs and the VQC.
- **How the binary task is judged:** attack recall at a fixed false-positive rate (0.1% and 1%; threshold set on
  the validation block, scored on the test block) and ROC-AUC. Binary F1 and the false-positive rate at the
  default decision are reported but never used to judge.

**Why, with the measurement** (protocol A, gap 100, all features):

| Weighting | 8-class F1 | 7-class F1 | Web F1 | BruteForce F1 | FPR at default decision | Attack recall at 1% FPR | at 0.1% FPR | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| balanced (equal family weight) | 0.575 | 0.615 | 0.10 | 0.08 | 33.2% | 0.983 | 0.981 | 0.997 |
| **sqrt** | **0.607** | **0.642** | 0.20 | 0.16 | 39.8% | **0.983** | **0.980** | **0.997** |
| none | 0.613 | 0.645 | 0.25 | 0.24 | 52.2% | 0.983 | 0.980 | 0.997 |

- **The binary task does not depend on the weighting.** At a fixed alarm rate all three are the same (recall
  0.983 at 1% FPR, 0.980–0.981 at 0.1%, ROC-AUC 0.997). The 33–52% false-positive rates at the default decision
  come from the class mix of the training sample: sampling per attack type gives Recon (5 types) about 4× the
  rows of benign (1 type), so benign rows are pushed to Recon. That is a threshold effect, not model quality,
  and it is why the binary task is judged at a fixed alarm rate.
- **The multi-class task does depend on it:** macro-F1 moves by up to 0.04, as much as the whole random-vs-A
  effect. So the weighting is fixed once for all models. `sqrt` is chosen because it is the rule every PyTorch
  model already trains with; `none` is 0.006 higher on this one fold, which is within noise.

**What still fails (per attack type, `sqrt`):** DDoS-UDP / TCP / SYN floods (34–69% recall, mostly to DoS),
DoS-UDP_Flood (32% to Mirai), DoS-SYN (46% to DDoS); SqlInjection (6%, to Recon), DictionaryBruteForce (23%),
the other Web types (38–52%); benign 60% (33% to Recon). The remaining types are at 74–100%; the ICMP /
RSTFIN / PSHACK / fragmentation floods and all Mirai types are above 99%.

**Revisit if:** part 2 changes the training sample (for example a per-family cap), or other folds show `none`
clearly above `sqrt`.

---

## D17 — The 7-class task (DDoS and DoS merged) is reported beside the 8-class task

**Date:** 18 Nov 2026 · **Owners:** Ganna + Nouran · **Status:** proposed from D8 and Checkpoint 1.

**Decision.** Every E0 setting is scored on both the 8-class task and `c7m` (label-map variant: Benign,
DDoS_DoS, Recon, Web, BruteForce, Spoofing, Mirai). The 8 families stay the primary label map
(INTERFACES 1.1); which task the EXP-1 win rules use is set in EXP-1.

**Why, with the measurement.**
- D8: half of all DoS rows are byte-identical to DDoS rows in the 37 features.
- Checkpoint 1 (protocol A, gap 100, balanced): 33% of DDoS rows are predicted DoS, still 21% on rows with no
  conflict at all. Per attack type the floods that use the same protocol fail (DDoS-UDP 26% recall, DDoS-TCP 39%,
  DDoS-SYN 42%, DDoS-SynonymousIP 39%, all mostly to DoS); merged, DDoS_DoS reaches F1 0.988.
- A distributed and a single-source flood of the same protocol differ in how many machines send, which
  per-window averages cannot see. The 8-class score therefore mixes model quality with a limit of the features;
  the 7-class score removes that limit.

**Revisit if:** a feature that sees the number of sources is added.

---

## D12 — Federated tree models: per-client ensembles now, histogram-based XGBoost only with a federated build

**Date:** 24 Nov 2026 · **Owner:** Kareem · **Status:** proposed (FED-5); reviewer Nouran.

**Decision.**
1. E1 reports XGBoost and Random Forest federated as a **per-client ensemble**: every client trains its own model
   on its own rows, once; the server averages the clients' class probabilities weighted by row count
   (`experiments/fed5_trees.py`, `fed_xgb_ens`, `fed_rf`). No row leaves a client; one round; no new library.
2. Centralised XGBoost stays the **ceiling** (E0), and the gap to it is reported per split.
3. **MLP-large and MLP-3k under FedAvg** are the strong iteratively-trained federated baselines beside the VQC,
   because a one-round ensemble cannot be updated round by round like the other federated models.
4. Histogram-based federated XGBoost (sites send gradient histograms, every site grows the same trees) is **not used
   in Phase 1 or E1**: the pinned wheel is built without it (see below). It comes back only through a requirements
   pull request that adds a federated build, with its own test against central XGBoost.

**Why, with the measurement** (`results/fed5/trees_A_f0.json`; protocol A fold 0, 5,000 rows per attack type,
top-8 stand-in features, validation block, seed 0; 8-class macro-F1):

| Split | Central XGBoost | Central RF | Fed XGBoost ensemble | Fed RF ensemble | Local-only RF (mean / worst client) |
|---|---|---|---|---|---|
| IID, 50 clients | 0.584 | 0.582 | 0.576 | 0.569 | 0.536 / 0.515 |
| Sites, 10 × 8 attack types | 0.584 | 0.582 | 0.525 | 0.517 | 0.347 / 0.263 |
| Whole-family sites, 10 × 3 families | 0.584 | 0.582 | 0.444 | 0.438 | 0.229 / 0.157 |

- The ensemble is within 0.01 of pooled training on IID data, loses 0.06 on the sites split and 0.14 when sites hold
  whole families. It beats the mean site training alone by 0.04 on IID and by 0.18 and 0.21 on the two sites splits,
where federation is needed. Under whole-family sites it loses DoS and Web
  entirely (recall 0.00): the sites that never saw a family outvote the ones that did.
- Histogram-based federation: XGBoost 3.2.0 in the image (`xgboost-cpu`, H-9) reports `USE_FEDERATED: False` in
  `xgboost.build_info()`; `experiments/fed5_xgb_federated.py` (10 site processes and a federated server) stops at
  `assert is_distributed()`. Using it needs XGBoost built with the federated plugin or NVIDIA FLARE.
- Flower's `FedXgbBagging` (every sampled client adds trees to the shared model each round) and `FedXgbCyclic`
  (clients boost in turn) exist and Flower is in the image, but they are not measured here. Bagging run for one
  round is close to the ensemble above.
- Caveats: one fold, one seed, validation block; `Number` is among the 8 stand-in features until DATA-4 decides
  whether it is a shortcut. The decision rests on the ordering of the options, which is large, not on the third
  decimal.

**Revisit if:** a federated XGBoost build is added to the lock file; Flower bagging over several rounds closes the
whole-family gap in a measured run; or the E1 design needs trees updated every round.

---

## Pending decisions

| # | Decision | Owner | Due |
|---|---|---|---|
| D11 (part 2) | Tuning per fold, fixed VQC depth, MLP size ladder, training-sample size per model (part 1, weighting, is above) | Nouran | Mon 23 Nov |
| D14 | Where runs happen; result files; environment tags | Sheghaf | Mon 23 Nov |
| E5 | Shot training: shots during training (parameter-shift, very expensive), or shots only at evaluation plus a small shot-trained subset, or simulated shot sampling in the PyTorch simulator | Nouran + Sheghaf | Wed 25 Nov |
| D13 | δ, the batch sampling method for DP-SGD, and the privacy unit | Nouran | Fri 27 Nov |
| D16 | Second dataset: whether CICIoMT2024's feature columns match CICIoT2023's, and the shared classes | Ganna | Fri 27 Nov |
| D10 | CICIDS2017 labels and split | Ganna | Fri 18 Dec |
