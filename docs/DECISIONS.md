# FQML-NIDS — Decision Log

One entry per decision that changes how the project is built or run.
Format: what was decided, why (with the measurement), who decided, and what would make us revisit it.

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

## D3 — Federated simulation for experiment sweeps: batched FedAvg loop (PROPOSED)

**Date:** 17 Oct 2026 · **Owner:** Kareem (federation) · **Status:** proposed — needs team agreement

**Proposal.** For E1–E5 and E8, simulate federated rounds with our own loop that trains all sampled
clients of a round together in one GPU batch (each client keeps its own weights). Keep Flower for
E7 (post-quantum channel, real messages) and the defense demo.

**Why:** the GPU speed in D1 only appears when many samples run together (batch ≥ 1,280 =
20 clients × 64). Flower trains clients one by one, which would lose most of that gain.

**Effect on compute (plan version B: full rounds and clients, E5 shots only at evaluation):**
~950 h (Mac, old way) → ~87 h (Mac, D2) → ~12–17 h (one Kaggle T4 with D1 + D3).

**Evidence (Spike S2, 20 clients x 50 steps x batch 64 per round):** equivalence test batched =
sequential passes (6 of 6, PR #1). Laptop CPU, 1 thread: VQC 16.9 s per round batched vs 25.4 s
sequential; MLP 0.06 s. Kaggle T4 GPU: VQC 1.8 s per round.

**Open questions:** how Byzantine attacks, robust aggregators and DP-SGD plug into the batched loop;
check results match a Flower run on a small case.

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

## Pending decisions

- **D6, dead VQC parameters (proposed):** the last layer's 8 RZ angles cannot change the output (only
  CNOTs and the Z readout follow), so 8 parameters never train (binary: 42 of 50 learn). Proposed fix:
  drop that RZ, giving 42 parameters (binary) and 56 (8 classes); change `vqc.py`, `torch_sim.py`,
  `qiskit_circuit.py` and the matched MLP together. Owners: Sheghaf + Kareem. Must land before S4.
- **E5 shot training:** train with shots (parameter-shift, very expensive) vs. shots only at
  evaluation plus a small shot-trained subset vs. simulated shot sampling in the PyTorch simulator.
  Owners: Nouran + Sheghaf; to agree with Prof. Samir at Gate 0.
