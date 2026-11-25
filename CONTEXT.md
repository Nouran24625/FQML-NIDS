# FQML-NIDS — Team context (read this first)

**Last updated:** Wed 25 Nov 2026 · **Status covered up to:** Wed 25 Nov 2026 (GitHub audit and branch fix, Nouran's week-2 tasks, the `Number` check, university workstation set up and checked)
**Rule:** this file is the single source of truth for project status. Whoever finishes a work session that changes status, a decision or a deadline updates it in the same pull request (see "How to keep this file current" at the end). Details live in the files linked from each section.

---

## 1. The project in one paragraph

FQML-NIDS is an E-JUST graduation project (Sep 2026 – Jun 2027): a **leakage-aware benchmark** testing whether a small quantum model (a variational quantum circuit, VQC) adds anything to **federated network intrusion detection**, plus tests of privacy (DP-SGD, membership inference, quantum shot noise vs DP noise), robustness to attacks on federated learning, and a post-quantum secure channel. Training is simulated; real quantum hardware (IBM Heron, Quantum Inspire Tuna-17) is used only for inference later. **We claim no quantum advantage.** If the VQC loses to classical models of the same size, that is a valid result and the fair, leakage-free test setup is the contribution.

**Phase 1 question:** on CICIoT2023, tested without leakage, does the VQC beat classical models of the same size, and how far is it from the best classical models?

## 2. Team and areas

| Member | Area | Owns |
|---|---|---|
| **Nouran** | AI and quantum ML (lead) | Quantum and classical models, training, tuning, DP-SGD, privacy pilot, EXP-1, E0 report. Only member with AI/QML background. |
| **Ganna** | Attack data | Datasets, cleaning, splits/folds, data quality and leakage checks, CICIoMT2024, CICIDS2017 |
| **Kareem** | Attacks and defences | Threat model, attacks on federated learning, defences, privacy attacks, client partitions, federated trial run |
| **Sheghaf** | Crypto and infrastructure | Post-quantum crypto, experiment runner, metrics module, results saving, tables/charts, IBM jobs |

`docs/INTERFACES.md` section owners: 1 data Ganna · 2 models Nouran · 3 federated Kareem · 4 privacy Nouran, post-quantum Sheghaf · 5 results logging Sheghaf.

**Working rules**
- Every pull request gets one reviewer from a **different area**. Nobody approves or merges their own PR. Changes to another area's code also need that owner's approval.
- When one area depends on another, the providing area writes a short spec first (`docs/specs/`).
- Interface changes go in one PR titled "Interface change: …" with a version bump of `INTERFACES.md`, approved by all four.
- Everything runs in Docker: `docker compose -f docker/compose.yml exec dev <command>` from the repo root. MLflow UI on port 5001.
- New libraries: add to `requirements.txt` in a PR; never hand-edit `requirements/lock.txt`.
- Never commit tokens or keys (env vars / Colab Secrets only).
- A design change is adopted only if a checked fact shows it raises the value of the results; time and compute cost are not reasons to reject it.
- Thursday sync (30 min) every week; the Scribe updates `docs/DECISIONS.md`, the Integrator keeps `main` green.
- Each member writes the thesis chapter for their own area.

## 3. Where things are

- **Repo:** `github.com/Nouran24625/FQML-NIDS` — **publicly readable as of 24 Nov** (anonymous clone works); change under Settings → General if it should be private. CI = GitHub Actions (Docker build, pytest, PQ smoke test).
- **Decisions log:** `docs/DECISIONS.md` (every D-number below is explained there).
- **Interfaces:** `docs/INTERFACES.md`: 2.3.0 after the federated PRs; Q-1's finite-shot mode becomes 2.4.0 when #21 is merged.
- **Datasets list:** `docs/DATASETS.md`. **Specs:** `docs/specs/` (EVAL-1, H-10, DATA-6, EVAL-2). **Pre-registration:** `docs/PREREG_E0.md` (final, 24 Nov), `docs/PREREG_MIA.md`. **Notes:** `docs/notes/Q1b_shot_noise_vs_dp.md`. **Thesis outlines:** `docs/thesis/`. **Team notebook:** `notebooks/LEARN-1_E0_ideas.ipynb` (everyone runs it and answers its 5 questions).
- **Cleaned data:** `data/interim/ciciot2023_v1/ciciot2023_clean_v1.parquet` (1.5 GB, not in git). Shared copy on Drive: `FQML-NIDS-data/2023 cleaned + readme/` (checksum in `data/checksums.json`, verified 20 Nov). Report: `results/data_reports/ciciot2023_clean_v1.json`.
- **Folds:** `data/folds/ciciot2023_A.json`, `data/folds/ciciot2023_B.json`.
- **University workstation (CSIT compute pool, set up 24 Nov by Kareem):** account `sheghaf.320230169`, login node csit-csc, work through Slurm (`-p cpu`). The account has no Docker access, so the same `docker/Dockerfile` is built with rootless Podman (storage on local disk via `~/.config/containers/storage.conf`) and converted to `/pool/scratch/$USER/images/fqml-nids-dev.sif`; `~/bin/fqml <command>` runs a command in it with the repo at `/workspace`, and `~/FQML-NIDS/runs/job-template.sh` is a Slurm template. Repo at `~/FQML-NIDS` (update with `git pull`; rebuild the image only when `requirements/` or the Dockerfile change). Checked on main d2f6073 (25 Nov): 364 tests pass, 2 skipped, and `pip freeze` is identical to the laptop image. Same-seed A8 run vs a laptop (Docker): every score identical once the MLP twin is seeded (fix in `demo/a8_vqc_train.py`); one run is no faster on the workstation (159 s vs 141 s). **Fast path:** a run of this size gains nothing from 2–8 threads (300 steps: 19.2 s at 1 thread, 24.1 s at 8) and peaks at ~1.4 GB, and the binding cap is the `student` QOS: 16 CPU threads and 1 GPU per user at a time (2 days per job), with at least 2 threads per task, so experiments go out as job arrays of 1-core, 2 GB tasks (`runs/array-template.sh`, at most 8 at once). Measured: 15 full A8 runs finished in 621 s as one array running 4 at a time (an interactive shell held the other 8 threads), against ~40 min one after another on the laptop. The cleaned parquet is at `~/FQML-NIDS/data/interim/ciciot2023_v1/` (SHA-256 matches `data/checksums.json`, 24 Nov) with the fold manifests in `data/folds/`. **Speed test (24 Nov):** `/pool` storage writes ~10 MB/s and reads ~14–28 MB/s (the parquet takes ~110 s to read), node-local disk 2–3 GB/s (1.4 s), so `fqml` keeps a checked copy of the parquet on each node's local disk (`/var/tmp/$USER/data`, staged on all four nodes) and binds it at the usual path; internet on the login node 28–40 MB/s down, ~7 MB/s up, but downloads from GPU nodes crawl (~0.1 MB/s), so fetch packages on the login node. CPU: 170 GFLOP/s on 1 thread, 620 on 8. **GPU test (A8 script, same seed, Quadro RTX 5000, torch 2.14.1+cu130 swapped into the team image for the test only):** batch 256, VQC 81 s on GPU vs 113 s on 1 CPU thread; batch 4096, VQC 82 s on GPU vs 866 s on 4 CPU threads (10.5×); same scores on GPU and CPU; the small MLP is slower on GPU. CUDA wheels for that test are kept in `/pool/scratch/$USER/wheels/cu130`. The frozen image stays CPU PyTorch (D5); a CUDA image would be a requirements PR.
- **Plans:** "FQML-NIDS Phase 1 Plan v3 — Whole Team" and the four personal v3 plans (Nouran, Ganna, Kareem, Sheghaf). Older v1/v2 plans are kept for history only.
- **Longer records** (in the `claude/` notes): Design changes 16 Nov · Phase 0 amendments 16 Nov · Checkpoint 1 results 18 Nov (corrected 24 Nov) · Phase 1 plan v3 (18 Nov) · Q-1 results 21 Nov · GitHub status and branch fix 24 Nov · Phase headers (whole-project roadmap).

## 4. Where we are (as of Tue 24 Nov, evening)

**Phase 0: closed.** Docker/CI/MLflow, PyTorch VQC simulator, batched FedAvg (matches sequential on real data 8/8), per-sample gradients, circuit v2 (0 dead parameters), CNOT chain mapping to IBM Heron and Tuna-17 with 0 extra gates. Phase 0 amendments merged as PR #16 (17 Nov).

**Phase 1: running Thu 19 Nov → Gate 1 Fri 18 Dec 2026** (spare days 21–24 Dec). SYNC-1 (19 Nov) accepted D8, D8a, D9, D11 part 1 and D17 (all four). Integrator and Scribe for this month: not recorded yet — whoever knows, add them here.

**How the branches stand (read before merging anything).** Nothing from Phase 1 reached `main` until #19 merges: the Checkpoint 1 chain lives on `data/checkpoint1`, and later work was merged into it on purpose (#22, #24, #33, #37). Twice on 24 Nov work was stranded by merging into a branch whose own PR was already merged (#27, #28, #32, #34); it was rescued as `fed/rescue`. **Rule: before clicking Merge, check that the PR's base branch is not already merged; if it is, change the base first. Merge with "Create a merge commit", never squash or rebase.** Order to finish: `fed/rescue` and `data/no-number-check` → `data/checkpoint1`, then #19 → `main` (Ganna), then `fix_q1_after_merges.sh` updates #21 (Q-1), then delete merged branches.

**Done:**
- Data: DATA-2 cleaner v1 (D8), DATA-3 folds A and B (D8a), DATA-5 Checkpoint 1, DATA-4 quality report + DATA-1b (Ganna, #33), the `Number` check (24 Nov).
- Models and privacy (Nouran): Q-1 finite shots (#21) and its batch-256 re-run; Q-2; EXP-1 final pre-registration (#20); D11 part 2 with committed tuning trials (#22); DATA-6 spec and EVAL-2 spec, Q-1b note, LEARN-1 notebook (#36); Q-3 RFF model, CLS-1 model factory, PRIV-2 DP-SGD (#37); PRIV-3 checks 1–3 code (check 2 in CI; checks 1 and 3 have a real-data script); WRITE outline of the AI/QML chapter.
- Federated and security (Kareem): H-8 scope (#26), SEC-1 + FED-2 (#24), FED-1/3/7 (#25, INTERFACES 2.2.0), FED-4, FED-5 (D12), FED-6 (INTERFACES 2.3.0) and the attack grid (in `fed/rescue`), PRIV-5 (#29), ABS-1 (#30), attacks chapter outline (#31).
- Team: CONTEXT.md and CLAUDE.md (#23); H-1, H-2.

**Not started on GitHub (24 Nov):** D14 (Sheghaf, was due 23 Nov), **H-10 training loop and EVAL-1 metrics (Sheghaf, due Wed 25)**, **EXP-0 runner and EVAL-3 (Sheghaf, Fri 27)**, **DATA-6 build (Ganna, Fri 27)**, DATA-8a and IAT-DOC (Ganna, Fri 27). Everything left on Nouran's list (tuning, first pass, PRIV-4, E0 report) waits for H-10, EVAL-1, EXP-0 and DATA-6.

## 5. Key results so far

**Checkpoint 1 (XGBoost, validation blocks, 18 Nov; corrected 24 Nov):**
- Random row split inflates 8-class macro-F1 by ~5 points (0.628 → 0.575 under protocol A). Gap of 100 rows is enough. Dropping IAT costs nothing.
- Binary F1 (~0.995) hides a 33–52% benign false-alarm rate at the default threshold; at a fixed 1% false-alarm rate, 98.3% of attacks are caught. → Binary is judged by recall at fixed FPR, never by F1.
- **`Number` is the packet-window size (DATA-4) and was a shortcut.** Without it (24 Nov, `sqrt`, fold 0): protocol A barely changes (8-class 0.607 → 0.587; recall at 0.1% FPR 0.979 → 0.952), but **under protocol B unseen attacks caught at 1% FPR fall from 98.0% to 51.9%** (ROC-AUC 0.994 → 0.946). The old "unseen attacks detected as well as seen ones (98.7%)" was mostly the shortcut. Family assignment of unseen types is unchanged (60% 8-class, 98% 7-class: errors are DDoS vs DoS).
- What fails: same-protocol floods (DDoS vs DoS vs Mirai) and low-volume families (SQL injection, brute force, Web, benign vs Recon).
- DATA-4: identical rows with different labels cap any model at 0.955 8-class / 0.997 7-class macro-F1; 67–73% of protocol A test rows have an exact twin in training.

**Q-1 (shots, 21 and 24 Nov):** at batch 64 a shot-trained step costs the same as exact on a T4; at batch 256 (the E0 batch) about 3×. Gradient RMS error at 1,024 shots: ~16–21% for `vqc`. Shot noise vs DP noise: equal at S* = κB/(σ²C²) (κ = 1.31 `vqc`, 0.75 `vqc_lin`, 0.66 `vqc_noent`); at B = 256, C = 1, σ = 1 and 1,024 shots, DP noise dominates (shot noise adds ~15% to the noise std for `vqc`). Gradient norms are from synthetic inputs; re-measure after DATA-6.

## 6. Design decisions in force

- **Datasets:** CICIoT2023 → CICIoMT2024 (second campaign, enables protocol C and a two-site federation) → CICIDS2017 (best effort). Edge-IIoTset and UNSW-NB15 fingerprinted, unused.
- **Evaluation protocols:** **A** time-blocked 5 folds inside each attack type, gap 100 · **B** unseen attack type within a family · **C** train CICIoT2023, test CICIoMT2024 shared classes (to build, DATA-8b). Random split (R) only as a leakage reference.
- **Runs:** 5 folds × 3 seeds, paired comparison with the fold as the unit. Run ID `E{n}-{dataset}-{model}-{variant}-{protocol}-f{fold}-s{seed}` (`fqml.utils.run_id`).
- **Features:** IAT never a model input (D9, accepted); `Number` dropped too (D9 addition, proposed 24 Nov with the measurement above; approval Thu 26 Nov) → 35 features. Per-fold preprocessing to [0, π], top 8 by mutual information with a redundancy rule (DATA-6 spec). Duplicates marked with `dup_group`, not deleted; scores also reported on test rows with no exact twin in training.
- **Training sample and tuning:** per-attack-type sample spread over time, class weights ∝ 1/√(family count) (D11 part 1). N = 5,000 rows per type for every compared model; reference models also at 50,000. 12 pre-committed trials per model per fold (`configs/tuning/`), early stopping (patience 10), batch 256, VQC depth 3 (D11 part 2). Only 8-class models are trained; 7-class and binary come from their predictions.
- **Metrics:** 8-class and 7-class (`c7m`, DDoS+DoS merged, computed from 8-class predictions) macro-F1; binary = attack recall at 0.1% and 1% FPR + ROC-AUC; per-attack-type scores averaged over types (D17).
- **Models:** `vqc` (CNOT chain + local readout, 64 parameters, main model) · `vqc_lin` (chain + linear readout, 120) · `vqc_noent` (no CNOTs + linear readout, 120). Six pre-registered comparisons: `vqc` vs `mlp_p64` and vs RFF, `vqc_lin` vs `vqc_noent`, each on 8 and 7 classes; corrected paired t-test over folds, Holm correction (PREREG_E0). Classical: LR, MLP size ladder (64, 120, 128, 256, 512, 1k, 3k), MLP-large, XGBoost, RF, RFF. All built by name with `fqml.models.classical.build`.
- **Shots (E5, proposed, Sheghaf to confirm):** E0 trains VQCs exactly; every VQC also scored with 1,024 shots at test time; shot-trained subset `vqc` + `vqc_lin`, protocol A, 5 folds (~5 GPU-hours).
- **Federation:** batched FedAvg loop for experiments; Flower only for the PQC benchmark and demo. Local-only baseline and "sites" partition planned.
- **Privacy (D13, proposed):** Phase 1 runs are centralised, so the privacy unit is one row and every ε is example-level; device or site level comes with federated DP. δ = 1e-6, Poisson sampling, B = 256, Opacus RDP accountant (+ PRV cross-check), C = 1.0 fixed, ε ∈ {∞, 8, 4, 1}. DP-SGD code: `fqml/privacy/dpsgd.py`; checks: equal to Opacus, per-sample gradients, one-run canary audit (`fqml/privacy/audit.py`).
- **Post-quantum channel:** X25519 + ML-KEM-768 key exchange, ML-DSA-65 model signing.
- **Claims:** no "first"; no "post-quantum secured federation" unless built and measured; no quantum advantage claim.
- **Not in Phase 1:** federated E1, attack/defence experiments, full privacy study, noisy simulation and hardware runs, QCNN, secure aggregation, compression, PQC benchmark E7, dashboard, EVE-NG demo.

## 7. Next deadlines (Gate 1 critical chain in bold)

| Date | What | Owner |
|---|---|---|
| **Now** | **Merge `fed/rescue` and `data/no-number-check` into `data/checkpoint1`, then #19 → `main`** | Kareem / Ganna |
| Wed 25 Nov | **H-10 training loop and EVAL-1 metrics module** (from the specs) | Sheghaf |
| Wed 25 Nov | E5 confirmed | Sheghaf (with Nouran) |
| Thu 26 Nov | SYNC-2: approve the D9 addition (`Number`), D11 part 2, D13, E5; record Integrator and Scribe | All |
| Fri 27 Nov | **DATA-6 build** from the spec; DATA-8a (CICIoMT2024 fingerprint); IAT-DOC | Ganna |
| Fri 27 Nov | **EXP-0 runner**; EVAL-3 result files; D14 (overdue) | Sheghaf |
| Fri 27 Nov | D16 (do CICIoMT2024 columns match?) | Ganna |
| Fri 27 Nov | LEARN-1 notebook run, 5 questions answered | Ganna, Kareem, Sheghaf |
| After DATA-6 | Q-1b re-measured on real rows; PRIV-3 real-data script (`python -m experiments.priv3_checks`) | Nouran |
| Fri 4 Dec | **CLS-2 and Q-4 tuning; first-pass check** (needs runner) | Nouran |
| Fri 4 Dec | EVAL-2 statistics module (from the spec) | Sheghaf |
| 7–9 Dec | **E0 under protocol A**; PRIV-4 privacy pilot (Wed 9) | All / Nouran |
| 10–11 Dec | **E0 under protocol B** | All |
| Thu 17 Dec | Gate 1 dry run | All |
| Fri 18 Dec | **Gate 1; E0 report** (Nouran); D10 | All |

**Gate 1 must-pass (status 24 Nov):** 1 fold manifests — accepted, on `data/checkpoint1` · 2 quality report — done (DATA-4) · 3 Checkpoint 1 — done, D9 and D17 accepted (`Number` addition pending) · 4 CICIoMT2024 — open · 5 E0 protocol A — open (needs runner) · 6 E0 protocol B — open · 7 metrics module and runner — specs done, build not started · 8 DP-SGD checks — code and tests done (check 2 in CI); real-data run after DATA-6.

## 8. How to keep this file current

After any session that changes status, a decision or a date:
1. Edit the relevant section (don't append a diary — replace the old status).
2. Bump "Last updated" and "Status covered up to" at the top.
3. Commit it in the same PR as the work, or as a small PR titled "Context update: …".

If you used Claude for the session, ask it at the end: *"Update CONTEXT.md with what changed in this session"* and paste the result.
