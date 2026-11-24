# FQML-NIDS — Team context (read this first)

**Last updated:** Sun 22 Nov 2026 · **Status covered up to:** Q-1 results, Sat 21 Nov 2026
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

- **Repo:** `github.com/Nouran24625/FQML-NIDS` (private). CI = GitHub Actions (Docker build, pytest, PQ smoke test).
- **Decisions log:** `docs/DECISIONS.md` (every D-number below is explained there).
- **Interfaces:** `docs/INTERFACES.md` (currently 2.1.0 pending, after Q-1).
- **Datasets list:** `docs/DATASETS.md`. **Specs:** `docs/specs/` (EVAL-1, H-10).
- **Cleaned data:** `data/interim/ciciot2023_v1/ciciot2023_clean_v1.parquet` (not in git; run on a team machine). Report: `results/data_reports/ciciot2023_clean_v1.json`.
- **Folds:** `data/folds/ciciot2023_A.json`, `data/folds/ciciot2023_B.json`.
- **Plans:** "FQML-NIDS Phase 1 Plan v3 — Whole Team" and the four personal v3 plans (Nouran, Ganna, Kareem, Sheghaf). Older v1/v2 plans are kept for history only.
- **Longer records** (in the `claude/` notes): Design changes 16 Nov · Phase 0 amendments 16 Nov · Checkpoint 1 results 18 Nov · Phase 1 plan v3 (18 Nov) · Q-1 results 21 Nov · Phase headers (whole-project roadmap).

## 4. Where we are (as of 21 Nov)

**Phase 0: closed.** Docker/CI/MLflow, PyTorch VQC simulator, batched FedAvg (matches sequential on real data 8/8), per-sample gradients, circuit v2 (0 dead parameters), CNOT chain mapping to IBM Heron and Tuna-17 with 0 extra gates. Phase 0 amendments merged as PR #16 (17 Nov).

**Phase 1: running Thu 19 Nov → Gate 1 Fri 18 Dec 2026** (spare days 21–24 Dec).

Done 16–21 Nov:
- DATA-2: cleaner v1 on CICIoT2023, 46,775,660 rows (#17).
- DATA-3: protocol A folds (#18) and protocol B folds (#19).
- DATA-5: Checkpoint 1 under random split, A and B (#19).
- H-9 / H-9b (xgboost-cpu 3.2.0, opacus 1.6.0, interface PR).
- Q-2: fair entanglement-free control (`vqc_lin` vs `vqc_noent`).
- EVAL-1 and H-10 specs written.
- Q-1 (Nouran): finite-shot training for the VQC, branch `q1/finite-shots`, PR "Interface change: Q-1 finite-shot mode for TorchVQC (INTERFACES 2.1.0)", reviewer Sheghaf.

**Not yet recorded here:** the outcome of SYNC-1 (Thu 19 Nov) — whether D8, D8a, D9, D11 part 1 and D17 were approved, and who holds Integrator and Scribe this month. Whoever knows: update this section.

## 5. Key results so far

**Checkpoint 1 (XGBoost, validation blocks, 18 Nov):**
- Random row split inflates 8-class macro-F1 by ~5 points (0.628 → 0.575 under protocol A). Gap of 100 rows between blocks is enough. Dropping IAT costs nothing.
- Binary F1 (~0.995) hides a 33–52% benign false-alarm rate at the default threshold; at a fixed 1% false-alarm rate, 98.3% of attacks are caught. → Binary is judged by recall at fixed FPR, never by F1.
- Unseen attack types (protocol B) are detected as well as seen ones (98.7% at 1% FPR). Family errors are almost all DDoS vs DoS; merged 7-class gets 97.5% right.
- What fails: same-protocol floods (DDoS vs DoS vs Mirai) and low-volume families (SQL injection, brute force, Web, benign vs Recon).
- **Flag:** XGBoost puts 40–47% of its importance on `Number` (probably packet-window size, 10 vs 100) — a possible shortcut, screened in DATA-4 before top-8 features are chosen.

**Q-1 (shots, 21 Nov):** shot-based training costs no extra time on a T4 GPU at batch 64. Gradient noise vs exact: ~58% at 128 shots, ~21% at 1,024 for `vqc`. At 1,024 shots, shot noise is < 10% of DP noise at σ = 1, C = 1; the break-even shot count grows as 1/C², so the clip norm decides whether shots matter. Open question for E5: does 20–60% gradient noise cost accuracy?

## 6. Design decisions in force

- **Datasets:** CICIoT2023 → CICIoMT2024 (second campaign, enables protocol C and a two-site federation) → CICIDS2017 (best effort). Edge-IIoTset and UNSW-NB15 fingerprinted, unused.
- **Evaluation protocols:** **A** time-blocked 5 folds inside each attack type, gap 100 · **B** unseen attack type within a family · **C** train CICIoT2023, test CICIoMT2024 shared classes (to build, DATA-8b). Random split (R) only as a leakage reference.
- **Runs:** 5 folds × 3 seeds, paired comparison with the fold as the unit. Run ID `E{n}-{dataset}-{model}-{variant}-{protocol}-f{fold}-s{seed}` (`fqml.utils.run_id`).
- **Features:** IAT never a model input (D9; 36 features remain). Duplicates marked with `dup_group`, not deleted; scores also reported on test rows with no exact twin in training.
- **Training sample:** per-attack-type sample spread over time, class weights ∝ 1/√(family count), identical for every model (D11 part 1).
- **Metrics:** 8-class and 7-class (`c7m`, DDoS+DoS merged, computed from 8-class predictions) macro-F1; binary = attack recall at 0.1% and 1% FPR + ROC-AUC; per-attack-type scores averaged over types (D17).
- **Models:** `vqc` (CNOT chain + local readout, main model) · `vqc_lin` (chain + linear readout) · `vqc_noent` (no CNOTs + linear readout). Entanglement test = `vqc_lin` vs `vqc_noent`. Classical: LR, MLP size ladder (50/64, 128, 256, 512, 1k, 3k), XGBoost, RF, Fourier-feature model.
- **Federation:** batched FedAvg loop for experiments; Flower only for the PQC benchmark and demo. Local-only baseline and "sites" partition planned.
- **Privacy:** privacy unit is the device or site, not a single flow. One-run canary audit for DP-SGD.
- **Post-quantum channel:** X25519 + ML-KEM-768 key exchange, ML-DSA-65 model signing.
- **Claims:** no "first"; no "post-quantum secured federation" unless built and measured; no quantum advantage claim.
- **Not in Phase 1:** federated E1, attack/defence experiments, full privacy study, noisy simulation and hardware runs, QCNN, secure aggregation, compression, PQC benchmark E7, dashboard, EVE-NG demo.

## 7. Next deadlines (Gate 1 critical chain in bold)

| Date | What | Owner |
|---|---|---|
| Mon 23 Nov | **DATA-4** quality report + window-size / `Number` screen | Ganna |
| Mon 23 Nov | Decisions D11 part 2 (tuning, VQC depth, size ladder, sample size), D14 | Team |
| Tue 24 Nov | **DATA-6 spec** (top-8 feature selection) | Nouran |
| Wed 25 Nov | **H-10 training loop and EVAL-1 metrics module** | Sheghaf |
| Wed 25 Nov | **EXP-1 final**; Q-1b (shot noise on real rows) | Nouran |
| Wed 25 Nov | Decision E5 (shots) | Nouran + Sheghaf |
| Thu 26 Nov | SYNC-2 | All |
| Fri 27 Nov | **DATA-6 build**; DATA-8a (CICIoMT2024 fingerprint) | Ganna |
| Fri 27 Nov | **EXP-0 runner**; EVAL-3 conventions | Sheghaf |
| Fri 27 Nov | CLS-1 (MLP size ladder) | Nouran |
| Fri 27 Nov | Decisions D13, D16 (do CICIoMT2024 columns match?) | Team |
| Fri 4 Dec | **CLS-2 and Q-4 tuning** | Nouran |
| Fri 4 Dec | FED-4 real-data federated trial run | Kareem |
| 7–9 Dec | **E0 under protocol A** | All |
| 10–11 Dec | **E0 under protocol B**; D12 (Fri 11) | All |
| Thu 17 Dec | Gate 1 dry run | All |
| Fri 18 Dec | **Gate 1; E0 report** (Nouran); D10 | All |

**Gate 1 must-pass (status 18 Nov):** 1 fold manifests — built, awaiting approval · 2 quality report — cleaning done, DATA-4 open · 3 Checkpoint 1 — done, awaiting approval · 4 CICIoMT2024 — open · 5 E0 protocol A — open · 6 E0 protocol B — open · 7 metrics module and runner — specs done, build open · 8 DP-SGD checks — open.

## 8. How to keep this file current

After any session that changes status, a decision or a date:
1. Edit the relevant section (don't append a diary — replace the old status).
2. Bump "Last updated" and "Status covered up to" at the top.
3. Commit it in the same PR as the work, or as a small PR titled "Context update: …".

If you used Claude for the session, ask it at the end: *"Update CONTEXT.md with what changed in this session"* and paste the result.
