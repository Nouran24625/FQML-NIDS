# FQML-NIDS — Current scope (H-8)

**Owner:** Kareem · **Approval:** all four members · **Written:** 23 Nov 2026, from Phase 1 plan v3 (18 Nov),
`docs/DECISIONS.md` and `CONTEXT.md`.

This page is the one agreed list of what the project does now, what comes after the core, and what is out.
Where an older plan (plans v1 and v2, the September plan, the action plan, the work headers, the Track A
handoff) says something different, this page and plan v3 replace it. Changes to this page go through a pull
request all four approve.

## In: Phase 1 core (to Gate 1, Fri 18 Dec 2026)

**Question.** On CICIoT2023, tested without leakage, does the VQC detect attacks better than classical models of
the same size; which MLP size does it equal; how far is it from XGBoost; does entanglement help
(`vqc_lin` vs `vqc_noent`)?

- **Data:** CICIoT2023 cleaned with cleaner v1 (D8); IAT dropped (D9); `Number` screened before the top-8
  features are fixed (DATA-4, DATA-6).
- **Protocols:** A, time-blocked, gap 100 (primary, D8a); B, unseen attack type. Random split only as a leakage
  reference.
- **Models (E0):** `vqc`, `vqc_lin`, `vqc_noent`, RFF, MatchedMLP, the MLP size ladder, XGBoost; LR, Random Forest
  and MLP-large as other classical models.
- **Metrics:** 8-class and 7-class (`c7m`) macro-F1 (D17); binary attack recall at 0.1% and 1% FPR plus ROC-AUC,
  never binary F1 (D11 part 1); scores per attack type averaged over types; scores without twins.
- **Runs:** 5 folds × 3 seeds, paired by fold, judged by the pre-registered rules in `docs/PREREG_E0.md`
  (EXP-1).
- **Tools built in Phase 1:** training loop (H-10), metrics module (EVAL-1), runner (EXP-0), statistics
  (EVAL-2), tables made by script (EVAL-4), DP-SGD module checked against Opacus and by a canary audit
  (PRIV-2, PRIV-3).
- **Federated pipeline, made ready but not reported:** partitions (FED-3), local-only baseline (FED-7), shared
  metrics (FED-1), trial run on real data (FED-4), threat model and attack definitions (SEC-1, FED-2).
- **Best effort:** protocol C on CICIoMT2024 (DATA-8a/8b, DATA-9), privacy pilot at ε ∈ {∞, 8, 4, 1}
  (PRIV-4), shot-noise note and E5, robust aggregators (FED-6), two-lab dry run (FED-8), federated trees
  decision (D12), membership-inference plan (PRIV-5), CICIDS2017 cleaner (DATA-7), abstract v2 (ABS-1).

## After the core (later phases, only once Gate 1 passes)

- **E1, federated experiment:** VQC and classical models under FedAvg against local-only and centralised
  training, on the sites split, IID and the two-lab federation (CICIoT2023 + CICIoMT2024).
- **Attacks and defences:** label flipping and class suppression with 20% malicious clients; median and trimmed
  mean, each measured with zero attackers first.
- **Privacy study:** DP-SGD accuracy against ε; membership-inference pilot with canaries, giving an empirical ε
  for exact, finite-shot and DP training.
- **Shots and hardware:** shot-based training per E5; noisy simulation; inference on IBM Heron and
  Quantum Inspire Tuna-17 (E6, D7).
- **Post-quantum channel benchmark (E7, D15):** cost of the hybrid ML-KEM-768 + X25519 channel with ML-DSA-65
  signing on real federated messages.
- **More data:** protocol C completed; E0 on CICIDS2017 (D10).

## Out (not planned; added back only by an all-four pull request)

- QCNN and other quantum models besides the VQC family.
- Secure aggregation and update compression.
- Training on quantum hardware (hardware is for inference only).
- Edge-IIoTset and UNSW-NB15 (fingerprinted, unused).
- The dashboard and the EVE-NG demo, unless the team decides to bring them back at a sync.
- Claims of quantum advantage, any "first" claim, and "post-quantum secured federation" unless it is built
  and measured (E7).

## Minimum defensible thesis

If time runs short, the thesis still stands on these five items, and work is protected in this order:

1. **E0 under protocols A and B:** the full core comparison, size ladder and XGBoost on 5 folds × 3 seeds,
   judged by EXP-1. A VQC that loses is a valid negative result.
2. **One federated experiment with a local-only baseline:** FedAvg against local-only and centralised
   training on the sites split, with recall on attack types each site never held (FED-7).
3. **One DP curve:** accuracy against ε for the VQC and its classical controls, with the DP-SGD module
   checked against Opacus and a canary audit.
4. **One hardware inference run:** a trained VQC evaluated on IBM Heron or Tuna-17 against its simulated
   result.
5. **One-command reproduction:** a fresh clone, `docker compose` build and one command per run
   (`python -m fqml.exp.e0 ...`), with every table and chart rebuilt by script from `results/results_log.csv`.
