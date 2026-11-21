# E0 pre-registration: how the centralised baselines are judged (EXP-1)

| | |
|---|---|
| **Task** | EXP-1 |
| **Author** | Nouran (AI and quantum ML) |
| **Reviewers** | Ganna, Kareem |
| **Version** | Draft 1, 21 Nov 2026. Final version due Wed 25 Nov 2026, before any tuning |
| **Depends on** | D8, D8a, D9, D11 part 1, D17 (approved at SYNC-1); D11 part 2 (Mon 23 Nov); D16 (Fri 27 Nov) |

## 0. Purpose

E0 answers the Phase 1 question:

> On CICIoT2023, tested without data leakage, does the VQC detect attacks better than classical models of the same size, and how far is it from the best classical models?

This document fixes, before any model is tuned or any test block is scored, which data each result is measured on, which metrics decide the outcome, which comparisons are made, how "beats" is decided, and what is concluded in each case. Either outcome is a valid result: a win is reported as a win, a loss as a loss, and the leakage-free benchmark is a contribution in both cases.

Any change after Wed 25 Nov is recorded in §7 with its date and reason, and the E0 report lists every change.

## 1. Evaluation protocols

- **Protocol A (primary).** Time-blocked 5-fold split inside every attack type and the benign capture, gap 100 rows (D8a, `data/folds/ciciot2023_A.json`). All six primary comparisons (§3) are decided on protocol A only.
- **Protocol B (strict).** Unseen attack type within a family, 5 folds (`data/folds/ciciot2023_B.json`). Reported for every model with the same metrics and intervals; it can qualify a protocol A result but cannot create a win.
- **Protocol C (independent).** Train on CICIoT2023, test on the CICIoMT2024 classes shared with CICIoT2023 (Benign, DDoS, DoS, Recon, Spoofing). The protocol A models (one per fold and seed) are each scored on CICIoMT2024, giving five paired scores per model. Decision thresholds come from CICIoT2023 validation blocks; nothing is tuned on CICIoMT2024. Fallback, fixed now: if D16 finds that not all eight selected features exist in CICIoMT2024, C uses the shared features only and is labelled so; if the classes cannot be matched, C moves to Phase 2 and the E0 report says so.
- **Test data is locked.** Tuning and early stopping use validation blocks only. Each final configuration is scored on test blocks once, by script, and logged to MLflow and `results_log.csv`.
- **Replication.** 5 folds × 3 seeds (0, 1, 2). The fold is the unit of comparison; seeds are averaged within a fold. Run ID: `E0-{dataset}-{model}-{variant}-{protocol}-f{fold}-s{seed}`.

## 2. Metrics

All metrics are computed on test blocks, per fold and seed, by the EVAL-1 metrics module.

- **Deciding metrics** (used for the six comparisons in §3): 8-class macro-F1, and 7-class macro-F1 (`c7m`, DDoS and DoS merged) computed from the 8-class predictions (D17).
- **Reported metrics** (with intervals, never used to declare a win):
  - Attack recall at 1% and 0.1% FPR. Attack score = 1 − P(Benign); the threshold is chosen on the fold's validation block to give that FPR on validation benign rows, then applied unchanged to the test block (D11 part 1).
  - ROC-AUC of the attack score.
  - Mean detection over attack types: recall at the 1% FPR threshold for each of the 33 attack types, unweighted mean. Types with fewer than 100 test rows in a fold are flagged.
  - Protocol B: recall at the 1% FPR threshold on held-out attack types, and the share assigned to the right family (8-class and 7-class).
  - Per-class F1 and the confusion matrix.
  - Binary F1, for comparison with published work only.
- **Subsets.** Every metric is also reported on test rows with no exact feature twin in the training rows (`dup_group`) and on rows with `conflict == 0`.
- **Fixed for all models:** class weights ∝ 1/√(family count), the training sample per attack type (D11 part 2), the 36 features without IAT (D9), and the same top-8 features for every 8-feature model (DATA-6, fitted on each fold's training rows only).

## 3. Primary comparisons

Six comparisons, all under protocol A, each on 8-class and 7-class macro-F1:

| # | Model | Rival | Question |
|---|---|---|---|
| 1, 2 | `vqc` | MatchedMLP (`mlp_p50` / `mlp_p64`: same top-8 inputs, matched parameter count) | Is the VQC better than a classical network of equal size? |
| 3, 4 | `vqc` | RFF (Q-3: integer frequencies ≤ 3 per feature, linear output layer) | Is the VQC better than a classical model of the circuit's own frequency spectrum? |
| 5, 6 | `vqc_lin` | `vqc_noent` (same circuit without CNOTs, linear readout) | Does entanglement help? |

`vqc_lin` vs MatchedMLP is reported as an extra comparison and does not count toward the six.

**Decision rule.**
1. Per fold, average each model's score over seeds 0, 1, 2. Take the paired difference d_k = model − rival for folds k = 1…5.
2. Test mean(d) = 0 with the corrected resampled t-test (Nadeau and Bengio, 2003): variance factor (1/5 + n_test/n_train), with n_test and n_train the fold's test and training row counts; 4 degrees of freedom; two-sided.
3. Adjust the six p-values with Holm's method at family-wise α = 0.05.
4. **Wins:** adjusted p < 0.05 and mean(d) > 0. **Loses:** adjusted p < 0.05 and mean(d) < 0. **Otherwise:** no detectable difference. A win with mean(d) < 0.01 is reported as "detectable but negligible".
5. Report mean(d), the 95% interval and the Bonferroni-level (99.17%) interval for every comparison, whatever the outcome.

Protocols B and C get the same calculation and are reported beside A; they do not change the outcome (§1).

## 4. Classical-equivalent size

The MLP size ladder (`fqml.models.sizes`: 50/64, 128, 256, 512, 1,000 and 3,000 parameters, same top-8 inputs) gives a curve of score against parameter count. The VQC's classical-equivalent size is where its score falls on that curve.

1. Per fold, average each ladder model's 8-class macro-F1 over seeds.
2. Make the curve non-decreasing in parameter count (isotonic fit), so a noisy ladder point cannot create two answers.
3. Read off the parameter count at which the curve reaches the VQC's score, interpolating linearly in log(parameters) between neighbouring ladder points. Below the smallest point it is reported as "< 50"; above the largest as "> 3,000".
4. Report the median and the range over the five folds, for `vqc`, `vqc_lin` and `vqc_noent`, on 8-class and 7-class, under protocols A and B.

The gap to the best classical models (XGBoost and Random Forest on all 36 features and on the top 8) is reported per metric as a description, without a test.

## 5. Outcomes and stop rule

| Result of comparisons 1–4 under A | Conclusion stated in the E0 report |
|---|---|
| Wins over MatchedMLP and over RFF | The VQC beats classical models of equal size and of its own function class on this benchmark. |
| Wins over MatchedMLP only | The VQC beats an equal-size network, but a classical model of its function class does as well: the gain is not specific to the quantum circuit. |
| Wins over RFF only | Reported as found; checked against the size ladder before any interpretation. |
| No win over either | **Negative result:** under leakage-free testing the VQC shows no advantage over classical models of equal size or of its own function class. |

Comparisons 5–6 are reported on their own: entanglement helps, hurts, or shows no detectable effect.

**Stop rule.** A negative result is final for E0. The VQC is not re-tuned, re-designed or given a larger budget than the classical models to change the outcome; every model gets the same tuning budget (D11 part 2). Later phases (federation, attacks, privacy, hardware) continue as planned, because their questions do not depend on a quantum win.

## 6. Models in E0

| Group | Models |
|---|---|
| Quantum | `vqc` (chain entangler, local readout), `vqc_lin` (chain, linear readout), `vqc_noent` (no CNOTs, linear readout); 3 layers (2 and 4 as a side check, D11 part 2) |
| Matched classical | MatchedMLP (`mlp_p50` / `mlp_p64`), RFF |
| Size ladder | MLPs at 50/64, 128, 256, 512, 1,000, 3,000 parameters |
| Reference classical | Logistic regression, MLP-large (2 × 256), XGBoost, Random Forest (all 36 features and top 8) |

Tuning: 12 trials per fold per model family on that fold's validation block, with early stopping (D11 part 2). The quantum models are trained in exact simulation; whether shots are used during training is decision E5 (Wed 25 Nov).

## 7. Changes after the final version

| Date | Section | Change | Reason |
|---|---|---|---|
| | | | |
