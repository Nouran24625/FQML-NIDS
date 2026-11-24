# E0 pre-registration: how the centralised baselines are judged (EXP-1)

| | |
|---|---|
| **Task** | EXP-1 |
| **Author** | Nouran (AI and quantum ML) |
| **Reviewers** | Ganna, Kareem |
| **Version** | Final, 24 Nov 2026 (draft 1: 21 Nov). Fixed before any tuning; later changes go in §7 |
| **Depends on** | D8, D8a, D9, D11 part 1, D17 (accepted 19 Nov); D9 addition on `Number`, D11 part 2, D13, E5 (proposed); DATA-4 (quality report); DATA-6 (preprocessing spec); D16 and DATA-9 (protocol C) |

## 0. Purpose

E0 answers the Phase 1 question:

> On CICIoT2023, tested without data leakage, does the VQC detect attacks better than classical models of the same size, and how far is it from the best classical models?

This document fixes, before any model is tuned or any test block is scored, which data each result is measured on, which metrics decide the outcome, which comparisons are made, how "beats" is decided, and what is concluded in each case. Either outcome is a valid result: a win is reported as a win, a loss as a loss, and the leakage-free benchmark is a contribution in both cases.

Any change after Wed 25 Nov is recorded in §7 with its date and reason, and the E0 report lists every change.

## 1. Evaluation protocols

- **Protocol A (primary).** Time-blocked 5-fold split inside every attack type and the benign capture, gap 100 rows (D8a, `data/folds/ciciot2023_A.json`). All six primary comparisons (§3) are decided on protocol A only.
- **Protocol B (strict).** Unseen attack type within a family, 5 folds (`data/folds/ciciot2023_B.json`). Reported for every model with the same metrics and intervals; it can qualify a protocol A result but cannot create a win.
- **Protocol C (independent).** Train on CICIoT2023, test on the CICIoMT2024 classes shared with CICIoT2023 (Benign, DDoS, DoS, Recon, Spoofing). The protocol A models (one per fold and seed) are each scored on CICIoMT2024, giving five paired scores per model. Decision thresholds come from CICIoT2023 validation blocks; nothing is tuned on CICIoMT2024. Fallback, fixed now: if D16 or the feature-stability audit DATA-9 finds that some of a fold's top-8 features do not exist in CICIoMT2024 or do not carry over, the protocol C models are retrained under protocol A on the DATA-9 feature list (same tuned settings, labelled "C-features"); if the classes cannot be matched, C moves to Phase 2 and the E0 report says so.
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
- **Subsets.** Every metric is also reported on test rows with **no twin** and on rows with `conflict == 0`. A *twin* is a test row whose values on the 35 input features (§2, last item) are identical to a row anywhere in the fold's **training block** (all training rows, not only the training sample), computed with the DATA-4 feature keys (`fqml.data.quality.feature_set_keys`). DATA-4 measured 67–73% of protocol A test rows with a twin (56–65% under protocol B, folds 1, 2, 4), so the no-twin scores are the measure of what a model learned rather than memorised.
- **Ceiling.** Beside every 8-class and 7-class score the report shows the best score any model can reach given identical rows with different labels (DATA-4: 0.955 8-class, 0.997 `c7m` on protocol A test folds).
- **Shots (E5).** Every VQC is also scored with 1,024 shots at test time; reported beside its exact score, never used in §3.
- **Fixed for all models:** class weights ∝ 1/√(family count) (D11 part 1); the training sample per attack type, N = 5,000 for every model in §3 and on the ladder (D11 part 2); the 35 features without `IAT` and `Number` (D9 and its addition); the same per-fold preprocessing and top-8 features for every 8-feature model (DATA-6, fitted on each fold's training rows only); only 8-class models are trained, and the 7-class and binary results come from their predictions.

## 3. Primary comparisons

Six comparisons, all under protocol A, each on 8-class and 7-class macro-F1:

| # | Model | Rival | Question |
|---|---|---|---|
| 1, 2 | `vqc` | MatchedMLP (`mlp_p64`: same top-8 inputs, 64 parameters like `vqc`) | Is the VQC better than a classical network of equal size? |
| 3, 4 | `vqc` | RFF (Q-3: integer frequencies ≤ 3 per feature, linear output layer) | Is the VQC better than a classical model of the circuit's own frequency spectrum? |
| 5, 6 | `vqc_lin` | `vqc_noent` (same circuit without CNOTs, linear readout) | Does entanglement help? |

`vqc_lin` vs its own same-size twin `mlp_p120` (120 parameters) is reported as an extra comparison and does not count toward the six.

**Decision rule.**
1. Per fold, average each model's score over seeds 0, 1, 2. Take the paired difference d_k = model − rival for folds k = 1…5.
2. Test mean(d) = 0 with the corrected resampled t-test (Nadeau and Bengio, 2003): variance factor (1/5 + n_test/n_train), with n_test and n_train the fold's test and training row counts; 4 degrees of freedom; two-sided.
3. Adjust the six p-values with Holm's method at family-wise α = 0.05.
4. **Wins:** adjusted p < 0.05 and mean(d) > 0. **Loses:** adjusted p < 0.05 and mean(d) < 0. **Otherwise:** no detectable difference. A win with mean(d) < 0.01 is reported as "detectable but negligible".
5. Report mean(d), the 95% interval and the Bonferroni-level (99.17%) interval for every comparison, whatever the outcome.

Protocols B and C get the same calculation and are reported beside A; they do not change the outcome (§1).

## 4. Classical-equivalent size

The MLP size ladder (D11 part 2: 64, 120, 128, 256, 512, 1,000 and 3,000 parameters, 8-class, same top-8 inputs) gives a curve of score against parameter count. The VQC's classical-equivalent size is where its score falls on that curve.

1. Per fold, average each ladder model's 8-class macro-F1 over seeds.
2. Make the curve non-decreasing in parameter count (isotonic fit), so a noisy ladder point cannot create two answers.
3. Read off the parameter count at which the curve reaches the VQC's score, interpolating linearly in log(parameters) between neighbouring ladder points. Below the smallest point it is reported as "< 64"; above the largest as "> 3,000".
4. Report the median and the range over the five folds, for `vqc`, `vqc_lin` and `vqc_noent`, on 8-class and 7-class, under protocol A (the ladder runs under A only).

The gap to the best classical models (XGBoost and Random Forest on all 35 features and on the top 8, at N = 5,000 and N = 50,000) is reported per metric as a description, without a test.

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
| Quantum | `vqc` (chain entangler, local readout, 64 parameters), `vqc_lin` (chain, linear readout, 120), `vqc_noent` (no CNOTs, linear readout, 120); 3 layers (2 and 4 as a side check, D11 part 2); exact training, 1,024-shot test scores, and a shot-trained `vqc` subset (E5) |
| Matched classical | MatchedMLP (`mlp_p64`, and `mlp_p120` for the extra comparison), RFF |
| Size ladder | MLPs at 64, 120, 128, 256, 512, 1,000, 3,000 parameters |
| Reference classical | Logistic regression, MLP-large (2 × 256), XGBoost, Random Forest (all 35 features and top 8), each at N = 5,000 and N = 50,000 rows per attack type |

Tuning: 12 pre-committed trials per model per fold (`configs/tuning/`), best on that fold's validation block, early stopping on a fixed validation sample (D11 part 2). Protocol B reuses the setting chosen most often under A.

## 7. Changes after the final version

| Date | Section | Change | Reason |
|---|---|---|---|
| 24 Nov 2026 | 1, 2, 3, 4, 6 | Final version: 35 features (`Number` dropped, DATA-4); twin definition and ceiling (DATA-4); N, ladder rungs, tuning (D11 part 2); shot scores (E5); protocol C fallback via DATA-9; MatchedMLP named `mlp_p64` | Draft → final before any tuning; no result had been seen |
