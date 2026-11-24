# Spec EVAL-1 — the shared metrics module

**Author:** Nouran (AI and quantum ML) · **Builds it:** Sheghaf · **Reviewer:** Nouran · **Due:** Wed 25 Nov 2026
**File:** `fqml/eval/metrics.py`, tests in `tests/test_metrics.py` · **Used by:** every model, the experiment
runner (EXP-0), the federated loop (FED-1), the tables script (EVAL-4).

Checkpoint 1 (`docs/DECISIONS.md`, D8a, D11, D17) decided what this module must measure. Its code in
`fqml/eval/checkpoint1.py` (`score`, `recall_at_fpr`, `per_type_recall`, `unseen_scores`) is the **reference
implementation**: port it here, then add what is listed below. After that, `checkpoint1.py` imports from this
module.

## 1. Rules every function follows

1. **Thresholds are chosen on the validation block and applied to the test block. Never the reverse, never on
   test alone.** Any function that sets a threshold returns it, so the caller can apply it to test.
2. **Inputs are plain arrays:** true family `y` (int, 0 = Benign), predicted family `p`, and
   `attack_score = 1 − P(benign)` (float in [0, 1]). Models give probabilities; the runner computes the score.
   No metric function sees a model.
3. **Row weights** `w = 1 / sample_rate` are accepted everywhere (default: all 1). With cleaner v1 and D8 every
   `sample_rate` is 1, so results do not change, but the code must not assume it.
4. **Families come from the label map** (`fqml.data.label_maps.load_label_map`). The 7-class scores use variant
   `c7m` through `family_index`, not a hand-written merge.
5. **Every result is plain JSON** (numbers, lists, strings) with `"metrics_version": 1`.
6. **Undefined values are `NaN`, never 0.** A class or type with no rows gives `NaN` and is left out of macro
   averages; the result records which were left out.

## 2. Functions

### 2.1 `multiclass(y, p, families, w=None) -> dict`
Accuracy, **macro-F1**, per-class precision / recall / F1 / support, confusion matrix (rows = true). Macro-F1
averages over the classes that have true rows in `y`; it lists the classes it skipped.

### 2.2 `threshold_at_fpr(attack_score_val, y_val, fpr) -> float`
The alarm threshold at which at most `fpr` of the validation **benign** rows score above it (same rule as
`recall_at_fpr` in Checkpoint 1: `np.quantile(benign_scores, 1 − fpr, method="higher")`, alarm when
`score > threshold`). Used for `fpr` = 0.01 and 0.001.

### 2.3 `binary(y, attack_score, thresholds, w=None) -> dict`
Attack = `y != 0`. For each named threshold (`{"fpr_1pct": t1, "fpr_0.1pct": t2}`): attack recall, the false-
positive rate actually reached, precision. Plus **ROC-AUC**, **benign PR-AUC** (benign as the positive class),
attack PR-AUC, and F1 / FPR at the default decision `p != 0` (reported, never used to judge: D11).

### 2.4 `per_type(row_ids, y, p, attack_score, label_ranges, thresholds) -> dict`
For every native attack type present (types are `row_id` ranges, `fqml.data.clean_ciciot2023_v1.label_ranges`):
rows, family recall, the family it is most often mistaken for and that share, and **detection** (share with
`score > threshold`) at each threshold. Plus two macro averages over attack types (benign excluded):
**mean detection over types at 1% FPR** and **mean family recall over types**. These count a 225-row
Web type as much as a 7-million-row flood. Protocol B showed why this is needed: detection counted over rows
was 98.7%, dominated by the large floods.

### 2.5 `unseen(row_ids, y, p, attack_score, label_ranges, unseen_types, thresholds, families) -> dict`
Protocol B only, on the types the model never saw: rows, detection at each threshold (over rows and averaged
over types), right family (8-class and `c7m`). Same as `unseen_scores` in Checkpoint 1, plus the per-type
average.

### 2.6 `evaluate(...) -> dict` — the one call the runner makes
```python
evaluate(y, p, attack_score, *, families, row_ids, label_ranges,
         thresholds,                   # from threshold_at_fpr on the validation block
         train_dup_groups, dup_group, conflict,   # for the subsets in 2.7
         unseen_types=None, w=None) -> dict
```
Returns `{"metrics_version": 1, "all": {...}, "no_twin": {...}, "no_conflict": {...}, "per_type": {...},
"unseen": {... or absent}, "thresholds": {...}, "rows": n}`. Each subset holds `multiclass` for 8 classes,
`multiclass` for `c7m`, and `binary`.

### 2.7 Subsets
- **no_twin:** test rows whose `dup_group` does not occur among the rows the model trained on
  (`fqml.data.folds.twin_mask`). The runner passes the training rows' `dup_group` values.
- **no_conflict:** rows with `conflict == 0` (D8).

## 3. The numbers that judge a model (for EXP-1)
Fixed by D11 and D17, and listed here so the tables script picks the same ones:
- **Multi-class:** macro-F1, 8-class and `c7m`.
- **Binary:** attack recall at 0.1% FPR and at 1% FPR (threshold from validation), ROC-AUC.
- **Per type:** mean detection over attack types at 1% FPR.
- **Protocol B:** detection of unseen types at 1% FPR (over rows and averaged over types); right family (`c7m`).

Binary F1 and FPR at the default decision are always reported and never used to judge.

## 4. Tests (`tests/test_metrics.py`)
1. **Hand-computed cases:** a 3-class example where every number (macro-F1, confusion, per-class) is checked
   by hand; a binary example with known ROC-AUC.
2. **Threshold rule:** with 1,000 benign scores spread over [0, 1], the threshold at 1% FPR gives at most 10
   alarms; with all scores tied, no alarm is raised and recall is 0, not an error.
3. **Thresholds come from validation:** a test where validation and test benign scores differ; the FPR reached
   on test is reported as measured, not forced to 1%.
4. **c7m matches the label map:** merged scores equal those from `family_index(native, "ciciot2023", "c7m")`.
5. **Weights:** doubling every weight changes nothing; weighting one class 2× equals duplicating its rows.
6. **NaN rule:** a family with no rows is `NaN` and listed as skipped, not counted as 0 in the macro average.
7. **Agreement with Checkpoint 1:** on a small saved example, `evaluate` reproduces the `score()` and
   `unseen_scores()` numbers of `fqml/eval/checkpoint1.py` exactly (when its threshold is fitted on the same
   rows).
8. **JSON:** `json.dumps(evaluate(...))` works and contains `metrics_version`.

## 5. Not in this task
Statistics across folds (paired comparison, bootstrap, classical-equivalent size): **EVAL-2**. Logging the
result file per run: **EVAL-3**. Tables and charts: **EVAL-4**.
