# Spec EVAL-2 — statistics for E0: paired comparisons, correction, classical-equivalent size

**Author:** Nouran (AI and quantum ML) · **Builds it:** Sheghaf · **Reviewer:** Nouran · **Due:** Fri 4 Dec 2026
**File:** `fqml/eval/stats.py`, tests in `tests/test_stats.py`
**Implements:** `docs/PREREG_E0.md` §3 (decision rule) and §4 (classical-equivalent size), exactly as written there.
**Inputs:** `results/results_log.csv` (EVAL-3), one row per run (model, protocol, fold, seed, metrics, row counts).

## 1. Why these methods
- **The unit is the fold.** Rows inside a test block are consecutive windows of one recording, so they are not
  independent; a row-level bootstrap would give intervals that are far too narrow. Each model gets one score per
  fold (mean of seeds 0, 1, 2), and two models are compared on the same folds.
- **Folds overlap in training data** (each trains on 3 of 5 blocks), so a plain paired t-test is over-confident.
  The corrected resampled t-test of Nadeau and Bengio (2003) inflates the variance for that overlap.
- **Five folds are too few** for a bootstrap or a Wilcoxon test (Wilcoxon's smallest two-sided p with n = 5 is
  0.0625, so it could never declare a win at 0.05).
- **Six comparisons** are tested together, so Holm's method keeps the chance of any false win at 5%.

## 2. Functions
```python
fold_scores(log: pd.DataFrame, model: str, metric: str, protocol: str, **filters) -> pd.Series
    # index = fold (0..4), value = mean over seeds of `metric`; raises if a fold or a seed is missing

paired_test(a: pd.Series, b: pd.Series, n_train: pd.Series, n_test: pd.Series, alpha: float = 0.05,
            n_comparisons: int = 6) -> dict
    # d_k = a_k - b_k over the folds present in both (must be 5)
    # mean d, sd d (ddof 1), r = mean_k(n_test_k / n_train_k)
    # se = sqrt((1/K + r) * sd^2), t = mean / se, df = K - 1, p = 2 * (1 - T_df(|t|))
    # returns {mean, sd, se, t, df, p, ci95: (lo, hi), ci_bonf: (lo, hi) at 1 - alpha / n_comparisons, d: list}

holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]
    # Holm step-down; returns per comparison {p, p_adjusted, reject}; p_adjusted is monotone and <= 1

outcome(test: dict, adjusted: dict, margin: float = 0.01) -> str
    # "wins" | "loses" | "no detectable difference"; a win with mean < margin -> "wins (negligible)"

e0_primary(log, protocol="A") -> pd.DataFrame
    # the six comparisons of PREREG §3: (vqc, mlp_p64), (vqc, rff), (vqc_lin, vqc_noent), each on the 8-class
    # and the 7-class macro-F1; one row each with every field above, Holm applied across the six
    # protocols B and C: the same table, labelled "reported, does not decide"

classical_equivalent_size(ladder: dict[int, pd.Series], vqc: pd.Series) -> dict
    # per fold: ladder scores (params -> score), made non-decreasing by isotonic regression
    # (sklearn.isotonic.IsotonicRegression, increasing=True, on log(params)); the VQC score is located on that
    # curve and the parameter count is interpolated linearly in log(params) between neighbouring rungs.
    # Below the first rung -> "< {min}", above the last -> "> {max}". Returns {per_fold: [...], median, min, max}.
```
`n_train` and `n_test` are the fold's **training-sample** and **test-block** row counts from the run log (the
training sample is what the model learned from, so it sets the overlap term).

## 3. Tests (`tests/test_stats.py`)
1. **Correction:** with n_test / n_train = 0 the corrected test equals `scipy.stats.ttest_rel`; with r > 0 its
   standard error is larger by exactly sqrt((1/K + r) / (1/K)).
2. **Known values:** d = [0.02, 0.03, 0.01, 0.025, 0.015], r = 1/3 gives the hand-computed t and p (in the test).
3. **Holm:** against a hand-worked example of six p-values; adjusted values monotone; with one p below 0.05/6
   and five at 0.5, only the first is rejected.
4. **Outcomes:** a clear positive difference → "wins"; clear negative → "loses"; noisy → "no detectable
   difference"; positive but mean 0.004 and significant → "wins (negligible)".
5. **Classical-equivalent size:** on a ladder 64 → 0.50, 128 → 0.60, 256 → 0.70, a VQC at 0.60 gives 128, at 0.65
   gives sqrt(128 · 256) ≈ 181 (log interpolation); a non-monotone ladder is smoothed first; below 0.50 gives
   "< 64".
6. **Missing data:** a missing fold or seed raises with the run IDs that are missing.

## 4. Output
`scripts/make_e0_tables.py` (EVAL-4) calls `e0_primary` and `classical_equivalent_size` and writes the tables; no
number in the E0 report is typed by hand.
