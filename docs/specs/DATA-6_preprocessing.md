# Spec DATA-6 — per-fold preprocessing: features, clipping, scaling to [0, π]

**Author:** Nouran (AI and quantum ML) · **Builds it:** Ganna · **Reviewer:** Nouran · **Due:** Fri 27 Nov 2026
**Files:** `fqml/data/preprocess.py`, tests in `tests/test_preprocess.py`, fitted settings in
`results/preprocess/<dataset>_<protocol>_f<fold>.json` (committed, small)
**Used by:** every E0 model through the runner (EXP-0), the federated runs (FED-4), DP-SGD (PRIV-2)
**Interfaces:** INTERFACES 1.3 (owner Ganna). The pull request updates 1.3 to match this spec ("Interface change").

The VQC encodes each input as a rotation angle, so inputs must lie in [0, π]. Every model, quantum or classical,
gets the **same** inputs from the **same** fitted settings: that is what keeps the comparison fair. Everything is
fitted on one fold's **training** rows only; validation and test rows are only transformed.

## 1. Features entering preprocessing
- Start from the 37 features of cleaner v1 (`report["features"]`).
- **Excluded:** `IAT` (D9, accepted 19 Nov) and `Number` (DATA-4: it is the packet-window size, 100 for
  DDoS/DoS/Mirai and 10 for the rest, a recording setting; proposed as an addition to D9, approval at the
  Thursday sync on 26 Nov). `Time_To_Live` is not in cleaner v1 and stays out (DATA-4).
- **35 features remain.** The exclusion list is a parameter (`exclude=("IAT", "Number")`), so a re-run without
  excluding `Number` is one argument if the team decides otherwise.
- Never inputs: `row_id`, `file`, `file_seq`, `row_index`, `dup_group`, `conflict`, `sample_rate`, `ts`, `y`.

## 2. The fitting sample
- Fitted on a **fixed sample of the fold's training rows**: at most 20,000 rows per native attack type, spread
  evenly over time (`fqml.data.rows.cap_per_label(train_rows, label_ranges, 20_000)`, no randomness).
- The same fitting sample serves every model, whatever its own training-sample size (D11 part 2: N = 5,000 or
  50,000). Fitting on each model's own sample would give models different inputs.
- About 600,000 rows per fold (fewer for small attack types). Labels for selection: the 8-class family `y`.

## 3. Steps, in order (all fitted on the fitting sample)
1. **Log for heavy tails.** A feature gets `log1p` if its training minimum is ≥ 0 **and** its 99th percentile is
   more than 100 times its median (median 0 counts as heavy if the 99th percentile is > 0 and the feature is not
   binary). Fixed rule, decided per fold from training rows only, recorded per feature. Reason: `Rate`, `Tot sum`,
   `Header_Length` and similar span several orders of magnitude; min-max on the raw values would put most rows
   at angle ≈ 0 and waste the VQC's input range.
2. **Clip** each feature (after step 1) to its 1st and 99th training percentiles `[lo, hi]`.
3. **Constant features:** if `hi == lo`, the feature is constant on this fold. It maps to 0 in the all-features
   set and can never be chosen for the top 8.
4. **Feature ranking (top 8).** Mutual information between each feature and `y`, from a **binned estimate**:
   the clipped feature is cut into 32 equal-count bins (ties kept in one bin), MI computed from the bin-by-class
   counts, in nats. Deterministic and fast (no k-NN estimator).
5. **Redundancy rule.** Take features in decreasing MI. Skip a feature if its absolute Spearman correlation with
   an already chosen feature is ≥ 0.95 on the fitting sample; continue until 8 are chosen. Reason: several
   CICIoT2023 features are near-copies (`Tot sum`, `Tot size`, `AVG`, `Max`, `Min`); 8 qubits should not spend
   two inputs on one signal. Skipped features and the correlation that skipped them are recorded.
6. **Scale** to [0, π]: `x' = π · (clip(x) − lo) / (hi − lo)`, float32. Validation and test rows are clipped to
   the training `[lo, hi]` first, so every output is inside [0, π].

Two feature sets come out of one fit: **`all`** (35 features, fixed order of `report["features"]`) and **`top8`**
(8 features in rank order). Both use the same per-feature settings.

## 4. API
```python
from fqml.data.preprocess import fit, Preprocessor

pre = fit(train_df, y_train, features=report_features, exclude=("IAT", "Number"),
          top_k=8, log_ratio=100.0, n_bins=32, max_abs_rho=0.95)
X = pre.transform(df, feature_set="top8")        # float32 [n, 8], values in [0, π]
X = pre.transform(df, feature_set="all")         # float32 [n, 35]
pre.to_json(path); pre = Preprocessor.from_json(path)
```
`Preprocessor` holds, per feature: `log` (bool), `lo`, `hi`, `constant` (bool), `mi`; plus `top8` (names in rank
order), `skipped` (name, the chosen feature it duplicated, ρ), `exclude`, the fitting-sample size per family,
`dataset`, `protocol`, `fold`, a format tag `fqml-preprocess/1`, and the git commit.

Command line (fits and saves every fold of one protocol, about a minute per fold):
```
python -m fqml.data.preprocess --dataset ciciot2023 --protocol A            # all 5 folds
python -m fqml.data.preprocess --dataset ciciot2023 --protocol B --fold 0
```
Output: `results/preprocess/ciciot2023_A_f0.json` … and a printed table of the top 8 per fold.

## 5. Tests (`tests/test_preprocess.py`)
1. **Range:** on training, validation and made-up out-of-range rows, every output is finite, float32, in [0, π].
2. **Training rows only:** changing validation or test rows changes nothing in the fitted settings; changing a
   training row in the fitting sample can.
3. **Log rule:** a heavy-tailed made-up feature gets `log=True`; a uniform one and a 0/1 flag get `log=False`.
4. **Constant feature:** maps to 0 and is never in `top8`.
5. **MI ranking:** with one feature equal to the label (plus noise) and seven noise features, the label feature
   ranks first; MI of a pure-noise feature is near 0.
6. **Redundancy:** an exact copy of the top feature is skipped and recorded in `skipped`.
7. **Exclusion:** `IAT` and `Number` never appear in `all` or `top8`; `exclude=("IAT",)` keeps `Number`.
8. **Round trip:** `to_json` → `from_json` gives identical `transform` output.
9. **Determinism:** two fits on the same rows give identical JSON (apart from the commit field).

## 6. What the E0 report shows from it
The top-8 lists of all folds side by side (stability of the choice across folds), and the MI values. If the
top 8 differ between folds, that is reported, not hidden: each fold uses its own list, as fitted on its own
training rows.

## 7. Not in this task
Protocol C features (they follow DATA-9's stability audit of CICIoMT2024); federated preprocessing (each client
fitting its own settings) is out of Phase 1.
