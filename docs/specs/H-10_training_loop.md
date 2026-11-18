# Spec H-10 — the shared training loop: validation, early stopping, probabilities

**Author:** Nouran (AI and quantum ML) · **Builds it:** Sheghaf · **Reviewer:** Nouran · **Due:** Wed 25 Nov 2026
**File:** `fqml/train/loop.py` (extended, the old `train()` keeps working), tests in `tests/test_train_loop.py`
**Used by:** every PyTorch model (VQC, entanglement-free VQC, MLP size ladder, RFF), the experiment runner (EXP-0),
DP-SGD (PRIV-2).

The loop is the same for every model: that is what makes the quantum–classical comparison fair. Today it trains
for a fixed number of epochs and has no validation. It needs the four things below.

## 1. What changes

### 1.1 Class weights: D11 rule, unchanged
Weighted cross-entropy with weights ∝ 1/√(family count in the training sample), mean 1 (the current
`class_weights`). Decision D11 (part 1) chose this rule for every model, XGBoost included.
- **Keep the function as it is**, and add `class_weights(y, n_classes, allow_absent=False)`: under protocol B a
  family can be absent from training (Spoofing in fold 0), so a zero count must not divide by zero. An absent
  family gets weight 0 and the model still has an output for it.

### 1.2 Validation and early stopping
```python
fit(model, X, y, n_classes, *, X_val, y_val, max_epochs=100, batch_size=64, lr=0.01, seed=0,
    patience=10, eval_every=1, select="macro_f1", max_seconds=None, on_epoch=None) -> FitResult
```
- After every `eval_every` epochs, score the **validation block** with `fqml.eval.metrics.multiclass` and keep a
  copy of the weights with the best `select` score (`macro_f1` of the model's own task; for the binary task
  attack recall at 1% FPR is not usable here because it needs a threshold, so binary models select on macro-F1
  of the 2 classes).
- Stop when the score has not improved for `patience` evaluations, at `max_epochs`, or after `max_seconds`
  (VQC runs on Colab or Kaggle need a time limit).
- **At the end the model holds the best weights, not the last ones.**
- `FitResult`: `best_epoch`, `best_score`, `stopped_because` (`"patience" | "max_epochs" | "max_seconds"`),
  `history` (per evaluation: epoch, training loss, validation macro-F1, seconds), `n_params`.
- The validation block is used **only** to choose the epoch (and, in D11 part 2, the tuning trial). It never
  becomes training data. The test block is never seen by `fit`.

### 1.3 Probabilities
```python
predict_proba(model, X, batch_size=8192) -> np.ndarray   # [n, n_classes], rows sum to 1, float32
```
Softmax of the logits, computed in batches under `torch.no_grad()` with the model in eval mode. The runner
turns it into `p = proba.argmax(1)` and `attack_score = 1 − proba[:, 0]` for `fqml.eval.metrics`.
`evaluate()` stays for old callers but returns probabilities too.

### 1.4 Determinism
- Same `seed`, same data, same machine ⇒ same weights, same `FitResult` (the batch order comes from a
  `torch.Generator` seeded with `seed`; the model's starting weights come from the model's own `seed`).
- `fit` never touches the global torch random state (the MLP `seed=` argument already follows this rule).
- Threads: `torch.set_num_threads(1)` is the caller's job (decision D2), not `fit`'s.

## 2. Inputs
- `X`: float32 in [0, π], `[n, n_features]` (INTERFACES 2.1), already preprocessed per fold (DATA-6).
- `y`: int64 family index for the task (`y` for 8-class, `family_index(..., "c7m")` for 7-class,
  `(y != 0)` for binary).
- The training rows are a per-attack-type sample spread over time (D11, `fqml.data.rows.cap_per_label`); the
  sample size per model is set in D11 part 2.

## 3. Tests (`tests/test_train_loop.py`)
1. **Early stopping restores the best weights:** on a small dataset where validation macro-F1 rises then falls,
   the returned model's validation score equals `best_score` and `best_epoch` is not the last epoch.
2. **Patience:** with `patience=2` and a validation score that never improves after epoch 1, training stops at
   epoch 3 with `stopped_because == "patience"`.
3. **Time limit:** `max_seconds=0.5` on a slow setting stops with `"max_seconds"`.
4. **Determinism:** two `fit` calls with the same seed give identical weights and history; a different seed
   gives different weights; the global torch RNG is the same before and after `fit`.
5. **Every model:** `fit` runs for `TorchVQC` (local and linear readout, chain and no entangler), `MatchedMLP`
   at two ladder sizes, binary and 8-class.
6. **Probabilities:** `predict_proba` rows sum to 1, shape `[n, n_classes]`, and `argmax` equals `model(X).argmax`.
7. **Absent family:** `class_weights(..., allow_absent=True)` with a missing class gives weight 0 for it and
   finite weights elsewhere; `fit` then trains without error.
8. **Old API:** the current `train()` and `evaluate()` calls in the repository still work.

## 4. Not in this task
Tuning trials and their budget (D11 part 2), DP-SGD (PRIV-2), finite shots (Q-1), the federated loop's own
evaluation (FED-1 calls `fqml.eval.metrics`).
