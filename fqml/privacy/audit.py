"""One-run privacy audit with canaries (task PRIV-3, check 3). Owner: Nouran.

Steinke, Nasr and Jagielski (NeurIPS 2023), "Privacy auditing with one (1) training run": plant m canary rows, include
each one in the training set independently with probability 1/2, train ONCE, then guess for each canary whether it
was included, from how well the trained model fits it. Under epsilon-DP the number of correct guesses among r guesses
is stochastically dominated by Binomial(r, e^eps / (1 + e^eps)). Observing many correct guesses therefore rules out
small epsilons; the largest epsilon ruled out at confidence 1 - beta is an empirical LOWER bound on the epsilon the
training actually spent.

PRIV-3 passes if, for every private run, the empirical lower bound stays below the accountant's epsilon (an empirical
bound above it means the DP-SGD code leaks more than it claims). A non-private run (epsilon = infinity) is audited too,
to show the audit has the power to detect leakage at all.

Canaries: points drawn uniformly in [0, pi]^d with a uniformly random label, so the only way to fit one is to
memorise it. Score: minus the cross-entropy of the trained model on the canary's own label. Guesses: the k highest
scores are called "included", the k lowest "excluded", the rest abstain (k fixed in advance, k = m / 10).
The delta term of the theorem is handled conservatively by testing at beta - m * delta.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def epsilon_lower_bound(correct: int, guesses: int, beta: float = 0.05, delta: float = 0.0, m: int = 0,
                        eps_max: float = 30.0) -> float:
    """Largest epsilon ruled out at confidence 1 - beta by `correct` right guesses out of `guesses`."""
    from scipy.stats import binom

    target = beta - delta * m
    if guesses <= 0 or correct <= 0 or target <= 0:
        return 0.0

    def tail(eps):                                    # P[Binomial(r, p_eps) >= correct]
        p = 1.0 / (1.0 + math.exp(-eps))
        return float(binom.sf(correct - 1, guesses, p))

    if tail(0.0) > target:                            # not even epsilon = 0 can be ruled out
        return 0.0
    if tail(eps_max) <= target:
        return eps_max
    lo, hi = 0.0, eps_max                             # tail(lo) <= target < tail(hi)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tail(mid) <= target else (lo, mid)
    return lo


def make_canaries(m: int, n_features: int, n_classes: int, seed: int = 0):
    """(X [m, d] float32 in [0, pi], y [m] int64, included [m] bool) with P(included) = 1/2, independently."""
    rng = np.random.default_rng(seed)
    X = (rng.random((m, n_features)) * np.pi).astype(np.float32)
    y = rng.integers(0, n_classes, m).astype(np.int64)
    included = rng.random(m) < 0.5
    return X, y, included


@torch.no_grad()
def canary_scores(model, X, y) -> np.ndarray:
    """Minus the cross-entropy of the model on each canary's own label: higher = fits it better = more likely in."""
    model.eval()
    dev = next(model.parameters()).device
    logits = model(torch.as_tensor(X, device=dev))
    loss = F.cross_entropy(logits, torch.as_tensor(y, device=dev), reduction="none")
    model.train()
    return (-loss).cpu().numpy()


def guess_and_count(scores: np.ndarray, included: np.ndarray, k: int) -> tuple[int, int]:
    """Top-k scores guessed 'in', bottom-k guessed 'out'. Returns (correct, guesses)."""
    order = np.argsort(scores)
    out_guess, in_guess = order[:k], order[-k:]
    correct = int(included[in_guess].sum() + (~included[out_guess]).sum())
    return correct, 2 * k


def run_audit(build_model, X, y, n_classes: int, *, m: int = 1000, k: int | None = None, target_epsilon: float,
              delta: float = 1e-6, beta: float = 0.05, seed: int = 0, **dp_kwargs) -> dict:
    """Train once with the included canaries added to (X, y), with fqml.privacy.dpsgd.dp_fit; audit the result.

    build_model: callable(seed) -> a fresh model (the same architecture the pilot uses)."""
    from fqml.privacy.dpsgd import dp_fit

    Xc, yc, inc = make_canaries(m, X.shape[1], n_classes, seed)
    Xt = np.concatenate([np.asarray(X, np.float32), Xc[inc]])
    yt = np.concatenate([np.asarray(y, np.int64), yc[inc]])
    model = build_model(seed)
    res = dp_fit(model, Xt, yt, n_classes, target_epsilon=target_epsilon, delta=delta, seed=seed, **dp_kwargs)
    k = k or max(1, m // 10)
    correct, guesses = guess_and_count(canary_scores(model, Xc, yc), inc, k)
    eps_low = epsilon_lower_bound(correct, guesses, beta=beta, delta=0.0 if math.isinf(target_epsilon) else delta,
                                  m=m)
    return {"target_epsilon": target_epsilon, "accountant_epsilon": res.epsilon, "noise_multiplier": res.noise_multiplier,
            "empirical_epsilon_lower_bound": eps_low, "correct": correct, "guesses": guesses,
            "canaries": m, "canaries_included": int(inc.sum()), "beta": beta, "delta": delta, "steps": res.steps,
            "passes": bool(math.isinf(target_epsilon) or eps_low <= res.epsilon)}
