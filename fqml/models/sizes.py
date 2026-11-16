"""Model sizes for the classical comparison (Phase 0 amendment 6, task CLS-1). Owner: Nouran.

No torch import here, so run planning and tests can use it anywhere.

The VQC has 50 trainable parameters (binary) or 64 (8 families); with the linear readout (vqc_lin,
vqc_noent) 57 or 120. One MLP "of the same size" is a weak
comparison: a reader can say a 50-parameter MLP is simply too small. So the MLP is trained at a LADDER
of sizes, the first rung matched to the VQC:

    VQC's own count (50 / 64), 128, 256, 512, 1,000, 3,000 parameters

and the result is reported as the VQC's "classical-equivalent size": the smallest rung whose score is
not lower than the VQC's (rule fixed in docs/PREREG_E0.md, task EXP-1). Every rung is the same model
family (one hidden layer, ReLU); only the hidden width changes.

A one-hidden-layer MLP has n_in*h + h + h*n_out + n_out parameters, so sizes move in steps of
(n_in + n_out + 1): 11 for 8 inputs and 2 outputs, 17 for 8 and 8. Targets are therefore met only
approximately; always log the ACTUAL count (`params`), never the target.
"""
from __future__ import annotations

LADDER_TARGETS = (128, 256, 512, 1_000, 3_000)   # rungs above the VQC-matched one


def mlp_params(n_in: int, hidden: int, n_out: int) -> int:
    return n_in * hidden + hidden + hidden * n_out + n_out


def matched_hidden(n_in: int, n_out: int, target: int) -> int:
    """Hidden width (>= 1) whose parameter count is closest to `target`; ties go to the smaller width."""
    step = n_in + n_out + 1
    guess = max(1, (target - n_out) // step)
    return min((h for h in (guess - 1, guess, guess + 1) if h >= 1),
               key=lambda h: (abs(mlp_params(n_in, h, n_out) - target), h))


def size_ladder(n_in: int, n_classes: int, vqc_params: int) -> list[dict]:
    """The rungs for one task: [{"target", "hidden", "params", "token"}], smallest first.

    `vqc_params` is the VQC's count for the same task (fqml.models.vqc.n_params). `token` is the model
    name used in run IDs (fqml.utils.run_id), e.g. "mlp_p128".
    """
    targets = [vqc_params] + [t for t in LADDER_TARGETS if t > vqc_params]
    rungs = []
    for t in targets:
        h = matched_hidden(n_in, n_classes, t)
        rungs.append({"target": t, "hidden": h, "params": mlp_params(n_in, h, n_classes),
                      "token": f"mlp_p{t}"})
    return rungs
