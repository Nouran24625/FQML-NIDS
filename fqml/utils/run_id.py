"""Run IDs (INTERFACES 5; Phase 0 amendment 4, task EVAL-3). Owner: Sheghaf.

    E{n}-{dataset}-{model}-{variant}-{protocol}-f{fold}-s{seed}
    E0-ciciot2023-vqc-c8.top8-A-f0-s0

  experiment  E0, E1, ...
  dataset     ciciot2023, ciciomt2024, cicids2017
  model       vqc, vqc_noent, mlp_p50 ... mlp_p3000 (fqml.models.sizes), rff, lr, xgb, rf, ...
  variant     everything else that defines the setting, joined with dots: task (c2 binary, c8 eight
              families, c7m DDoS and DoS merged), features (top8, all), federated split (iid, dir0.5,
              sites), privacy (eps8), ...
  protocol    A time-blocked folds, B unseen attack type, C independent dataset,
              R old random row split (leakage reference only, never a reported result)
  fold        0..4 for protocol A; index of the held-out group for B; 0 for C and R
  seed        0, 1, 2

Every reported setting runs on all of its folds x seeds 0, 1, 2 (protocol A: 5 x 3 = 15 runs). The
fold is the unit of comparison: seeds are averaged inside a fold first. Fold 0, seed 0 runs first; the
rest only after it looks right. This replaces the Phase 0 convention "seeds 0, 1, 2 on one split".

The parts are separated by "-", so no part may contain one; use "_" or "." inside a part.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from itertools import product

PROTOCOLS = ("A", "B", "C", "R")
SEEDS = (0, 1, 2)
N_FOLDS = {"A": 5, "C": 1, "R": 1}          # B: one fold per held-out group, set by its manifest

_PART = re.compile(r"^[A-Za-z0-9_.+]+$")
_ID = re.compile(r"^E(\d+)-([^-]+)-([^-]+)-([^-]+)-([ABCR])-f(\d+)-s(\d+)$")


@dataclass(frozen=True)
class RunID:
    experiment: int
    dataset: str
    model: str
    variant: str
    protocol: str
    fold: int
    seed: int

    def __post_init__(self):
        for name in ("dataset", "model", "variant"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _PART.match(value):
                raise ValueError(f"{name}={value!r}: letters, digits, '_', '.', '+' only (no '-')")
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"protocol={self.protocol!r}: one of {PROTOCOLS}")
        for name in ("experiment", "fold", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name}={value!r}: a whole number >= 0")

    def __str__(self):
        return (f"E{self.experiment}-{self.dataset}-{self.model}-{self.variant}-"
                f"{self.protocol}-f{self.fold}-s{self.seed}")

    def asdict(self) -> dict:
        return asdict(self)


def make_run_id(experiment, dataset, model, variant, protocol, fold, seed) -> str:
    return str(RunID(experiment, dataset, model, variant, protocol, fold, seed))


def parse_run_id(run_id: str) -> RunID:
    m = _ID.match(run_id)
    if not m:
        raise ValueError(f"not a run ID: {run_id!r} (expected E<n>-<dataset>-<model>-<variant>-"
                         f"<protocol>-f<fold>-s<seed>)")
    e, dataset, model, variant, protocol, fold, seed = m.groups()
    return RunID(int(e), dataset, model, variant, protocol, int(fold), int(seed))


def setting_runs(experiment, dataset, model, variant, protocol, n_folds=None, seeds=SEEDS) -> list[str]:
    """All run IDs of one setting, in launch order: fold 0 / seed 0 first, then by seed, then by fold."""
    if n_folds is None:
        if protocol not in N_FOLDS:
            raise ValueError("protocol B: pass n_folds (the number of held-out groups in its manifest)")
        n_folds = N_FOLDS[protocol]
    return [make_run_id(experiment, dataset, model, variant, protocol, f, s)
            for s, f in product(seeds, range(n_folds))]
