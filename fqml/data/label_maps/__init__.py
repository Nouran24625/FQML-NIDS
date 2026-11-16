from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

_DIR = Path(__file__).parent


def load_label_map(name: str, variant: str | None = None) -> tuple[list[str], dict[str, str]]:
    """(families, native label -> family). Family index = position in `families`; 0 is always Benign.

    variant=None is the primary map. A variant (e.g. "c7m" for CICIoT2023: DDoS and DoS merged, 7
    classes) replaces the family list and renames the merged families.
    """
    spec = yaml.safe_load((_DIR / f"{name}.yaml").read_text())
    families, mapping = spec["families"], spec["map"]
    if variant is not None:
        variants = spec.get("variants") or {}
        if variant not in variants:
            raise KeyError(f"{name}: no label-map variant {variant!r} (have: {sorted(variants)})")
        v = variants[variant]
        merge = v.get("merge", {})
        families = v["families"]
        mapping = {native: merge.get(fam, fam) for native, fam in mapping.items()}
    unknown = sorted(set(mapping.values()) - set(families))
    if unknown or families[0] != "Benign" or len(set(families)) != len(families):
        raise ValueError(f"{name}/{variant}: bad family list {families} (unmapped: {unknown})")
    return families, mapping


def family_index(native_labels, name: str, variant: str | None = None) -> np.ndarray:
    """Class index (int64) for an array / Series of native labels under the chosen map or variant."""
    families, mapping = load_label_map(name, variant)
    index = {native: families.index(fam) for native, fam in mapping.items()}
    native = np.asarray(native_labels, dtype=object)
    out = np.fromiter((index.get(v, -1) for v in native), dtype=np.int64, count=len(native))
    if (out < 0).any():
        raise ValueError(f"unknown native labels: {sorted(set(native[out < 0]))[:5]}")
    return out
