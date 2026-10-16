from __future__ import annotations

from pathlib import Path

import yaml

_DIR = Path(__file__).parent


def load_label_map(name: str) -> tuple[list[str], dict[str, str]]:
    spec = yaml.safe_load((_DIR / f"{name}.yaml").read_text())
    return spec["families"], spec["map"]
