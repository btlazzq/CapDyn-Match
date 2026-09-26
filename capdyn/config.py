"""YAML config loading. Paths in configs are user-supplied, never baked in."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    cfg["_config_path"] = str(path)
    return cfg


def require(cfg: dict, key: str):
    if key not in cfg:
        raise KeyError(f"missing config key: {key}")
    return cfg[key]
