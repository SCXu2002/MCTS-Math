"""Project configuration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_API_CONFIG_PATH = Path(__file__).resolve().parent.parent / "api_config.json"


def load_api_config(config_path: str | None = None) -> dict[str, Any]:
    """Load API settings from a JSON config file if it exists."""
    path = Path(config_path) if config_path else DEFAULT_API_CONFIG_PATH
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"API config must be a JSON object: {path}")

    return config
