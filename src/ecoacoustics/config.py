# Generated with Claude Code (Anthropic): load config.yaml and resolve repo-relative paths.
"""Load the project configuration and resolve paths relative to the repo root."""

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML configuration file."""
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Return the absolute path of an entry in the ``paths`` section."""
    return REPO_ROOT / config["paths"][key]
