# Generated with Claude Code (Anthropic): load config.yaml and resolve repo-relative paths.
"""Load the project configuration and resolve paths relative to the repo root."""

import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
# The tests run the real stages on a reduced configuration by naming its file in this variable.
CONFIG_ENV_VAR = "ECOACOUSTICS_CONFIG"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Read the YAML configuration file.

    That is ``path`` if given, else the file named by ``ECOACOUSTICS_CONFIG``, else config.yaml.
    """
    path = path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Return the absolute path of an entry in the ``paths`` section."""
    return REPO_ROOT / config["paths"][key]
