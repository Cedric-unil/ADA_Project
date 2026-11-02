# Generated with Claude Code (Anthropic): test config loading and path resolution.
"""Tests for configuration loading."""

from ecoacoustics.config import REPO_ROOT, load_config, resolve_path


def test_config_paths_stay_inside_the_repo() -> None:
    config = load_config()
    for key in config["paths"]:
        assert resolve_path(config, key).is_relative_to(REPO_ROOT)
