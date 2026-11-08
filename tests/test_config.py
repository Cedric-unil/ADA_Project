# Generated with Claude Code (Anthropic): test config loading and path resolution.
"""Tests for configuration loading."""

from pathlib import Path

import pytest

from ecoacoustics.config import CONFIG_ENV_VAR, REPO_ROOT, load_config, resolve_path


def test_config_paths_stay_inside_the_repo() -> None:
    config = load_config()
    for key in config["paths"]:
        assert resolve_path(config, key).is_relative_to(REPO_ROOT)


def test_an_environment_variable_can_name_another_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "reduced.yaml"
    other.write_text("seed: 7\n", encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(other))

    assert load_config() == {"seed": 7}


def test_an_explicit_path_comes_before_the_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    named, explicit = tmp_path / "named.yaml", tmp_path / "explicit.yaml"
    named.write_text("seed: 7\n", encoding="utf-8")
    explicit.write_text("seed: 8\n", encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(named))

    assert load_config(explicit) == {"seed": 8}
