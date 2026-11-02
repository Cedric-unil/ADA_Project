# Generated with Claude Code (Anthropic): test that one seed call reproduces draws.
"""Tests for the seeding helper."""

import random

import numpy as np
import torch

from ecoacoustics.utils import set_seeds


def _draw() -> tuple[float, np.ndarray, torch.Tensor]:
    return random.random(), np.random.rand(3), torch.rand(3)  # noqa: NPY002


def test_same_seed_reproduces_draws() -> None:
    set_seeds(123)
    first = _draw()
    set_seeds(123)
    second = _draw()
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])


def test_different_seeds_give_different_draws() -> None:
    set_seeds(123)
    first = _draw()
    set_seeds(124)
    second = _draw()
    assert first[0] != second[0]
