# Generated with Claude Code (Anthropic): test the same-period sensitivity check.
"""Tests for the same-period check. They use small synthetic data, never the real data."""

import numpy as np
import pytest

from ecoacoustics.sensitivity import (
    ALL_OTHER_TAPES,
    RANDOM_DROP,
    SHARED_DIGITS_DROP,
    comparable_cuts,
    first_two_digits,
    random_tapes,
    same_period_check,
    tapes_sharing_digits,
    validation_overlap,
)

SPECIES_ON = {
    "60001": {0},
    "60002": {0},
    "60003": {1},
    "61001": {0},
    "61002": {0},
    "62001": {1},
    "62002": {1},
}


def test_first_two_digits_of_tape_numbers() -> None:
    assert first_two_digits(np.array(["72021", "61079"])).tolist() == ["72", "61"]


def test_comparable_cuts_need_two_values_of_the_first_digits() -> None:
    y = np.array([0, 0, 1, 1])
    tapes = np.array(["60001", "61001", "60002", "60003"])

    assert comparable_cuts(y, tapes).tolist() == [True, True, False, False]


def test_tapes_sharing_digits_leave_out_the_tape_itself() -> None:
    assert tapes_sharing_digits("60001", SPECIES_ON) == ["60002", "60003"]
    assert tapes_sharing_digits("61001", SPECIES_ON) == ["61002"]


@pytest.mark.parametrize("seed", range(5))
def test_random_tapes_drop_as_many_tapes_of_each_kind(seed: int) -> None:
    rng = np.random.default_rng(seed)

    # Tape 60001 shares its digits with one tape of its species (60002) and one other (60003).
    chosen = random_tapes("60001", SPECIES_ON, rng)

    assert "60001" not in chosen
    assert len(set(chosen)) == len(chosen) == 2
    assert sum(SPECIES_ON[tape] == {0} for tape in chosen) == 1
    assert sum(SPECIES_ON[tape] == {1} for tape in chosen) == 1


def _period_task() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three species, each sounding different in each of two periods: tapes must share a period."""
    rng = np.random.default_rng(0)
    x, y, tapes = [], [], []
    for species in range(3):
        for period in range(2):
            centre = rng.normal(scale=5.0, size=6)
            for tape in range(3):
                x.append(centre + rng.normal(scale=0.3, size=(4, 6)))
                y += [species] * 4
                tapes += [f"{60 + period}{species}{tape:02d}"] * 4
    return np.vstack(x), np.array(y), np.array(tapes)


def test_the_check_shows_when_tapes_sharing_their_digits_matter() -> None:
    x, y, tapes = _period_task()

    table = same_period_check(x, y, tapes, 3, 1.0, 2000, n_repeats=3, seed=0)

    assert table["Trained on"].tolist() == [ALL_OTHER_TAPES, RANDOM_DROP, SHARED_DIGITS_DROP]
    all_other, random_drop, sharing_drop = table["Macro-F1"]
    assert all_other > 0.9
    # Dropping the same number of tapes at random usually leaves one from the same period.
    assert sharing_drop < random_drop
    assert table.loc[1, "Lowest repeat"] <= random_drop <= table.loc[1, "Highest repeat"]
    assert table[["Lowest repeat", "Highest repeat"]].isna().sum().tolist() == [2, 2]


def test_the_check_is_reproducible_with_the_same_seed() -> None:
    x, y, tapes = _period_task()

    first = same_period_check(x, y, tapes, 3, 1.0, 2000, n_repeats=2, seed=1)
    second = same_period_check(x, y, tapes, 3, 1.0, 2000, n_repeats=2, seed=1)

    assert first.equals(second)


def test_validation_overlap_counts_matching_training_tapes() -> None:
    y = np.array([0, 0, 0, 1])
    tapes = np.array(["60001", "60002", "61001", "60003"])
    folds = np.array([0, 1, 1, 1])

    table = validation_overlap(y, tapes, folds)

    # Cut 0 and cut 1 each have the other's tape in training; cuts 2 and 3 have no match.
    assert table.iloc[0].tolist() == [4, 2, 50.0]
