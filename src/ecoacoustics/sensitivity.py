# Generated with Claude Code (Anthropic): the same-period sensitivity check on training data.
"""How much do tapes that share their first two digits help, beyond simply being more data?

Whole tapes are kept apart in every split, but tapes with close numbers may come from the same
period, trip or equipment. The first two digits of a tape number are only possibly its recording
year, which is unconfirmed, so this module speaks of tapes that share their first two digits.

Each tape is held out in turn. A logistic regression on whole-cut statistics predicts its cuts
after training on one of three sets of tapes:

- every other tape;
- every other tape except those sharing its first two digits;
- every other tape except the same number of tapes, of the same kinds (sharing a species with the
  held-out tape or not), chosen at random whatever their first two digits, repeated several times.

The last two sets leave the same number of tapes of each kind, so a gap between them does not come
from having fewer tapes. Scores cover the species whose tapes have at least two different values
of the first two digits, because the others have nothing left to train on in the second set.

Only training rows may be passed in: this check is not allowed to see the test set.
"""

import numpy as np
import pandas as pd

from ecoacoustics.evaluation import macro_f1
from ecoacoustics.models import fit_predict_logistic

ALL_OTHER_TAPES = "All other tapes"
RANDOM_DROP = "Same number of tapes dropped at random"
SHARED_DIGITS_DROP = "Tapes sharing the first two digits dropped"


def first_two_digits(tapes: np.ndarray) -> np.ndarray:
    """The first two digits of each tape number."""
    return np.array([tape[:2] for tape in tapes])


def comparable_cuts(y: np.ndarray, tapes: np.ndarray) -> np.ndarray:
    """Which cuts belong to a species whose tapes have at least two values of the first digits."""
    values = pd.Series(first_two_digits(tapes)).groupby(y).nunique()
    return np.isin(y, values[values >= 2].index.to_numpy())


def tapes_sharing_digits(tape: str, species_on: dict[str, set[int]]) -> list[str]:
    """The other tapes with the same first two digits as ``tape``."""
    return [other for other in species_on if other != tape and other[:2] == tape[:2]]


def random_tapes(
    tape: str, species_on: dict[str, set[int]], rng: np.random.Generator
) -> list[str]:
    """As many tapes as :func:`tapes_sharing_digits`, of the same kinds, chosen at random."""
    sharing = set(tapes_sharing_digits(tape, species_on))
    others = [other for other in species_on if other != tape]
    same_species = [other for other in others if species_on[other] & species_on[tape]]
    other_species = [other for other in others if not species_on[other] & species_on[tape]]
    chosen: list[str] = []
    for pool in (same_species, other_species):
        n_to_drop = sum(other in sharing for other in pool)
        if n_to_drop:
            chosen += rng.choice(pool, n_to_drop, replace=False).tolist()
    return chosen


def same_period_check(
    x: np.ndarray,
    y: np.ndarray,
    tapes: np.ndarray,
    n_classes: int,
    c: float,
    max_iter: int,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Macro-F1 of the three training sets, as a table with the range of the random repeats.

    ``x`` holds one row of whole-cut statistics per training cut, ``y`` its class and ``tapes``
    its tape number; ``c`` and ``max_iter`` are the logistic regression's settings.
    """
    rng = np.random.default_rng(seed)
    species_on = {tape: set(y[tapes == tape].tolist()) for tape in sorted(set(tapes.tolist()))}
    scored = comparable_cuts(y, tapes)

    def predict(tape: str, dropped: list[str]) -> np.ndarray:
        kept = ~np.isin(tapes, [tape, *dropped])
        held_out = tapes == tape
        return fit_predict_logistic(
            x[kept], y[kept], x[held_out], n_classes, c, max_iter
        ).argmax(axis=1)

    all_other = np.empty(len(y), dtype=int)
    sharing_dropped = np.empty(len(y), dtype=int)
    random_dropped = np.empty((n_repeats, len(y)), dtype=int)
    for tape in species_on:
        held_out = tapes == tape
        all_other[held_out] = predict(tape, [])
        sharing_dropped[held_out] = predict(tape, tapes_sharing_digits(tape, species_on))
        for repeat in range(n_repeats):
            random_dropped[repeat, held_out] = predict(tape, random_tapes(tape, species_on, rng))

    random_f1 = [macro_f1(y[scored], run[scored]) for run in random_dropped]
    return pd.DataFrame(
        [
            {"Trained on": ALL_OTHER_TAPES, "Macro-F1": macro_f1(y[scored], all_other[scored])},
            {
                "Trained on": RANDOM_DROP,
                "Macro-F1": float(np.mean(random_f1)),
                "Lowest repeat": min(random_f1),
                "Highest repeat": max(random_f1),
            },
            {
                "Trained on": SHARED_DIGITS_DROP,
                "Macro-F1": macro_f1(y[scored], sharing_dropped[scored]),
            },
        ]
    )


def validation_overlap(y: np.ndarray, tapes: np.ndarray, folds: np.ndarray) -> pd.DataFrame:
    """How many validation cuts have a training tape of the same species and first two digits.

    A one-row table, computed over the cross-validation folds of the training cuts.
    """
    digits = first_two_digits(tapes)
    matching = 0
    for fold in np.unique(folds):
        is_validation = folds == fold
        in_training = set(zip(y[~is_validation], digits[~is_validation], strict=True))
        pairs = zip(y[is_validation], digits[is_validation], strict=True)
        matching += sum(pair in in_training for pair in pairs)
    return pd.DataFrame(
        [
            {
                "Validation cuts": len(y),
                "With a training tape of the same species and first two digits": matching,
                "Share (%)": 100 * matching / len(y),
            }
        ]
    )
