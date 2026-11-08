# Generated with Claude Code (Anthropic): clean the inventory and build the grouped split.
"""Clean the inventory and build a grouped train/test split with cross-validation folds.

Cuts are grouped by the tape they were recorded on, and a tape never appears on both sides
of a split. The test set is held out once. Hyperparameters are tuned by grouped
cross-validation on the training set.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# A Watkins cut is named by the five-digit number of its tape, followed by a three-character cut
# number: tape 72021 in 72021005.wav. The first two digits may encode the recording year (72 =
# 1972), but that meaning is unconfirmed.
WATKINS_FILE_NAME = r"(?P<tape_id>\d{5})[0-9A-Za-z]{3}\.wav"
TEST_FOLD = -1


def require_grouping_key(prepare_config: dict[str, Any]) -> str:
    """Return the configured grouping column, refusing to go on without one."""
    key = prepare_config.get("grouping_key")
    if not key:
        raise ValueError(
            "prepare.grouping_key is not set in config.yaml. "
            "Stage 02 never falls back to a random split."
        )
    return key


def add_tape_id(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``cut_id`` (the file name without extension) and ``tape_id``."""
    parts = inventory["original_path"].str.extract(f"^{WATKINS_FILE_NAME}$")
    unexpected = inventory.loc[parts["tape_id"].isna(), "original_path"]
    if len(unexpected):
        raise ValueError(f"File names not in the Watkins format: {unexpected.tolist()[:5]}")
    out = inventory.copy()
    out.insert(0, "cut_id", out["original_path"].str.removesuffix(".wav"))
    out.insert(1, "tape_id", parts["tape_id"])
    return out


def drop_label_conflicts(cuts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop audio labelled with more than one species, and extra copies of repeated audio.

    Audio that carries two species labels cannot be classified correctly under both, so every
    copy of it is dropped. Returns the kept rows and the dropped rows with a ``reason``.
    """
    conflict = cuts.groupby("audio_sha1")["species"].transform("nunique") > 1
    repeat = ~conflict & cuts.duplicated("audio_sha1")
    dropped = pd.concat(
        [
            cuts[conflict].assign(reason="audio labelled with more than one species"),
            cuts[repeat].assign(reason="repeated copy of the same audio"),
        ]
    )
    return cuts[~conflict & ~repeat].copy(), dropped


def filter_species(
    cuts: pd.DataFrame, group_col: str, min_cuts: int, min_groups: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep species with at least ``min_cuts`` cuts and ``min_groups`` distinct groups.

    Returns the kept rows and the dropped rows with a ``reason``.
    """
    by_species = cuts.groupby("species")[group_col]
    too_few_cuts = by_species.transform("size") < min_cuts
    too_few_groups = ~too_few_cuts & (by_species.transform("nunique") < min_groups)
    dropped = pd.concat(
        [
            cuts[too_few_cuts].assign(reason=f"species has fewer than {min_cuts} cuts"),
            cuts[too_few_groups].assign(
                reason=f"species has fewer than {min_groups} groups ({group_col})"
            ),
        ]
    )
    return cuts[~too_few_cuts & ~too_few_groups].copy(), dropped


def choose_test_groups(
    cuts: pd.DataFrame, group_col: str, test_fraction: float, min_train_groups: int
) -> set[str]:
    """Choose whole groups for the test set so that every species is in it.

    Species are visited from fewest groups to most. For each one, its groups are moved to
    the test set one at a time while that brings the test shares closer to
    ``test_fraction``. A move is scored over every species in the group, not only the one
    being visited, so a group shared by two species is not moved for one of them at the
    other's expense. Every species gets at least one test group, and a group is moved only
    if every species in it keeps ``min_train_groups`` groups for training. The choice is
    deterministic: ties go to the lowest group id.
    """
    sizes = cuts.groupby(["species", group_col]).size()
    groups_of = {
        species: sorted(part.index.get_level_values(group_col))
        for species, part in sizes.groupby(level="species")
    }
    species_in = cuts.groupby(group_col)["species"].unique()
    n_cuts = sizes.groupby(level="species").sum()
    target = test_fraction * n_cuts
    test: set[str] = set()
    test_cuts = dict.fromkeys(groups_of, 0)

    def can_move(group: str) -> bool:
        return all(
            sum(g not in test for g in groups_of[species]) > min_train_groups
            for species in species_in[group]
        )

    def cost(group: str) -> float:
        """Change in the summed gap between each species' test share and the target."""
        total = 0.0
        for species in species_in[group]:
            before = abs(test_cuts[species] - target[species])
            after = abs(test_cuts[species] + sizes[species, group] - target[species])
            total += (after - before) / n_cuts[species]
        return total

    for species in sorted(groups_of, key=lambda name: (len(groups_of[name]), name)):
        while True:
            candidates = [g for g in groups_of[species] if g not in test and can_move(g)]
            if not candidates:
                break
            score, best = min((cost(g), g) for g in candidates)
            if test_cuts[species] and score >= 0:
                break
            test.add(best)
            for other in species_in[best]:
                test_cuts[other] += sizes[other, best]
        if not test_cuts[species]:
            raise ValueError(
                f"No group of {species} can go to the test set while every species keeps "
                f"{min_train_groups} groups for training"
            )
    return test


def make_split(
    cuts: pd.DataFrame,
    group_col: str,
    test_fraction: float,
    min_train_groups: int,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    """Return a copy with a ``split`` column ("train" or "test") and a ``cv_fold`` column.

    ``cv_fold`` numbers the grouped, species-stratified cross-validation folds of the
    training set from 0, and is ``TEST_FOLD`` for test cuts.
    """
    test_groups = choose_test_groups(cuts, group_col, test_fraction, min_train_groups)
    out = cuts.copy()
    is_test = out[group_col].isin(test_groups)
    out["split"] = np.where(is_test, "test", "train")
    out["cv_fold"] = TEST_FOLD

    train = out[~is_test]
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_rows) in enumerate(
        folds.split(train, train["species"], groups=train[group_col])
    ):
        out.loc[train.index[val_rows], "cv_fold"] = fold
    return out


def check_split(meta: pd.DataFrame, group_col: str, n_folds: int) -> None:
    """Raise ``ValueError`` listing every way the split breaks its guarantees."""
    problems = []
    if meta["cut_id"].duplicated().any():
        problems.append("cut ids are not unique")
    if meta["audio_sha1"].duplicated().any():
        problems.append("the same audio appears in more than one row")
    if not set(meta["split"]) <= {"train", "test"}:
        problems.append("split holds values other than train and test")
    if (meta.groupby(group_col)["split"].nunique() > 1).any():
        problems.append(f"a {group_col} appears in both train and test")

    train = meta[meta["split"] == "train"]
    test = meta[meta["split"] == "test"]
    if (train.groupby(group_col)["cv_fold"].nunique() > 1).any():
        problems.append(f"a {group_col} appears in more than one fold")
    if not train["cv_fold"].between(0, n_folds - 1).all() or test["cv_fold"].ne(TEST_FOLD).any():
        problems.append("fold numbers are out of range")

    all_species = set(meta["species"])
    for name, part in [("train", train), ("test", test)]:
        if missing := all_species - set(part["species"]):
            problems.append(f"species missing from {name}: {sorted(missing)}")
    for fold in range(n_folds):
        if missing := all_species - set(train.loc[train["cv_fold"] != fold, "species"]):
            problems.append(f"species missing from training in fold {fold}: {sorted(missing)}")

    if problems:
        raise ValueError("Split check failed:\n- " + "\n- ".join(problems))


def summarise_split(meta: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Return cuts and groups per species in each split, with a last row for all species."""

    def describe(part: pd.DataFrame) -> dict[str, float]:
        train = part[part["split"] == "train"]
        test = part[part["split"] == "test"]
        return {
            "cuts": len(part),
            "groups": part[group_col].nunique(),
            "train_cuts": len(train),
            "train_groups": train[group_col].nunique(),
            "test_cuts": len(test),
            "test_groups": test[group_col].nunique(),
            "test_share": len(test) / len(part),
        }

    rows = [{"species": species, **describe(part)} for species, part in meta.groupby("species")]
    rows.append({"species": "All species", **describe(meta)})
    return pd.DataFrame(rows)
