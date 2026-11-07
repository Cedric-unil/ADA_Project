# Generated with Claude Code (Anthropic): test the stage 02 cleaning and grouped split.
"""Tests for the cleaning and grouped split. They use synthetic metadata, never the real data."""

import pandas as pd
import pytest

from ecoacoustics.splits import (
    TEST_FOLD,
    add_tape_id,
    check_split,
    choose_test_groups,
    drop_label_conflicts,
    filter_species,
    make_split,
    require_grouping_key,
)

GROUP = "tape_id"
N_FOLDS = 3


def _cuts() -> pd.DataFrame:
    """Five species on four tapes each, with tapes of different sizes and one shared tape."""
    rows = []
    for s in range(5):
        for t in range(4):
            tape = f"{s + 1}{t:04d}"
            rows += [
                {"cut_id": f"{tape}{c:03d}", "tape_id": tape, "species": f"sp{s}"}
                for c in range(2 + 3 * t)
            ]
    rows += [
        {"cut_id": f"90000{c:03d}", "tape_id": "90000", "species": f"sp{c % 2}"} for c in range(4)
    ]
    cuts = pd.DataFrame(rows)
    cuts["audio_sha1"] = cuts["cut_id"]
    return cuts


def _split(cuts: pd.DataFrame | None = None) -> pd.DataFrame:
    return make_split(
        _cuts() if cuts is None else cuts,
        GROUP,
        test_fraction=0.15,
        min_train_groups=2,
        n_folds=N_FOLDS,
        seed=0,
    )


def test_tape_id_is_the_first_five_digits_of_the_file_name() -> None:
    inventory = pd.DataFrame({"original_path": ["72021005.wav", "7801800H.wav", "6107900M.wav"]})
    out = add_tape_id(inventory)
    assert out["tape_id"].tolist() == ["72021", "78018", "61079"]
    assert out["cut_id"].tolist() == ["72021005", "7801800H", "6107900M"]


def test_unexpected_file_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="Watkins format"):
        add_tape_id(pd.DataFrame({"original_path": ["72021005.wav", "cut.wav"]}))


def test_audio_with_two_species_is_dropped_entirely() -> None:
    cuts = pd.DataFrame(
        {
            "audio_sha1": ["a", "a", "b", "b", "c"],
            "species": ["seal", "whale", "seal", "seal", "whale"],
        }
    )
    kept, dropped = drop_label_conflicts(cuts)
    assert kept["audio_sha1"].tolist() == ["b", "c"]
    assert sorted(dropped["audio_sha1"]) == ["a", "a", "b"]


def test_species_need_enough_cuts_and_tapes() -> None:
    cuts = pd.DataFrame(
        {
            "species": ["few_cuts"] * 2 + ["few_tapes"] * 4 + ["kept"] * 4,
            "tape_id": ["1", "2", "3", "3", "4", "4", "5", "6", "7", "7"],
        }
    )
    kept, dropped = filter_species(cuts, GROUP, min_cuts=3, min_groups=3)
    assert set(kept["species"]) == {"kept"}
    assert dropped.groupby("species")["reason"].first().str.contains("cuts").to_dict() == {
        "few_cuts": True,
        "few_tapes": False,
    }


def test_split_keeps_tapes_together_and_covers_every_species() -> None:
    meta = _split()
    check_split(meta, GROUP, N_FOLDS)

    train = meta[meta["split"] == "train"]
    test = meta[meta["split"] == "test"]
    assert not set(train[GROUP]) & set(test[GROUP])
    assert set(test["species"]) == set(meta["species"])
    assert test["cv_fold"].eq(TEST_FOLD).all()
    for fold in range(N_FOLDS):
        assert set(train.loc[train["cv_fold"] != fold, "species"]) == set(meta["species"])


def test_every_species_keeps_two_training_tapes() -> None:
    train = _split().query("split == 'train'")
    assert train.groupby("species")[GROUP].nunique().min() >= 2


def test_shared_tape_is_not_moved_at_the_other_species_expense() -> None:
    # Tape "s" holds 1 cut of B but 20 of A. Moving it would suit B and overload A's test share.
    tapes = {"A": {"a1": 10, "a2": 10, "a3": 10, "s": 20}, "B": {"b1": 8, "b2": 8, "b3": 8, "s": 1}}
    cuts = pd.DataFrame(
        [
            {"species": species, "tape_id": tape}
            for species, sizes in tapes.items()
            for tape, n in sizes.items()
            for _ in range(n)
        ]
    )
    assert choose_test_groups(cuts, GROUP, test_fraction=0.15, min_train_groups=2) == {"a1", "b1"}


def test_split_is_reproducible() -> None:
    pd.testing.assert_frame_equal(_split(), _split())


def test_check_split_catches_a_tape_on_both_sides() -> None:
    meta = _split()
    row = meta.index[meta["split"] == "train"][0]
    meta.loc[row, ["split", "cv_fold"]] = ["test", TEST_FOLD]
    with pytest.raises(ValueError, match="both train and test"):
        check_split(meta, GROUP, N_FOLDS)


def test_species_on_too_few_tapes_cannot_be_split() -> None:
    cuts = _cuts()
    cuts = cuts[~cuts[GROUP].isin(["10000", "10001", "90000"])]
    with pytest.raises(ValueError, match="No group of sp0"):
        _split(cuts)


def test_grouping_key_must_be_set() -> None:
    with pytest.raises(ValueError, match="grouping_key"):
        require_grouping_key({"grouping_key": None})
