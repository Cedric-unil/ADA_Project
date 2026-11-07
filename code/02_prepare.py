# Generated with Claude Code (Anthropic): stage 02, clean the inventory and save the grouped split.
"""Stage 02: drop unusable cuts, group cuts by tape, and save the train/test split and folds."""

import pandas as pd

from ecoacoustics.config import load_config, resolve_path
from ecoacoustics.plotting import save_table
from ecoacoustics.splits import (
    add_tape_id,
    check_split,
    drop_label_conflicts,
    filter_species,
    make_split,
    require_grouping_key,
    summarise_split,
)
from ecoacoustics.utils import setup_logging

FIRST_COLUMNS = ["cut_id", "tape_id", "species", "split", "cv_fold"]


def flow_row(step: str, cuts: pd.DataFrame) -> dict[str, object]:
    """One row of the table that follows the cuts through each step."""
    return {
        "Step": step,
        "Cuts": len(cuts),
        "Species": cuts["species"].nunique(),
        "Tapes": cuts["tape_id"].nunique(),
    }


def format_summary(summary: pd.DataFrame, group_label: str) -> pd.DataFrame:
    """Give the split summary readable species names and column headings for the paper."""
    table = summary.copy()
    table["species"] = table["species"].str.replace("_", " ")
    table["test_share"] = (100 * table["test_share"]).round(1)
    return table.rename(
        columns={
            "species": "Species",
            "cuts": "Cuts",
            "groups": group_label,
            "train_cuts": "Train cuts",
            "train_groups": f"Train {group_label.lower()}",
            "test_cuts": "Test cuts",
            "test_groups": f"Test {group_label.lower()}",
            "test_share": "Test share (%)",
        }
    )


def main() -> None:
    config = load_config()
    logger = setup_logging("02_prepare", resolve_path(config, "logs_dir"))
    prep = config["prepare"]
    group_col = require_grouping_key(prep)
    min_cuts, min_groups = prep["min_cuts_per_species"], prep["min_groups_per_species"]
    n_folds = prep["n_cv_folds"]
    interim_dir = resolve_path(config, "interim_dir")

    inventory = add_tape_id(pd.read_csv(interim_dir / "inventory.csv"))
    if group_col not in inventory.columns:
        raise ValueError(f"prepare.grouping_key {group_col!r} is not a column of the inventory")
    flow = [flow_row("Inventory from stage 01", inventory)]

    cuts, conflicts = drop_label_conflicts(inventory)
    flow.append(flow_row("Drop audio labelled with more than one species", cuts))

    cuts, rare = filter_species(cuts, group_col, min_cuts, min_groups)
    flow.append(
        flow_row(f"Keep species with at least {min_cuts} cuts and {min_groups} tapes", cuts)
    )

    meta = make_split(
        cuts, group_col, prep["test_fraction"], min_groups - 1, n_folds, config["seed"]
    )
    check_split(meta, group_col, n_folds)
    for split in ("train", "test"):
        flow.append(flow_row(f"{split.capitalize()} set", meta[meta["split"] == split]))

    rest = [column for column in meta.columns if column not in FIRST_COLUMNS]
    meta = meta[[*FIRST_COLUMNS, *rest]].sort_values(["split", "cv_fold", "cut_id"])
    meta.to_csv(interim_dir / "metadata.csv", index=False)
    excluded = pd.concat([conflicts, rare])[["cut_id", "tape_id", "species", "reason"]]
    excluded.to_csv(interim_dir / "excluded.csv", index=False)

    tables_dir = resolve_path(config, "tables_dir")
    group_label = "Tapes" if group_col == "tape_id" else f"Groups ({group_col})"
    summary = summarise_split(meta, group_col)
    save_table(pd.DataFrame(flow), "data_flow", tables_dir)
    save_table(format_summary(summary, group_label), "split_summary", tables_dir, "%.1f")

    logger.info("Dropped %d rows whose audio carries more than one species label", len(conflicts))
    for species, reason in rare.groupby("species")["reason"].first().items():
        logger.info("Dropped %s: %s", species, reason)
    logger.info(
        "Kept %d cuts of %d species on %d tapes",
        len(meta),
        meta["species"].nunique(),
        meta["tape_id"].nunique(),
    )
    test = meta[meta["split"] == "test"]
    shares = summary["test_share"].iloc[:-1]
    logger.info(
        "Test set: %d cuts (%.1f%%) on %d tapes; per-species test share %.1f%% to %.1f%%",
        len(test),
        100 * len(test) / len(meta),
        test[group_col].nunique(),
        100 * shares.min(),
        100 * shares.max(),
    )
    train = meta[meta["split"] == "train"]
    for fold, part in train.groupby("cv_fold"):
        logger.info(
            "Fold %d validation: %d cuts on %d tapes, %d species",
            fold,
            len(part),
            part[group_col].nunique(),
            part["species"].nunique(),
        )
    logger.info("Saved metadata, exclusions and tables to %s and %s", interim_dir, tables_dir)


if __name__ == "__main__":
    main()
