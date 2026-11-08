# Generated with Claude Code (Anthropic): stage 05a, the same-period sensitivity check.
"""Stage 05a: check, on training data only, how much tapes sharing their first two digits help."""

import json

from ecoacoustics.config import load_config, resolve_path
from ecoacoustics.features import load_clip_stats
from ecoacoustics.io import read_metadata
from ecoacoustics.plotting import save_table
from ecoacoustics.sensitivity import comparable_cuts, same_period_check, validation_overlap
from ecoacoustics.utils import setup_logging


def main() -> None:
    config = load_config()
    logger = setup_logging("05a_same_period", resolve_path(config, "logs_dir"))
    settings = config["sensitivity"]
    baseline = config["baseline"]

    # Only the training rows go any further: this check never touches the test set.
    meta = read_metadata(resolve_path(config, "interim_dir") / "metadata.csv")
    train = meta[meta["split"] == "train"].reset_index(drop=True)
    stats = load_clip_stats(resolve_path(config, "features_dir"), config["features"])
    x = stats.set_index("cut_id").loc[train["cut_id"]].to_numpy()
    species = sorted(train["species"].unique())
    y = train["species"].map({name: k for k, name in enumerate(species)}).to_numpy()
    tapes = train["tape_id"].to_numpy(dtype=str)

    # The penalty stage 05 chose for the logistic regression on whole cuts.
    chosen_path = resolve_path(config, "outputs_dir") / "chosen_settings.json"
    chosen = json.loads(chosen_path.read_text(encoding="utf-8"))
    c = chosen["Logistic regression (whole cuts)"]["C"]

    scored = comparable_cuts(y, tapes)
    logger.info(
        "Scoring %d of %d training cuts: the %d of %d species whose tapes have at least two values "
        "of the first two digits. C = %s",
        scored.sum(),
        len(y),
        len(set(y[scored].tolist())),
        len(species),
        c,
    )
    check = same_period_check(
        x,
        y,
        tapes,
        len(species),
        c,
        baseline["max_iter"],
        settings["same_period_repeats"],
        config["seed"],
    )
    overlap = validation_overlap(y, tapes, train["cv_fold"].to_numpy())

    tables_dir = resolve_path(config, "tables_dir")
    save_table(check, "same_period_check", tables_dir, "%.3f")
    save_table(overlap, "same_period_overlap", tables_dir, "%.1f")
    logger.info("Macro-F1 by training set:\n%s", check.round(3).to_string(index=False))
    logger.info("Validation cuts:\n%s", overlap.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
