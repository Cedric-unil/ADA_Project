# Generated with Claude Code (Anthropic): stage 05b, the pilot behind the LSTM design, recovered.
"""Stage 05b: regenerate the pilot behind the LSTM design, on one validation fold of training cuts.

This is the pilot that chose the learning rate and averaging the LSTM outputs, before the
cross-validation of stage 05. It looks at validation fold 0 only, so that fold also helped choose
the design: the LSTMs' cross-validated scores are slightly optimistic for that reason. The logistic
regressions on the same fold are there for reference.
"""

import time

import numpy as np
import pandas as pd

from ecoacoustics.config import load_config, resolve_path
from ecoacoustics.evaluation import macro_f1
from ecoacoustics.features import (
    cut_windows,
    load_clip_stats,
    load_frames,
    seconds_to_frames,
    window_statistics,
)
from ecoacoustics.io import read_metadata
from ecoacoustics.models import fit_predict_logistic, fit_predict_window_logistic, train_lstm
from ecoacoustics.plotting import save_table
from ecoacoustics.utils import setup_logging


def variant_name(variant: dict[str, object]) -> str:
    """A readable name for one LSTM variant of the pilot."""
    output = "last output" if variant["use_last_output"] else "average of outputs"
    order = ", shuffled frames" if variant["shuffle_order"] else ""
    return f"LSTM, {output}{order}"


def main() -> None:
    config = load_config()
    logger = setup_logging("05b_lstm_pilot", resolve_path(config, "logs_dir"))
    features, lstm, pilot = config["features"], config["lstm"], config["lstm_pilot"]
    baseline = config["baseline"]
    features_dir = resolve_path(config, "features_dir")

    # Only the training rows go any further: the pilot never touches the test set.
    meta = read_metadata(resolve_path(config, "interim_dir") / "metadata.csv")
    train = meta[meta["split"] == "train"].reset_index(drop=True)
    species = sorted(train["species"].unique())
    n_classes = len(species)
    y = train["species"].map({name: k for k, name in enumerate(species)}).to_numpy()
    frames_by_cut = load_frames(features_dir, features)
    frames = [frames_by_cut[cut_id] for cut_id in train["cut_id"]]
    stats = load_clip_stats(features_dir, features)
    x_clip = stats.set_index("cut_id").loc[train["cut_id"]].to_numpy()
    window = seconds_to_frames(lstm["window_seconds"], features)
    hop = seconds_to_frames(lstm["window_hop_seconds"], features)
    x_windows = [window_statistics(cut_windows(f, window, hop)) for f in frames]

    is_validation = train["cv_fold"].to_numpy() == pilot["fold"]
    fit, ev = np.flatnonzero(~is_validation), np.flatnonzero(is_validation)
    fit_frames, ev_frames = [frames[i] for i in fit], [frames[i] for i in ev]
    logger.info("Fold %d: %d cuts to train on, %d to predict", pilot["fold"], len(fit), len(ev))

    rows, curves = [], []
    for c in baseline["c_grid"]:
        whole = fit_predict_logistic(
            x_clip[fit], y[fit], x_clip[ev], n_classes, c, baseline["max_iter"]
        )
        windows = fit_predict_window_logistic(
            [x_windows[i] for i in fit],
            y[fit],
            [x_windows[i] for i in ev],
            n_classes,
            c,
            baseline["max_iter"],
        )
        for name, probs in [("whole cuts", whole), ("windows", windows)]:
            f1 = macro_f1(y[ev], probs.argmax(axis=1))
            rows.append(
                {"Model": f"Logistic regression ({name})", "Setting": f"C = {c}", "Macro-F1": f1}
            )

    for variant in pilot["variants"]:
        start = time.perf_counter()
        settings = {
            **lstm,
            "learning_rate": variant["learning_rate"],
            "use_last_output": variant["use_last_output"],
        }
        history = train_lstm(
            fit_frames,
            y[fit],
            ev_frames,
            n_classes,
            settings,
            window,
            hop,
            variant["epochs"],
            variant["shuffle_order"],
            config["seed"],
        )
        f1 = [macro_f1(y[ev], probs.argmax(axis=1)) for probs in history]
        name = variant_name(variant)
        setting = f"learning rate {variant['learning_rate']}, {variant['epochs']} epochs"
        best = int(np.argmax(f1))
        rows.append(
            {"Model": name, "Setting": setting, "Macro-F1": f1[best], "Best epoch": best + 1}
        )
        curves += [
            {"model": name, "setting": setting, "epoch": epoch + 1, "macro_f1": value}
            for epoch, value in enumerate(f1)
        ]
        logger.info(
            "%s, %s: best macro-F1 %.3f at epoch %d (%.0f s)",
            name,
            setting,
            f1[best],
            best + 1,
            time.perf_counter() - start,
        )

    table = pd.DataFrame(rows)
    save_table(table, "lstm_pilot", resolve_path(config, "tables_dir"), "%.3f")
    outputs_dir = resolve_path(config, "outputs_dir")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curves).to_csv(outputs_dir / "lstm_pilot_curves.csv", index=False)
    logger.info("Pilot on fold %d:\n%s", pilot["fold"], table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
