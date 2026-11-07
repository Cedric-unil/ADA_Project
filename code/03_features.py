# Generated with Claude Code (Anthropic): stage 03, compute and cache the MFCC features.
"""Stage 03: resample every kept cut to one rate, peak-normalise it, and cache its MFCC features."""

import time

import numpy as np

from ecoacoustics.config import load_config, resolve_path
from ecoacoustics.features import mfcc_frames, prepare_waveform, save_features
from ecoacoustics.io import read_cut_audio, read_metadata
from ecoacoustics.utils import setup_logging

LOG_EVERY = 200


def main() -> None:
    config = load_config()
    logger = setup_logging("03_features", resolve_path(config, "logs_dir"))
    settings = config["features"]
    rate = settings["sample_rate"]
    meta = read_metadata(resolve_path(config, "interim_dir") / "metadata.csv")
    logger.info("Computing features for %d cuts at %d Hz", len(meta), rate)

    start = time.perf_counter()
    frames_by_cut = {}
    for cut_id, waveform, native_rate in read_cut_audio(meta, resolve_path(config, "raw_dir")):
        audio = prepare_waveform(waveform, native_rate, rate, settings["peak_normalise"])
        frames_by_cut[cut_id] = mfcc_frames(audio, settings)
        if len(frames_by_cut) % LOG_EVERY == 0:
            logger.info("%d of %d cuts done", len(frames_by_cut), len(meta))

    missing = set(meta["cut_id"]) - set(frames_by_cut)
    if missing:
        raise RuntimeError(f"No audio found for {len(missing)} cuts, e.g. {sorted(missing)[:5]}")
    # Save in the order of metadata.csv rather than the order of the Parquet files.
    frames_by_cut = {cut_id: frames_by_cut[cut_id] for cut_id in meta["cut_id"]}
    features_dir = resolve_path(config, "features_dir")
    save_features(frames_by_cut, features_dir, settings)

    native = meta["sample_rate"]
    logger.info(
        "Resampled %d cuts up from below %d Hz and %d cuts down; %d were already at %d Hz",
        (native < rate).sum(),
        rate,
        (native > rate).sum(),
        (native == rate).sum(),
        rate,
    )
    lengths = np.array([len(frames) for frames in frames_by_cut.values()])
    logger.info(
        "Frames per cut: median %d, min %d, max %d; %d frames in all",
        np.median(lengths),
        lengths.min(),
        lengths.max(),
        lengths.sum(),
    )
    logger.info(
        "Saved the features of %d cuts to %s in %.0f s",
        len(frames_by_cut),
        features_dir,
        time.perf_counter() - start,
    )


if __name__ == "__main__":
    main()
