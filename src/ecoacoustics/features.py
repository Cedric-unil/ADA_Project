# Generated with Claude Code (Anthropic): compute and cache MFCC features.
"""MFCC and delta frames per cut, clip-level summary statistics, and their cache on disk.

With the settings in config.yaml, one frame describes 32 ms of sound and frames are 10 ms
apart, so a 1 s cut gives 101 frames. Stage 03 saves four files to ``data/features/``:

- ``frames.npy``: the frames of every cut stacked into one float32 array, one row per frame,
  columns as in :func:`feature_names`;
- ``frame_index.csv``: for each cut, the row where its frames start and how many there are;
- ``clip_stats.csv``: one row per cut, the mean and standard deviation of every column;
- ``settings.json``: the feature settings used, so that later stages refuse a stale cache.

No scaling is applied here: scalers are fitted later, on training data only.
"""

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd

FRAMES_FILE = "frames.npy"
INDEX_FILE = "frame_index.csv"
STATS_FILE = "clip_stats.csv"
SETTINGS_FILE = "settings.json"


def prepare_waveform(
    waveform: np.ndarray, native_rate: int, target_rate: int, peak_normalise: bool
) -> np.ndarray:
    """Resample a cut to ``target_rate`` and, if asked, scale it so its loudest sample is 1.

    Peak normalisation removes differences in recording gain between tapes while keeping how
    loudness changes within the cut. It comes after resampling, which can shift the peak
    slightly. A silent cut is left as it is.
    """
    audio = np.asarray(waveform, dtype=np.float32)
    if native_rate != target_rate:
        audio = librosa.resample(audio, orig_sr=native_rate, target_sr=target_rate)
    if peak_normalise:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
    return audio


def feature_names(n_mfcc: int) -> list[str]:
    """Names of the frame columns: the MFCCs first, then their deltas."""
    return [f"mfcc_{i:02d}" for i in range(n_mfcc)] + [f"delta_{i:02d}" for i in range(n_mfcc)]


def mfcc_frames(audio: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    """Return MFCCs and their deltas as an array of shape ``(n_frames, 2 * n_mfcc)``.

    ``settings`` is the ``features`` section of config.yaml and ``audio`` must already be at
    ``settings["sample_rate"]``. A delta measures how its MFCC changes around each frame,
    estimated over ``delta_width`` frames.
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=settings["sample_rate"],
        n_mfcc=settings["n_mfcc"],
        n_fft=settings["n_fft"],
        hop_length=settings["hop_length"],
        n_mels=settings["n_mels"],
        fmin=settings["fmin"],
        fmax=settings["fmax"],
    )
    width = settings["delta_width"]
    # The default delta needs at least `width` frames; shorter cuts repeat their edge frames.
    mode = "interp" if mfcc.shape[1] >= width else "nearest"
    delta = librosa.feature.delta(mfcc, width=width, mode=mode)
    return np.vstack([mfcc, delta]).T.astype(np.float32)


def clip_statistics(frames: np.ndarray, names: list[str]) -> dict[str, float]:
    """Mean and standard deviation of every column over all frames of one cut."""
    means = frames.mean(axis=0, dtype=np.float64)
    stds = frames.std(axis=0, dtype=np.float64)
    stats = {f"mean_{name}": float(value) for name, value in zip(names, means, strict=True)}
    stats |= {f"std_{name}": float(value) for name, value in zip(names, stds, strict=True)}
    return stats


def save_features(
    frames_by_cut: dict[str, np.ndarray], features_dir: Path, settings: dict[str, Any]
) -> None:
    """Save the frames, their index, the clip statistics and the settings, in the given order."""
    names = feature_names(settings["n_mfcc"])
    cut_ids = list(frames_by_cut)
    lengths = np.array([len(frames_by_cut[cut_id]) for cut_id in cut_ids])
    all_frames = np.concatenate([frames_by_cut[cut_id] for cut_id in cut_ids])
    if not np.isfinite(all_frames).all():
        raise ValueError("Some features are not finite; check the audio of the cuts involved")

    features_dir.mkdir(parents=True, exist_ok=True)
    np.save(features_dir / FRAMES_FILE, all_frames)
    index = pd.DataFrame(
        {"cut_id": cut_ids, "start": np.cumsum(lengths) - lengths, "n_frames": lengths}
    )
    index.to_csv(features_dir / INDEX_FILE, index=False)
    stats = pd.DataFrame(
        [{"cut_id": cut_id, **clip_statistics(frames_by_cut[cut_id], names)} for cut_id in cut_ids]
    )
    stats.to_csv(features_dir / STATS_FILE, index=False)
    (features_dir / SETTINGS_FILE).write_text(
        json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8"
    )


def check_settings(features_dir: Path, settings: dict[str, Any]) -> None:
    """Raise if the cached features were computed with settings other than ``settings``."""
    saved = json.loads((features_dir / SETTINGS_FILE).read_text(encoding="utf-8"))
    if saved != settings:
        raise ValueError(
            "The cached features were computed with different settings. "
            "Run stage 03 again (make features)."
        )


def load_clip_stats(features_dir: Path, settings: dict[str, Any]) -> pd.DataFrame:
    """Read the clip statistics, one row per cut, keeping cut ids as strings."""
    check_settings(features_dir, settings)
    return pd.read_csv(features_dir / STATS_FILE, dtype={"cut_id": str})


def load_frames(features_dir: Path, settings: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return each cut's frames, keyed by cut id, in the order they were saved."""
    check_settings(features_dir, settings)
    frames = np.load(features_dir / FRAMES_FILE)
    index = pd.read_csv(features_dir / INDEX_FILE, dtype={"cut_id": str})
    return {row.cut_id: frames[row.start : row.start + row.n_frames] for row in index.itertuples()}
