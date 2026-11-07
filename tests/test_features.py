# Generated with Claude Code (Anthropic): test the MFCC features on synthetic audio.
"""Tests for the feature functions. They use synthetic tones, never the real data."""

from pathlib import Path

import numpy as np
import pytest

from ecoacoustics.config import load_config
from ecoacoustics.features import (
    clip_statistics,
    feature_names,
    load_clip_stats,
    load_frames,
    mfcc_frames,
    prepare_waveform,
    save_features,
)

SETTINGS = load_config()["features"]
RATE = SETTINGS["sample_rate"]
N_COLUMNS = 2 * SETTINGS["n_mfcc"]


def _tone(
    seconds: float, rate: int = RATE, frequency: float = 440.0, amplitude: float = 0.5
) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_frames_have_the_expected_shape_for_a_known_length() -> None:
    frames = mfcc_frames(_tone(1.0), SETTINGS)

    assert frames.shape == (1 + RATE // SETTINGS["hop_length"], N_COLUMNS)
    assert frames.dtype == np.float32
    assert len(feature_names(SETTINGS["n_mfcc"])) == N_COLUMNS


@pytest.mark.parametrize("native_rate", [600, 8000, 44100])
def test_resampling_keeps_the_duration(native_rate: int) -> None:
    audio = prepare_waveform(
        _tone(2.0, rate=native_rate, frequency=100.0), native_rate, RATE, peak_normalise=False
    )

    assert len(audio) == 2 * RATE


def test_peak_normalisation_removes_recording_gain() -> None:
    quiet, loud = _tone(1.0, amplitude=0.01), _tone(1.0, amplitude=0.8)

    normalised = [prepare_waveform(audio, RATE, RATE, True) for audio in (quiet, loud)]
    raw = [prepare_waveform(audio, RATE, RATE, False) for audio in (quiet, loud)]

    assert np.abs(normalised[0]).max() == pytest.approx(1.0)
    np.testing.assert_allclose(
        mfcc_frames(normalised[0], SETTINGS), mfcc_frames(normalised[1], SETTINGS), atol=1e-3
    )
    assert not np.allclose(mfcc_frames(raw[0], SETTINGS), mfcc_frames(raw[1], SETTINGS), atol=1.0)


def test_silent_and_very_short_cuts_give_finite_features() -> None:
    silent = prepare_waveform(np.zeros(RATE // 2, dtype=np.float32), RATE, RATE, True)
    short = _tone(0.05)  # 6 frames, fewer than the delta width

    for audio in (silent, short):
        frames = mfcc_frames(audio, SETTINGS)
        assert np.isfinite(frames).all()
        assert frames.shape[1] == N_COLUMNS


def test_features_are_identical_when_computed_twice() -> None:
    first = mfcc_frames(prepare_waveform(_tone(1.5, rate=44100), 44100, RATE, True), SETTINGS)
    second = mfcc_frames(prepare_waveform(_tone(1.5, rate=44100), 44100, RATE, True), SETTINGS)

    np.testing.assert_array_equal(first, second)


def test_clip_statistics_are_column_means_and_standard_deviations() -> None:
    frames = np.arange(12, dtype=np.float32).reshape(4, 3)

    stats = clip_statistics(frames, ["a", "b", "c"])

    assert list(stats) == ["mean_a", "mean_b", "mean_c", "std_a", "std_b", "std_c"]
    assert stats["mean_a"] == pytest.approx(4.5)
    assert stats["std_c"] == pytest.approx(np.std([2, 5, 8, 11]))


def test_saved_features_load_back_and_refuse_other_settings(tmp_path: Path) -> None:
    frames_by_cut = {
        "72021005": mfcc_frames(_tone(1.0), SETTINGS),
        "7801800H": mfcc_frames(_tone(0.5, frequency=880.0), SETTINGS),
    }

    save_features(frames_by_cut, tmp_path, SETTINGS)
    frames = load_frames(tmp_path, SETTINGS)
    stats = load_clip_stats(tmp_path, SETTINGS)

    assert list(frames) == ["72021005", "7801800H"]
    for cut_id, expected in frames_by_cut.items():
        np.testing.assert_array_equal(frames[cut_id], expected)
    assert stats["cut_id"].tolist() == ["72021005", "7801800H"]
    assert stats.shape == (2, 1 + 2 * N_COLUMNS)
    with pytest.raises(ValueError, match="different settings"):
        load_frames(tmp_path, {**SETTINGS, "n_mfcc": 20})
