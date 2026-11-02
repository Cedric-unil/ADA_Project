# Generated with Claude Code (Anthropic): test the inventory on a small synthetic Parquet file.
"""Tests for the raw data inventory. They use synthetic audio, never the real data."""

import io
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from ecoacoustics.io import build_inventory, check_inventory


def _wav_bytes(n_frames: int, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(n_frames), sample_rate, format="WAV")
    return buffer.getvalue()


def _write_parquet(path: Path, frames: list[int]) -> None:
    records = [
        {
            "audio": {"bytes": _wav_bytes(n), "path": f"{n}cut.wav"},
            "species": "Walrus",
            "label": 28,
        }
        for n in frames
    ]
    pq.write_table(pa.Table.from_pylist(records), path)


def test_inventory_has_one_row_per_cut_and_correct_durations(tmp_path: Path) -> None:
    path = tmp_path / "train-00000-of-00001.parquet"
    _write_parquet(path, [8000, 16000, 32000])

    inventory = build_inventory([path])

    assert len(inventory) == 3
    assert inventory["duration_s"].tolist() == [0.5, 1.0, 2.0]
    assert inventory["sample_rate"].eq(16000).all()
    assert inventory["hf_split"].eq("train").all()
    assert inventory["parquet_row"].tolist() == [0, 1, 2]
    assert inventory["original_path"].tolist() == ["8000cut.wav", "16000cut.wav", "32000cut.wav"]


def test_identical_audio_gets_the_same_hash(tmp_path: Path) -> None:
    path = tmp_path / "test-00000-of-00001.parquet"
    _write_parquet(path, [8000, 8000, 16000])

    hashes = build_inventory([path])["audio_sha1"]

    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]


def test_check_inventory_reports_wrong_count_and_repeated_audio(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "train-00000-of-00001.parquet"
    _write_parquet(path, [8000, 8000])

    with caplog.at_level(logging.WARNING):
        check_inventory(build_inventory([path]), expected_cuts=3)

    assert "expected 3" in caplog.text
    assert "2 rows share their audio" in caplog.text
