# Generated with Claude Code (Anthropic): download the raw data and read cuts from it.
"""Download the raw data, build a per-cut inventory from it, and read the audio back."""

import hashlib
import io
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


def download_raw_data(repo_id: str, revision: str, raw_dir: Path) -> list[Path]:
    """Download the Parquet files at a pinned revision and return their paths.

    Files already present and unchanged are not downloaded again.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns="data/*.parquet",
        local_dir=raw_dir,
    )
    paths = sorted((raw_dir / "data").glob("*.parquet"))
    logger.info("%d Parquet files in %s", len(paths), raw_dir)
    return paths


def audio_hash(audio_bytes: bytes) -> str:
    """Fingerprint of one encoded audio cut, used to spot repeats and to check reads."""
    return hashlib.sha1(audio_bytes, usedforsecurity=False).hexdigest()


def describe_cut(audio_bytes: bytes) -> dict[str, object]:
    """Return format details and a content hash for one encoded audio cut."""
    info = sf.info(io.BytesIO(audio_bytes))
    return {
        "sample_rate": info.samplerate,
        "n_frames": info.frames,
        "n_channels": info.channels,
        "duration_s": info.frames / info.samplerate,
        "audio_format": info.format,
        "audio_subtype": info.subtype,
        "n_bytes": len(audio_bytes),
        "audio_sha1": audio_hash(audio_bytes),
    }


def build_inventory(parquet_paths: list[Path]) -> pd.DataFrame:
    """Return one row per cut with its labels, source path and audio details.

    ``parquet_file`` and ``parquet_row`` record where to find the cut's audio later.
    """
    rows = []
    for path in parquet_paths:
        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_SIZE):
            for record in batch.to_pylist():
                rows.append(
                    {
                        "original_path": record["audio"]["path"],
                        "species": record["species"],
                        "label": record["label"],
                        "hf_split": path.name.split("-")[0],
                        "parquet_file": path.name,
                        "parquet_row": row_index,
                        **describe_cut(record["audio"]["bytes"]),
                    }
                )
                row_index += 1
    return pd.DataFrame(rows)


def check_inventory(inventory: pd.DataFrame, expected_cuts: int) -> None:
    """Log any departure from the expected number of cuts, and any repeated audio."""
    if len(inventory) != expected_cuts:
        logger.warning("Found %d cuts, expected %d", len(inventory), expected_cuts)
    repeated = inventory.duplicated("audio_sha1", keep=False).sum()
    if repeated:
        logger.warning(
            "%d rows share their audio with another row (%d distinct cuts in all)",
            repeated,
            inventory["audio_sha1"].nunique(),
        )


def read_metadata(path: Path) -> pd.DataFrame:
    """Read the metadata saved by stage 02, keeping cut and tape ids as strings."""
    return pd.read_csv(path, dtype={"cut_id": str, "tape_id": str})


def read_cut_audio(meta: pd.DataFrame, raw_dir: Path) -> Iterator[tuple[str, np.ndarray, int]]:
    """Yield ``(cut_id, waveform, native sample rate)`` for every cut in ``meta``.

    ``meta`` needs the ``parquet_file``, ``parquet_row`` and ``audio_sha1`` columns saved by
    stage 01. Each Parquet file is read once, in small batches, so memory use stays low. Every
    cut's audio is checked against its stage 01 hash, so a cut is never paired with the wrong
    audio. Cuts come out in Parquet order, not in the order of ``meta``.
    """
    for parquet_file, cuts in meta.groupby("parquet_file", sort=True):
        wanted = {
            int(row.parquet_row): (row.cut_id, row.audio_sha1) for row in cuts.itertuples()
        }
        parquet = pq.ParquetFile(raw_dir / "data" / parquet_file)
        row_index = 0
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=["audio"]):
            for record in batch.column("audio").to_pylist():
                if row_index in wanted:
                    cut_id, expected_hash = wanted[row_index]
                    if audio_hash(record["bytes"]) != expected_hash:
                        raise ValueError(f"Audio of cut {cut_id} does not match its stage 01 hash")
                    waveform, sample_rate = sf.read(io.BytesIO(record["bytes"]), dtype="float32")
                    if waveform.ndim > 1:
                        waveform = waveform.mean(axis=1)
                    yield cut_id, waveform, sample_rate
                row_index += 1
