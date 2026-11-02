# Generated with Claude Code (Anthropic): download the raw data and read cuts from it.
"""Download the raw data and build a per-cut inventory from it."""

import hashlib
import io
import logging
from pathlib import Path

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
        "audio_sha1": hashlib.sha1(audio_bytes, usedforsecurity=False).hexdigest(),
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
