# Generated with Claude Code (Anthropic): stage 01, download the data and save an inventory.
"""Stage 01: download the raw Parquet files and save a per-cut inventory."""

from ecoacoustics.config import load_config, resolve_path
from ecoacoustics.io import build_inventory, check_inventory, download_raw_data
from ecoacoustics.utils import setup_logging


def main() -> None:
    config = load_config()
    logger = setup_logging("01_get_data", resolve_path(config, "logs_dir"))

    data = config["data"]
    parquet_paths = download_raw_data(
        data["hf_repo_id"], data["hf_revision"], resolve_path(config, "raw_dir")
    )
    inventory = build_inventory(parquet_paths)
    check_inventory(inventory, data["expected_cuts"])

    interim_dir = resolve_path(config, "interim_dir")
    interim_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(interim_dir / "inventory.csv", index=False)

    logger.info("Saved inventory of %d cuts to %s", len(inventory), interim_dir)
    logger.info("Species: %d", inventory["species"].nunique())
    logger.info("Native sample rates: %s", inventory["sample_rate"].value_counts().to_dict())
    logger.info("Duration in seconds:\n%s", inventory["duration_s"].describe().round(2))


if __name__ == "__main__":
    main()
