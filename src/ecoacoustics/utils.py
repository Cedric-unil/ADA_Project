# Generated with Claude Code (Anthropic): central seeding and logging setup for all stages.
"""Seeding and logging helpers shared by every stage."""

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and PyTorch from one place.

    scikit-learn estimators and the GMM take ``random_state=config["seed"]`` directly.
    PyTorch is imported here rather than at module level so stages without it start fast.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002

    import torch

    torch.manual_seed(seed)


def setup_logging(stage: str, logs_dir: Path) -> logging.Logger:
    """Log to the console and to ``<logs_dir>/<stage>.log``, replacing earlier handlers."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs_dir / f"{stage}.log", mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger(stage)
