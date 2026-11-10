# Generated with Claude Code (Anthropic): run the real stages twice on a reduced configuration.
"""Running the stages twice with the same seed gives identical outputs.

The real stage 05 takes about 5 minutes, so these tests run the real stage files on a REDUCED
configuration: a small synthetic data set, 2 seeds, 2 epochs and a tiny LSTM. They never read the
real data. Stage 01 is left out because it downloads; its inventory is built from the synthetic
Parquet file instead.

The test set here is synthetic too, so the test-set cells of stage 05 run as well. They are never
run on the real test set by the tests.
"""

import hashlib
import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf
import yaml

from ecoacoustics.config import CONFIG_ENV_VAR, REPO_ROOT, load_config
from ecoacoustics.evaluation import macro_f1
from ecoacoustics.io import build_inventory

SPECIES = ["Seal_A", "Whale_B", "Dolphin_C"]
DOUBTFUL = "Seal_A"
SEEDS = [42, 43]
# Dolphin_C is recorded below 16 kHz, so the robustness subset leaves it out.
RATES = {"Seal_A": 16000, "Whale_B": 32000, "Dolphin_C": 8000}
STAGES = [
    "02_prepare.py",
    "03_features.py",
    "05_supervised.ipynb",
    "05a_same_period.py",
    "05b_lstm_pilot.py",
]


def _wav_bytes(seconds: float, rate: int, frequency: float, rng: np.random.Generator) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    wave = 0.5 * np.sin(2 * np.pi * frequency * t) + 0.05 * rng.normal(size=len(t))
    buffer = io.BytesIO()
    sf.write(buffer, wave, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _write_raw_data(raw_dir: Path) -> Path:
    """Three species, each on four tapes with two values of the first two digits, 5 cuts a tape."""
    rng = np.random.default_rng(0)
    records = []
    for s, species in enumerate(SPECIES):
        for digits in (60 + s, 70 + s):
            for tape in (1, 2):
                for cut in range(5):
                    seconds = (0.5, 1.0, 2.0, 3.5, 5.0)[cut]
                    frequency = 300.0 * (s + 1) + 20.0 * tape + 7.0 * cut
                    audio = _wav_bytes(seconds, RATES[species], frequency, rng)
                    name = f"{digits}{tape:03d}{cut:03d}.wav"
                    records.append(
                        {"audio": {"bytes": audio, "path": name}, "species": species, "label": s}
                    )
    (raw_dir / "data").mkdir(parents=True)
    path = raw_dir / "data" / "train-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist(records), path)
    return path


def _reduced_config(root: Path, evaluate_test: bool) -> dict[str, Any]:
    """The real settings, cut down so that the stages run in seconds, with paths under ``root``."""
    config = load_config()
    config["seed"] = SEEDS[0]
    folders = {key: root / key.removesuffix("_dir") for key in config["paths"]}
    folders["raw_dir"] = root.parent / "raw"
    folders["interim_dir"] = root.parent / "interim"
    folders["features_dir"] = root.parent / "features"
    config["paths"] = {key: str(folder) for key, folder in folders.items()}
    config["prepare"] |= {
        "min_cuts_per_species": 5,
        "min_groups_per_species": 3,
        "test_fraction": 0.25,
        "n_cv_folds": 2,
    }
    config["baseline"] |= {"c_grid": [1.0, 10.0], "max_iter": 500}
    config["lstm"] |= {
        "hidden_size": 4,
        "batch_size": 8,
        "eval_batch_size": 16,
        "max_epochs": 2,
        "num_threads": 1,
        # Two processes, so that the LSTMs of stages 05 and 05b also train at the same time here.
        "n_jobs": 2,
        "n_seeds": len(SEEDS),
    }
    config["evaluation"] |= {
        "evaluate_test": evaluate_test,
        "n_bootstrap": 10,
        "n_chance_draws": 3,
        "doubtful_species": [DOUBTFUL],
    }
    config["sensitivity"] = {"same_period_repeats": 2}
    config["lstm_pilot"] = {
        "fold": 0,
        "variants": [
            {"use_last_output": True, "learning_rate": 0.01, "epochs": 2, "shuffle_order": False},
            {"use_last_output": False, "learning_rate": 0.01, "epochs": 2, "shuffle_order": True},
        ],
    }
    return config


def _write_config(config: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run_stage(stage: str, config_path: Path, notebooks_dir: Path) -> None:
    """Run one real stage in its own process, as the runner does."""
    env = {
        **os.environ,
        CONFIG_ENV_VAR: str(config_path),
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "PYTHONIOENCODING": "utf-8",
    }
    path = REPO_ROOT / "code" / stage
    command = [sys.executable, str(path)]
    if path.suffix == ".ipynb":
        command = [sys.executable, "-m", "ecoacoustics.pipeline", str(path), str(notebooks_dir)]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, f"{stage} failed:\n{result.stdout}\n{result.stderr}"


def _fingerprints(*folders: Path) -> dict[str, str]:
    """A hash of every output file, so that any changed byte shows."""
    return {
        file.relative_to(folder.parent).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for folder in folders
        for file in sorted(folder.rglob("*"))
        if file.is_file()
    }


class Reduced:
    """The folders of the reduced project, and the outputs of its two runs."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.root = base / "unlocked"
        self.config_path = _write_config(
            _reduced_config(self.root, evaluate_test=True), self.root / "config.yaml"
        )
        self.tables = self.root / "tables"
        self.outputs = self.root / "outputs"
        self.notebooks = self.root / "executed"
        self.folders = (base / "interim", base / "features", self.tables, self.outputs)
        self.first: dict[str, str] = {}
        self.second: dict[str, str] = {}


@pytest.fixture(scope="module")
def reduced(tmp_path_factory: pytest.TempPathFactory) -> Reduced:
    """Run the reduced stages twice, keeping a fingerprint of the outputs of each run."""
    project = Reduced(tmp_path_factory.mktemp("reduced"))
    parquet = _write_raw_data(project.base / "raw")
    (project.base / "interim").mkdir()
    build_inventory([parquet]).to_csv(project.base / "interim" / "inventory.csv", index=False)
    for stage in STAGES:
        _run_stage(stage, project.config_path, project.notebooks)
    project.first = _fingerprints(*project.folders)
    for stage in STAGES:
        _run_stage(stage, project.config_path, project.notebooks)
    project.second = _fingerprints(*project.folders)
    return project


def test_running_the_stages_twice_gives_identical_outputs(reduced: Reduced) -> None:
    expected = {
        "interim/metadata.csv",
        "features/frames.npy",
        "tables/cv_results.csv",
        "tables/cv_seed_gaps.csv",
        "tables/test_results.csv",
        "tables/test_gaps.csv",
        "tables/same_period_check.csv",
        "tables/same_period_overlap.csv",
        "tables/lstm_pilot.csv",
        "outputs/cv_predictions.csv",
        "outputs/lstm_learning_curves.csv",
        "outputs/chosen_settings.json",
        "outputs/test_predictions.csv",
        "outputs/lstm_pilot_curves.csv",
    }
    assert expected <= reduced.first.keys()
    assert reduced.first == reduced.second


def test_lstm_scores_report_the_mean_and_spread_over_the_seeds(reduced: Reduced) -> None:
    table = pd.read_csv(reduced.tables / "cv_results.csv").set_index("Model")
    curves = pd.read_csv(reduced.outputs / "lstm_learning_curves.csv")
    chosen = yaml.safe_load((reduced.outputs / "chosen_settings.json").read_text(encoding="utf-8"))

    for model in ("LSTM", "LSTM (shuffled frames)"):
        assert chosen[model]["seeds"] == SEEDS
        runs = curves[curves["model"] == model]
        assert sorted(runs["seed"].unique()) == SEEDS
        # The model's own average curve chooses its number of epochs.
        average = runs.groupby("epoch")["macro_f1"].mean()
        assert chosen[model]["epochs"] == average.idxmax()
        at_chosen = runs[runs["epoch"] == chosen[model]["epochs"]]["macro_f1"]
        assert table.loc[model, "Macro-F1"] == pytest.approx(at_chosen.mean())
        assert table.loc[model, "Macro-F1 SD"] == pytest.approx(at_chosen.std(ddof=1))
        assert table.loc[model, "Macro-F1 lowest"] == pytest.approx(at_chosen.min())
        assert table.loc[model, "Macro-F1 highest"] == pytest.approx(at_chosen.max())
    assert pd.isna(table.loc["Majority class", "Macro-F1 SD"])


def test_seed_table_pairs_the_gap_seed_by_seed(reduced: Reduced) -> None:
    table = pd.read_csv(reduced.tables / "cv_seed_gaps.csv", dtype={"Seed": str})

    assert table["Seed"].tolist() == [str(seed) for seed in SEEDS] + [
        "Mean",
        "SD",
        "Lowest",
        "Highest",
    ]
    per_seed = table.iloc[: len(SEEDS)]
    expected = per_seed["LSTM"] - per_seed["LSTM (shuffled frames)"]
    np.testing.assert_allclose(per_seed["LSTM minus shuffled"], expected)
    assert table.loc[len(SEEDS), "LSTM minus shuffled"] == pytest.approx(expected.mean())


def test_test_scores_are_reported_with_and_without_doubtful_species(reduced: Reduced) -> None:
    table = pd.read_csv(reduced.tables / "test_results.csv")
    gaps = pd.read_csv(reduced.tables / "test_gaps.csv")
    saved = pd.read_csv(reduced.outputs / "test_predictions.csv", dtype={"tape_id": str})
    without = "Without Seal A"

    assert set(table["Species"]) == {"All species", without}
    assert set(table["Data"]) == {"All cuts", "Cuts at 16000 Hz or more"}
    assert len(table) == 2 * 2 * 7
    assert len(gaps) == 2 * 2 * 3

    # Recompute the LSTM's mean score from the saved predictions, without the doubtful species.
    cuts = saved[(saved["data"] == "All cuts") & (saved["species"] != DOUBTFUL)]
    expected = np.mean([macro_f1(cuts["species"], cuts[f"LSTM (seed {s})"]) for s in SEEDS])
    row = table[
        (table["Data"] == "All cuts") & (table["Species"] == without) & (table["Model"] == "LSTM")
    ]
    assert row["Macro-F1"].item() == pytest.approx(expected)
    assert row["Macro-F1 SD"].notna().all()


def test_a_locked_test_set_writes_no_test_outputs(reduced: Reduced) -> None:
    root = reduced.base / "locked"
    config_path = _write_config(_reduced_config(root, evaluate_test=False), root / "config.yaml")

    _run_stage("05_supervised.ipynb", config_path, root / "executed")

    written = {file.name for file in root.rglob("*") if file.is_file()}
    assert "cv_results.csv" in written
    assert not {name for name in written if name.startswith("test_")}
    # The cross-validation does not depend on the test set being unlocked.
    locked = (root / "tables" / "cv_results.csv").read_bytes()
    assert locked == (reduced.tables / "cv_results.csv").read_bytes()
