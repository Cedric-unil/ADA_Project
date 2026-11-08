<!-- Generated with Claude Code (Anthropic): running log of AI assistance for the paper appendix. -->
# AI usage log

Running log of AI assistance, for the paper appendix. The tool used is Claude Code (Anthropic). The "What I changed" column is for the author to fill in; an empty cell means nothing has been recorded yet.

## Generated files

| Date | File | What was asked | What I changed |
|---|---|---|---|
| 2026-11-02 | `.gitignore`, `.gitattributes` | Keep data, caches, checkpoints, environments and editor settings out of git; strip notebook outputs | |
| 2026-11-02 | `pyproject.toml` | Make `src/` importable; configure ruff and pytest | |
| 2026-11-02 | `requirements.txt` | Pin the packages needed, CPU-only | |
| 2026-11-02 | `config.yaml` | Hold all paths, seeds and hyperparameters in one place | |
| 2026-11-02 | `Makefile` | Provide `make data / features / analysis / figures / all` | |
| 2026-11-02 | `README.md` | README skeleton covering the course's required sections | |
| 2026-11-02 | `AI_USAGE.md` | Running log of AI assistance | |
| 2026-11-02 | `data/README.md`, `paper/README.md` | Explain what belongs in each folder | |
| 2026-11-02 | `src/ecoacoustics/__init__.py`, `config.py`, `utils.py` | Load config, set all seeds from one place, set up logging | |
| 2026-11-02 | `src/ecoacoustics/io.py`, `splits.py`, `features.py`, `models.py`, `evaluation.py`, `plotting.py` | Package module placeholders, filled in at their stages | |
| 2026-11-02 | `tests/test_config.py`, `tests/test_utils.py` | Test config path resolution and seed reproducibility | |
| 2026-11-02 | `code/01_get_data.py`, `src/ecoacoustics/io.py`, `tests/test_io.py` | Download the data at a pinned revision and save a per-cut inventory; test it on synthetic audio | |
| 2026-11-06 | `code/02_prepare.py`, `src/ecoacoustics/splits.py`, `src/ecoacoustics/plotting.py`, `src/ecoacoustics/io.py`, `tests/test_splits.py`, `tests/test_plotting.py`, `tests/test_io.py`, `config.yaml`, `requirements.txt`, `README.md`, `data/README.md` | Stage 02: drop audio with conflicting species labels, group cuts by tape, keep species with at least 20 cuts and 3 tapes, hold out a grouped test set covering every species, assign grouped cross-validation folds, check the split, and save the data-flow and split-summary tables | |

| 2026-11-07 | `code/03_features.py`, `src/ecoacoustics/features.py`, `src/ecoacoustics/io.py`, `tests/test_features.py`, `tests/test_io.py`, `config.yaml`, `README.md`, `data/README.md` | Stage 03: resample every cut to 16 kHz, peak-normalise it, compute 13 MFCCs and their deltas per frame and their mean and standard deviation per cut, cache them with the settings used, and check each cut's audio against its stage 01 hash; test on synthetic audio | |
| 2026-11-07 | `CLAUDE.md` | Bring it up to date with the repository after stages 01-03 and the proposal's approval | |
| 2026-11-07 | `code/05_supervised.ipynb`, `src/ecoacoustics/models.py`, `src/ecoacoustics/evaluation.py`, `src/ecoacoustics/features.py`, `tests/test_models.py`, `tests/test_evaluation.py`, `tests/test_features.py`, `config.yaml`, `README.md`, `CLAUDE.md` | Stage 05: grouped cross-validation of chance, majority, sample-rate-only, logistic regression on whole cuts and on windows, an LSTM and an LSTM on shuffled frames; tune `C` and the number of epochs; test-set comparison with a tape-level bootstrap and a 16 kHz robustness check, locked until `evaluate_test` is true. Includes a pilot on validation fold 0 that changed the LSTM to average its outputs and raised its learning rate to 0.003 | |

| 2026-11-08 | `src/ecoacoustics/pipeline.py`, `src/ecoacoustics/menu.py`, `run.py`, `menu.ipynb`, `tests/test_pipeline.py`, `Makefile`, `requirements.txt`, `README.md`, `CLAUDE.md` | An interactive menu with buttons that runs the whole pipeline and stays future-proof: stages found automatically in `code/`, run one, several, all or only the out-of-date ones, with live output, Stop, tests and lint; out-of-date detection from each stage's code, imported package modules, config sections and input files; the same menu in a terminal | |
| 2026-11-08 | `src/ecoacoustics/menu.py`, `src/ecoacoustics/pipeline.py`, `run.py`, `tests/test_pipeline.py`, `requirements.txt`, `README.md`, `CLAUDE.md` | Open the menu with a terminal command instead of a notebook: `python run.py` now opens a desktop window with the same buttons (Tkinter). Removed `menu.ipynb` and the text menu; the direct commands stay | |

## Code adapted from external sources

None so far.
