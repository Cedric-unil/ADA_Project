<!-- Generated with Claude Code (Anthropic): README skeleton, updated after each stage. -->
# Spectral Shape or Temporal Structure? Species Identity in Marine Mammal Vocalizations

## What this project does

This project asks whether species identity in marine mammal calls is carried mainly by spectral shape or whether temporal structure adds information, by comparing a logistic regression on time-averaged MFCC statistics with a small LSTM on MFCC sequences. It also tests whether clusters found without labels recover species, using the Watkins Marine Mammal Sound Database, with a held-out test set and cross-validation folds grouped by recording tape.

## Install

Requires Python 3.13. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
nbstripout --install --attributes .gitattributes
```

## What to run, in order

Run from the repository root with the environment active. Each stage reads only what the previous stage saved.

| Stage | Command | Equivalent without Make | Approximate time |
|---|---|---|---|
| Get data | `make data` | `python code/01_get_data.py` | about 2 min, nearly all of it the download |
| Prepare | `make prepare` | `python code/02_prepare.py` | a few seconds |
| Features | `make features` | `python code/03_features.py` | to be measured |
| Unsupervised | `make unsupervised` | `jupyter nbconvert --to notebook --execute --output-dir results/logs/notebooks code/04_unsupervised.ipynb` | to be measured |
| Supervised | `make supervised` | `jupyter nbconvert --to notebook --execute --output-dir results/logs/notebooks code/05_supervised.ipynb` | to be measured |
| Figures | `make figures` | `jupyter nbconvert --to notebook --execute --output-dir results/logs/notebooks code/06_figures.ipynb` | to be measured |

`make all` runs every stage in order, and `make analysis` runs the unsupervised and supervised stages. `make test` runs the tests and `make lint` runs ruff.

## Outputs

- `results/figures/`: every figure in the paper.
- `results/tables/`: every table in the paper, as `.csv` and `.tex`.
- `results/models/` and `results/logs/`: trained models and run logs, not tracked in git.
- `data/`: downloaded audio and cached features, not tracked in git. See [data/README.md](data/README.md).

## How the data are cleaned and split

Stage 02 makes these choices, all set in `config.yaml`:

- Audio that appears under two species labels is dropped entirely (25 files, 50 rows). A single-label model cannot be right on both copies.
- Cuts are grouped by recording tape, the first five digits of the Watkins file name (tape 72021 in `72021005.wav`). No tape is in both the training and the test set, or in two cross-validation folds.
- A species needs at least 20 cuts and 3 tapes, so that it can have a test tape and two training tapes. 23 species remain.
- Whole tapes holding about 15% of the cuts form the test set, which includes every species. The rest is split into 5 grouped cross-validation folds for tuning.

`results/tables/data_flow` and `results/tables/split_summary` give the counts at each step and per species. Some species have only one or two cuts in the test set because of how their tapes are sized, so per-species test results are noisy.

## Data source and licence

The data are the "best of" cuts of the Watkins Marine Mammal Sound Database, downloaded from the Hugging Face mirror [`confit/wmms-parquet`](https://huggingface.co/datasets/confit/wmms-parquet) at the revision pinned in `config.yaml`. The mirror's card also points to an [archive.org copy](https://archive.org/details/watkins_202104) of the database.

The card declares 16 kHz, but the stored WAV files keep their native sample rates, from 600 Hz to 166.6 kHz.

Credit: "Watkins Marine Mammal Sound Database, Woods Hole Oceanographic Institution and the New Bedford Whaling Museum."

**Licence:** free for personal or academic use only, not commercial use, and reposting is forbidden. The audio is therefore never committed to this repository; `code/01_get_data.py` downloads it instead.

## Resources beyond a CPU-only cloud workspace

None. The pipeline needs no GPU. The download is about 1.2 GB.

## Attribution

Code generated with AI assistance carries a comment at the top of each file and is logged in [AI_USAGE.md](AI_USAGE.md), along with any code adapted from external sources.
