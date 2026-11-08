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
| Features | `make features` | `python code/03_features.py` | about 10 s |
| Unsupervised | `make unsupervised` | `jupyter nbconvert --to notebook --execute --output-dir results/logs/notebooks code/04_unsupervised.ipynb` | to be measured |
| Supervised | `make supervised` | `jupyter nbconvert --to notebook --execute --output-dir results/logs/notebooks code/05_supervised.ipynb` | about 9 min on a 16-core laptop, nearly all of it the LSTMs; a few minutes more once the test set is unlocked |
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

## How the features are computed

Stage 03 makes these choices, all set in `config.yaml`:

- Every cut is resampled to 16 kHz. The 237 cuts recorded below that rate (every fin whale cut, and nearly every harp seal, humpback and northern right whale cut) have nothing above half their native rate, so a model could recognise them by the empty high frequencies alone. Stage 05 checks how much the sample rate on its own reveals about species, and repeats the main comparison without these cuts.
- Every cut is scaled so its loudest sample is 1. This removes differences in recording gain between tapes but keeps how loudness changes within a cut.
- Every 10 ms, 13 MFCCs (40 mel bands up to 8 kHz, 32 ms frames) and their deltas are computed, giving 26 numbers per frame. The LSTM reads these frame sequences.
- Each cut is also summarised by the mean and standard deviation of those 26 numbers over all its frames (52 numbers per cut). The logistic regression and the clustering use these.

Nothing is standardised at this stage: scalers are fitted later, on training data only.

## How the models are compared

Stage 05 compares seven models, with settings in `config.yaml`:

| Model | What it sees |
|---|---|
| Chance | nothing: a uniformly random guess |
| Majority class | nothing: always the most common training species |
| Sample rate only | the native sample rate of the recording, nothing about the sound |
| Logistic regression (whole cuts) | mean and standard deviation of each MFCC and delta over the whole cut |
| Logistic regression (windows) | the same statistics over each 3 s window; a cut's windows are averaged |
| LSTM (shuffled frames) | the frames of each 3 s window, in random order |
| LSTM | the frames of each 3 s window, in order |

- **Scoring.** Every model is scored by grouped 5-fold cross-validation on the training set. The out-of-fold predictions of all 1,113 training cuts are pooled, and macro-F1 (the plain average of each species' F1 score) is the main score, so every species counts equally.
- **Weights.** Training gives every species the same total weight. For the models that read windows, every cut also counts once, however long it is, so the three 20-minute sperm whale cuts do not dominate.
- **Windows.** Each LSTM epoch shows the model one randomly placed 3 s window of every training cut; cuts shorter than 3 s are used whole. A cut's prediction averages all its windows, taken every 1.5 s.
- **LSTM design.** The classifier reads the average of the LSTM's outputs over the window. That averaging and the learning rate of 0.003 were chosen in a pilot on validation fold 0, because the first design (last output only, learning rate 0.001) was still far from converged after 80 epochs.
- **Tuning.** One setting per model is tuned on the pooled cross-validated macro-F1: `C` for the logistic regressions (0.01 to 10) and the number of epochs for the LSTMs (1 to 40). The window length is fixed at 3 s rather than tuned, to keep the stage within minutes.
- **Order.** The shuffled LSTM is the same model, with the same seed, as the LSTM, but the frames in each window are put in random order. The gap between the two measures what order adds beyond the roughly 40 ms that each frame's deltas already describe.
- **Test set.** It is used once, when `evaluation.evaluate_test` is set to true. Gaps in macro-F1 get a 95% bootstrap interval that resamples whole test tapes, and the comparison is repeated on the cuts recorded at 16 kHz or above.

## Data source and licence

The data are the "best of" cuts of the Watkins Marine Mammal Sound Database, downloaded from the Hugging Face mirror [`confit/wmms-parquet`](https://huggingface.co/datasets/confit/wmms-parquet) at the revision pinned in `config.yaml`. The mirror's card also points to an [archive.org copy](https://archive.org/details/watkins_202104) of the database.

The card declares 16 kHz, but the stored WAV files keep their native sample rates, from 600 Hz to 166.6 kHz.

Credit: "Watkins Marine Mammal Sound Database, Woods Hole Oceanographic Institution and the New Bedford Whaling Museum."

**Licence:** free for personal or academic use only, not commercial use, and reposting is forbidden. The audio is therefore never committed to this repository; `code/01_get_data.py` downloads it instead.

## Resources beyond a CPU-only cloud workspace

None. The pipeline needs no GPU. The download is about 1.2 GB.

## Attribution

Code generated with AI assistance carries a comment at the top of each file and is logged in [AI_USAGE.md](AI_USAGE.md), along with any code adapted from external sources.
