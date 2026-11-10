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

## Running the pipeline: menu or direct commands

- **Menu:** run `python run.py` from the repository root. A window opens that lists every stage with its status and has buttons to run one stage, the ticked stages, all of them, or only the out-of-date ones, plus Stop, the tests and the linter. Output appears live in the window. Close the window to get the terminal back. If a stage is still running, the window offers to stop it first.
- **Direct commands:** `python run.py list` (status), `python run.py all`, `python run.py outdated`, `python run.py 03 05` or `python run.py features`. Run `python run.py --help` for more. These need no window and no Make.

**A new stage needs no extra setup.** A stage is any file in `code/` named `NN_name.py` or `NN_name.ipynb` (a letter may follow the number, as in `04a_name.ipynb`). The menu, `run.py` and `make all` find stages by themselves and run them in number order. The menu's description comes from the script's docstring or the notebook's first markdown cell. A window that is already open shows a new stage after you press **Refresh**.

**What "out of date" means.** After every successful run, the runner records what the stage depends on. A stage is out of date when any of these changed since its last successful run:

- its code;
- a module of `src/ecoacoustics/` that it imports, directly or through another module;
- a section of `config.yaml` that it reads;
- a file it could read: new data, or the outputs of an earlier stage that re-ran.

A stage is also out of date when one of its outputs was deleted. **Run out-of-date** re-runs exactly these stages, in order, plus any later stage whose inputs they rewrite. Changes to line endings alone are ignored. Data and results folders are taken from the `paths` section of `config.yaml`, so a folder added there is tracked too.

Each run is logged in `results/logs/runs/`, along with the history used to decide what is out of date. The executed copy of each notebook stage, with its outputs, is saved in `results/logs/notebooks/`. None of this is tracked in git.

## What to run, in order

Run from the repository root with the environment active. Each stage reads only what earlier stages saved.

| Stage | Command | Equivalent without Make | Approximate time |
|---|---|---|---|
| Get data | `make data` | `python code/01_get_data.py` | about 2 min, nearly all of it the download |
| Prepare | `make prepare` | `python code/02_prepare.py` | a few seconds |
| Features | `make features` | `python code/03_features.py` | about 10 s |
| Unsupervised | `make unsupervised` | `jupyter nbconvert --to notebook --execute --output-dir results/logs/notebooks code/04_unsupervised.ipynb` | to be measured |
| Supervised | `make supervised` | `jupyter nbconvert --to notebook --execute --output-dir results/logs/notebooks code/05_supervised.ipynb` | about 18 min with the test set locked, nearly all of it the LSTMs; longer once it is unlocked (not measured, because the test set is still unused) |
| Same-period check | `make same_period` | `python code/05a_same_period.py` | about 1 min |
| LSTM pilot | `make lstm_pilot` | `python code/05b_lstm_pilot.py` | about 1 min |
| Figures | `make figures` | `jupyter nbconvert --to notebook --execute --output-dir results/logs/notebooks code/06_figures.ipynb` | to be measured |

All times were measured on one computer with an AMD Ryzen 7 9800X3D CPU (8 cores, 16 threads) and no GPU.

`make all` (or `python run.py all`) runs every stage in order, `make outdated` runs only the out-of-date ones, and `make analysis` runs the unsupervised, supervised, same-period and pilot stages. `make test` runs the tests and `make lint` runs ruff. The notebook commands in the table give the same executed notebook as the runner, which also prints each cell's output as it runs.

## Outputs

- `results/figures/`: every figure in the paper.
- `results/tables/`: every table in the paper, as `.csv` and `.tex`.
- `results/outputs/`: the saved outputs that back the results, committed so that they can be checked without re-training: the LSTM learning curves of every seed, the per-cut predictions and the chosen settings. They hold cut ids and species labels, never audio.
- `results/models/` and `results/logs/`: trained models and run logs, not tracked in git.
- `data/`: downloaded audio and cached features, not tracked in git. See [data/README.md](data/README.md).

## How the data are cleaned and split

Stage 02 makes these choices, all set in `config.yaml`:

- Audio that appears under two species labels is dropped entirely (25 files, 50 rows). A single-label model cannot be right on both copies.
- Cuts are grouped by recording tape, the first five digits of the Watkins file name (tape 72021 in `72021005.wav`). No tape is in both the training and the test set, or in two cross-validation folds.
- A species needs at least 20 cuts and 3 tapes, so that it can have a test tape and two training tapes. 23 species remain.
- Whole tapes holding about 15% of the cuts form the test set, which includes every species. The rest is split into 5 grouped cross-validation folds for tuning.

`results/tables/data_flow` and `results/tables/split_summary` give the counts at each step and per species. Some species have only one or two cuts in the test set because of how their tapes are sized, so per-species test results are noisy.

**Doubtful labels.** All the test cuts of Bearded Seal (3, on tape 78018) and of Long-Finned Pilot Whale (9, on tape 77028) are on tapes where other cuts, dropped here, also carried a second species label (Bowhead Whale and Sperm Whale). These tapes may hold both species. They stay in the test set, and stage 05 reports test scores both with and without these two species.

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
- **Seeds.** The LSTM and the shuffled LSTM are each trained on the same ten seeds, derived from the seed in `config.yaml` (42 to 51; five until 10 Nov 2026), because the seed alone moved the best macro-F1 by 0.017, more than the gap being measured. Their scores are the mean over the seeds, with the spread across them: the standard deviation, the lowest and the highest. `results/tables/cv_seed_gaps` gives every score and paired gap seed by seed.
- **Weights.** Training gives every species the same total weight. For the models that read windows, every cut also counts once, however long it is, so the three 20-minute sperm whale cuts do not dominate.
- **Windows.** Each LSTM epoch shows the model one randomly placed 3 s window of every training cut; cuts shorter than 3 s are used whole. A cut's prediction averages all its windows, taken every 1.5 s.
- **LSTM design.** The classifier reads the average of the LSTM's outputs over the window. That averaging and the learning rate of 0.003 were chosen in a pilot on validation fold 0. On that fold the first design (last output only, learning rate 0.001) reached a best macro-F1 of 0.301, at epoch 41 of 80, while averaging with a learning rate of 0.003 reached 0.399, at the last of its 50 epochs; the other variants reached 0.318 to 0.345. `code/05b_lstm_pilot.py` regenerates the pilot, and `results/tables/lstm_pilot` gives its scores.
- **Tuning.** One setting per model is tuned on the pooled cross-validated macro-F1: `C` for the logistic regressions (0.01, 0.1, 1, 10, 30, 100) and the number of epochs for the LSTMs (1 to 80), each LSTM on its own validation curve averaged over the seeds. Both ranges were widened on 10 Nov 2026, before the test set was used, because the earlier choices (`C` = 10 and 38 of 40 epochs) sat at the edge of the ranges. The LSTMs therefore had more selection opportunity than the logistic regressions: 80 epochs, after the pilot, against 6 values of `C`. The window length is fixed at 3 s rather than tuned, to limit the run time.
- **Order.** The shuffled LSTM is the same model as the LSTM. For a given seed the two start from the same weights and see the same windows in the same batches, but the frames in each window are put in random order. The gap between the two measures what order adds beyond the roughly 40 ms that each frame's deltas already describe. It is reported seed by seed, but the two models' training paths diverge within a few epochs, so pairing them by seed does not make the gap more precise.
- **Reproducibility.** The same seed gives different LSTM results on a different number of CPU threads, through rounding, so every LSTM training always runs on 2 threads (`lstm.num_threads` in `config.yaml`). The trainings run in separate processes, 4 at once (`lstm.n_jobs`, lowered automatically on a computer with fewer CPUs); that changes only the speed, never the results, because each training sets its own seed and threads. A different CPU may still give slightly different numbers.
- **Test set.** It is used once, when `evaluation.evaluate_test` is set to true. Gaps in macro-F1 get a 95% bootstrap interval that resamples whole test tapes, with the spread of the paired gap across seeds. Every score and gap is reported with all species, which is the primary result, and again without Bearded Seal and Long-Finned Pilot Whale, whose labels are doubtful, as a sensitivity check; the comparison is also repeated on the cuts recorded at 16 kHz or above. The test set is small: 17 of the 23 species have all their test cuts on one tape, so its intervals are expected to be wide.
- **Same-period check.** Whole tapes are kept apart, but tapes with close numbers may come from the same period, trip or equipment. Stage 05a holds out each training tape in turn and compares a logistic regression trained on all other tapes, on all but those sharing the first two digits of the held-out tape, and on all but the same number of tapes chosen at random (`results/tables/same_period_check`). It uses training data only, and the logistic regression on whole cuts only; its scores (one tape held out at a time, 18 species) are not on the same scale as the main cross-validated scores. The first two digits are only possibly the recording year, which is unconfirmed. Macro-F1 falls further when the tapes sharing the first two digits are dropped than when the same number of tapes are dropped at random, so the main scores probably overstate performance on a new period.

## Data source and licence

The data are the "best of" cuts of the Watkins Marine Mammal Sound Database, downloaded from the Hugging Face mirror [`confit/wmms-parquet`](https://huggingface.co/datasets/confit/wmms-parquet) at the revision pinned in `config.yaml`. The mirror's card also points to an [archive.org copy](https://archive.org/details/watkins_202104) of the database.

The card declares 16 kHz, but the stored WAV files keep their native sample rates, from 600 Hz to 166.6 kHz.

Credit: "Watkins Marine Mammal Sound Database, Woods Hole Oceanographic Institution and the New Bedford Whaling Museum."

**Licence:** free for personal or academic use only, not commercial use, and reposting is forbidden. The audio is therefore never committed to this repository; `code/01_get_data.py` downloads it instead.

## Resources beyond a CPU-only cloud workspace

The pipeline needs no GPU, and the download is about 1.2 GB. Stage 05 trains 100 LSTMs (two models, ten seeds, five folds, 80 epochs each) and takes about 18 minutes on the CPU named above, where it runs 4 trainings at once on 2 threads each. On a computer with fewer CPUs it runs fewer at once (`lstm.n_jobs` is lowered automatically), which takes longer but does not change the results. The outputs that back the results are committed in `results/tables/` and `results/outputs/`, so the results can be checked without re-training. Setting `lstm.n_seeds` to 1 in `config.yaml` gives a run of about 2 minutes, but it does not reproduce the reported ten-seed table.

## Attribution

Code generated with AI assistance carries a comment at the top of each file and is logged in [AI_USAGE.md](AI_USAGE.md), along with any code adapted from external sources.
