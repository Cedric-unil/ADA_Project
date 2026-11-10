# CLAUDE.md

Context for AI coding assistants working in this repository. Read it fully before making changes.

## The project in one paragraph

Individual semester project for **Advanced Data Analytics** (HEC Lausanne, autumn 2026), author
Cédric Nim. It is the only assessment in the course (100% of the grade). Title: **"Spectral Shape or
Temporal Structure? Species Identity in Marine Mammal Vocalizations."** We classify marine mammal
species from recordings of their calls and ask whether a model needs the *order* of sounds
(temporal structure) or whether the overall timbre (spectral shape) is enough. An unsupervised
part asks whether clusters found without labels recover species.

The author is a finance master's student, not a data scientist. Explain decisions in plain
language, keep code readable over clever, and ask before making any research choice.

## Research question and hypotheses

- **H1 (supervised):** an LSTM on MFCC frame sequences achieves higher macro-F1 than a
  multinomial logistic regression on time-averaged MFCC statistics.
- **H2 (unsupervised):** Gaussian mixture clusters (on PCA-reduced clip-level features) agree with
  species (ARI and NMI clearly above zero), more than with recording era or equipment.
- A null result is a valid finding ("spectral shape suffices at this data scale"). Never tune,
  filter or re-split data to make an effect appear.

## Data

- **Source:** Watkins Marine Mammal Sound Database, Woods Hole Oceanographic Institution (WHOI),
  "best of" cuts: about 1,700 expert-labelled cuts, 32 species.
- **Download:** Hugging Face mirror `confit/wmms-parquet` via `code/01_get_data.py`, at the
  revision pinned in `config.yaml` (WAV bytes, columns `audio`, `species`, `label`; its own
  random 80/20 split is **ignored**). The card says 16 kHz, but each cut keeps its native sample
  rate, from 600 Hz to 166.6 kHz. Fallbacks: WHOI repository https://darchive.mblwhoilibrary.org/handle/1912/5877
  and the main site https://whoicf2.whoi.edu/science/B/whalesounds.
- **Licence:** free for personal or academic, non-commercial use; reposting is not permitted.
  Credit: "Watkins Marine Mammal Sound Database, Woods Hole Oceanographic Institution and the New
  Bedford Whaling Museum." **The audio must never be committed.** `data/raw/` is gitignored; the
  download script plus this licence statement is what the repository contains (requirement 4.15).
- **Grouping key (confirmed):** `tape_id`, the first five digits of the Watkins file name
  (`72021005.wav` → tape `72021`), saved in `data/interim/metadata.csv`. The first two digits
  may encode the recording year (`72` = 1972); **unconfirmed**, check WHOI Technical Report
  92-31 before using it. Repository upload dates are not recording dates.
- **After stage 02:** 1,316 cuts, 23 species, 134 tapes (`results/tables/data_flow`,
  `split_summary`). 50 rows (25 audio files) carried two species labels and were dropped.
- **Known issues:**
  - Clip lengths run from 0.22 s to about 20 min. 72% are shorter than 3 s; the longest are
    sperm whale and humpback cuts.
  - Native sample rate is confounded with species: 237 cuts are below 16 kHz, including every
    fin whale cut (about 600 Hz), and nearly every harp seal, humpback and northern right whale
    cut. After resampling they have empty high frequencies.
  - Heavy class imbalance; small classes; species partly confounded with recording era and
    equipment.
  - Doubtful test labels: all the test cuts of Bearded Seal (tape 78018) and Long-Finned Pilot
    Whale (tape 77028) are on tapes where other cuts also carried a second species label
    (decision C below).

## Pipeline and repository layout

```
.
├── CLAUDE.md
├── README.md                 # 2-sentence summary, install, run order, runtimes, outputs, data + licence
├── requirements.txt          # pinned packages
├── pyproject.toml            # makes src/ importable: pip install -e .
├── config.yaml               # ALL paths, seeds, feature and model hyperparameters
├── run.py                    # python run.py opens the menu window; or run.py list|all|outdated|03 05
├── Makefile                  # make data / features / analysis / figures / all, via run.py
├── AI_USAGE.md               # running log of AI assistance (feeds the paper appendix)
├── .gitignore
├── data/                     # gitignored except data/README.md
│   ├── raw/                  # downloaded Parquet files with the audio (never committed)
│   ├── interim/              # inventory.csv, metadata.csv (split + folds), excluded.csv
│   └── features/             # cached MFCC frames and clip statistics (never committed)
├── src/ecoacoustics/         # importable package: config, io, splits, features,
│                             #   models, evaluation, sensitivity, plotting, utils;
│                             #   pipeline (find/run stages, out-of-date checks), menu (window)
├── code/                     # thin numbered stages; each loads the previous stage's output
│   ├── 01_get_data.py
│   ├── 02_prepare.py         # filtering, metadata, grouped folds saved to file
│   ├── 03_features.py        # MFCC + deltas per frame, clip-level stats; cached once
│   ├── 04_unsupervised.ipynb # PCA, GMM (K by BIC), ARI/NMI
│   ├── 05_supervised.ipynb   # baselines, logistic regression, LSTM on five seeds, comparison
│   ├── 05a_same_period.py    # same-period sensitivity check, training data only
│   ├── 05b_lstm_pilot.py     # the pilot on validation fold 0 behind the LSTM design
│   └── 06_figures.ipynb      # every paper figure and table
├── results/
│   ├── figures/              # committed
│   ├── tables/               # committed, .csv and .tex
│   ├── outputs/              # committed: learning curves, per-cut predictions, chosen settings
│   ├── models/               # gitignored
│   └── logs/                 # gitignored
├── tests/                    # pytest
└── paper/                    # paper.pdf (written in Overleaf, SIAM template)
```

Stage scripts stay thin: read config, call package functions, save outputs. Shared code lives in
`src/ecoacoustics/` and is imported, never copied (requirement 4.9).

**Stages are discovered, not listed.** Any `code/NN_name.py` or `code/NN_name.ipynb` (optionally
`NNa_name`) is a stage. `run.py` (its menu window and its commands) and `make all` find stages
automatically and run them in number order, so adding a stage needs no other change. Give each new
stage a module docstring (script) or a first markdown cell (notebook): its first line is the menu
description. A stage's
config dependencies are detected from `config["section"]` / `config.get("section")` /
`resolve_path(config, ...)` in its code, so read config that way. Handing the whole `config` to a
function makes the stage depend on every section. Run history and logs live in
`results/logs/runs/`, and executed notebooks in `results/logs/notebooks/`.

## Methods (course weeks in brackets)

1. **Features (done, stage 03):** each cut is resampled to 16 kHz and peak-normalised (loudest
   sample = 1, removes recording gain), then `librosa` MFCCs (13 coefficients) + deltas every
   10 ms; clip-level summary = mean and std of each coefficient and delta (52 numbers). Cached
   in `data/features/`; read with `load_frames` / `load_clip_stats`. Windows of 3 s every
   1.5 s are cut at training time from the cached frames (fixed, not tuned, to keep stage 05
   within minutes).
2. **PCA (11)** on standardized clip-level statistics, not on raw MFCC frames (MFCCs are already
   decorrelated). Report explained variance.
3. **GMM (12)**, diagonal covariance, K chosen by BIC over a range; evaluate with ARI and NMI against
   species, and against tape/era if available.
4. **Baselines (6):** chance (~1/number of classes), majority class, and multinomial logistic
   regression on time-averaged features. All computed in code, never typed in.
   **Sample-rate checks (decided 7 Nov 2026):** keep all 23 species in the main analysis; add a
   reference model that sees only each cut's native sample rate, and repeat the main comparison
   without the cuts below 16 kHz as a robustness check.
5. **LSTM (7–8):** PyTorch, 1 layer, 64 hidden units, dropout 0.3; the classifier reads the
   average of the LSTM outputs over the window (padding left out by a mask; no packing, which
   gives the same outputs but is about 5 times slower on the CPU). Each epoch = one random 3 s
   window per training cut; clip prediction = average over all its windows. Learning rate
   0.003 and output averaging chosen in a pilot on validation fold 0 (regenerated by stage 05b).
   Trained on ten seeds (decision A below). Each training runs in its own process on 2 threads
   (`lstm.num_threads`), 4 at once (`lstm.n_jobs`, which changes only the speed); until 10 Nov
   2026 they ran one at a time on 8 threads. The number of epochs (1–80, decision D) is tuned
   on pooled cross-validated macro-F1, as `C` is for logistic regression (0.01 to 100), each
   LSTM on its own validation curve averaged over the seeds; this replaces early stopping.
   **Controls (decided 7 Nov 2026):** logistic regression on the same windows' statistics (same
   input as the LSTM), and the same LSTM on frames shuffled within each window (isolates
   order). For a given seed both LSTMs start from the same weights and see the same windows in
   the same batches. Implemented in `code/05_supervised.ipynb`.
6. **Diagnostics:** per-class confusion matrices, learning curve, accuracy vs estimated
   signal-to-noise ratio, bootstrap 95% interval on the LSTM–baseline macro-F1 gap.

## Methodological rules (non-negotiable)

- **Grouped splits only.** Whole tapes holding about 15% of the cuts form a held-out test set
  that covers every species; the remaining tapes get 5 `StratifiedGroupKFold` folds for tuning.
  No tape appears in both training and evaluation data. Split and folds are saved in
  `data/interim/metadata.csv` (`split`, `cv_fold`); every stage uses them. The test set is used
  once, when `evaluation.evaluate_test` is set to true after hyperparameters are frozen. If no grouping key exists, **stop and tell the author**; never
  silently fall back to a random split.
- Each fold starts from a fresh, untrained model.
- Fit scalers, PCA and normalization **on training data only**, inside each fold.
- Tune only on validation data carved from the training folds; held-out data is for final scores.
- Drop species below a configurable minimum (default 20 cuts) and log what was dropped and why.
- Windows from one clip always stay in the same split as the clip.
- Report macro-F1 (primary), accuracy and confusion matrices; state the averaging method explicitly.
- One base seed, set in `config.yaml`, applied to Python, NumPy, PyTorch, scikit-learn and GMM
  init. The LSTM and its shuffled control are trained on ten seeds derived from it (the seed,
  then the next integers), all at the same fixed number of threads (`lstm.num_threads`), because
  the seed effect (0.017) is larger than the gap being measured (decision A below).
- **Never invent numbers, results or paper text.** Build and run the pipeline; the author
  interprets the results. Data or results adjusted to produce an effect are academic misconduct
  (requirement 7.3).

## Decisions after the audit (8 Nov 2026, amended 10 Nov 2026)

The author took decisions A to C after a read-only audit of the project, and the amendments and
decision D on 10 Nov 2026, after the cross-validation results and before the test set was used.
The paper is not written yet, so what it must say is listed here.

- **A. Five LSTM seeds (ten since 10 Nov 2026).** The LSTM and the shuffled control run on the
  same five seeds derived from the config seed (`lstm.n_seeds`: seeds 42 to 46), all at the same
  fixed number of threads (`lstm.num_threads`: 8 when decided, 2 since the speed-up of 10 Nov
  2026), so the gap is paired by seed. Each model picks its number of epochs on its own average validation
  curve. Tables report the mean and the spread across seeds (`cv_results`, `cv_seed_gaps`).
  *Why:* the seed effect is larger than the gap being measured. In the audit the best pooled LSTM
  macro-F1 was 0.323, 0.306 and 0.310 for seeds 42, 1 and 2, a spread of 0.017, while LSTM minus
  shuffled was 0.004 to 0.013. (Those runs came before shuffling got its own random generator, and
  seeds 1 and 2 ran on 4 threads, which is itself a source of differences.) A 5-epoch rolling
  average did not steady the choice of epoch (26, 38, 36), so it is not used.
  *Amended 10 Nov 2026: ten seeds* (`lstm.n_seeds: 10`, seeds 42 to 51), now that a run is cheap.
  In the five-seed run the LSTM minus shuffled gap was +0.009 on average but ranged from -0.022 to
  +0.050 across seeds (SD 0.026), and rounding alone changed about half of the cut-level
  predictions after a few epochs, so single runs are unreliable. *The paper must not present the
  pairing by seed as a gain in precision:* the two models start from the same weights and batches,
  but their training paths diverge within a few epochs, and the per-seed gap varied about as much
  as for unrelated runs (SD 0.026, against about 0.023 expected for independent runs).
- **B. Keep the approved tape split and add a same-period check.** Stage 05a holds out each
  training tape in turn and compares a logistic regression trained on all other tapes, on all but
  those sharing the first two digits of the held-out tape, and on all but the same number of tapes
  chosen at random (`same_period_check`, `same_period_overlap`). It uses training data only.
  *Why:* tapes with close numbers may share a period, trip or equipment, so a split by tape does
  not make them independent; re-splitting would change the approved design. *The paper must say*
  that scores probably overstate performance on a new period, and that the first two digits are
  only possibly the recording year.
  *Amended 10 Nov 2026: B is kept, and the paper must also say* that the check covers the
  logistic regression on whole cuts only (repeating it for the LSTM would take thousands of
  trainings; all models read the same features, so the effect plausibly applies to them, but this
  is untested), and that its scores (leave one tape out, 866 cuts of 18 species) are not on the
  same scale as the main cross-validated scores and must not be compared with them. Whether the
  first two digits are the recording year is still to be checked in WHOI Technical Report 92-31
  (see the open questions below).
- **C. Keep both doubtful test tapes and disclose them.** All 3 Bearded Seal test cuts are on tape
  78018, where 6 dropped cuts also carried the Bowhead Whale label; all 9 Long-Finned Pilot Whale
  test cuts are on tape 77028, where 5 dropped cuts also carried the Sperm Whale label. These tapes
  may hold both species. Stage 05 reports every test score and gap both with all species and
  without these two (`evaluation.doubtful_species`): their cuts leave the evaluation, while the
  models still train on these species. *Why:* the labels may be wrong, but the approved split stays.
  Both reports are fixed now, before unlocking, so that neither can be chosen after seeing test
  scores. *The paper must disclose* the doubtful labels.
  *Amended 10 Nov 2026: the report with all species is the primary result;* the one without the
  two doubtful species is a sensitivity check. *Expect a noisy test set:* 17 of the 23 species
  have all their test cuts on a single tape, some have only 1 to 3 test cuts (Harp Seal 1,
  Bottlenose Dolphin 1, Bearded Seal 3), and one species can move macro-F1 by up to 1/23 (about
  0.04), more than the cross-validated gaps. The tape bootstrap will likely give wide intervals;
  the author decides before unlocking how an interval that includes zero will be reported.
- **D. Widen the tuning ranges chosen at their edge (10 Nov 2026).** The LSTMs' number of epochs
  now runs from 1 to 80 (`lstm.max_epochs`, was 40) and the logistic regressions' `C` grid is
  0.01, 0.1, 1, 10, 30 and 100 (`baseline.c_grid`, was 0.01 to 10). *Why:* in the five-seed
  cross-validation the LSTM chose 38 of 40 epochs, the window logistic regression chose `C` = 10,
  the top of its grid, and the whole-cut one tied between 1 and 10; a choice at the edge of a
  range suggests an under-tuned model. Both model families are widened, so neither gains extra
  tuning over the other. Decided before the new results were seen; the outcome is reported
  whatever it is. The pilot (stage 05b) keeps its own epochs, as the record of how the design
  was chosen.

## Binding course requirements (summary; the Requirements PDF wins if in doubt)

- **Deadlines:** proposal by Mon 9 Nov 2026, 23:59 Zurich time; full project by **Sun 20 Dec 2026,
  23:59 Zurich time**. Late work gets zero.
- **Submission** = a public GitHub repository plus an email to the TA with the repo URL and the full
  40-character commit SHA. Only that commit is graded. Never rewrite history or force-push after
  submission.
- The repository holds: the paper PDF (~10 pages, at least 8 excluding references, SIAM style), the
  data or its download code, the source code, and a ~15-minute recording (or a link to it).
- **Paper sections, in order:** Abstract; Introduction; Research question and literature;
  Methodology; Data (source and cleaning); Implementation; Results; Conclusion; Appendix:
  declaration of tools used.
- Each stage can be found and run on its own. **No single file contains the whole analysis.** Every
  notebook runs top to bottom in a freshly started kernel.
- Every figure and table in the paper is produced by code in the repository.
- No caches, checkpoints, virtual environments, editor settings (`.vscode/`), data or model files
  in git.
- The README states
  approximate runtimes.
- No secrets or keys in the code or history.

## Attribution (requirements 5 and 6)

- Libraries used as intended need no attribution.
- **AI-generated code:** put a comment at the top of each generated file or function:
  `# Generated with <tool> (<what was requested>).` and add an entry to `AI_USAGE.md` (date,
  file, request, what the author changed). In this repository the comment reads
  `# Generated with Claude Code (Anthropic): <what was requested>.`
- **Course material** (lecture notebooks, exercise solutions): reusable, but identify it where
  used.
- **Anything else not written by the author** (docs examples, repos, forum answers, papers):
  a comment naming the source and link where it appears, and an entry in `AI_USAGE.md` with
  its licence. Do not use code whose licence you cannot state.
- Never use code from other students' projects in this course or previous years.

## Conventions

- Python 3, type hints, short docstrings, small functions.
- `pathlib` paths relative to the repo root; no absolute paths.
- `logging` in the package; plain prints are fine in stage scripts.
- Lowercase file names, no spaces or accents.
- `ruff` for linting; notebooks committed with outputs cleared (nbstripout).
- Figures: axis labels with units, readable fonts, saved as PDF and PNG to `results/figures/`.
- Tables: saved as `.csv` and `.tex` to `results/tables/`.
- Small, focused commits with clear messages. Commit and push regularly.

## Tests (`tests/`)

- No group (recording) appears in more than one fold.
- Scalers/PCA are fitted on training indices only. PCA joins these tests with stage 04.
- Feature shapes are as expected for a known clip length.
- Running a stage twice with the same seed gives identical outputs. Stage 05 takes about 18 minutes,
  so `tests/test_stage_rerun.py` runs the real stages on a reduced configuration and synthetic
  data, named by the `ECOACOUSTICS_CONFIG` environment variable. Its synthetic test set also
  exercises the test-set cells of stage 05; no test touches the real test set.

## Common commands

```bash
pip install -r requirements.txt && pip install -e .
python run.py           # opens the menu: a window with buttons (Tkinter)
python run.py list      # status of every stage and why it is out of date
python run.py outdated  # run only what changed (make outdated)
python run.py 03 05     # run chosen stages, by number or name
make data        # 01_get_data.py
make prepare     # 02_prepare.py
make features    # 03_features.py
make analysis    # stages 04, 05, 05a and 05b
make figures     # 06 notebook
make all
pytest
```

`make` is not installed on the author's Windows machine; use `python run.py` there.
Locally the packages are installed in the global Python 3.13 (no virtual environment). Until
`pip install -e .` has been run there, tests need `src` on the path:
`PYTHONPATH=src python -m pytest`.

## Current status and open questions

- Proposal approved (by 7 Nov 2026). Stages 01 (download), 02 (clean, split), 03 (features), the
  cross-validation part of 05 (supervised, ten seeds, wider tuning ranges: decisions A and D of
  10 Nov 2026), 05a (same-period check) and 05b (LSTM pilot) are done. **The test set is still locked** (`evaluation.evaluate_test: false`) and stays
  locked until the author explicitly says otherwise. Next: stage 04 (unsupervised), then 06
  (figures), then unlock the test set once all settings are final.
- The ten-seed run may be slow or impractical in the course workspace, so the outputs that back
  the results are committed in `results/tables/` and `results/outputs/`.
- Open: is the recording year available (first two digits of the tape id)? Needed for the
  "era" comparison in H2; otherwise compare clusters with tapes. The answer should be in WHOI
  Technical Report 92-31 ("SOUND database of marine animal vocalizations: structure and
  operations", Watkins et al., 1992): https://cis.whoi.edu/science/B/whalesounds/WHOI-92-31.pdf,
  also DTIC ADA256661. On 10 Nov 2026 these sites refused automated access (maintenance page, 403),
  so the author checks the report by hand.
- When an open question is resolved, update this file.
