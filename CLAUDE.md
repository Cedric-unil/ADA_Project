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

## Pipeline and repository layout

```
.
├── CLAUDE.md
├── README.md                 # 2-sentence summary, install, run order, runtimes, outputs, data + licence
├── requirements.txt          # pinned packages
├── pyproject.toml            # makes src/ importable: pip install -e .
├── config.yaml               # ALL paths, seeds, feature and model hyperparameters
├── Makefile                  # make data / features / analysis / figures / all
├── AI_USAGE.md               # running log of AI assistance (feeds the paper appendix)
├── .gitignore
├── data/                     # gitignored except data/README.md
│   ├── raw/                  # downloaded Parquet files with the audio (never committed)
│   ├── interim/              # inventory.csv, metadata.csv (split + folds), excluded.csv
│   └── features/             # cached MFCC frames and clip statistics (never committed)
├── src/ecoacoustics/         # importable package: config, io, splits, features,
│                             #   models, evaluation, plotting, utils
├── code/                     # thin numbered stages; each loads the previous stage's output
│   ├── 01_get_data.py
│   ├── 02_prepare.py         # filtering, metadata, grouped folds saved to file
│   ├── 03_features.py        # MFCC + deltas per frame, clip-level stats; cached once
│   ├── 04_unsupervised.ipynb # PCA, GMM (K by BIC), ARI/NMI
│   ├── 05_supervised.ipynb   # baselines, logistic regression, LSTM, comparison
│   └── 06_figures.ipynb      # every paper figure and table
├── results/
│   ├── figures/              # committed
│   ├── tables/               # committed, .csv and .tex
│   ├── models/               # gitignored
│   └── logs/                 # gitignored
├── tests/                    # pytest
└── paper/                    # paper.pdf (written in Overleaf, SIAM template)
```

Stage scripts stay thin: read config, call package functions, save outputs. Shared code lives in
`src/ecoacoustics/` and is imported, never copied (requirement 4.9).

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
   average of the LSTM outputs over the window (packing + mask). Each epoch = one random 3 s
   window per training cut; clip prediction = average over all its windows. Learning rate
   0.003 and output averaging chosen in a pilot on validation fold 0. The number of epochs
   (1–40) is tuned on pooled cross-validated macro-F1, as `C` is for logistic regression; this
   replaces early stopping.
   **Controls (decided 7 Nov 2026):** logistic regression on the same windows' statistics (same
   input as the LSTM), and the same LSTM on frames shuffled within each window (isolates
   order). Implemented in `code/05_supervised.ipynb`.
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
- One seed, set in `config.yaml`, applied to Python, NumPy, PyTorch, scikit-learn and GMM init.
- **Never invent numbers, results or paper text.** Build and run the pipeline; the author
  interprets the results. Data or results adjusted to produce an effect are academic misconduct
  (requirement 7.3).

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
- Everything must run in the provided **Nuvolos cloud workspace: CPU only, modest resources**. No GPU
  assumptions. Keep each stage to minutes and cache intermediate results. The README states
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
- Scalers/PCA are fitted on training indices only.
- Feature shapes are as expected for a known clip length.
- Running a stage twice with the same seed gives identical outputs.

## Common commands

```bash
pip install -r requirements.txt && pip install -e .
make data        # 01_get_data.py
make prepare     # 02_prepare.py
make features    # 03_features.py
make analysis    # 04 and 05 notebooks
make figures     # 06 notebook
make all
pytest
```

Locally the packages are installed in the global Python 3.13 (no virtual environment). Until
`pip install -e .` has been run there, tests need `src` on the path:
`PYTHONPATH=src python -m pytest`.

## Current status and open questions

- Proposal approved (by 7 Nov 2026). Stages 01 (download), 02 (clean, split), 03 (features)
  and the cross-validation part of 05 (supervised) are done. **The test set is still locked**
  (`evaluation.evaluate_test: false`). Next: stage 04 (unsupervised), then 06 (figures), then
  unlock the test set once all settings are final.
- Open: is the recording year available (first two digits of the tape id)? Needed for the
  "era" comparison in H2; otherwise compare clusters with tapes.
- When an open question is resolved, update this file.
