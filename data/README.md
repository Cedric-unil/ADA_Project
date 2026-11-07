<!-- Generated with Claude Code (Anthropic): explain the contents of the data folder. -->
# Data

Everything in this folder except this file is gitignored and is rebuilt by the pipeline. **The audio must never be committed:** the licence allows personal or academic use only and forbids reposting.

| Folder | Written by | Contents |
|---|---|---|
| `raw/` | `code/01_get_data.py` | Parquet files downloaded from the Hugging Face mirror at a pinned revision |
| `interim/` | `code/01_get_data.py`, `code/02_prepare.py` | `inventory.csv` (every cut), `metadata.csv` (the cuts kept, with tape, split and fold) and `excluded.csv` (the cuts dropped, with the reason) |
| `features/` | `code/03_features.py` | Cached MFCC frame sequences and clip-level statistics |

`interim/inventory.csv` has one row per cut: labels, source file name, native sample rate, duration, a hash of the audio, and where to find the cut in `raw/`. The stored audio keeps its native sample rate, whatever the dataset card declares.

`interim/metadata.csv` adds `cut_id`, `tape_id`, `split` (`train` or `test`) and `cv_fold` (0 to 4 for training cuts, -1 for test cuts). Read it with `ecoacoustics.io.read_metadata`, which keeps the ids as strings.

Run `make data` or `python code/01_get_data.py` to download the audio. See the main README for the source and licence.
