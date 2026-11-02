<!-- Generated with Claude Code (Anthropic): explain the contents of the data folder. -->
# Data

Everything in this folder except this file is gitignored and is rebuilt by the pipeline. **The audio must never be committed:** the licence allows personal or academic use only and forbids reposting.

| Folder | Written by | Contents |
|---|---|---|
| `raw/` | `code/01_get_data.py` | Parquet files downloaded from the Hugging Face mirror at a pinned revision |
| `interim/` | `code/01_get_data.py`, `code/02_prepare.py` | Per-cut inventory, filtered metadata and the saved split assignment |
| `features/` | `code/03_features.py` | Cached MFCC frame sequences and clip-level statistics |

Run `make data` or `python code/01_get_data.py` to download the audio. See the main README for the source and licence.
