# Generated with Claude Code (Anthropic): save figures and tables to results/.
"""Figure and table output helpers. Filled in as stages need them."""

from pathlib import Path

import pandas as pd


def save_table(
    table: pd.DataFrame, name: str, tables_dir: Path, float_format: str = "%.2f"
) -> None:
    """Save ``table`` as ``<name>.csv`` and ``<name>.tex``.

    The LaTeX version escapes special characters and uses booktabs rules, so the paper needs
    ``\\usepackage{booktabs}``.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables_dir / f"{name}.csv", index=False)
    table.to_latex(
        tables_dir / f"{name}.tex", index=False, escape=True, float_format=float_format
    )
