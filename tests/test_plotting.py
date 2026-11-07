# Generated with Claude Code (Anthropic): test that tables are saved as CSV and escaped LaTeX.
"""Tests for the table output helper."""

from pathlib import Path

import pandas as pd

from ecoacoustics.plotting import save_table


def test_table_is_saved_as_csv_and_escaped_latex(tmp_path: Path) -> None:
    table = pd.DataFrame({"Species": ["Grampus, Rissos_Dolphin"], "Test share (%)": [15.04]})

    save_table(table, "summary", tmp_path, float_format="%.1f")

    assert pd.read_csv(tmp_path / "summary.csv").equals(table)
    tex = (tmp_path / "summary.tex").read_text(encoding="utf-8")
    assert r"Rissos\_Dolphin" in tex
    assert r"Test share (\%)" in tex
    assert "15.0" in tex
