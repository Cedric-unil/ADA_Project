# Generated with Claude Code (Anthropic): test stage discovery, the runner and out-of-date checks.
"""Tests for the pipeline runner. They build small throwaway projects, never the real data."""

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from ecoacoustics.pipeline import (
    FAILED,
    NOT_RUN,
    OK,
    OUT_OF_DATE,
    STOPPED,
    UP_TO_DATE,
    WAITING,
    Project,
    Runner,
    config_sections,
    describe,
    discover_stages,
    hash_text,
    load_history,
    package_sources,
    pipeline_status,
    select_stages,
)

CONFIG = "first:\n  value: 1\nsecond:\n  value: 2\n"
MAKE = '''"""Stage 01: write the first value."""
from pathlib import Path

import yaml

config = yaml.safe_load(Path("config.yaml").read_text())
Path("data/made.txt").write_text(str(config["first"]["value"]))
print("made it")
'''
USE = '''"""Stage 02: copy it, twice, to the results."""
from pathlib import Path

Path("results/used.txt").write_text(Path("data/made.txt").read_text() * 2)
'''


SLOW = 'import time\nprint("started", flush=True)\ntime.sleep(120)\n'


def _project(root: Path, stages: dict[str, str] | None = None) -> Project:
    """A small project with its own config, folders and stage files."""
    for folder in ("code", "data", "results", "logs", "src/pkg"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(CONFIG, encoding="utf-8")
    for name, text in (stages or {}).items():
        (root / "code" / name).write_text(text, encoding="utf-8")
    return Project(
        root=root,
        code_dir=root / "code",
        package_dir=root / "src" / "pkg",
        config_path=root / "config.yaml",
        logs_dir=root / "logs",
        watched_dirs=(root / "data", root / "results"),
    )


def _states(project: Project) -> dict[str, str]:
    return {stage.file: status.state for stage, status in pipeline_status(project)}


def _reasons(project: Project, file: str) -> str:
    return " | ".join(dict((s.file, st) for s, st in pipeline_status(project))[file].reasons)


def _rewrite(path: Path, text: str) -> None:
    """Change a file and move its modification time on, so the change is always visible."""
    path.write_text(text, encoding="utf-8")
    later = time.time() + 5
    os.utime(path, (later, later))


def _notebook(*cells: tuple[str, str]) -> str:
    return json.dumps(
        {
            "cells": [
                {"cell_type": kind, "id": f"cell{index}", "metadata": {}, "source": source}
                | ({"outputs": [], "execution_count": None} if kind == "code" else {})
                for index, (kind, source) in enumerate(cells)
            ],
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )


# -- finding stages ------------------------------------------------------------------------------


def test_discover_stages_keeps_numbered_scripts_and_notebooks_in_order(tmp_path: Path) -> None:
    names = ["10_last.py", "02_b.py", "01_a.py", "04b_y.ipynb", "04a_x.py", "notes.py", "03_x.txt"]
    for name in names:
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "05_folder.py").mkdir()
    stages = discover_stages(tmp_path)
    assert [stage.file for stage in stages] == [
        "01_a.py",
        "02_b.py",
        "04a_x.py",
        "04b_y.ipynb",
        "10_last.py",
    ]
    assert [stage.kind for stage in stages][2:4] == ["script", "notebook"]


def test_select_stages_by_number_id_or_name(tmp_path: Path) -> None:
    names = ["01_get_data.py", "04a_x.py", "04b_y.py", "05_supervised.ipynb", "06_unsupervised.py"]
    for name in names:
        (tmp_path / name).write_text("", encoding="utf-8")
    stages = discover_stages(tmp_path)

    def files(*selectors: str) -> list[str]:
        return [stage.file for stage in select_stages(stages, selectors)]

    assert files("4") == ["04a_x.py", "04b_y.py"]
    assert files("04a") == ["04a_x.py"]
    assert files("supervised") == ["05_supervised.ipynb"]
    assert files("06", "1") == ["01_get_data.py", "06_unsupervised.py"]  # pipeline order
    with pytest.raises(ValueError, match="No stage matches"):
        files("07")


def test_describe_reads_docstring_or_first_markdown_cell(tmp_path: Path) -> None:
    script = tmp_path / "01_a.py"
    script.write_text('"""Stage 01: download the data.\n\nMore."""\n', encoding="utf-8")
    notebook = tmp_path / "05_b.ipynb"
    notebook.write_text(_notebook(("markdown", "# Stage 05: supervised models\n\nText")), "utf-8")
    bare = tmp_path / "02_c.py"
    bare.write_text("x = 1\n", encoding="utf-8")
    assert describe(script) == "Download the data."
    assert describe(notebook) == "Supervised models"
    assert describe(bare) == ""


# -- what a stage depends on -------------------------------------------------------------------


def test_config_sections_found_in_the_code() -> None:
    config = {"paths": {}, "lstm": {}, "data": {}, "seed": 1}
    assert config_sections('x = config["lstm"]\nresolve_path(config, "a")', [], config) == [
        "lstm",
        "paths",
    ]
    assert config_sections('cfg = load_config()\ncfg.get("data")', [], config) == ["data"]
    assert config_sections("print(1)", [], config) == []


def test_config_sections_count_everything_when_unclear() -> None:
    config = {"paths": {}, "lstm": {}}
    assert config_sections("config = load_config()\nrun(config)", [], config) == ["lstm", "paths"]
    assert config_sections("config = load_config()\nprint(len(config))", [], config) == [
        "lstm",
        "paths",
    ]


def test_package_sources_follow_imports_between_modules(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "a.py").write_text("from pkg.b import f\n", encoding="utf-8")
    (package / "b.py").write_text("from .c import g\n", encoding="utf-8")
    (package / "c.py").write_text("g = 1\n", encoding="utf-8")
    (package / "unused.py").write_text("", encoding="utf-8")
    assert list(package_sources("from pkg.a import x", package)) == [
        "__init__.py",
        "a.py",
        "b.py",
        "c.py",
    ]
    assert package_sources("import numpy", package) == {}


def test_hash_ignores_line_endings() -> None:
    assert hash_text("a\r\nb\r\n") == hash_text("a\nb\n")
    assert hash_text("a\nb\n") != hash_text("a\nc\n")


# -- running and out-of-date detection -------------------------------------------------------------


def test_stages_are_up_to_date_after_a_run(tmp_path: Path) -> None:
    project = _project(tmp_path, {"01_make.py": MAKE, "02_use.py": USE})
    assert set(_states(project).values()) == {NOT_RUN}
    lines: list[str] = []
    assert Runner(project, on_output=lines.append).run_all()
    assert "made it" in lines
    assert (tmp_path / "results" / "used.txt").read_text() == "11"
    assert _states(project) == {"01_make.py": UP_TO_DATE, "02_use.py": UP_TO_DATE}
    records = load_history(project)["stages"]
    assert records["01_make.py"]["last_success"]["outputs"] == ["data/made.txt"]
    assert records["02_use.py"]["last_success"]["outputs"] == ["results/used.txt"]
    assert records["01_make.py"]["last_success"]["fingerprint"]["config"].keys() == {"first"}


def test_changed_input_makes_only_later_stages_out_of_date(tmp_path: Path) -> None:
    project = _project(tmp_path, {"01_make.py": MAKE, "02_use.py": USE})
    Runner(project, on_output=lambda line: None).run_all()
    _rewrite(tmp_path / "data" / "made.txt", "99")
    assert _states(project) == {"01_make.py": UP_TO_DATE, "02_use.py": OUT_OF_DATE}
    assert "outputs of 01_make.py changed: data/made.txt" in _reasons(project, "02_use.py")


def test_new_file_from_outside_counts_once_it_has_been_read(tmp_path: Path) -> None:
    project = _project(tmp_path, {"01_make.py": MAKE, "02_use.py": USE})
    Runner(project, on_output=lambda line: None).run_all()
    _rewrite(tmp_path / "data" / "extra.csv", "new data")  # appears after the runs: not an input
    assert _states(project)["02_use.py"] == UP_TO_DATE
    Runner(project, on_output=lambda line: None).run_all()
    _rewrite(tmp_path / "data" / "extra.csv", "newer data")  # was there at the last run
    assert _states(project)["01_make.py"] == OUT_OF_DATE
    assert "outside the pipeline" in _reasons(project, "01_make.py")


def test_settings_change_affects_only_stages_that_read_them(tmp_path: Path) -> None:
    project = _project(tmp_path, {"01_make.py": MAKE, "02_use.py": USE})
    Runner(project, on_output=lambda line: None).run_all()
    _rewrite(project.config_path, CONFIG.replace("value: 2", "value: 3"))  # section "second"
    assert set(_states(project).values()) == {UP_TO_DATE}
    _rewrite(project.config_path, CONFIG.replace("value: 1", "value: 5"))  # section "first"
    assert _states(project) == {"01_make.py": OUT_OF_DATE, "02_use.py": WAITING}
    assert "settings changed in config.yaml: first" in _reasons(project, "01_make.py")


def test_edited_stage_is_out_of_date_but_line_endings_alone_are_not(tmp_path: Path) -> None:
    project = _project(tmp_path, {"01_make.py": MAKE, "02_use.py": USE})
    Runner(project, on_output=lambda line: None).run_all()
    (tmp_path / "code" / "02_use.py").write_bytes(USE.replace("\n", "\r\n").encode())
    assert _states(project)["02_use.py"] == UP_TO_DATE
    _rewrite(tmp_path / "code" / "02_use.py", USE + "# a comment\n")
    assert _states(project)["02_use.py"] == OUT_OF_DATE
    assert "02_use.py was edited" in _reasons(project, "02_use.py")


def test_missing_output_makes_a_stage_out_of_date(tmp_path: Path) -> None:
    project = _project(tmp_path, {"01_make.py": MAKE, "02_use.py": USE})
    Runner(project, on_output=lambda line: None).run_all()
    (tmp_path / "results" / "used.txt").unlink()
    assert _states(project)["02_use.py"] == OUT_OF_DATE
    assert "outputs missing: results/used.txt" in _reasons(project, "02_use.py")


def test_run_out_of_date_reruns_what_is_needed_and_nothing_else(tmp_path: Path) -> None:
    project = _project(tmp_path, {"01_make.py": MAKE, "02_use.py": USE})
    lines: list[str] = []
    runner = Runner(project, on_output=lines.append)
    runner.run_all()
    _rewrite(project.config_path, CONFIG.replace("value: 1", "value: 7"))
    lines.clear()
    assert runner.run_out_of_date()
    assert any(line.startswith("=== 01_make.py") for line in lines)
    assert any(line.startswith("=== 02_use.py") for line in lines)  # its input was rewritten
    assert (tmp_path / "results" / "used.txt").read_text() == "77"
    assert set(_states(project).values()) == {UP_TO_DATE}
    lines.clear()
    assert runner.run_out_of_date()
    assert lines == ["Every stage is up to date."]


def test_a_failing_stage_stops_the_pipeline(tmp_path: Path) -> None:
    stages = {"01_make.py": MAKE, "02_fail.py": "raise SystemExit(3)\n", "03_use.py": USE}
    project = _project(tmp_path, stages)
    lines: list[str] = []
    assert not Runner(project, on_output=lines.append).run_all()
    expected = {"01_make.py": UP_TO_DATE, "02_fail.py": FAILED, "03_use.py": NOT_RUN}
    assert _states(project) == expected
    assert load_history(project)["stages"]["02_fail.py"]["exit_code"] == 3
    assert "Full log: logs/runs/02_fail.log" in lines


def test_stop_ends_the_running_stage_and_skips_the_rest(tmp_path: Path) -> None:
    project = _project(tmp_path, {"01_slow.py": SLOW, "02_use.py": USE})
    started = threading.Event()
    runner = Runner(project, on_output=lambda line: started.set() if line == "started" else None)
    result: list[bool] = []
    thread = threading.Thread(target=lambda: result.append(runner.run_all()))
    thread.start()
    assert started.wait(30)
    runner.stop()
    thread.join(30)
    assert not thread.is_alive()
    assert result == [False]
    assert _states(project) == {"01_slow.py": STOPPED, "02_use.py": NOT_RUN}


def test_notebook_stage_runs_and_saves_its_executed_copy(tmp_path: Path) -> None:
    good = _notebook(("markdown", "# Stage 01: say hello"), ("code", 'print("hello from it")'))
    bad = _notebook(("code", "x = 1"), ("code", 'raise ValueError("broken on purpose")'))
    project = _project(tmp_path, {"01_good.ipynb": good, "02_bad.ipynb": bad})
    lines: list[str] = []
    runner = Runner(project, on_output=lines.append)
    assert runner.run(select_stages(discover_stages(project.code_dir), ["01"]))
    assert "hello from it" in lines
    assert any(line.startswith("--- cell 1 of 1") for line in lines)
    assert (project.notebooks_dir / "01_good.ipynb").is_file()

    lines.clear()
    assert not runner.run(select_stages(discover_stages(project.code_dir), ["02"]))
    assert any("broken on purpose" in line for line in lines)
    assert (project.notebooks_dir / "02_bad.ipynb").is_file()  # saved even though it failed
    assert load_history(project)["stages"]["02_bad.ipynb"]["status"] == FAILED
    assert load_history(project)["stages"]["01_good.ipynb"]["status"] == OK


# -- the menu window -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tk_root() -> Iterator[Any]:
    """Tcl started once for all the window tests, as the real menu starts it once.

    Starting it again in the same process fails now and then ("couldn't read file init.tcl").
    The tests are skipped on a computer without a screen; any other error fails them.
    """
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        if "display" not in str(error):
            raise
        pytest.skip(f"cannot open a window: {error}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def window(tk_root: Any) -> Iterator[Any]:
    """A hidden window for one test, with its timers cancelled at the end."""
    import tkinter as tk

    top = tk.Toplevel(tk_root)
    top.withdraw()
    yield top
    for timer in tk_root.tk.splitlist(tk_root.tk.call("after", "info")):
        top.after_cancel(timer)
    top.destroy()


def _pump_until(root: Any, done: Callable[[], bool]) -> None:
    """Let the window work, as it does when open, until a condition holds."""
    deadline = time.time() + 60
    while not done():
        assert time.time() < deadline, "timed out"
        root.update()
        time.sleep(0.01)


def _log_text(menu: Any) -> str:
    return menu.log.get("1.0", "end")


def test_window_runs_all_stages_and_shows_their_output(window: Any, tmp_path: Path) -> None:
    from ecoacoustics.menu import PipelineMenu

    project = _project(tmp_path, {"01_make.py": MAKE, "02_use.py": USE})
    menu = PipelineMenu(window, project)
    assert len(menu.row_buttons) == 2
    menu.run_all_button.invoke()
    _pump_until(window, lambda: not menu.busy)
    assert "finished" in menu.status_line.cget("text")
    assert "made it" in _log_text(menu)
    assert "logs/runs/02_use.log" in menu.log_line.cget("text")
    assert set(_states(project).values()) == {UP_TO_DATE}


def test_window_finds_a_new_stage_on_refresh(window: Any, tmp_path: Path) -> None:
    from ecoacoustics.menu import PipelineMenu

    project = _project(tmp_path, {"01_make.py": MAKE})
    menu = PipelineMenu(window, project)
    assert list(menu.ticked) == ["01_make.py"]
    (project.code_dir / "02_use.py").write_text(USE, encoding="utf-8")
    menu.refresh_button.invoke()
    assert list(menu.ticked) == ["01_make.py", "02_use.py"]
    assert len(menu.row_buttons) == 2


def test_window_runs_one_stage_and_only_the_ticked_stages(window: Any, tmp_path: Path) -> None:
    from ecoacoustics.menu import PipelineMenu

    stages = {"01_make.py": MAKE, "02_use.py": USE, "03_other.py": "print('other')\n"}
    project = _project(tmp_path, stages)
    menu = PipelineMenu(window, project)
    menu.run_selected_button.invoke()
    assert "Tick at least one" in menu.status_line.cget("text")
    assert not menu.busy

    menu.ticked["01_make.py"].set(True)
    menu.ticked["03_other.py"].set(True)
    menu.run_selected_button.invoke()
    _pump_until(window, lambda: not menu.busy)
    states = _states(project)
    assert (states["01_make.py"], states["02_use.py"]) == (UP_TO_DATE, NOT_RUN)
    assert states["03_other.py"] == UP_TO_DATE

    menu.row_buttons[1].invoke()  # the run button on the row of 02_use.py
    _pump_until(window, lambda: not menu.busy)
    assert _states(project)["02_use.py"] == UP_TO_DATE
    assert menu.ticked["01_make.py"].get()  # the ticks survive the redraw


def test_window_stop_button_stops_the_stage_and_buttons_follow(window: Any, tmp_path: Path) -> None:
    from ecoacoustics.menu import PipelineMenu

    project = _project(tmp_path, {"01_slow.py": SLOW, "02_use.py": USE})
    menu = PipelineMenu(window, project)
    assert menu.stop_button.instate(["disabled"])
    menu.run_all_button.invoke()
    _pump_until(window, lambda: "started" in _log_text(menu))
    assert menu.run_all_button.instate(["disabled"])
    assert all(button.instate(["disabled"]) for button in menu.row_buttons)
    assert not menu.stop_button.instate(["disabled"])
    menu.stop_button.invoke()
    _pump_until(window, lambda: not menu.busy)
    assert "stopped" in menu.status_line.cget("text")
    assert not menu.run_all_button.instate(["disabled"])
    assert _states(project) == {"01_slow.py": STOPPED, "02_use.py": NOT_RUN}


def test_closing_during_a_run_stops_it_and_records_it(window: Any, tmp_path: Path) -> None:
    from ecoacoustics.menu import PipelineMenu

    project = _project(tmp_path, {"01_slow.py": SLOW})
    menu = PipelineMenu(window, project)
    menu.run_all_button.invoke()
    _pump_until(window, lambda: "started" in _log_text(menu))
    menu.stop_and_close()
    _pump_until(window, lambda: not window.winfo_exists())  # closes once the run is recorded
    assert _states(project) == {"01_slow.py": STOPPED}
