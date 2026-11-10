# Generated with Claude Code (Anthropic): find the stages, run them, tell which are out of date.
"""Find the pipeline stages, run them, and tell which ones are out of date.

A stage is any file in ``code/`` whose name starts with a number and an underscore and ends in
``.py`` or ``.ipynb``, such as ``01_get_data.py`` or ``04a_extra.ipynb``. Stages run in the
order of their numbers, so a new stage joins the pipeline as soon as it is saved there. Its
description is the first line of its docstring (scripts) or of its first markdown cell
(notebooks).

Each run's output goes to ``results/logs/runs/<stage>.log``. After a successful run,
``results/logs/runs/history.json`` records fingerprints of the stage's code, of the package
modules it imports and of the ``config.yaml`` sections it reads, and the size and modification
time of every file in the data and results folders, noting which ones the run created or changed
(its outputs).

A stage is out of date when, since its last successful run:

- its code, a package module it imports or a config section it reads changed;
- a file it could have read changed or disappeared (any file in the data and results folders
  except its own outputs and those of later stages), for example new data or a re-run of an
  earlier stage;
- one of its own outputs disappeared.

Run as ``python -m ecoacoustics.pipeline NOTEBOOK OUTPUT_DIR``, the module executes one notebook;
this is how notebook stages are run.
"""

import ast
import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ecoacoustics.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config, resolve_path

PACKAGE_DIR = Path(__file__).resolve().parent

# Stage states shown in the menus.
UP_TO_DATE = "up to date"
OUT_OF_DATE = "out of date"
WAITING = "waiting"
NOT_RUN = "not run"
FAILED = "failed"
STOPPED = "stopped"
OK = "ok"

STAGE_NAME = re.compile(r"^(?P<id>(?P<number>\d+)[a-z]?)_(?P<name>\w+)$", re.IGNORECASE)
STAGE_PREFIX = re.compile(r"^stage\s+\w+\s*[:.-]\s*", re.IGNORECASE)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# Notices that every notebook kernel prints. The kernel only listens on this computer, so they
# are harmless; they are kept in the log file but not shown on screen.
QUIET_LINES = ("Kernel is running over TCP without encryption",)


@dataclass(frozen=True)
class Project:
    """Where the stages, the shared package, the settings, the data and the logs are."""

    root: Path
    code_dir: Path
    package_dir: Path
    config_path: Path
    logs_dir: Path
    watched_dirs: tuple[Path, ...]

    @classmethod
    def default(cls) -> "Project":
        """This repository: every folder in the ``paths`` section of config.yaml is watched."""
        config = load_config()
        logs_dir = resolve_path(config, "logs_dir")
        watched = {resolve_path(config, key) for key in config["paths"]} - {logs_dir}
        return cls(
            root=REPO_ROOT,
            code_dir=REPO_ROOT / "code",
            package_dir=PACKAGE_DIR,
            config_path=DEFAULT_CONFIG_PATH,
            logs_dir=logs_dir,
            watched_dirs=tuple(sorted(watched)),
        )

    @property
    def runs_dir(self) -> Path:
        return self.logs_dir / "runs"

    @property
    def notebooks_dir(self) -> Path:
        """Where the executed copy of each notebook stage is saved, with its outputs."""
        return self.logs_dir / "notebooks"

    @property
    def history_path(self) -> Path:
        return self.runs_dir / "history.json"

    def relative(self, path: Path) -> str:
        """Path relative to the project root, with forward slashes on every system."""
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()


# ---------------------------------------------------------------------------------------------
# Finding the stages


def python_command(path: Path, project: Project) -> list[str]:
    return [sys.executable, "-u", str(path)]


def notebook_command(path: Path, project: Project) -> list[str]:
    output_dir = str(project.notebooks_dir)
    return [sys.executable, "-u", "-m", "ecoacoustics.pipeline", str(path), output_dir]


# How each kind of stage file is run. Adding an entry here makes files of that type stages too,
# for example ".R": lambda path, project: ["Rscript", str(path)].
RUNNERS: dict[str, Callable[[Path, Project], list[str]]] = {
    ".py": python_command,
    ".ipynb": notebook_command,
}

# Commands offered next to the stages.
CHECKS: dict[str, list[str]] = {
    "tests": [sys.executable, "-m", "pytest"],
    "lint": [sys.executable, "-m", "ruff", "check", "."],
}


@dataclass(frozen=True)
class Stage:
    """One numbered file in ``code/``."""

    path: Path
    id: str
    number: int
    name: str

    @property
    def file(self) -> str:
        return self.path.name

    @property
    def kind(self) -> str:
        return "notebook" if self.path.suffix == ".ipynb" else "script"

    @property
    def title(self) -> str:
        return self.name.replace("_", " ").capitalize()

    @property
    def description(self) -> str:
        return describe(self.path)

    def command(self, project: Project) -> list[str]:
        return RUNNERS[self.path.suffix](self.path, project)


def discover_stages(code_dir: Path) -> list[Stage]:
    """Every stage in ``code_dir``, in the order they run: by number, then letter, then name."""
    if not code_dir.is_dir():
        return []
    stages = []
    for path in code_dir.iterdir():
        match = STAGE_NAME.match(path.stem)
        if match and path.is_file() and path.suffix in RUNNERS:
            stages.append(Stage(path, match["id"], int(match["number"]), match["name"]))
    return sorted(stages, key=lambda stage: (stage.number, stage.id.lower(), stage.name))


def select_stages(stages: list[Stage], selectors: Iterable[str]) -> list[Stage]:
    """The stages matching any selector, in pipeline order.

    A selector is a stage number (``5`` or ``05``, which also matches ``05a``, ``05b``), an id
    (``05a``), a name (``supervised``) or a file name (``05_supervised.ipynb``).
    """
    chosen = set()
    for selector in selectors:
        key = selector.lower()
        matches = [
            stage
            for stage in stages
            if key in {stage.id.lower(), stage.name.lower(), stage.path.stem.lower(), stage.file}
            or (key.isdigit() and int(key) == stage.number)
        ]
        if not matches:
            known = ", ".join(f"{stage.id} ({stage.name})" for stage in stages)
            raise ValueError(f"No stage matches {selector!r}. Stages: {known or 'none'}")
        chosen.update(matches)
    return [stage for stage in stages if stage in chosen]


def describe(path: Path) -> str:
    """First line of a script's docstring or of a notebook's first markdown cell."""
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".ipynb":
            cells = json.loads(text).get("cells", [])
            text = next((_cell_text(c) for c in cells if c.get("cell_type") == "markdown"), "")
        else:
            text = ast.get_docstring(ast.parse(text)) or ""
    except (OSError, ValueError, SyntaxError):
        return ""
    lines = [line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()]
    line = STAGE_PREFIX.sub("", lines[0] if lines else "")
    return line[:1].upper() + line[1:]


def _cell_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


# ---------------------------------------------------------------------------------------------
# Fingerprints: what a stage depends on


def hash_text(text: str) -> str:
    """Short hash of a text, ignoring the difference between Windows and Unix line endings."""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()[:16]


def stage_source(path: Path) -> str:
    """The code a stage runs: the whole script, or the code cells of a notebook."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix != ".ipynb":
        return text
    try:
        cells = json.loads(text).get("cells", [])
    except ValueError:
        return text
    return "\n\n".join(_cell_text(cell) for cell in cells if cell.get("cell_type") == "code")


def imported_modules(source: str, package: str) -> set[str]:
    """Names of the package modules that some source code imports."""
    names = set(re.findall(rf"\b{package}\.(\w+)", source))
    names |= set(re.findall(r"^\s*from\s+\.(\w+)\s+import\b", source, re.MULTILINE))
    pattern = rf"^\s*from\s+{package}\s+import\s+\(?([\w\s,]+)\)?"
    for group in re.findall(pattern, source, re.MULTILINE):
        names |= {item.split()[0] for item in group.split(",") if item.strip()}
    return names


def package_sources(source: str, package_dir: Path) -> dict[str, str]:
    """Source of every package module a stage imports, directly or through other modules."""
    package = package_dir.name
    found: dict[str, str] = {}
    pending = [source]
    while pending:
        for name in imported_modules(pending.pop(), package):
            path = package_dir / f"{name}.py"
            if f"{name}.py" not in found and path.is_file():
                found[f"{name}.py"] = path.read_text(encoding="utf-8", errors="replace")
                pending.append(found[f"{name}.py"])
    init = package_dir / "__init__.py"
    if found and init.is_file():
        found["__init__.py"] = init.read_text(encoding="utf-8", errors="replace")
    return dict(sorted(found.items()))


def config_sections(
    source: str, module_sources: Iterable[str], config: dict[str, Any]
) -> list[str]:
    """Top-level sections of config.yaml that a stage reads.

    Looks for ``config["section"]`` and ``config.get("section")`` in the stage and in the package
    modules it imports, and for ``resolve_path(...)``, which reads ``paths``. When the stage hands
    the whole configuration to another function, or loads it but reads no section that can be
    seen, every section counts.
    """
    text = "\n".join([source, *module_sources])
    names = {"config", *re.findall(r"(\w+)\s*=\s*load_config\(", source)}
    keys: set[str] = set()
    for name in names:
        keys |= set(re.findall(rf"""\b{name}\s*(?:\[\s*|\.get\(\s*)["'](\w+)["']""", text))
        callers = re.findall(rf"\b(\w+)\(\s*{name}\s*[,)]", source)
        if any(caller != "resolve_path" for caller in callers):
            return sorted(config)
    if "resolve_path(" in text:
        keys.add("paths")
    if not keys & config.keys() and "load_config(" in source:
        return sorted(config)
    return sorted(keys & config.keys())


def fingerprint(stage: Stage, project: Project, config: dict[str, Any]) -> dict[str, Any]:
    """Hashes of the code, package modules and config sections a stage depends on."""
    source = stage_source(stage.path)
    modules = package_sources(source, project.package_dir)
    sections = config_sections(source, modules.values(), config)
    return {
        "code": hash_text(source),
        "modules": {name: hash_text(text) for name, text in modules.items()},
        "config": {key: hash_text(json.dumps(config[key], sort_keys=True)) for key in sections},
    }


def snapshot(project: Project) -> dict[str, tuple[int, int]]:
    """Size and modification time of every file in the watched folders.

    Hidden files and folders and the logs folder are skipped.
    """
    files = {}
    logs_dir = project.logs_dir.resolve()
    for folder in project.watched_dirs:
        for dirpath, dirnames, filenames in os.walk(folder):
            here = Path(dirpath)
            dirnames[:] = [
                name
                for name in dirnames
                if not name.startswith(".")
                and name != "__pycache__"
                and (here / name).resolve() != logs_dir
            ]
            for name in filenames:
                if name.startswith("."):
                    continue
                try:
                    stat = (here / name).stat()
                except OSError:  # removed while scanning
                    continue
                files[project.relative(here / name)] = (stat.st_size, stat.st_mtime_ns)
    return files


# ---------------------------------------------------------------------------------------------
# Run history and status


def load_history(project: Project) -> dict[str, Any]:
    """The recorded runs, or an empty history when there are none (or the file is damaged)."""
    try:
        history = json.loads(project.history_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        history = {}
    history.setdefault("stages", {})
    return history


def save_history(history: dict[str, Any], project: Project) -> None:
    """Write the history in one step, so a crash never leaves half a file."""
    project.runs_dir.mkdir(parents=True, exist_ok=True)
    temporary = project.history_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(history, indent=1), encoding="utf-8")
    # Windows refuses to replace a file while anything is reading it, even for an instant.
    for _ in range(20):
        with contextlib.suppress(PermissionError):
            temporary.replace(project.history_path)
            return
        time.sleep(0.05)
    temporary.replace(project.history_path)


@dataclass
class StageStatus:
    """Whether a stage needs to run, and why."""

    state: str
    reasons: list[str] = field(default_factory=list)

    @property
    def up_to_date(self) -> bool:
        return self.state == UP_TO_DATE


def _outputs(record: dict[str, Any] | None) -> set[str]:
    return set(((record or {}).get("last_success") or {}).get("outputs", []))


def _preview(paths: list[str], limit: int = 3) -> str:
    shown = ", ".join(paths[:limit])
    return shown + (f" and {len(paths) - limit} more" if len(paths) > limit else "")


def _changed(old: dict[str, str], new: dict[str, str]) -> list[str]:
    return sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))


def _own_status(
    stage: Stage,
    record: dict[str, Any] | None,
    earlier: dict[str, set[str]],
    later_outputs: set[str],
    files: dict[str, tuple[int, int]],
    config: dict[str, Any],
    project: Project,
) -> StageStatus:
    """Status of one stage from its own record, ignoring whether earlier stages must run first."""
    if record is None:
        return StageStatus(NOT_RUN, ["no run recorded yet"])
    if record["status"] != OK:
        return StageStatus(record["status"], [f"last run {record['status']}; see {record['log']}"])

    success = record["last_success"]
    then, now = success["fingerprint"], fingerprint(stage, project, config)
    reasons = []
    if then["code"] != now["code"]:
        reasons.append(f"{stage.file} was edited")
    if modules := _changed(then["modules"], now["modules"]):
        reasons.append(f"package code changed: {', '.join(modules)}")
    if sections := _changed(then["config"], now["config"]):
        reasons.append(f"settings changed in config.yaml: {', '.join(sections)}")

    own = set(success["outputs"])
    changed: dict[str, list[str]] = {}
    for path, stat in success["files"].items():
        if path not in own and path not in later_outputs and files.get(path) != tuple(stat):
            source = next((name for name, outs in earlier.items() if path in outs), None)
            changed.setdefault(source or "", []).append(path)
    for source, paths in sorted(changed.items()):
        where = f"outputs of {source}" if source else "files changed outside the pipeline"
        reasons.append(f"{where} changed: {_preview(paths)}")
    if missing := sorted(own - files.keys()):
        reasons.append(f"outputs missing: {_preview(missing)}")
    return StageStatus(OUT_OF_DATE if reasons else UP_TO_DATE, reasons)


def pipeline_status(project: Project) -> list[tuple[Stage, StageStatus]]:
    """Every stage with its status, in pipeline order.

    A stage that is up to date on its own but reads the outputs of an earlier stage that must
    run first is marked as waiting for it.
    """
    stages = discover_stages(project.code_dir)
    records = load_history(project)["stages"]
    files = snapshot(project)
    config = load_config(project.config_path)
    outputs = {stage.file: _outputs(records.get(stage.file)) for stage in stages}

    results: list[tuple[Stage, StageStatus]] = []
    for index, stage in enumerate(stages):
        record = records.get(stage.file)
        earlier = {s.file: outputs[s.file] for s in stages[:index]}
        later = set().union(*(outputs[s.file] for s in stages[index + 1 :]))
        status = _own_status(stage, record, earlier, later, files, config, project)
        if status.up_to_date:
            inputs = set(record["last_success"]["files"]) - outputs[stage.file] - later
            blockers = [
                s.file for s, st in results if not st.up_to_date and outputs[s.file] & inputs
            ]
            if blockers:
                reason = f"reads outputs of {', '.join(blockers)}, which must run first"
                status = StageStatus(WAITING, [reason])
        results.append((stage, status))
    return results


def format_seconds(seconds: float) -> str:
    """Duration in words: '45 s', '2 min 3 s', '1 h 5 min'."""
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"


def last_run_text(record: dict[str, Any] | None) -> str:
    """When a stage last ran and how long it took, for the menus."""
    if not record:
        return "never"
    started = datetime.fromisoformat(record["started"]).strftime("%d %b %H:%M")
    return f"{started}, {format_seconds(record['seconds'])}"


# ---------------------------------------------------------------------------------------------
# Running


class Runner:
    """Runs stages one at a time, passing on their output line by line, and records each run.

    ``on_output`` receives every line the stage prints; ``on_stage_done`` is called after each
    stage. ``stop()`` may be called from another thread.
    """

    def __init__(
        self,
        project: Project | None = None,
        on_output: Callable[[str], None] = print,
        on_stage_done: Callable[[], None] | None = None,
    ) -> None:
        self.project = project or Project.default()
        self.on_output = on_output
        self.on_stage_done = on_stage_done
        self.current: str | None = None
        self.current_log: Path | None = None
        self._process: subprocess.Popen[str] | None = None
        self._stop = threading.Event()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        """Stop the running stage, and do not start any more."""
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            _kill_tree(process)

    def run(self, stages: list[Stage]) -> bool:
        """Run stages in order, stopping at the first that fails. True if every one succeeded."""
        self._stop.clear()
        for stage in stages:
            if self._stop.is_set():
                return False
            if self._run_stage(stage) != OK:
                return False
        return True

    def run_all(self) -> bool:
        return self.run(discover_stages(self.project.code_dir))

    def run_out_of_date(self) -> bool:
        """Run, in order, every stage that is not up to date.

        Each stage's status is checked just before its turn, so a stage whose inputs were just
        rewritten by an earlier stage runs too.
        """
        self._stop.clear()
        ran = 0
        for stage in discover_stages(self.project.code_dir):
            if self._stop.is_set():
                return False
            status = dict((s.file, st) for s, st in pipeline_status(self.project)).get(stage.file)
            if status is None or status.up_to_date:
                continue
            ran += 1
            if self._run_stage(stage) != OK:
                return False
        self.on_output("Every stage is up to date." if ran == 0 else f"Ran {ran} stage(s).")
        return True

    def run_check(self, name: str) -> bool:
        """Run one of ``CHECKS`` (the tests or the linter)."""
        self._stop.clear()
        self.current = name
        self.on_output(f"=== {name}: {' '.join(CHECKS[name][1:])} ===")
        try:
            code = self._execute(CHECKS[name], self.project.runs_dir / f"{name}.log")
        finally:
            self.current = None
        self.on_output(f"=== {name}: {'passed' if code == 0 else 'failed'} ===")
        return code == 0

    def _run_stage(self, stage: Stage) -> str:
        project = self.project
        prints = fingerprint(stage, project, load_config(project.config_path))
        before = snapshot(project)
        log_path = project.runs_dir / f"{stage.path.stem}.log"
        self.current = stage.file
        self.on_output(f"=== {stage.file}: {stage.description or stage.title} ===")
        started, start = datetime.now(), time.perf_counter()
        try:
            code = self._execute(stage.command(project), log_path)
        finally:
            self.current = None
        seconds = time.perf_counter() - start
        state = STOPPED if self._stop.is_set() else OK if code == 0 else FAILED

        history = load_history(project)
        previous = history["stages"].get(stage.file, {})
        record: dict[str, Any] = {
            "status": state,
            "started": started.isoformat(timespec="seconds"),
            "seconds": round(seconds, 1),
            "exit_code": code,
            "log": project.relative(log_path),
        }
        if state == OK:
            after = snapshot(project)
            changed = {path for path, stat in after.items() if before.get(path) != stat}
            outputs = (changed | _outputs(previous)) & after.keys()
            record["last_success"] = {
                "fingerprint": prints,
                "files": after,
                "outputs": sorted(outputs),
            }
        elif previous.get("last_success"):
            record["last_success"] = previous["last_success"]
        history["stages"][stage.file] = record
        save_history(history, project)

        self.on_output(f"=== {stage.file}: {state} after {format_seconds(seconds)} ===")
        if state == FAILED:
            self.on_output(f"Full log: {record['log']}")
        if self.on_stage_done is not None:
            self.on_stage_done()
        return state

    def _execute(self, command: list[str], log_path: Path) -> int:
        """Run a command from the project root, passing on and logging its output."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_log = log_path
        # On Linux and macOS the command gets its own process group, so Stop ends its children.
        extra = {"start_new_session": True} if os.name == "posix" else {}
        process_output = subprocess.Popen(
            command,
            cwd=self.project.root,
            env=self._environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **extra,
        )
        # Leaving this block closes the log and the pipe from the process.
        with log_path.open("w", encoding="utf-8") as log, process_output as process:
            self._process = process
            if self._stop.is_set():  # Stop was pressed just as the stage was starting
                _kill_tree(process)
            try:
                for raw_line in process.stdout:  # type: ignore[union-attr]
                    line = ANSI_ESCAPE.sub("", raw_line.rstrip("\n"))
                    log.write(line + "\n")
                    log.flush()
                    if not any(quiet in line for quiet in QUIET_LINES):
                        self.on_output(line)
                return process.wait()
            except KeyboardInterrupt:  # Ctrl+C in a terminal
                self.stop()
                return process.wait()
            finally:
                self._process = None

    def _environment(self) -> dict[str, str]:
        """The current environment, with the package importable even before ``pip install -e``."""
        env = os.environ.copy()
        paths = [str(PACKAGE_DIR.parent), str(self.project.package_dir.parent)]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        return env


def _kill_tree(process: subprocess.Popen[str]) -> None:
    """End a process and every process it started (for a notebook, its kernel)."""
    if os.name == "nt":
        command = ["taskkill", "/F", "/T", "/PID", str(process.pid)]
        subprocess.run(command, capture_output=True, check=False)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)


# ---------------------------------------------------------------------------------------------
# Executing a notebook stage


def execute_notebook(path: Path, output_dir: Path) -> bool:
    """Run a notebook top to bottom in a fresh kernel, printing its output as it comes.

    Equivalent to ``jupyter nbconvert --to notebook --execute``. The executed copy, with all
    outputs, is saved in ``output_dir`` even when a cell fails. True if every cell succeeded.
    """
    import warnings

    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    # pyzmq on Windows warns about its event loop; the warning is harmless.
    warnings.filterwarnings("ignore", message=".*Proactor event loop.*")

    class PrintingClient(NotebookClient):
        """Prints each cell's output while it runs, as a terminal would."""

        def output(self, outs, msg, display_id, cell_index):  # type: ignore[no-untyped-def]
            content = msg["content"]
            if msg["msg_type"] == "stream":
                print(content["text"], end="", flush=True)
            elif msg["msg_type"] in {"execute_result", "display_data"}:
                data = content.get("data", {})
                text = data.get("text/plain", "")
                # Tables print as text; figures ("<Figure ...>") and widgets are skipped.
                widget = "application/vnd.jupyter.widget-view+json" in data
                if text and not text.startswith("<") and not widget:
                    print(text, flush=True)
            return super().output(outs, msg, display_id, cell_index)

    notebook = nbformat.read(path, as_version=4)
    code_cells = [i for i, cell in enumerate(notebook.cells) if _runs(cell)]

    def announce(cell: Any, cell_index: int) -> None:
        lines = [line for line in cell.source.splitlines() if line.strip()]
        first = next((line for line in lines if not line.lstrip().startswith("#")), lines[0])
        number = code_cells.index(cell_index) + 1
        print(f"--- cell {number} of {len(code_cells)}: {first.strip()[:70]}", flush=True)

    client = PrintingClient(
        notebook,
        timeout=None,
        on_cell_execute=announce,  # called only for the code cells that run
        resources={"metadata": {"path": str(path.parent)}},
    )
    try:
        client.execute()
        return True
    except CellExecutionError as error:
        print(error, flush=True)
        return False
    finally:
        output_dir.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output_dir / path.name)
        print(f"Executed copy saved to {output_dir / path.name}", flush=True)


def _runs(cell: Any) -> bool:
    return cell.cell_type == "code" and bool(cell.source.strip())


def main(argv: list[str] | None = None) -> int:
    notebook, output_dir = (Path(arg) for arg in (sys.argv[1:] if argv is None else argv))
    return 0 if execute_notebook(notebook, output_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
