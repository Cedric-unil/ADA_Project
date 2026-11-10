# Generated with Claude Code (Anthropic): a desktop window of buttons that runs the pipeline.
"""A window of buttons that runs the pipeline, opened from a terminal by ``python run.py``.

The stages are found again at every refresh, so a stage saved in ``code/`` appears after pressing
Refresh, and after every run. Stages run in a background thread, so the window and its Stop button
keep working meanwhile. Tkinter lets only the main thread touch the window, so the background
thread hands its output over through a queue, which the main thread reads several times a second.
"""

import contextlib
import queue
import signal
import sys
import threading
import time
import tkinter as tk
from collections.abc import Callable
from tkinter import font as tkfont
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from ecoacoustics.pipeline import (
    FAILED,
    NOT_RUN,
    OUT_OF_DATE,
    STOPPED,
    UP_TO_DATE,
    WAITING,
    Project,
    Runner,
    Stage,
    StageStatus,
    discover_stages,
    format_seconds,
    last_run_text,
    load_history,
    pipeline_status,
)

POLL_MS = 100
MAX_LOG_LINES = 5000
INTRO = (
    "Stages are the numbered files in code/ (NN_name.py or NN_name.ipynb), run in number order. "
    "Save a new one there and press Refresh."
)
MUTED = "#5f5f5f"
# Symbol and colour of each state.
STATE_STYLE = {
    UP_TO_DATE: ("✔", "#1a7f45"),
    OUT_OF_DATE: ("⚠", "#a86400"),
    WAITING: ("⏳", "#a86400"),
    NOT_RUN: ("○", MUTED),
    FAILED: ("✖", "#c62828"),
    STOPPED: ("■", MUTED),
}
RUNNING_STYLE = ("▶", "#1d5fd1")
OUTCOME_STYLE = {
    "finished": STATE_STYLE[UP_TO_DATE],
    "stopped": STATE_STYLE[STOPPED],
    "failed": STATE_STYLE[FAILED],
}


class PipelineMenu:
    """The buttons, the table of stages and the live output, in one window."""

    def __init__(self, root: tk.Tk, project: Project | None = None) -> None:
        self.root = root
        self.project = project or Project.default()
        self.lines: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.runner = Runner(self.project, self.lines.put, self._request_refresh)
        self.worker: threading.Thread | None = None
        self.ticked: dict[str, tk.BooleanVar] = {}
        self.row_buttons: list[ttk.Button] = []
        self._job = ""
        self._job_start = 0.0
        self._outcome = "failed"
        self._shown_stage: str | None = None
        self._refresh_due = False
        self._close_when_done = False
        self._build_window()
        self.refresh()
        self._set_buttons()
        self._poll()

    @property
    def busy(self) -> bool:
        return self.worker is not None

    # -- building the window -----------------------------------------------------------------------

    def _build_window(self) -> None:
        root = self.root
        root.title("Pipeline menu")
        root.protocol("WM_DELETE_WINDOW", self.request_close)
        width = min(root.winfo_fpixels("11.5i"), 0.9 * root.winfo_screenwidth())
        height = min(root.winfo_fpixels("8i"), 0.85 * root.winfo_screenheight())
        root.geometry(f"{int(width)}x{int(height)}")
        self.bold = tkfont.nametofont("TkDefaultFont").copy()
        self.bold.configure(weight="bold")

        ttk.Label(root, text=INTRO, foreground=MUTED, padding=(10, 8, 10, 0)).pack(fill="x")
        self._build_toolbar()
        panes = ttk.PanedWindow(root, orient="vertical")
        panes.pack(fill="both", expand=True, padx=10)
        panes.add(self._build_table(panes), weight=3)
        panes.add(self._build_log(panes), weight=2)
        self.status_line = ttk.Label(root, padding=(10, 6, 10, 0))
        self.status_line.pack(fill="x")
        self.log_line = ttk.Label(root, foreground=MUTED, padding=(10, 0, 10, 8))
        self.log_line.pack(fill="x")

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(fill="x")

        def button(label: str, command: Callable[[], None]) -> ttk.Button:
            widget = ttk.Button(toolbar, text=label, command=command)
            widget.pack(side="left", padx=(0, 6))
            return widget

        self.run_all_button = button(
            "▶ Run all", lambda: self._start("All stages", self.runner.run_all)
        )
        self.run_outdated_button = button(
            "▶ Run out-of-date",
            lambda: self._start("Out-of-date stages", self.runner.run_out_of_date),
        )
        self.run_selected_button = button("▶ Run selected", self._run_selected)
        self.stop_button = button("■ Stop", self._stop)
        self.refresh_button = button("⟳ Refresh", self.refresh)
        self.tests_button = button("Run tests", lambda: self._start_check("tests"))
        self.lint_button = button("Lint", lambda: self._start_check("lint"))

    def _build_table(self, parent: tk.Misc) -> ttk.Frame:
        """A scrollable table, so that any number of stages fits."""
        frame = ttk.Frame(parent)
        background = ttk.Style().lookup("TFrame", "background")
        self.canvas = tk.Canvas(frame, highlightthickness=0, background=background, height=1)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.table = ttk.Frame(self.canvas)
        window = self.canvas.create_window((0, 0), window=self.table, anchor="nw")
        self.table.bind(
            "<Configure>",
            lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(window, width=e.width))
        self.root.bind_all("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return frame

    def _build_log(self, parent: tk.Misc) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=(0, 6, 0, 0))
        self.log = ScrolledText(frame, height=8, wrap="word", state="disabled", font="TkFixedFont")
        heading = tkfont.nametofont("TkFixedFont").copy()
        heading.configure(weight="bold")
        self.log.tag_configure("heading", font=heading)
        self._log_heading_font = heading  # Tk forgets a font that Python no longer refers to
        self.log.pack(fill="both", expand=True)
        return frame

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        over_table = str(event.widget).startswith(str(self.canvas))
        if over_table and self.canvas.yview() != (0.0, 1.0):
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    # -- the table of stages -----------------------------------------------------------------------

    def refresh(self) -> None:
        """Look for stages again and redraw the table with their current status."""
        for child in self.table.winfo_children():
            child.destroy()
        self.row_buttons = []
        try:
            rows = pipeline_status(self.project)
            records = load_history(self.project)["stages"]
        except Exception as error:  # a broken config.yaml, for example
            message = f"Cannot read the pipeline: {error}"
            ttk.Label(self.table, text=message, wraplength="8i").grid(sticky="w")
            return
        self.ticked = {
            stage.file: self.ticked.get(stage.file) or tk.BooleanVar(self.root, False)
            for stage, _ in rows
        }
        self._draw_header()
        for index, (stage, status) in enumerate(rows, start=1):
            self._draw_row(2 * index, stage, status, records.get(stage.file))
        if not rows:
            ttk.Label(self.table, text="No stages found in code/.").grid(columnspan=6, sticky="w")
        self.table.columnconfigure(3, weight=3)
        self.table.columnconfigure(4, weight=2)

    def _draw_header(self) -> None:
        select_all = tk.BooleanVar(self.root, False)
        self._select_all = select_all  # Tk forgets a variable that Python no longer refers to

        def tick_all() -> None:
            for variable in self.ticked.values():
                variable.set(select_all.get())

        ttk.Checkbutton(self.table, variable=select_all, command=tick_all).grid(row=0, column=0)
        for column, text in enumerate(("Stage", "What it does", "Status", "Last run"), start=2):
            label = ttk.Label(self.table, text=text, font=self.bold)
            label.grid(row=0, column=column, sticky="w", padx=6, pady=(0, 4))

    def _draw_row(self, row: int, stage: Stage, status: StageStatus, record: dict | None) -> None:
        table = self.table
        ttk.Separator(table).grid(row=row - 1, column=0, columnspan=6, sticky="ew")
        ttk.Checkbutton(table, variable=self.ticked[stage.file]).grid(row=row, column=0)
        run = ttk.Button(
            table,
            text="▶",
            width=3,
            state="disabled" if self.busy else "normal",
            command=lambda s=stage: self._start(s.file, lambda: self.runner.run([s])),
        )
        run.grid(row=row, column=1, padx=2, pady=4)
        self.row_buttons.append(run)
        ttk.Label(table, text=stage.id, font=self.bold).grid(row=row, column=2, padx=6)

        what = ttk.Frame(table)
        what.grid(row=row, column=3, sticky="ew", padx=6, pady=4)
        title = f"{stage.title}  ({stage.kind}, {stage.file})"
        ttk.Label(what, text=title, font=self.bold).pack(anchor="w")
        description = stage.description or stage.title
        ttk.Label(what, text=description, foreground=MUTED, wraplength="4.5i").pack(anchor="w")

        self._draw_status(row, status, running=self.runner.current == stage.file)
        last_run = ttk.Label(table, text=last_run_text(record), foreground=MUTED)
        last_run.grid(row=row, column=5, sticky="w", padx=6)

    def _draw_status(self, row: int, status: StageStatus, running: bool) -> None:
        symbol, colour = RUNNING_STYLE if running else STATE_STYLE.get(status.state, ("?", MUTED))
        state = "running" if running else status.state
        cell = ttk.Frame(self.table)
        cell.grid(row=row, column=4, sticky="ew", padx=6, pady=4)
        ttk.Label(cell, text=f"{symbol} {state}", foreground=colour, font=self.bold).pack(
            anchor="w"
        )
        if status.reasons and not status.up_to_date and not running:
            reasons = "\n".join(status.reasons)
            ttk.Label(cell, text=reasons, foreground=MUTED, wraplength="3i").pack(anchor="w")

    def _set_buttons(self) -> None:
        for button in (
            self.run_all_button,
            self.run_outdated_button,
            self.run_selected_button,
            self.refresh_button,
            self.tests_button,
            self.lint_button,
            *self.row_buttons,
        ):
            button.configure(state="disabled" if self.busy else "normal")
        self.stop_button.configure(state="normal" if self.busy else "disabled")

    # -- running -----------------------------------------------------------------------------------

    def _run_selected(self) -> None:
        ticked = {name for name, variable in self.ticked.items() if variable.get()}
        stages = [s for s in discover_stages(self.project.code_dir) if s.file in ticked]
        if not stages:
            self.status_line.configure(text="Tick at least one stage first.", foreground=MUTED)
            return
        self._start(f"{len(stages)} selected stage(s)", lambda: self.runner.run(stages))

    def _start_check(self, name: str) -> None:
        self._start(name.capitalize(), lambda: self.runner.run_check(name))

    def _start(self, label: str, job: Callable[[], bool]) -> None:
        """Run a job in a background thread, so the window and its Stop button stay usable."""
        if self.busy:
            return
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._job, self._job_start, self._shown_stage = label, time.perf_counter(), None
        self.worker = threading.Thread(target=self._work, args=(job,), daemon=True)
        self.worker.start()
        self._set_buttons()

    def _work(self, job: Callable[[], bool]) -> None:
        """Runs in the background thread, so it must not touch the window."""
        try:
            ok = job()
            self._outcome = (
                "finished" if ok else "stopped" if self.runner.stop_requested else "failed"
            )
        except Exception as error:
            self.lines.put(f"Error: {error!r}")
            self._outcome = "failed"

    def _request_refresh(self) -> None:
        """Called by the background thread after each stage; the main thread does the redraw."""
        self._refresh_due = True

    def _poll(self) -> None:
        """Several times a second, in the main thread: bring the window up to date."""
        self._show_new_lines()
        if self.worker is not None:
            current = self.runner.current
            if self._refresh_due or (current and current != self._shown_stage):
                self._refresh_due = False
                self._shown_stage = current
                self.refresh()
            if self.worker.is_alive():
                self._show_progress()
            else:
                self._finish()
                if self._close_when_done:
                    self.root.destroy()
                    return
        self.root.after(POLL_MS, self._poll)

    def _show_progress(self) -> None:
        symbol, colour = RUNNING_STYLE
        elapsed = format_seconds(time.perf_counter() - self._job_start)
        text = f"{symbol} Running {self.runner.current or self._job}, {elapsed} so far."
        self.status_line.configure(text=text, foreground=colour)
        self._show_log_path()

    def _finish(self) -> None:
        self._show_new_lines()
        self.worker = None
        self.refresh()
        self._set_buttons()
        symbol, colour = OUTCOME_STYLE[self._outcome]
        elapsed = format_seconds(time.perf_counter() - self._job_start)
        text = f"{symbol} {self._job} {self._outcome} after {elapsed}."
        self.status_line.configure(text=text, foreground=colour)
        self._show_log_path()

    def _show_log_path(self) -> None:
        log = self.runner.current_log
        if log is not None:
            self.log_line.configure(text=f"Full log: {self.project.relative(log)}")

    def _show_new_lines(self) -> None:
        lines = []
        with contextlib.suppress(queue.Empty):
            while True:
                lines.append(self.lines.get_nowait())
        if not lines:
            return
        at_bottom = self.log.yview()[1] >= 0.999
        self.log.configure(state="normal")
        for line in lines:
            self.log.insert("end", line + "\n", "heading" if line.startswith("===") else ())
        excess = int(self.log.index("end-1c").split(".")[0]) - MAX_LOG_LINES
        if excess > 0:
            self.log.delete("1.0", f"{excess + 1}.0")
        self.log.configure(state="disabled")
        if at_bottom:
            self.log.see("end")

    # -- stopping and closing ----------------------------------------------------------------------

    def _stop(self) -> None:
        self.stop_button.configure(state="disabled")
        self.status_line.configure(text="Stopping...", foreground=MUTED)
        self.runner.stop()

    def request_close(self) -> None:
        """The window's X: if a stage is running, ask first, then stop it before closing."""
        if not self.busy:
            self.root.destroy()
        elif messagebox.askyesno(
            "A stage is running", "Stop it and close the window?", parent=self.root
        ):
            self.stop_and_close()

    def stop_and_close(self) -> None:
        """Stop what is running, wait for its run to be recorded, then close the window."""
        self._close_when_done = True
        self.runner.stop()


def _draw_sharp_text() -> None:
    """On Windows, draw at the screen's real resolution rather than blurred and scaled up."""
    if sys.platform == "win32":
        import ctypes

        with contextlib.suppress(AttributeError, OSError):
            ctypes.windll.shcore.SetProcessDpiAwareness(1)


def open_menu(project: Project | None = None) -> None:
    """Open the menu window and wait until it is closed."""
    _draw_sharp_text()
    root = tk.Tk()
    menu = PipelineMenu(root, project)

    def on_ctrl_c(*_: object) -> None:
        if menu.busy:
            menu.stop_and_close()
        else:
            root.destroy()

    signal.signal(signal.SIGINT, on_ctrl_c)
    # Open in front of the editor, without staying on top of everything afterwards.
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))
    root.mainloop()
