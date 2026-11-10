# Generated with Claude Code (Anthropic): open the pipeline menu, or run stages, from a terminal.
"""Run the pipeline from a terminal.

    python run.py              open the menu: a window with buttons
    python run.py list         every stage and whether it is up to date
    python run.py all          every stage, in order
    python run.py outdated     only the stages that are not up to date
    python run.py 03 05        stages 03 and 05; names work too: python run.py features
    python run.py tests        the tests (pytest); "lint" runs ruff

Stages are the numbered files in code/ (NN_name.py or NN_name.ipynb), found anew each time, so a
new stage needs no change here. Ctrl+C stops the running stage.
"""

import sys
from pathlib import Path

# Lets this run before `pip install -e .` has been done.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ecoacoustics.pipeline import (
    CHECKS,
    Project,
    Runner,
    discover_stages,
    last_run_text,
    load_history,
    pipeline_status,
    select_stages,
)


def print_status(project: Project) -> None:
    """One line per stage: id, name, status and last run, with the reasons it is out of date."""
    rows = pipeline_status(project)
    records = load_history(project)["stages"]
    print()
    if not rows:
        print(f"  No stages found in {project.relative(project.code_dir)}/")
    for stage, status in rows:
        last = last_run_text(records.get(stage.file))
        print(f"  {stage.id:>4}  {stage.title:<20} {status.state:<12} last run: {last}")
        for reason in [] if status.up_to_date else status.reasons:
            print(f"{'':8}- {reason}")


def run_targets(runner: Runner, targets: list[str]) -> bool:
    """Run 'all', 'outdated', a check such as 'tests', or stages chosen by number or name."""
    words = [target.lower() for target in targets]
    if words == ["all"]:
        return runner.run_all()
    if words in (["outdated"], ["stale"]):
        return runner.run_out_of_date()
    if len(words) == 1 and words[0] in CHECKS:
        return runner.run_check(words[0])
    return runner.run(select_stages(discover_stages(runner.project.code_dir), targets))


def main(argv: list[str] | None = None) -> int:
    targets = sys.argv[1:] if argv is None else argv
    if any(target in {"-h", "--help", "help"} for target in targets):
        print(__doc__)
        return 0
    if not targets:
        # Imported here so that the commands below also work on a computer without a screen.
        from ecoacoustics.menu import open_menu

        print("Menu window opened. Close it to get the terminal back.")
        open_menu()
        return 0
    runner = Runner(Project.default())
    if targets == ["list"]:
        print_status(runner.project)
        return 0
    try:
        return 0 if run_targets(runner, targets) else 1
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
