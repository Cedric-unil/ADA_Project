# Generated with Claude Code (Anthropic): run the pipeline stages in order.
# Every stage target goes through run.py, which finds the numbered stages in code/ by itself, so
# `make all` includes any stage added there. Stage targets name stages by name, not number, so
# they survive renumbering. Stage 02 needs a confirmed grouping key in config.yaml. Stage 05
# reads the test set only when evaluation.evaluate_test is true.

PYTHON ?= python
RUN := $(PYTHON) run.py

.NOTPARALLEL:
.PHONY: all outdated status menu data prepare features unsupervised supervised same_period lstm_pilot analysis figures test lint

all:
	$(RUN) all

outdated:
	$(RUN) outdated

status:
	$(RUN) list

menu:
	$(RUN)

data:
	$(RUN) get_data

prepare:
	$(RUN) prepare

features:
	$(RUN) features

unsupervised:
	$(RUN) unsupervised

supervised:
	$(RUN) supervised

same_period:
	$(RUN) same_period

lstm_pilot:
	$(RUN) lstm_pilot

analysis:
	$(RUN) unsupervised supervised same_period lstm_pilot

figures:
	$(RUN) figures

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
