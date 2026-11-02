# Generated with Claude Code (Anthropic): run the pipeline stages in order.
# Stage 02 needs a confirmed grouping key in config.yaml. Stage 05 reads the test set
# only when evaluation.evaluate_test is true.

PYTHON ?= python
NB_OUT := results/logs/notebooks
NBCONVERT := $(PYTHON) -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir $(NB_OUT)

.NOTPARALLEL:
.PHONY: all data prepare features unsupervised supervised analysis figures test lint

all: data prepare features analysis figures

data:
	$(PYTHON) code/01_get_data.py

prepare:
	$(PYTHON) code/02_prepare.py

features:
	$(PYTHON) code/03_features.py

unsupervised:
	$(NBCONVERT) code/04_unsupervised.ipynb

supervised:
	$(NBCONVERT) code/05_supervised.ipynb

analysis: unsupervised supervised

figures:
	$(NBCONVERT) code/06_figures.ipynb

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
