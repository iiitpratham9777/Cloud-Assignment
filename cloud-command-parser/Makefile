SHELL := /bin/bash
SUBSET ?= 0
VERSION ?= v1
PY ?= python3

.PHONY: up down train train-all eval lint test clean-data package report setup check serve smoke audit

# --- local mode (no Docker / Kubernetes). Same commands work on Windows via `python scripts/dev.py ...`
setup:         ## create .venv and install dependencies
	$(PY) scripts/dev.py setup

check:         ## verify dependencies, datasets and model weights
	$(PY) scripts/dev.py check

serve:         ## run the API locally on http://127.0.0.1:8000/docs
	$(PY) scripts/dev.py serve

smoke:         ## tiny end-to-end training check on synthetic data
	$(PY) scripts/dev.py smoke

audit:         ## scan for secrets / personal paths before publishing
	$(PY) scripts/dev.py audit

# --- optional Kubernetes workflow (needs docker, k3d, kubectl, bash)

up:            ## build and deploy everything from a fresh clone
	bash scripts/up.sh

down:          ## delete the cluster
	k3d cluster delete cmdparser

train:         ## fine-tune Pythia-70M (reads training/config.yaml); SUBSET=50 for a smoke test
	$(PY) -m training.train --version $(VERSION) $(if $(filter-out 0,$(SUBSET)),--subset $(SUBSET),)

train-all:
	$(MAKE) train VERSION=v1
	$(MAKE) train VERSION=v2

clean-data:    ## offline cleaning of downloaded shards: make clean-data SHARDS=results/teacher
	$(PY) -m worker.clean --local $(SHARDS) --out data

lint:
	ruff check .

test:
	$(PY) -m pytest -q

report:        ## rebuild report/report.pdf from report/report.md
	bash scripts/build_report.sh
