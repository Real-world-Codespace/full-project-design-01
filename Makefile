PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
INSTALL_STAMP := $(VENV)/.installed

.PHONY: setup install sample test pipeline verify notebooks api docker-build

setup: $(INSTALL_STAMP)

install: setup

$(INSTALL_STAMP): pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -e '.[aws,api,notebooks,test]'
	touch $(INSTALL_STAMP)

sample: setup
	PYTHONPATH=src $(VENV_PYTHON) -m cooling_load.synthetic --output data/raw/cooling_load.csv

test: setup
	MPLBACKEND=Agg PYTHONPATH=src $(VENV_PYTHON) -m pytest

pipeline: setup
	MPLBACKEND=Agg $(VENV_PYTHON) scripts/run_notebooks.py

verify: test pipeline

notebooks: setup
	$(VENV_PYTHON) -m jupyter lab notebooks

api: setup
	PYTHONPATH=src $(VENV_PYTHON) -m uvicorn cooling_load.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t cooling-load-api:local .
