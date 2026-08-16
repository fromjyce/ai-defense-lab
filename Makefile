PYTHON311 := python3.11
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

CONFIG := config/default.yaml
DATA_DIR := data
RESULTS_DIR := results

TRANSACTIONS := $(DATA_DIR)/transactions.csv
HOLDOUT := $(DATA_DIR)/holdout.csv
MODEL := $(DATA_DIR)/models/detector.joblib

.PHONY: setup data train loop eval demo test clean

setup:
	$(PYTHON311) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	mkdir -p $(DATA_DIR)
	$(PY) -m generate.synth.generator --source synthetic --config $(CONFIG) --out $(TRANSACTIONS)

train: data
	mkdir -p $(DATA_DIR)/models
	$(PY) -m defend.transaction --data $(TRANSACTIONS) --config $(CONFIG) \
		--model-out $(MODEL) --holdout-out $(HOLDOUT)

loop: train
	mkdir -p $(RESULTS_DIR)
	$(PY) -m loop.orchestrator --data $(TRANSACTIONS) --holdout $(HOLDOUT) \
		--model $(MODEL) --config $(CONFIG) --results-dir $(RESULTS_DIR)

eval: train
	mkdir -p $(RESULTS_DIR)
	$(PY) -m defend.eval.metrics --holdout $(HOLDOUT) --model $(MODEL) \
		--config $(CONFIG) --out $(RESULTS_DIR)/metrics.json

demo:
	@echo "No web UI yet (web/ is a placeholder this pass)."
	@echo "Run 'make loop' and inspect results/evasion_curve.json for the headline chart data."

test:
	$(PY) -m pytest tests/ -v

clean:
	rm -rf $(DATA_DIR)
	rm -rf $(RESULTS_DIR)/*
	touch $(RESULTS_DIR)/.gitkeep
	rm -rf $(VENV)
