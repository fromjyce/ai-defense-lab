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

.PHONY: setup data train loop eval fidelity demo web test clean

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

fidelity:
	mkdir -p $(RESULTS_DIR)
	$(PY) -m defend.eval.fidelity --config $(CONFIG) --out $(RESULTS_DIR)/fidelity_report.json

demo:
	@echo "Run 'make web' after 'make loop eval fidelity' for the live dashboard (attacker curve,"
	@echo "detector metrics, live transaction scorer, mandate forgery demo, attack taxonomy)."
	@echo "The mock payment API alone is runnable after 'make train':"
	@echo "  $(PY) -m uvicorn generate.mock_api.app:app --reload"

web:
	$(PY) -m uvicorn web.server:app --reload --port 8000

test:
	$(PY) -m pytest tests/ -v

clean:
	rm -rf $(DATA_DIR)
	rm -rf $(RESULTS_DIR)/*
	touch $(RESULTS_DIR)/.gitkeep
	rm -rf $(VENV)
