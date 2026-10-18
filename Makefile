PYTHON ?= .venv/bin/python
SYSTEM_PYTHON ?= python3.12

.PHONY: install run-all ingest preprocess train-delay train-demand evaluate run-api run-app test lint
install:
	$(SYSTEM_PYTHON) -m venv .venv
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install --no-deps -e .
run-all:
	$(PYTHON) -m src.cli run-all
ingest:
	$(PYTHON) -m src.cli ingest
preprocess:
	$(PYTHON) -m src.cli preprocess
train-delay:
	$(PYTHON) -m src.cli train-delay
train-demand:
	$(PYTHON) -m src.cli train-demand
evaluate:
	$(PYTHON) -m src.cli evaluate
run-api:
	$(PYTHON) -m uvicorn src.api:app --host 0.0.0.0 --port 8000
run-app:
	$(PYTHON) -m streamlit run src/dashboard.py --server.address 0.0.0.0 --server.port 8501
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check .
