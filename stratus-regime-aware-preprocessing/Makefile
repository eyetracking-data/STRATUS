.PHONY: install test validate-data run

install:
	python -m pip install -e ".[notebook,dev]"

test:
	pytest

validate-data:
	python scripts/validate_data.py

run: validate-data
	python scripts/run_experiment.py
