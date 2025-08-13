PYTHON ?= python
PIP ?= pip

.PHONY: setup setup-dev test lint typecheck audit security reproduce smoke figures clean licenses precommit-install

setup:
	$(PIP) install -r requirements.txt

setup-dev: setup
	$(PIP) install -r requirements-dev.txt
	pre-commit install || true

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check .
	black --check .
	isort --check-only .

typecheck:
	mypy qtrust scripts || true

audit:
	bandit -q -r qtrust || true
	pip-audit -r requirements.txt || true
	safety check -r requirements.txt || true

licenses:
	$(PYTHON) -m piplicenses --format=markdown --with-authors --with-description --with-urls > THIRD_PARTY_LICENSES.md || true

reproduce:
	$(PYTHON) run_experiment.py --config configs/minimal.yaml --output artifacts
	$(PYTHON) run_experiment.py --config configs/minimal_rainbow.yaml --output artifacts
	$(PYTHON) -m scripts.eval artifacts results
	$(PYTHON) -m scripts.plot_figures artifacts results/figures
	$(PYTHON) -m scripts.export_tables results results/tables
	$(PYTHON) scripts/check_claims.py --config configs/default.yaml --metrics results/metrics.json || true
	@echo "To generate a locked requirements file with hashes: pip-compile --generate-hashes -o requirements.lock.txt requirements.txt"

smoke:
	$(PYTHON) -m scripts.smoke_suite --base_config configs/minimal.yaml --artifacts artifacts --results results
	$(PYTHON) scripts/check_claims.py --config configs/default.yaml --metrics results/metrics.json || true

figures:
	$(PYTHON) scripts/plot_figures.py artifacts results/figures

paper-assets:
	$(PYTHON) -m scripts.paper_assets --asset fig2 --base_config configs/minimal.yaml --artifacts artifacts --out results/paper_assets
	$(PYTHON) -m scripts.paper_assets --asset table_vi --base_config configs/minimal.yaml --artifacts artifacts --out results/paper_assets
	$(PYTHON) -m scripts.paper_assets --asset table_viii --out results/paper_assets
	$(PYTHON) -m scripts.paper_assets --asset table_ix --base_config configs/minimal.yaml --artifacts artifacts --out results/paper_assets
	$(PYTHON) -m scripts.paper_assets --asset table_x --base_config configs/minimal.yaml --artifacts artifacts --out results/paper_assets

precommit-install:
	pre-commit install

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache dist build


