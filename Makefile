.PHONY: help install lint format typecheck test coverage clean run run-all

PYTHON := python
CONFIG := configs/base.yaml

help:
	@echo "Bitcoin Sentiment Trader Performance — Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install       Install Python dependencies"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Run black + isort"
	@echo "  make typecheck     Run mypy --strict"
	@echo "  make test          Run pytest (unit + integration)"
	@echo "  make coverage      Run pytest with coverage report"
	@echo "  make run           Run a single pipeline stage (set STAGE=...)"
	@echo "  make run-all       Run the full pipeline (Phases 01-09)"
	@echo "  make clean         Remove generated artifacts (outputs, logs, cache)"
	@echo ""
	@echo "Examples:"
	@echo "  make run STAGE=ingestion"
	@echo "  make run-all"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

lint:
	ruff check src/ tests/ pipelines/

format:
	black --line-length 100 src/ tests/ pipelines/
	isort --profile black src/ tests/ pipelines/

typecheck:
	mypy --strict src/sentiment_trader_analytics/

test:
	python -m pytest tests/unit/ tests/integration/ -v --tb=short

coverage:
	python -m pytest tests/ \
		--cov=src/sentiment_trader_analytics/ \
		--cov-fail-under=85 \
		--cov-report=term-missing

run:
	python pipelines/run_$(STAGE)_pipeline.py --config $(CONFIG)

run-all:
	python pipelines/run_full_pipeline.py --config $(CONFIG)

clean:
	rm -rf outputs/figures outputs/tables outputs/reports outputs/presentation_assets
	rm -rf logs/*.log
	rm -rf data/interim data/processed data/features data/generated
	rm -rf experiments/mlruns
	rm -rf .coverage .pytest_cache .mypy_cache .ruff_cache
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Generated artifacts cleaned. Raw data retained."
