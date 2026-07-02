# Bitcoin Sentiment Trader Performance — Documentation

## Project Overview

This repository is a **production-grade quantitative research system** that rigorously investigates whether, and how, Bitcoin market sentiment — as measured by the Fear & Greed Index — influences the trading behavior and profitability of accounts trading on Hyperliquid.

All outputs (statistical findings, business insights, and optionally predictive models) are defensible under scrutiny from a trading desk, a risk committee, or a panel of quantitative researchers.

---

## Quick Navigation

| Document | Description |
|---|---|
| [Architecture](architecture.md) | System design, module layout, and pipeline architecture |
| [Methodology](methodology.md) | Analytical methodology, decisions, and assumptions |
| [Data Dictionary](data_dictionary.md) | Schema definitions, feature catalog, and lineage |
| [Phased Execution Plan](phased_execution_plan.md) | Master execution roadmap with phases, agents, and gates |
| [API Reference](api_reference/) | Auto-generated code documentation |

---

## Phase Documentation

| Phase | Document |
|---|---|
| Phase 01: Data Ingestion | [phases/phase_01_data_ingestion.md](phases/phase_01_data_ingestion.md) |
| Phase 02: Data Validation | [phases/phase_02_data_validation.md](phases/phase_02_data_validation.md) |
| Phase 03: Preprocessing & Cleaning | [phases/phase_03_preprocessing.md](phases/phase_03_preprocessing.md) |
| Phase 04: Feature Engineering | [phases/phase_04_feature_engineering.md](phases/phase_04_feature_engineering.md) |
| Phase 05: Exploratory Data Analysis | [phases/phase_05_eda.md](phases/phase_05_eda.md) |
| Phase 06: Statistical Analysis | [phases/phase_06_statistical_analysis.md](phases/phase_06_statistical_analysis.md) |
| Phase 07: Business Insight Synthesis | [phases/phase_07_business_insights.md](phases/phase_07_business_insights.md) |
| Phase 08: Machine Learning (Optional) | [phases/phase_08_machine_learning.md](phases/phase_08_machine_learning.md) |
| Phase 09: Visualization & Reporting | [phases/phase_09_reporting.md](phases/phase_09_reporting.md) |
| Phase 10: Documentation & Architecture | [phases/phase_10_documentation.md](phases/phase_10_documentation.md) |
| Phase 11: Testing, CI & Release | [phases/phase_11_testing_release.md](phases/phase_11_testing_release.md) |

---

## Governance

All development — by humans and AI agents — follows the [Engineering Standards (CLAUDE.md)](../.claude/CLAUDE.md). This binding document governs all coding conventions, pipeline rules, documentation standards, and analytical methodology.

---

## Quickstart

```bash
# 1. Set up environment
conda env create -f environment/conda.yaml
conda activate bst

# 2. Configure environment variables
cp .env.example .env
# Edit .env with required credentials/paths

# 3. Run the full pipeline
python pipelines/run_full_pipeline.py --config configs/base.yaml

# 4. Run tests
pytest tests/ --cov=src/ --cov-report=term-missing
```

---

## Business Objective

This system answers the following with statistical rigor:

- Does aggregate trader profitability shift systematically with sentiment regime (Fear vs. Greed)?
- Does Fear correlate with elevated losses, wider drawdowns, or panic-driven behavior?
- Does Greed correlate with elevated leverage, larger position sizing, or overtrading?
- Which trader-level characteristics are associated with consistent profitability across regimes?
- What concrete, risk-aware recommendations follow for a trading desk or product team?

---


