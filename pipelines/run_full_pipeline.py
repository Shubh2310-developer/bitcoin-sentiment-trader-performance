#!/usr/bin/env python3
# ruff: noqa: E402
"""Full pipeline orchestrator.

Chains all pipeline stages from ingestion through reporting in a single
sequential execution. Each stage is called as a subprocess to ensure
independent process isolation. A stage failure halts the pipeline.

Usage:
    python pipelines/run_full_pipeline.py --config configs/base.yaml
    python pipelines/run_full_pipeline.py --config configs/base.yaml --skip-ml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sentiment_trader_analytics.config import AppConfig, load_config  # noqa: E402
from sentiment_trader_analytics.utils.logging_utils import setup_logging  # noqa: E402

logger = setup_logging("full_pipeline", log_file="logs/pipeline.log")

STAGES: list[tuple[str, str]] = [
    ("Phase 01 — Ingestion", "run_ingestion_pipeline.py"),
    ("Phase 02 — Validation", "run_validation_pipeline.py"),
    ("Phase 03 — Preprocessing", "run_preprocessing_pipeline.py"),
    ("Phase 04 — Feature Engineering", "run_feature_pipeline.py"),
    ("Phase 05 — EDA", "run_eda_pipeline.py"),
    ("Phase 06 — Statistical Analysis", "run_statistical_pipeline.py"),
    ("Phase 07 — Machine Learning", "run_ml_pipeline.py"),
    ("Phase 08 — Reporting", "run_reporting_pipeline.py"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run the full end-to-end pipeline (Phases 01-09).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Skip the Machine Learning (Phase 08) and Reporting (Phase 09) stages",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Run a single stage by name (e.g., 'ingestion', 'ml'). Skips all others.",
    )
    return parser.parse_args()


def _run_stage(stage_name: str, script_name: str, pipeline_dir: Path, config_path: Path) -> None:
    """Execute a single pipeline stage and handle its result.

    Args:
        stage_name: Human-readable stage label for logging.
        script_name: Filename of the pipeline script.
        pipeline_dir: Directory containing pipeline scripts.
        config_path: Path to the YAML configuration file.

    Raises:
        SystemExit: If the stage exits with a non-zero return code.
    """
    script_path = pipeline_dir / script_name
    if not script_path.exists():
        logger.error("Script not found: %s — skipping stage '%s'", script_path, stage_name)
        return

    logger.info("=== Stage: %s ===", stage_name)
    stage_start = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path), "--config", str(config_path)],
        capture_output=False,
    )

    stage_elapsed = time.time() - stage_start

    if result.returncode != 0:
        logger.error(
            "Stage '%s' failed with exit code %d after %.2f seconds. Halting pipeline.",
            stage_name,
            result.returncode,
            stage_elapsed,
        )
        sys.exit(result.returncode)

    logger.info("Stage '%s' completed in %.2f seconds", stage_name, stage_elapsed)


def main() -> None:
    """Execute the full pipeline by chaining all stages."""
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    config: AppConfig = load_config(str(config_path))
    _ = config  # Validate config loads before proceeding.

    pipeline_dir = Path(__file__).resolve().parent
    start_time = time.time()
    logger.info("Full pipeline started (config: %s)", config_path)

    try:
        if args.stage:
            # Single-stage mode
            stage_lower = args.stage.lower()
            matched = False
            for stage_name, script_name in STAGES:
                if stage_lower in script_name.lower() or stage_lower in stage_name.lower():
                    _run_stage(stage_name, script_name, pipeline_dir, config_path)
                    matched = True
                    break
            if not matched:
                logger.error(
                    "No stage matched '%s'. Available stages: %s",
                    args.stage,
                    [s[1].replace("run_", "").replace("_pipeline.py", "") for s in STAGES],
                )
                sys.exit(1)
        else:
            # Full pipeline mode
            for stage_name, script_name in STAGES:
                # Skip ML and Reporting if --skip-ml
                if args.skip_ml and ("Machine Learning" in stage_name or "Reporting" in stage_name):
                    logger.info("Skipping stage: %s (--skip-ml)", stage_name)
                    continue

                _run_stage(stage_name, script_name, pipeline_dir, config_path)

    except SystemExit:
        logger.error("Pipeline halted due to stage failure.")
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error in pipeline orchestrator.")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info("Full pipeline completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
