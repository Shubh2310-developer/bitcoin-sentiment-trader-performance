#!/usr/bin/env python3
"""Statistical analysis pipeline entry point.

Orchestrates the full hypothesis test suite (HT-01 through HT-08),
followed by the extended analyses (ET-01 through ET-07), applies
multiple testing correction, and saves all results to
``outputs/tables/statistics/``.

Usage:
    python pipelines/run_statistical_pipeline.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd  # noqa: E402

from sentiment_trader_analytics.config import (  # noqa: E402
    AppConfig,
    StatConfig,
    load_config,
)
from sentiment_trader_analytics.statistics.correlation_analysis import (  # noqa: E402
    compute_correlation,
    compute_correlation_matrix,
)
from sentiment_trader_analytics.statistics.extended_analyses import (  # noqa: E402
    analyze_lagged_correlations,
    analyze_regime_duration,
    analyze_regime_transitions,
    analyze_trader_segments,
    analyze_volatility_interaction,
    analyze_win_streaks,
    build_findings_matrix,
    compute_power_analysis,
)
from sentiment_trader_analytics.statistics.hypothesis_tests import (  # noqa: E402
    StatTestResult,
    apply_multiple_testing_correction,
    chi_square_test,
    compare_multiple_groups,
    compare_two_groups,
)
from sentiment_trader_analytics.utils.logging_utils import setup_logging  # noqa: E402

logger = setup_logging("statistics", log_file="logs/pipeline.log")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run the statistical analysis pipeline (Phase 06)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file (default: configs/base.yaml)",
    )
    return parser.parse_args()


def _find_latest_features(features_dir: Path) -> Path:
    """Find the most recently created parquet file in the features directory.

    Args:
        features_dir: Directory containing feature store parquet files.

    Returns:
        Path to the latest parquet file.

    Raises:
        FileNotFoundError: If no parquet files are found.
    """
    parquet_files = sorted(features_dir.glob("feature_store_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No feature store parquet files found in {features_dir}. "
            "Run the feature engineering pipeline first."
        )
    return parquet_files[-1]


def _result_to_dict(result: StatTestResult) -> dict[str, Any]:
    """Convert a StatTestResult to a flat dictionary for CSV output.

    Args:
        result: The test result to convert.

    Returns:
        A dictionary with flattened fields.
    """
    return {
        "test_name": result.test_name,
        "metric": result.metric,
        "groups": "|".join(result.groups),
        "statistic": round(result.statistic, 6),
        "p_value": round(result.p_value, 6),
        "effect_size": round(result.effect_size, 6),
        "effect_size_measure": result.effect_size_measure,
        "confidence_interval_95": (
            f"({result.confidence_interval_95[0]:.6f}, {result.confidence_interval_95[1]:.6f})"
        ),
        "confidence_interval_lower": round(result.confidence_interval_95[0], 6),
        "confidence_interval_upper": round(result.confidence_interval_95[1], 6),
        "sample_sizes": ";".join(f"{k}={v}" for k, v in result.sample_sizes.items()),
        "normality_rejected": result.normality_rejected,
        "correction_applied": result.correction_applied or "",
        "corrected_threshold": (
            round(result.corrected_threshold, 6) if result.corrected_threshold is not None else ""
        ),
        "underpowered": result.underpowered,
    }


def main() -> None:
    """Execute the statistical analysis pipeline."""
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    config: AppConfig = load_config(str(config_path))
    stat_config: StatConfig = config.statistics
    start_time = time.time()

    tables_dir = Path(stat_config.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Statistical analysis pipeline started")
    logger.info(
        "Config: alpha=%.3f, correction=%s, min_sample_size=%d",
        stat_config.alpha,
        stat_config.correction_method,
        stat_config.min_sample_size,
    )

    try:
        # 1. Load the feature store
        features_dir = Path("data/features")
        input_path = _find_latest_features(features_dir)
        logger.info("Loading feature store from: %s", input_path)
        df = pd.read_parquet(input_path)
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

        # Exclude cold-start rows
        if "feature_cold_start" in df.columns:
            n_before = len(df)
            df = df[df["feature_cold_start"] == False].copy()  # noqa: E712
            logger.info(
                "Excluded %d cold-start rows; %d remaining",
                n_before - len(df),
                len(df),
            )
        else:
            logger.warning("Column 'feature_cold_start' not found; proceeding with all rows")

        alpha = stat_config.alpha
        min_n = stat_config.min_sample_size

        # ── HT-01: PnL in Fear vs. Greed ──────────────────────────
        logger.info("HT-01: PnL in Fear vs. Greed")
        fear_pnl = df.loc[df["sentiment_is_fear"] == True, "Closed PnL"].dropna()  # noqa: E712
        greed_pnl = df.loc[df["sentiment_is_greed"] == True, "Closed PnL"].dropna()  # noqa: E712
        ht01 = compare_two_groups(fear_pnl, greed_pnl, alpha, min_n)
        ht01.test_name = "HT-01"
        ht01.metric = "Closed PnL"
        ht01.groups = ["Fear", "Greed"]
        ht01.sample_sizes = {"Fear": len(fear_pnl), "Greed": len(greed_pnl)}
        logger.info(
            "  -> %s, p=%.4f, effect=%s=%.4f, n_Fear=%d, n_Greed=%d",
            ht01.test_name,
            ht01.p_value,
            ht01.effect_size_measure,
            ht01.effect_size,
            ht01.sample_sizes["Fear"],
            ht01.sample_sizes["Greed"],
        )

        # ── HT-02: PnL across all 5 regimes ──────────────────────
        logger.info("HT-02: PnL across all 5 regimes")
        regime_groups_pnl: dict[str, pd.Series] = {}
        for regime in df["sentiment_classification"].cat.categories:
            subset = df.loc[df["sentiment_classification"] == regime, "Closed PnL"].dropna()
            if len(subset) > 0:
                regime_groups_pnl[str(regime)] = subset
        ht02 = compare_multiple_groups(regime_groups_pnl, alpha, min_n)
        ht02.test_name = "HT-02"
        ht02.metric = "Closed PnL"
        ht02.groups = list(regime_groups_pnl.keys())
        logger.info(
            "  -> %s, p=%.4f, effect=%s=%.4f, groups=%s",
            ht02.test_name,
            ht02.p_value,
            ht02.effect_size_measure,
            ht02.effect_size,
            list(ht02.sample_sizes.keys()),
        )

        # ── HT-03: Leverage in Fear vs. Greed ─────────────────────
        logger.info("HT-03: Leverage in Fear vs. Greed")
        fear_lev = df.loc[df["sentiment_is_fear"] == True, "Leverage"].dropna()  # noqa: E712
        greed_lev = df.loc[df["sentiment_is_greed"] == True, "Leverage"].dropna()  # noqa: E712
        ht03 = compare_two_groups(fear_lev, greed_lev, alpha, min_n)
        ht03.test_name = "HT-03"
        ht03.metric = "Leverage"
        ht03.groups = ["Fear", "Greed"]
        ht03.sample_sizes = {"Fear": len(fear_lev), "Greed": len(greed_lev)}
        logger.info(
            "  -> %s, p=%.4f, effect=%s=%.4f",
            ht03.test_name,
            ht03.p_value,
            ht03.effect_size_measure,
            ht03.effect_size,
        )

        # ── HT-04: Trade size across all 5 regimes ────────────────
        logger.info("HT-04: Trade size across all 5 regimes")
        size_col = "Size USD" if "Size USD" in df.columns else "trader_avg_size_usd_7d"
        regime_groups_size: dict[str, pd.Series] = {}
        for regime in df["sentiment_classification"].cat.categories:
            subset = df.loc[df["sentiment_classification"] == regime, size_col].dropna()
            if len(subset) > 0:
                regime_groups_size[str(regime)] = subset
        ht04 = compare_multiple_groups(regime_groups_size, alpha, min_n)
        ht04.test_name = "HT-04"
        ht04.metric = size_col
        ht04.groups = list(regime_groups_size.keys())
        logger.info(
            "  -> %s, p=%.4f, effect=%s=%.4f",
            ht04.test_name,
            ht04.p_value,
            ht04.effect_size_measure,
            ht04.effect_size,
        )

        # ── HT-05: Trade Side vs. Regime ──────────────────────────
        logger.info("HT-05: Trade Side vs. Regime")
        side_col = "Side" if "Side" in df.columns else "Direction"
        if side_col in df.columns and "sentiment_classification" in df.columns:
            contingency = pd.crosstab(
                df[side_col].astype(str),
                df["sentiment_classification"].astype(str),
            )
            ht05 = chi_square_test(
                contingency,
                alpha,
                stat_config.min_expected_frequency,
            )
        else:
            ht05 = StatTestResult(
                test_name="HT-05",
                metric="contingency",
                groups=["Side", "sentiment_classification"],
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                effect_size_measure="cramers_v",
                confidence_interval_95=(0.0, 0.0),
                sample_sizes={"total": 0},
                normality_rejected=False,
                underpowered=True,
            )
        ht05.test_name = "HT-05"
        logger.info(
            "  -> %s, p=%.4f, effect=%s=%.4f",
            ht05.test_name,
            ht05.p_value,
            ht05.effect_size_measure,
            ht05.effect_size,
        )

        # ── HT-06: Sentiment vs. Closed PnL ───────────────────────
        logger.info("HT-06: Sentiment vs. Closed PnL")
        ht06_corr = compute_correlation(
            df["sentiment_value"].dropna(),
            df["Closed PnL"].dropna(),
            method="spearman",
            alpha=alpha,
        )
        ht06 = StatTestResult(
            test_name="HT-06",
            metric="sentiment_value vs Closed PnL",
            groups=["sentiment_value", "Closed PnL"],
            statistic=ht06_corr.coefficient,
            p_value=ht06_corr.p_value,
            effect_size=ht06_corr.coefficient,
            effect_size_measure="spearman_rho",
            confidence_interval_95=ht06_corr.confidence_interval_95,
            sample_sizes={"paired_n": ht06_corr.sample_size},
            normality_rejected=False,
            underpowered=ht06_corr.sample_size < min_n,
        )
        logger.info(
            "  -> %s, rho=%.4f, p=%.4f, n=%d",
            ht06.test_name,
            ht06_corr.coefficient,
            ht06_corr.p_value,
            ht06_corr.sample_size,
        )

        # ── HT-07: 7d Win Rate vs. 1d Sentiment Lag ──────────────
        logger.info("HT-07: Win rate vs. Sentiment lag")
        wr_col = "trader_win_rate_7d"
        sl_col = "sentiment_value_lag_1d"
        if wr_col in df.columns and sl_col in df.columns:
            ht07_corr = compute_correlation(
                df[wr_col].dropna(),
                df[sl_col].dropna(),
                method="spearman",
                alpha=alpha,
            )
        else:
            ht07_corr = None
        if ht07_corr is not None:
            ht07 = StatTestResult(
                test_name="HT-07",
                metric=f"{wr_col} vs {sl_col}",
                groups=[wr_col, sl_col],
                statistic=ht07_corr.coefficient,
                p_value=ht07_corr.p_value,
                effect_size=ht07_corr.coefficient,
                effect_size_measure="spearman_rho",
                confidence_interval_95=ht07_corr.confidence_interval_95,
                sample_sizes={"paired_n": ht07_corr.sample_size},
                normality_rejected=False,
                underpowered=ht07_corr.sample_size < min_n,
            )
        else:
            ht07 = StatTestResult(
                test_name="HT-07",
                metric=f"{wr_col} vs {sl_col}",
                groups=[wr_col, sl_col],
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                effect_size_measure="spearman_rho",
                confidence_interval_95=(0.0, 0.0),
                sample_sizes={"paired_n": 0},
                normality_rejected=False,
                underpowered=True,
            )
        logger.info(
            "  -> %s, rho=%.4f, p=%.4f, n=%d",
            ht07.test_name,
            ht07.statistic,
            ht07.p_value,
            ht07.sample_sizes.get("paired_n", 0),
        )

        # ── HT-08: Win rate across all 5 regimes ─────────────────
        logger.info("HT-08: Win rate across all 5 regimes")
        regime_groups_wr: dict[str, pd.Series] = {}
        for regime in df["sentiment_classification"].cat.categories:
            subset = df.loc[df["sentiment_classification"] == regime, wr_col].dropna()
            if len(subset) > 0:
                regime_groups_wr[str(regime)] = subset
        ht08 = compare_multiple_groups(regime_groups_wr, alpha, min_n)
        ht08.test_name = "HT-08"
        ht08.metric = wr_col
        ht08.groups = list(regime_groups_wr.keys())
        logger.info(
            "  -> %s, p=%.4f, effect=%s=%.4f",
            ht08.test_name,
            ht08.p_value,
            ht08.effect_size_measure,
            ht08.effect_size,
        )

        # ── Assemble results ──────────────────────────────────────
        all_results: list[StatTestResult] = [ht01, ht02, ht03, ht04, ht05, ht06, ht07, ht08]

        # Apply multiple testing correction
        logger.info(
            "Applying %s correction across %d tests",
            stat_config.correction_method,
            len(all_results),
        )
        all_results = apply_multiple_testing_correction(
            all_results, stat_config.correction_method, alpha
        )

        # Log underpowered tests
        for result in all_results:
            if result.underpowered:
                logger.warning(
                    "Underpowered test: %s (sample sizes: %s)",
                    result.test_name,
                    result.sample_sizes,
                )

        # Save individual HT results
        result_files = {
            "HT-01": "ht01_pnl_fear_vs_greed.csv",
            "HT-02": "ht02_pnl_across_regimes.csv",
            "HT-03": "ht03_leverage_fear_vs_greed.csv",
            "HT-04": "ht04_size_across_regimes.csv",
            "HT-05": "ht05_side_vs_regime.csv",
            "HT-06": "ht06_sentiment_pnl_correlation.csv",
            "HT-07": "ht07_winrate_sentiment_correlation.csv",
            "HT-08": "ht08_winrate_across_regimes.csv",
        }

        for result in all_results:
            fname = result_files.get(result.test_name)
            if fname:
                fpath = tables_dir / fname
                record = _result_to_dict(result)
                pd.DataFrame([record]).to_csv(fpath, index=False)
                logger.info("Saved: %s", fpath)

        # Save combined core results table
        combined_path = tables_dir / "all_hypothesis_tests.csv"
        combined_df = pd.DataFrame([_result_to_dict(r) for r in all_results])
        combined_df.to_csv(combined_path, index=False)
        logger.info(
            "Saved combined results: %s (%d tests)",
            combined_path,
            len(combined_df),
        )

        # Save Spearman correlation matrix
        corr_columns = [
            "sentiment_value",
            "Closed PnL",
            "Leverage",
            "trader_win_rate_7d",
            "trader_pnl_rolling_7d",
            "trader_pnl_rolling_30d",
            "trader_leverage_avg_24h",
            "trader_pnl_volatility_14d",
            "trader_trade_count_7d",
            "trader_avg_size_usd_7d",
        ]
        corr_path = tables_dir / "spearman_correlation_matrix.csv"
        compute_correlation_matrix(df, corr_columns, output_path=str(corr_path))
        logger.info("Saved correlation matrix: %s", corr_path)

        # ── Extended Analyses ET-01 through ET-07 ──────────────────────────
        extended_results: list[StatTestResult] = []
        if stat_config.run_extended_analyses:
            logger.info("=== Running extended analyses (ET-01 through ET-07) ===")

            # ET-01: Regime Transition Analysis
            logger.info("--- ET-01: Regime Transition Analysis ---")
            try:
                et01 = analyze_regime_transitions(
                    df,
                    alpha=alpha,
                    min_sample_size=stat_config.min_sample_size,
                )
                extended_results.append(et01)
                logger.info(
                    "ET-01: p=%.4f, effect_size=%.4f, test=%s",
                    et01.p_value,
                    et01.effect_size or 0.0,
                    et01.test_name,
                )
            except Exception:
                logger.exception("ET-01 failed; skipping.")

            # ET-02: Regime Duration Effects
            logger.info("--- ET-02: Regime Duration Effects ---")
            try:
                et02 = analyze_regime_duration(
                    df,
                    duration_buckets=stat_config.regime_duration_buckets,
                    alpha=alpha,
                    min_sample_size=stat_config.min_sample_size,
                )
                extended_results.append(et02)
                logger.info(
                    "ET-02: p=%.4f, effect_size=%.4f", et02.p_value, et02.effect_size or 0.0
                )
            except Exception:
                logger.exception("ET-02 failed; skipping.")

            # ET-03: Trader Segmentation
            logger.info("--- ET-03: Trader Segmentation ---")
            try:
                et03_list = analyze_trader_segments(
                    df,
                    k_min=stat_config.trader_segment_k_min,
                    k_max=stat_config.trader_segment_k_max,
                    alpha=alpha,
                    min_sample_size=stat_config.min_sample_size,
                )
                extended_results.extend(et03_list)
                logger.info("ET-03: produced %d results.", len(et03_list))
            except Exception:
                logger.exception("ET-03 failed; skipping.")

            # ET-04: Volatility Regime Interaction
            logger.info("--- ET-04: Volatility Regime Interaction ---")
            try:
                et04_list = analyze_volatility_interaction(
                    df,
                    min_sample_size=stat_config.min_sample_size,
                )
                extended_results.extend(et04_list)
                logger.info("ET-04: produced %d results.", len(et04_list))
            except Exception:
                logger.exception("ET-04 failed; skipping.")

            # ET-05: Extended Lag Analysis
            logger.info("--- ET-05: Extended Lag Analysis ---")
            try:
                et05_list = analyze_lagged_correlations(
                    df,
                    max_lag_days=stat_config.max_lag_days,
                    min_sample_size=stat_config.min_sample_size,
                )
                extended_results.extend(et05_list)
                logger.info("ET-05: produced %d lag results.", len(et05_list))
            except Exception:
                logger.exception("ET-05 failed; skipping.")

            # ET-07: Win-Streak Analysis
            logger.info("--- ET-07: Win-Streak Analysis ---")
            try:
                et07_list = analyze_win_streaks(
                    df,
                    min_sample_size=stat_config.min_sample_size,
                )
                extended_results.extend(et07_list)
                logger.info("ET-07: produced %d results.", len(et07_list))
            except Exception:
                logger.exception("ET-07 failed; skipping.")

            # Apply correction to extended results separately
            if extended_results:
                logger.info(
                    "Applying %s correction to %d extended results.",
                    stat_config.correction_method,
                    len(extended_results),
                )
                # Filter out NaN p-value results before correction
                valid_ext = [
                    r for r in extended_results if r.p_value == r.p_value  # NaN check
                ]
                if valid_ext:
                    valid_ext = apply_multiple_testing_correction(
                        valid_ext, stat_config.correction_method, alpha
                    )

                # Save extended results
                ext_path = tables_dir / "extended_analyses_results.csv"
                ext_df = pd.DataFrame([_result_to_dict(r) for r in extended_results])
                ext_df.to_csv(ext_path, index=False)
                logger.info("Saved extended analyses: %s (%d rows)", ext_path, len(ext_df))

            # ET-06: Power Analysis (over all core + extended results)
            logger.info("--- ET-06: Power Analysis ---")
            try:
                all_for_power = all_results + [
                    r for r in extended_results if r.p_value == r.p_value
                ]
                power_df = compute_power_analysis(all_for_power, alpha=alpha, power_target=0.80)
                power_path = tables_dir / "power_analysis.csv"
                power_df.to_csv(power_path, index=False)
                logger.info("Saved power analysis: %s", power_path)
            except Exception:
                logger.exception("ET-06 power analysis failed; skipping.")

        # ── Findings Matrix ────────────────────────────────────────────────
        try:
            all_combined = all_results + extended_results
            findings_df = build_findings_matrix(all_combined)
            findings_path = tables_dir / "findings_matrix.csv"
            findings_df.to_csv(findings_path, index=False)
            n_sig = int(findings_df["significant"].sum())
            n_total_fm = len(findings_df)
            logger.info(
                "Findings matrix: %d/%d tests significant → %s",
                n_sig,
                n_total_fm,
                findings_path,
            )
        except Exception:
            logger.exception("Findings matrix generation failed; skipping.")

    except Exception:
        logger.exception("Statistical analysis pipeline failed")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info("Statistical analysis pipeline completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
