#!/usr/bin/env python3
# ruff: noqa: E402
"""Reporting pipeline entry point (Phase 09).

Orchestrates the final deliverable generation:
1. Pre-check: verify all upstream artifacts exist before proceeding.
2. Presentation asset regeneration at 300 DPI.
3. Final report PDF assembly (full analytical narrative).
4. Executive summary PDF assembly (stakeholder-facing, ≤2 pages).

Usage:
    python pipelines/run_reporting_pipeline.py --config configs/base.yaml

Pipeline stage: §11.8 — Reporting
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Any

import pandas as pd

from sentiment_trader_analytics.config import AppConfig, load_config
from sentiment_trader_analytics.utils.logging_utils import setup_logging
from sentiment_trader_analytics.utils.pdf_renderer import ReportPDF

logger = setup_logging("reporting", log_file="logs/pipeline.log")

REQUIRED_ARTIFACTS: list[Path] = [
    Path("outputs/figures/eda/pnl_by_sentiment_boxplot.png"),
    Path("outputs/figures/eda/sentiment_value_timeseries.png"),
    Path("outputs/figures/eda/feature_correlation_heatmap.png"),
    Path("outputs/figures/eda/sentiment_regime_frequency_barplot.png"),
    Path("outputs/tables/statistics/all_hypothesis_tests.csv"),
    Path("outputs/tables/statistics/spearman_correlation_matrix.csv"),
    Path("outputs/reports/insights_draft.md"),
]

OPTIONAL_ARTIFACTS: list[Path] = [
    Path("outputs/tables/statistics/ml_evaluation_summary.csv"),
    Path("outputs/figures/ml/classification_feature_importance.png"),
    Path("outputs/figures/ml/regression_feature_importance.png"),
]


class ReportGenerationError(Exception):
    """Raised when report generation cannot proceed due to missing artifacts."""


def pre_check() -> None:
    """Verify all required upstream artifacts exist.

    Raises:
        ReportGenerationError: If any required artifact is missing.
    """
    missing = [p for p in REQUIRED_ARTIFACTS if not p.exists()]
    if missing:
        logger.error("Missing required artifacts: %s", [str(m) for m in missing])
        raise ReportGenerationError(
            f"Cannot generate report. Missing artifacts: {[str(m) for m in missing]}"
        )
    logger.info("All %d required artifacts present", len(REQUIRED_ARTIFACTS))

    optional_missing = [p for p in OPTIONAL_ARTIFACTS if not p.exists()]
    if optional_missing:
        logger.info(
            "Optional artifacts missing (ML section will be omitted): %s",
            [str(m) for m in optional_missing],
        )
    else:
        logger.info("All optional ML artifacts present")


def _load_insights() -> list[dict[str, Any]]:
    """Parse the insights draft into structured dictionaries.

    Returns:
        List of insight dictionaries with five-part structure.
    """
    content = Path("outputs/reports/insights_draft.md").read_text(encoding="utf-8")
    sections = content.split("### ")
    insights = []

    for section in sections[1:]:
        lines = section.strip().split("\n")
        insight_id_line = lines[0].strip()
        insight_id = insight_id_line.split(":")[0].strip()

        obs = ""
        evidence = {}
        interpretation = ""
        recommendation = ""
        limitation = ""
        current_field = None

        for line in lines:
            stripped = line.strip()
            if stripped == "## Observation":
                current_field = "observation"
                continue
            elif stripped == "## Statistical Evidence":
                current_field = "evidence"
                continue
            elif stripped == "## Business Interpretation":
                current_field = "interpretation"
                continue
            elif stripped == "## Practical Recommendation":
                current_field = "recommendation"
                continue
            elif stripped == "## Limitation":
                current_field = "limitation"
                continue
            elif stripped.startswith("---") or stripped == "":
                continue

            if current_field == "observation":
                obs = (obs + " " + stripped).strip()
            elif current_field == "evidence":
                if "|" in stripped and stripped.startswith("|"):
                    parts = [p.strip() for p in stripped.split("|") if p.strip()]
                    if len(parts) == 2:
                        key = parts[0].lower().replace(" ", "_")
                        evidence[key] = parts[1]
            elif current_field == "interpretation":
                interpretation = (interpretation + " " + stripped).strip()
            elif current_field == "recommendation":
                recommendation = (recommendation + " " + stripped).strip()
            elif current_field == "limitation":
                limitation = (limitation + " " + stripped).strip()

        insights.append(
            {
                "insight_id": insight_id,
                "observation": obs,
                "evidence": evidence,
                "interpretation": interpretation,
                "recommendation": recommendation,
                "limitation": limitation,
                "metric": insight_id_line.split(":")[-1].strip() if ":" in insight_id_line else "",
            }
        )

    return insights


def _load_hypothesis_tests() -> pd.DataFrame:
    return pd.read_csv("outputs/tables/statistics/all_hypothesis_tests.csv")


def _load_correlation_matrix() -> pd.DataFrame:
    return pd.read_csv("outputs/tables/statistics/spearman_correlation_matrix.csv", index_col=0)


def _load_ml_evaluation() -> pd.DataFrame | None:
    path = Path("outputs/tables/statistics/ml_evaluation_summary.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


def _insight_evidence_to_dict(evidence: dict[str, Any]) -> dict[str, Any]:
    """Convert flat evidence dict from Markdown table format."""
    ev = {}
    mapping = {
        "test": "test_name",
        "metric": "metric",
        "groups": "groups",
        "statistic": "statistic",
        "p-value": "p_value",
        "significance": "significance",
        "effect_size": "effect_size",
        "effect_size_measure": "effect_size_measure",
        "95% ci": "confidence_interval_95",
        "sample_sizes": "sample_sizes",
        "normality_rejected": "normality_rejected",
        "correction_applied": "correction_applied",
        "corrected_threshold": "corrected_threshold",
        "underpowered": "underpowered",
    }
    for md_key, ev_key in mapping.items():
        val = evidence.get(md_key, "")
        # Parse numeric values
        if ev_key in ("statistic", "p_value", "effect_size", "corrected_threshold"):
            try:
                val = float(val) if val and val != "N/A" else None
            except ValueError:
                val = None
        elif ev_key == "underpowered" or ev_key == "normality_rejected":
            val = val == "True"
        ev[ev_key] = val
    return ev


def build_final_report(_config: AppConfig) -> None:
    """Build the final report PDF from upstream artifacts."""
    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "final_report.pdf"

    logger.info("Building final report...")

    ht_df = _load_hypothesis_tests()
    corr_df = _load_correlation_matrix()
    ml_df = _load_ml_evaluation()
    insights = _load_insights()

    pdf = ReportPDF("Bitcoin Sentiment Trader Performance Analysis")
    pdf.add_title_page(
        subtitle="Final Report — Statistical & Behavioral Analysis",
        authors="",
    )

    # --- Executive Summary ---
    pdf.add_section("Executive Summary", level=1)
    pdf.add_paragraph(
        "This report investigates the relationship between Bitcoin market sentiment "
        "(measured by the Fear & Greed Index) and trader behavior on the Hyperliquid "
        "platform. The analysis spans from the dataset's earliest records through the "
        "available observation period, encompassing trades conducted under varying "
        "sentiment regimes: Extreme Fear, Fear, Neutral, Greed, and Extreme Greed."
    )
    pdf.add_paragraph(
        "Of eight hypothesis tests conducted at a Bonferroni-corrected significance "
        "threshold of 0.00625 (α = 0.05 ÷ 8), three tests produced statistically "
        "significant results: the association between trade direction (BUY vs SELL) and "
        "sentiment regime (HT-05, p = 0.0004, Cramér's V = 0.11), the negative "
        "correlation between lagged sentiment and 7-day rolling win rate (HT-07, "
        "p = 0.0054, ρ = -0.074), and the difference in win rate across sentiment "
        "regimes (HT-08, p = 0.0017, ε² = 0.004). The remaining five tests showed "
        "no statistically significant effects after correction, indicating that "
        "aggregate PnL, leverage, and position sizing do not vary systematically "
        "with sentiment regime in this dataset."
    )
    pdf.add_paragraph(
        "Key business implications include: (1) traders exhibit directional bias — "
        "buying more during Greed, selling more during Fear — which may represent "
        "contrarian opportunities; (2) lagged sentiment provides a weak but statistically "
        "significant signal for near-term win rates; and (3) there is no evidence to "
        "support sentiment-based dynamic adjustments to leverage limits or position "
        "sizing. All findings represent statistical associations, not causal "
        "relationships, and are subject to the limitations discussed in Section 8."
    )

    # --- Introduction ---
    pdf.add_section("Introduction & Business Objective", level=1)
    pdf.add_paragraph(
        "The Bitcoin Fear & Greed Index is a widely cited market sentiment indicator "
        "that distills multiple data sources (volatility, market momentum, social media, "
        "surveys, dominance) into a single 0–100 score. Market participants routinely "
        "reference this index to inform trading decisions, yet the empirical link between "
        "sentiment regime and actual trader profitability on derivative exchanges remains "
        "underspecified."
    )
    pdf.add_paragraph(
        "This study addresses that gap by answering the following business questions:"
    )
    pdf.add_bullet(
        "Does aggregate trader profitability shift systematically with sentiment regime?"
    )
    pdf.add_bullet(
        "Does Fear correlate with elevated losses, wider drawdowns, or panic-driven behavior?"
    )
    pdf.add_bullet("Does Greed correlate with elevated leverage, larger positions, or overtrading?")
    pdf.add_bullet(
        "Which trader-level characteristics are associated with consistent profitability "
        "across sentiment regimes?"
    )
    pdf.add_bullet("What concrete, risk-aware recommendations follow from the above?")

    # --- Methodology ---
    pdf.add_section("Data Sources & Methodology", level=1)

    pdf.add_section("Fear & Greed Index Dataset", level=2)
    pdf.add_paragraph(
        "The Fear & Greed Index dataset contains daily sentiment scores (0–100) with "
        "corresponding categorical classifications: Extreme Fear (0–25), Fear (25–45), "
        "Neutral (45–55), Greed (55–75), and Extreme Greed (75–100). The data is "
        "sourced from alternative.me and covers the full observation period."
    )

    pdf.add_section("Hyperliquid Trader History Dataset", level=2)
    pdf.add_paragraph(
        "The trader history dataset contains individual trade records from the "
        "Hyperliquid exchange, including fields such as Account ID, Timestamp, Side "
        "(BUY/SELL), Size USD, Execution Price, Closed PnL, and Leverage. The dataset "
        "was ingested with schema validation, duplicate detection, and timezone "
        "normalization to UTC."
    )

    pdf.add_section("Data Pipeline & Validation", level=2)
    pdf.add_paragraph(
        "All data passed through a multi-stage pipeline: ingestion → schema validation "
        "(pandera) → cleaning (missing value handling, deduplication) → feature "
        "engineering (rolling windows, lag features, regime encoding). Validation "
        "checks included date parseability, value range conformance, uniqueness of "
        "Trade IDs, and non-null Account fields. No silent failures were tolerated."
    )

    pdf.add_section("Feature Engineering", level=2)
    pdf.add_paragraph(
        "Engineered features include rolling trader metrics (7-day and 30-day PnL, "
        "7-day win rate, 7-day trade count, 14-day PnL volatility, 24-hour average "
        "leverage), sentiment lag features (1-day lag), regime ordinal encoding, and "
        "temporal features. All rolling windows were computed without look-ahead bias."
    )

    pdf.add_section("Statistical Methodology", level=2)
    pdf.add_paragraph(
        "The analysis follows the methodology defined in Section 8 (Statistical "
        "Analysis Standards) of the Engineering Standards document. Key parameters: "
        "α = 0.05, Bonferroni correction for 8 hypotheses (corrected α = 0.00625), "
        "minimum sample size of 30 per group. Test selection was determined by "
        "normality assessment (Shapiro-Wilk / D'Agostino-Pearson): parametric tests "
        "(t-test, ANOVA) for normal data, non-parametric (Mann-Whitney U, "
        "Kruskal-Wallis) otherwise. Effect sizes (Cohen's d, rank-biserial r, "
        "epsilon-squared, Cramér's V) and 95% confidence intervals accompany all results."
    )

    # --- EDA ---
    pdf.add_section("Exploratory Data Analysis", level=1)

    pdf.add_section("Sentiment Regime Distribution", level=2)
    pdf.add_paragraph(
        "The sentiment regime frequency barplot shows the distribution of trades "
        "across sentiment classifications. Fear dominates as the most frequently "
        "observed regime, followed by Greed, Extreme Greed, Neutral, and Extreme "
        "Fear (the latter not present in the trader data)."
    )
    pdf.add_figure(
        "outputs/presentation_assets/sentiment_regime_frequency_barplot.png",
        "Figure 1: Distribution of trades across sentiment regimes. "
        "Bar labels indicate trade counts per regime.",
    )

    pdf.add_section("Trader PnL Distribution", level=2)
    pdf.add_paragraph(
        "The box plot of Closed PnL by sentiment regime reveals substantial "
        "variance in profitability across regimes, with Extreme Greed showing "
        "the widest interquartile range. Statistical testing (Section 5) determines "
        "whether these observed differences exceed what would be expected by chance."
    )
    pdf.add_figure(
        "outputs/presentation_assets/pnl_by_sentiment_boxplot.png",
        "Figure 2: Trader Closed PnL grouped by sentiment regime. "
        "Boxes show median, IQR; whiskers extend to 1.5× IQR.",
    )

    pdf.add_section("Time Series Overview", level=2)
    pdf.add_paragraph(
        "The sentiment value time series illustrates the evolution of the Fear & "
        "Greed Index over the observation period. The index fluctuates between "
        "periods of Fear and Greed, with notable transitions between regimes."
    )
    pdf.add_figure(
        "outputs/presentation_assets/sentiment_value_timeseries.png",
        "Figure 3: Bitcoin Fear & Greed Index over time. "
        "Values range from 0 (Extreme Fear) to 100 (Extreme Greed).",
    )

    # --- Statistical Results ---
    pdf.add_section("Statistical Analysis Results", level=1)

    pdf.add_section("Hypothesis Tests (HT-01 through HT-08)", level=2)
    pdf.add_paragraph(
        "Eight hypothesis tests were conducted to assess the relationship between "
        "sentiment and trader behavior. Results are summarized in the table below."
    )

    # Summary table
    ht_headers = ["Test", "Metric", "p-value", "Significant", "Effect Size"]
    ht_rows = []
    for _, row in ht_df.iterrows():
        p = row["p_value"]
        threshold = row.get("corrected_threshold", 0.00625)
        sig = "Yes" if p < threshold else "No"
        es = f"{row['effect_size']:.4f} ({row['effect_size_measure']})"
        ht_rows.append(
            [
                row["test_name"],
                row["metric"][:35],
                f"{p:.6f}",
                sig,
                es,
            ]
        )
    pdf.add_table(ht_headers, ht_rows, col_widths=[18, 50, 28, 22, 72])

    pdf.add_paragraph(
        "After Bonferroni correction (α = 0.00625), three tests reached statistical "
        "significance: HT-05 (association between trade side and sentiment regime), "
        "HT-07 (lagged sentiment vs. win rate correlation), and HT-08 (win rate "
        "differences across regimes). HT-02 (PnL across regimes, p = 0.026) was "
        "not significant after correction."
    )

    pdf.add_section("Correlation Analysis", level=2)
    pdf.add_paragraph(
        "The Spearman correlation matrix reveals the pairwise relationships among "
        "key numeric features. Notable observations include:"
    )
    pdf.add_bullet(
        "Sentiment value shows weak positive correlation with Closed PnL "
        f"(ρ = {corr_df.loc['sentiment_value', 'Closed PnL']:.3f})."
    )
    pdf.add_bullet(
        "Sentiment value is negatively correlated with 7-day rolling win rate "
        f"(ρ = {corr_df.loc['sentiment_value', 'trader_win_rate_7d']:.3f})."
    )
    pdf.add_bullet(
        "Sentiment value shows weak negative correlation with 30-day rolling PnL "
        f"(ρ = {corr_df.loc['sentiment_value', 'trader_pnl_rolling_30d']:.3f})."
    )

    pdf.add_figure(
        "outputs/presentation_assets/feature_correlation_heatmap.png",
        "Figure 4: Spearman correlation matrix of numeric features. "
        "Values in cells represent correlation coefficients (range: -1 to +1).",
    )

    pdf.add_section("Multiple Testing Correction Applied", level=2)
    pdf.add_paragraph(
        "All eight hypothesis tests were subjected to Bonferroni correction to "
        "control the family-wise error rate. The corrected significance threshold "
        "is α_corrected = 0.05 / 8 = 0.00625. This conservative approach reduces "
        "the risk of false positives when testing multiple related hypotheses. "
        "Results that are significant at the uncorrected α = 0.05 level but not "
        "at the corrected threshold are reported as trends rather than confirmatory "
        "findings."
    )

    # --- Business Insights ---
    pdf.add_section("Business Insights & Recommendations", level=1)
    pdf.add_paragraph(
        "The following insights are derived from the hypothesis testing and "
        "correlation analyses. Each insight follows a five-part structure: "
        "Observation, Statistical Evidence, Business Interpretation, Practical "
        "Recommendation, and Limitation. Insights are numbered INS-01 through "
        "INS-08, corresponding to hypothesis tests HT-01 through HT-08."
    )

    for insight in insights:
        ev = _insight_evidence_to_dict(insight.get("evidence", {}))
        pdf.add_insight_box(
            insight_id=insight["insight_id"],
            observation=insight["observation"],
            evidence=ev,
            interpretation=insight["interpretation"],
            recommendation=insight["recommendation"],
            limitation=insight["limitation"],
            title=insight.get("metric", ""),
        )

    # --- ML Results ---
    ml_section_present = len(ml_df) > 0 if ml_df is not None else False
    if ml_section_present:
        pdf.add_section("Machine Learning Results", level=1)

        pdf.add_section("Model Performance vs. Baseline", level=2)
        assert ml_df is not None
        for _, row in ml_df.iterrows():
            task = row["task"]
            pdf.add_paragraph(
                f"For the {task} task, a Random Forest model was trained with "
                f"chronological train/test split (cutoff: {row['cutoff_date']}). "
                f"The model's performance was compared against a naive baseline "
                f"(DummyRegressor/DummyClassifier)."
            )

            if task == "regression":
                pdf.add_table(
                    ["Metric", "Model", "Baseline", "Lift"],
                    [
                        [
                            "RMSE",
                            f"{row['rmse']:.2f}",
                            f"{row['baseline_rmse']:.2f}",
                            f"{row['lift_rmse']:.2%}" if pd.notna(row.get("lift_rmse")) else "N/A",
                        ],
                        [
                            "MAE",
                            f"{row['mae']:.2f}",
                            f"{row['baseline_mae']:.2f}",
                            f"{row['lift_mae']:.2%}" if pd.notna(row.get("lift_mae")) else "N/A",
                        ],
                        [
                            "R²",
                            f"{row['r2']:.4f}",
                            f"{row['baseline_r2']:.4f}",
                            f"{row['lift_r2']:.2%}" if pd.notna(row.get("lift_r2")) else "N/A",
                        ],
                    ],
                    col_widths=[30, 40, 40, 40],
                )
            elif task == "classification":
                pdf.add_table(
                    ["Metric", "Model", "Baseline", "Lift"],
                    [
                        [
                            "Accuracy",
                            f"{row['accuracy']:.4f}",
                            f"{row['baseline_accuracy']:.4f}",
                            (
                                f"{row['lift_accuracy']:.2%}"
                                if pd.notna(row.get("lift_accuracy"))
                                else "N/A"
                            ),
                        ],
                        [
                            "Precision",
                            f"{row['precision']:.4f}",
                            f"{row['baseline_precision']:.4f}",
                            (
                                f"{row['lift_precision']:.2%}"
                                if pd.notna(row.get("lift_precision"))
                                else "N/A"
                            ),
                        ],
                        [
                            "Recall",
                            f"{row['recall']:.4f}",
                            f"{row['baseline_recall']:.4f}",
                            (
                                f"{row['lift_recall']:.2%}"
                                if pd.notna(row.get("lift_recall"))
                                else "N/A"
                            ),
                        ],
                        [
                            "F1",
                            f"{row['f1']:.4f}",
                            f"{row['baseline_f1']:.4f}",
                            f"{row['lift_f1']:.2%}" if pd.notna(row.get("lift_f1")) else "N/A",
                        ],
                        [
                            "ROC-AUC",
                            f"{row['roc_auc']:.4f}" if pd.notna(row.get("roc_auc")) else "N/A",
                            "N/A",
                            "N/A",
                        ],
                    ],
                    col_widths=[30, 40, 40, 40],
                )

        pdf.add_section("Feature Importance", level=2)
        pdf.add_paragraph(
            "Feature importance charts for both classification and regression "
            "tasks are included below. These were computed using permutation "
            "importance to avoid misleading impurity-based rankings."
        )

        clf_path = Path("outputs/presentation_assets/ml_classification_feature_importance.png")
        reg_path = Path("outputs/presentation_assets/ml_regression_feature_importance.png")
        if clf_path.exists():
            pdf.add_figure(
                str(clf_path),
                "Figure 5: Permutation feature importance for Random Forest "
                "classifier (profitable vs. unprofitable trade prediction).",
                width_mm=150,
            )
        if reg_path.exists():
            pdf.add_figure(
                str(reg_path),
                "Figure 6: Permutation feature importance for Random Forest "
                "regressor (PnL prediction).",
                width_mm=150,
            )

        pdf.add_section("Limitations of ML Approach", level=2)
        pdf.add_paragraph(
            "The Random Forest models were trained as a baseline comparison and "
            "should be interpreted with several caveats. First, the dataset size "
            "and temporal coverage may limit generalizability. Second, the R² "
            "values near or below zero for regression indicate the model fails to "
            "outperform a mean-predicting baseline, suggesting either weak signal "
            "or a need for more sophisticated feature engineering. Third, the "
            "classification model shows marginal lift over the baseline, with "
            "ROC-AUC barely exceeding 0.5. These results should not be interpreted "
            "as a deployable trading signal without substantial further investigation."
        )

    # --- Limitations ---
    pdf.add_section("Limitations & Caveats", level=1)
    pdf.add_paragraph(
        "The following limitations qualify the confidence of the findings presented "
        "in this report and should be considered when applying the recommendations."
    )
    pdf.add_bullet(
        "Correlation vs. Causation: All findings represent statistical associations, "
        "not causal relationships. Confounders including overall BTC price trend, "
        "market-wide volatility, and exchange-specific factors may influence both "
        "sentiment and trader behavior."
    )
    pdf.add_bullet(
        "Dataset Coverage: The analysis is limited to the Hyperliquid exchange and "
        "may not generalize to other trading venues. The observation period may not "
        "capture a full market cycle."
    )
    pdf.add_bullet(
        "Regime Imbalance: The distribution of trades across sentiment regimes is "
        "uneven (Fear dominates), reducing statistical power for rarer regimes such "
        "as Extreme Fear."
    )
    pdf.add_bullet(
        "Temporal Granularity: Sentiment is measured daily, while trading occurs "
        "continuously. Intra-day sentiment shifts are not captured, potentially "
        "masking short-term behavioral effects."
    )
    pdf.add_bullet(
        "Population Composition: The trader population on Hyperliquid may differ "
        "systematically from retail traders on other platforms, affecting "
        "generalizability of behavioral findings."
    )
    pdf.add_bullet(
        "ML Limitations: Predictive models were trained as a baseline exercise "
        "and should not be deployed without additional validation, feature "
        "engineering, and out-of-sample testing."
    )

    # --- Appendix ---
    pdf.add_section("Appendix: Full Statistical Tables", level=1)

    pdf.add_section("All Hypothesis Tests (HT-01 through HT-08)", level=2)
    pdf.add_paragraph(
        "The table below summarizes all eight hypothesis tests with full statistical details. "
        "Columns: Test ID, Metric (truncated), p-value, Significant?, Effect Size, Effect Measure, "
        "95% CI Lower, 95% CI Upper, Correction Applied, Corrected Threshold."
    )
    ht_appendix_headers = [
        "Test",
        "Metric",
        "p-value",
        "Sig?",
        "Effect Size",
        "ES Measure",
        "CI Low",
        "CI High",
        "Correction",
        "Threshold",
    ]
    ht_appendix_rows = []
    for _, row in ht_df.iterrows():
        ht_appendix_rows.append(
            [
                str(row.get("test_name", "")),
                str(row.get("metric", ""))[:18],
                f"{row['p_value']:.5f}" if pd.notna(row.get("p_value")) else "",
                (
                    "YES"
                    if pd.notna(row.get("p_value"))
                    and pd.notna(row.get("corrected_threshold"))
                    and row["p_value"] < row["corrected_threshold"]
                    else "no"
                ),
                f"{row['effect_size']:.4f}" if pd.notna(row.get("effect_size")) else "",
                str(row.get("effect_size_measure", ""))[:12],
                (
                    f"{row['confidence_interval_lower']:.4f}"
                    if pd.notna(row.get("confidence_interval_lower"))
                    else ""
                ),
                (
                    f"{row['confidence_interval_upper']:.4f}"
                    if pd.notna(row.get("confidence_interval_upper"))
                    else ""
                ),
                str(row.get("correction_applied", "")),
                (
                    f"{row['corrected_threshold']:.5f}"
                    if pd.notna(row.get("corrected_threshold"))
                    else ""
                ),
            ]
        )
    pdf.add_table(
        ht_appendix_headers, ht_appendix_rows, col_widths=[14, 30, 20, 10, 18, 22, 18, 18, 22, 18]
    )

    pdf.add_section("Spearman Correlation Matrix", level=2)
    pdf.add_paragraph(
        "Spearman rank correlation coefficients between key numeric features. "
        "Values close to +1 indicate strong positive rank correlation; close to -1 indicate "
        "strong negative correlation. Values near 0 indicate no monotonic relationship."
    )

    # Truncate column names for readability in the table
    def _short(s: str, n: int = 12) -> str:
        mapping = {
            "sentiment_value": "s_val",
            "Closed PnL": "PnL",
            "Leverage": "Lev",
            "trader_win_rate_7d": "win_7d",
            "trader_pnl_rolling_7d": "pnl_7d",
            "trader_pnl_rolling_30d": "pnl_30d",
            "trader_leverage_avg_24h": "lev_24h",
            "trader_pnl_volatility_14d": "vol_14d",
            "trader_trade_count_7d": "trds_7d",
            "trader_avg_size_usd_7d": "size_7d",
        }
        return mapping.get(s, s[:n])

    corr_short_cols = [_short(c) for c in corr_df.columns]
    corr_headers = ["Feature"] + corr_short_cols
    corr_rows = []
    for idx in corr_df.index:
        row_vals = [_short(str(idx), 16)] + [f"{corr_df.loc[idx, c]:.3f}" for c in corr_df.columns]
        corr_rows.append(row_vals)
    n_cols = len(corr_headers)
    w_first = 30
    w_rest = (190 - w_first) / max(1, n_cols - 1)
    corr_widths = [w_first] + [w_rest] * (n_cols - 1)
    pdf.add_table(corr_headers, corr_rows, col_widths=corr_widths)

    if ml_section_present:
        pdf.add_section("ML Evaluation Summary", level=2)
        assert ml_df is not None

        # Transpose the dataframe so metrics are rows and tasks are columns
        ml_t = ml_df.set_index("task").T
        ml_t.index.name = "Metric"
        ml_t = ml_t.reset_index()

        ml_headers_short = [str(c).title() for c in ml_t.columns]
        ml_rows = []
        for _, row in ml_t.iterrows():
            # Only add rows that have at least one non-null value in the task columns
            if row[1:].notna().any():
                ml_rows.append([str(v)[:20] if pd.notna(v) else "N/A" for v in row.values])

        n_ml_cols = len(ml_headers_short)
        ml_col_w = 190 / max(1, n_ml_cols)
        pdf.add_table(ml_headers_short, ml_rows, col_widths=[ml_col_w] * n_ml_cols)

    # Output
    pdf.output(str(output_path))
    logger.info("Final report saved to %s", output_path)


def build_executive_summary(_config: AppConfig) -> None:
    """Build the executive summary PDF (max 2 pages)."""
    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "executive_summary.pdf"

    logger.info("Building executive summary...")

    ht_df = _load_hypothesis_tests()
    insights = _load_insights()

    pdf = ReportPDF("Executive Summary | Bitcoin Sentiment Trader Analysis", section_numbering=True)
    pdf.add_title_page(
        subtitle="Executive Summary — Key Findings & Recommendations",
        authors="",
    )

    # --- Business Question & Answer ---
    pdf.add_section("Business Question & Core Answer", level=1)
    pdf.add_paragraph(
        "Does Bitcoin market sentiment (Fear & Greed Index) predictably influence "
        "trader profitability and behavior on Hyperliquid? "
        "Answer: Partially — sentiment correlates with trade direction and near-term "
        "win rates, but does not systematically predict aggregate PnL, leverage usage, "
        "or position sizing. The effects are statistically detectable but economically "
        "small, and all findings are associations, not causal relationships."
    )

    # --- Top 3 Insights ---
    pdf.add_section("Top 3 Actionable Insights", level=1)

    sig_insights = [
        i for i in insights if i.get("evidence", {}).get("significance", "").startswith("**")
    ]
    for i, insight in enumerate(sig_insights[:3], 1):
        pdf.add_paragraph(f"{i}. {insight['insight_id']}: {insight['observation']}")
        pdf.add_paragraph(f"   Recommendation: {insight['recommendation']}")
        pdf.add_paragraph(f"   Limitation: {insight['limitation']}")
        pdf.ln(2)

    # If we don't have 3 significant ones, add non-significant ones with the most notable
    if len(sig_insights) < 3:
        for insight in insights:
            if insight not in sig_insights[:3] and len(sig_insights) < 3:
                sig_insights.append(insight)

    # --- Key Statistical Findings ---
    pdf.add_section("Key Statistical Findings", level=1)
    ht_headers = ["Test", "Metric", "p-value", "Significant (α=0.00625)", "Effect Size"]
    ht_rows = []
    for _, row in ht_df.iterrows():
        p = row["p_value"]
        threshold = row.get("corrected_threshold", 0.00625)
        sig = "YES" if p < threshold else "no"
        es = f"{row['effect_size']:.4f} ({row['effect_size_measure']})"
        ht_rows.append(
            [
                row["test_name"],
                row["metric"][:30],
                f"{p:.6f}",
                sig,
                es,
            ]
        )
    pdf.add_table(ht_headers, ht_rows, col_widths=[16, 48, 26, 36, 64])

    # --- Limitations ---
    pdf.add_section("Limitations", level=1)
    pdf.add_paragraph(
        "All findings represent statistical associations, not causal relationships. "
        "Results are specific to the Hyperliquid exchange and may not generalize to "
        "other trading venues or time periods. The analysis is limited by daily "
        "sentiment granularity, uneven regime distribution, and potential confounders "
        "including overall BTC price trends and market volatility. ML models showed "
        "negligible predictive lift and should not be deployed without further validation."
    )

    # Output
    pdf.output(str(output_path))
    logger.info("Executive summary saved to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reporting pipeline (Phase 09).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--skip-precheck",
        action="store_true",
        help="Skip the artifact pre-check (for development use)",
    )
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Skip presentation asset regeneration",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Skip final report generation",
    )
    parser.add_argument(
        "--skip-executive",
        action="store_true",
        help="Skip executive summary generation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    config: AppConfig = load_config(str(config_path))
    start_time = time.time()

    try:
        # Step 1: Pre-check
        if not args.skip_precheck:
            logger.info("Step 1/4: Running artifact pre-check...")
            pre_check()
        else:
            logger.warning("Skipping artifact pre-check")

        # Step 2: Presentation assets
        if not args.skip_assets:
            logger.info("Step 2/4: Generating presentation assets (300 DPI)...")
            from pipelines.generate_presentation_assets import (
                generate_presentation_assets,
            )

            assets = generate_presentation_assets(config)
            logger.info("Generated %d presentation assets", len(assets))
        else:
            logger.warning("Skipping presentation asset generation")

        # Step 3: Final report
        if not args.skip_final:
            logger.info("Step 3/4: Building final report...")
            build_final_report(config)
        else:
            logger.warning("Skipping final report generation")

        # Step 4: Executive summary
        if not args.skip_executive:
            logger.info("Step 4/4: Building executive summary...")
            build_executive_summary(config)
        else:
            logger.warning("Skipping executive summary generation")

    except ReportGenerationError:
        logger.error("Report generation halted due to missing artifacts")
        sys.exit(1)
    except Exception:
        logger.exception("Reporting pipeline failed")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info("Reporting pipeline completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
