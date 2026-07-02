"""Business insight synthesis layer.

Translates statistical test results (StatTestResult from Phase 06) into
structured BusinessInsight objects with the mandatory five-part format:
observation, statistical_evidence, business_interpretation,
practical_recommendation, limitation. Also provides validation and
Markdown export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from sentiment_trader_analytics.statistics.hypothesis_tests import StatTestResult

logger = logging.getLogger(__name__)


class BusinessInsight(BaseModel):
    """A structured business insight derived from a statistical test result.

    Every insight follows the five-part format mandated by Section 18 of
    the Engineering Standards. An insight is ``report_ready`` only when
    all five components are present and the underlying ``effect_size``
    is not null.
    """

    insight_id: str
    title: str = ""
    observation: str
    statistical_evidence: StatTestResult
    business_interpretation: str
    practical_recommendation: str
    limitation: str
    report_ready: bool = False


# Human-readable insight titles mapped from test ID
_INSIGHT_TITLES: dict[str, str] = {
    "INS-01": "PnL Comparison: Fear vs. Greed Regime",
    "INS-02": "Trader PnL Across All Five Sentiment Regimes",
    "INS-03": "Leverage Usage: Fear vs. Greed Regime",
    "INS-04": "Position Sizing Across Sentiment Regimes",
    "INS-05": "Trade Directionality vs. Sentiment (Chi-Square)",
    "INS-06": "Continuous Sentiment Score vs. Closed PnL",
    "INS-07": "Lagged Sentiment Predicts 7-Day Rolling Win Rate [SIGNIFICANT]",
    "INS-08": "Win Rate Differs Across Sentiment Regimes [SIGNIFICANT]",
    "ET-01": "PnL Behavior Following Regime Transitions",
    "ET-02": "Impact of Regime Duration (Streak Length) on PnL",
    "ET-03": "Trader Segmentation: Archetypes & Sentiment Sensitivity",
    "ET-04": "Volatility & Sentiment Interaction Effects",
    "ET-05": "Multi-Day Lagged Correlations with PnL",
    "ET-06": "Statistical Power & Minimum Detectable Effects",
    "ET-07": "Win Streak Frequencies Across Sentiment Regimes",
}

# The 3-Act Narrative Structure mapping (Insight ID -> Act)
_NARRATIVE_ACTS: dict[str, str] = {
    "INS-03": "Act I — The Setup",
    "INS-04": "Act I — The Setup",
    "INS-05": "Act II — The Conflict",
    "INS-07": "Act II — The Conflict",
    "ET-01": "Act II — The Conflict",
    "ET-03": "Act II — The Conflict",
    "INS-01": "Act III — The Resolution",
    "INS-02": "Act III — The Resolution",
    "ET-07": "Act III — The Resolution",
}


def _build_observation(
    evidence: StatTestResult,
    threshold: float,
) -> str:
    """Build a clean, human-readable observation sentence.

    Args:
        evidence: The statistical test result.
        threshold: The corrected significance threshold.

    Returns:
        A well-formed observation string with inline statistics.
    """
    p = evidence.p_value
    es = evidence.effect_size
    es_measure = evidence.effect_size_measure or "ES"
    significant = p < threshold

    # Clean up groups — filter nan, limit to distinct labels
    raw_groups = evidence.groups or []
    clean_groups = [g for g in raw_groups if g and str(g).lower() not in ("nan", "none")]
    # For chi-square tests the groups list is the contingency axes — collapse to a short label
    if len(clean_groups) > 4:  # noqa: PLR2004
        groups_str = f"{len(clean_groups)} categories"
    else:
        groups_str = " vs. ".join(clean_groups) if clean_groups else "groups"

    metric = evidence.metric or "metric"
    # Make metric human-readable
    metric_label = (
        metric.replace("_", " ")
        .replace("trader win rate 7d", "7-day rolling win rate")
        .replace("sentiment value lag 1d", "1-day lagged sentiment")
        .replace("trader pnl rolling 7d", "7-day rolling PnL")
        .replace("trader avg size usd 7d", "7-day average position size (USD)")
        .replace("trader leverage avg 24h", "24-hour average leverage")
    )

    if significant:
        return (
            f"A statistically significant relationship was detected for {metric_label} "
            f"({groups_str}): p={p:.4f}, {es_measure}={es:.4f} "
            f"(corrected α={threshold:.5f})."
        )
    return (
        f"No statistically significant relationship was found between {metric_label} "
        f"and sentiment regime ({groups_str}): p={p:.4f}, {es_measure}={es:.4f} "
        f"— the observed difference is consistent with chance variation "
        f"(corrected α={threshold:.5f})."
    )


def build_insight(
    evidence: StatTestResult,
    interpretation: str,
    recommendation: str,
    limitation: str,
    insight_id: str | None = None,
    title: str | None = None,
) -> BusinessInsight:
    """Build a :class:`BusinessInsight` from statistical evidence and prose.

    Automatically populates the ``observation`` field using the evidence
    metadata and evaluates ``report_ready`` via :func:`validate_insight`.

    Args:
        evidence: The :class:`StatTestResult` from Phase 06.
        interpretation: Business interpretation (what it means).
        recommendation: Practical recommendation (what to do).
        limitation: Limitations and caveats.
        insight_id: Optional identifier; auto-generated from test_name if
            not provided.
        title: Optional human-readable title; auto-generated from insight_id
            if not provided.

    Returns:
        A :class:`BusinessInsight` with auto-populated observation and
        report_ready flag.
    """
    threshold = evidence.corrected_threshold or 0.05
    observation = _build_observation(evidence, threshold)

    if insight_id is None:
        insight_id = f"INS-{evidence.test_name.replace('HT-', '')}"

    resolved_title = title or _INSIGHT_TITLES.get(insight_id, insight_id)

    insight = BusinessInsight(
        insight_id=insight_id,
        title=resolved_title,
        observation=observation,
        statistical_evidence=evidence,
        business_interpretation=interpretation,
        practical_recommendation=recommendation,
        limitation=limitation,
        report_ready=False,
    )

    insight.report_ready = validate_insight(insight)
    return insight


def validate_insight(insight: BusinessInsight) -> bool:
    """Validate that a :class:`BusinessInsight` has all required components.

    Checks:
        1. All five string components are non-empty.
        2. The underlying ``effect_size`` in statistical_evidence is not None.

    Sets ``insight.report_ready`` to the result of validation.

    Args:
        insight: The :class:`BusinessInsight` to validate.

    Returns:
        ``True`` if the insight is report-ready, ``False`` otherwise.
    """
    required_text_fields = [
        ("observation", insight.observation),
        ("business_interpretation", insight.business_interpretation),
        ("practical_recommendation", insight.practical_recommendation),
        ("limitation", insight.limitation),
        ("insight_id", insight.insight_id),
    ]

    for field_name, value in required_text_fields:
        if not isinstance(value, str) or not value.strip():
            logger.warning(
                "Insight %s validation failed: field '%s' is empty",
                insight.insight_id,
                field_name,
            )
            insight.report_ready = False
            return False

    if insight.statistical_evidence.effect_size is None:
        logger.warning(
            "Insight %s validation failed: effect_size is None",
            insight.insight_id,
        )
        insight.report_ready = False
        return False

    insight.report_ready = True
    return True


def _serialize_evidence(evidence: StatTestResult) -> dict[str, Any]:
    """Serialize a :class:`StatTestResult` to a dict for Markdown rendering."""
    return {
        "test_name": evidence.test_name,
        "metric": evidence.metric,
        "groups": evidence.groups,
        "statistic": evidence.statistic,
        "p_value": evidence.p_value,
        "effect_size": evidence.effect_size,
        "effect_size_measure": evidence.effect_size_measure,
        "confidence_interval_95": evidence.confidence_interval_95,
        "sample_sizes": evidence.sample_sizes,
        "normality_rejected": evidence.normality_rejected,
        "correction_applied": evidence.correction_applied,
        "corrected_threshold": evidence.corrected_threshold,
        "underpowered": evidence.underpowered,
    }


def _insight_to_markdown(insight: BusinessInsight) -> str:
    """Render a single :class:`BusinessInsight` as a Markdown section."""
    ev = insight.statistical_evidence
    ci_low, ci_high = ev.confidence_interval_95
    sig_flag = (
        "**Significant**" if ev.p_value < (ev.corrected_threshold or 0.05) else "Not significant"
    )

    # Clean groups: filter nan/None, limit display
    raw_groups = ev.groups or []
    clean_groups = [g for g in raw_groups if g and str(g).lower() not in ("nan", "none")]
    groups_display = ", ".join(clean_groups) if clean_groups else "—"

    section_title = insight.title if insight.title else (ev.metric or "Analysis")

    lines = [
        f"### {insight.insight_id}: {section_title}",
        "",
        "---",
        "",
        "## Observation",
        "",
        insight.observation,
        "",
        "## Statistical Evidence",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Test | {ev.test_name} |",
        f"| Metric | {ev.metric} |",
        f"| Groups | {groups_display} |",
        f"| Statistic | {ev.statistic:.4f} |",
        f"| p-value | {ev.p_value:.6f} |",
        f"| Significance | {sig_flag} |",
        f"| Effect Size | {ev.effect_size:.4f} ({ev.effect_size_measure}) |",
        f"| 95% CI | ({ci_low:.4f}, {ci_high:.4f}) |",
        f"| Sample Sizes | {ev.sample_sizes} |",
        f"| Normality Rejected | {ev.normality_rejected} |",
        f"| Correction Applied | {ev.correction_applied or 'None'} |",
        f"| Corrected Threshold | {ev.corrected_threshold or 'N/A'} |",
        f"| Underpowered | {ev.underpowered} |",
        "",
        "## Business Interpretation",
        "",
        insight.business_interpretation,
        "",
        "## Practical Recommendation",
        "",
        insight.practical_recommendation,
        "",
        "## Limitation",
        "",
        insight.limitation,
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def export_insights_to_markdown(
    insights: list[BusinessInsight],
    output_path: Path,
) -> None:
    """Export a list of validated insights to a formatted Markdown document.

    Only insights with ``report_ready: True`` are included. The document
    contains a header, metadata table, and the five-part body for each
    insight.

    Args:
        insights: List of :class:`BusinessInsight` objects to export.
        output_path: Filesystem path for the output ``.md`` file.

    Raises:
        ValueError: If no report-ready insights are provided.
    """
    report_ready = [i for i in insights if i.report_ready]
    if not report_ready:
        raise ValueError(
            "Cannot export: no report-ready insights. "
            "Run validate_insight on each insight first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "# Business Insights Draft",
        "",
        f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Report-Ready Insights:** {len(report_ready)} / {len(insights)}",
        "",
        "---",
        "",
    ]

    toc_parts = ["## Table of Contents", ""]
    body_sections = []

    # Group insights by act
    acts: dict[str, list[BusinessInsight]] = {
        "Act I — The Setup": [],
        "Act II — The Conflict": [],
        "Act III — The Resolution": [],
        "Additional Findings": [],
    }

    for ins in report_ready:
        act = _NARRATIVE_ACTS.get(ins.insight_id, "Additional Findings")
        if act not in acts:
            acts["Additional Findings"].append(ins)
        else:
            acts[act].append(ins)

    for act_name, act_insights in acts.items():
        if not act_insights:
            continue

        toc_parts.append(f"### {act_name}")
        body_sections.append(f"# {act_name}")
        body_sections.append("")

        for ins in act_insights:
            anchor = ins.insight_id.lower()
            display_title = (
                ins.title if ins.title else (ins.statistical_evidence.metric or ins.insight_id)
            )
            toc_parts.append(f"- [{ins.insight_id}: {display_title}](#{anchor})")
            body_sections.append(_insight_to_markdown(ins))

        toc_parts.append("")

    toc_parts.append("---")
    toc_parts.append("")

    all_lines = header + toc_parts + body_sections
    content = "\n".join(all_lines)

    output_path.write_text(content, encoding="utf-8")
    logger.info(
        "Exported %d report-ready insights to %s",
        len(report_ready),
        output_path,
    )


def load_statistical_results(stats_dir: Path) -> dict[str, StatTestResult]:
    """Ingest both core hypothesis tests and extended analyses results.

    Args:
        stats_dir: Directory containing 'all_hypothesis_tests.csv' and
            'extended_analyses_results.csv'.

    Returns:
        Dictionary mapping test_name (e.g., 'HT-01', 'ET-01') to StatTestResult.
    """
    import pandas as pd

    results: dict[str, StatTestResult] = {}
    files_to_load = [
        stats_dir / "all_hypothesis_tests.csv",
        stats_dir / "extended_analyses_results.csv",
    ]

    for path in files_to_load:
        if not path.exists():
            logger.warning("Could not find results file: %s", path)
            continue

        df = pd.read_csv(path)
        for _, row in df.iterrows():
            test_name = str(row["test_name"])

            # Skip duplicated lags/results if we only want the primary test summary
            if test_name in results:
                continue

            groups = str(row["groups"]).split("|") if pd.notna(row.get("groups")) else ["unknown"]

            sample_sizes = {}
            if pd.notna(row.get("sample_sizes")):
                for part in str(row["sample_sizes"]).split(";"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        v_clean = v.replace("-", "").strip()
                        sample_sizes[k.strip()] = int(v_clean) if v_clean.isdigit() else 0

            ci_str = str(row.get("confidence_interval_95", "")).strip("()").replace(" ", "")
            ci_parts = ci_str.split(",") if ci_str else []
            ci_lower = float(ci_parts[0]) if len(ci_parts) > 0 and ci_parts[0] else 0.0
            ci_upper = float(ci_parts[1]) if len(ci_parts) > 1 and ci_parts[1] else 0.0

            results[test_name] = StatTestResult(
                test_name=test_name,
                metric=str(row["metric"]),
                groups=groups,
                statistic=(
                    float(row["statistic"]) if pd.notna(row.get("statistic")) else float("nan")
                ),
                p_value=float(row["p_value"]) if pd.notna(row.get("p_value")) else float("nan"),
                effect_size=(
                    float(row["effect_size"]) if pd.notna(row.get("effect_size")) else float("nan")
                ),
                effect_size_measure=str(row.get("effect_size_measure", "")),
                confidence_interval_95=(ci_lower, ci_upper),
                sample_sizes=sample_sizes,
                normality_rejected=bool(row.get("normality_rejected", False)),
                correction_applied=str(row.get("correction_applied", "")) or None,
                corrected_threshold=(
                    float(row["corrected_threshold"])
                    if pd.notna(row.get("corrected_threshold"))
                    else None
                ),
                underpowered=bool(row.get("underpowered", False)),
            )

    return results
