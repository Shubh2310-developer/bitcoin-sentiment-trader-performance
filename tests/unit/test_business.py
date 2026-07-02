"""Unit tests for the business insight synthesis layer.

Tests cover the BusinessInsight model, build_insight,
validate_insight, and export_insights_to_markdown.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentiment_trader_analytics.business.insight_generator import (
    BusinessInsight,
    build_insight,
    export_insights_to_markdown,
    validate_insight,
)
from sentiment_trader_analytics.statistics.hypothesis_tests import StatTestResult

# ── Helpers ──────────────────────────────────────────────────────────


def _make_evidence(
    effect_size: float | None = 0.5,
    p_value: float = 0.03,
    test_name: str = "HT-01",
    metric: str = "Closed PnL",
    groups: list[str] | None = None,
) -> StatTestResult:
    return StatTestResult(
        test_name=test_name,
        metric=metric,
        groups=groups or ["Fear", "Greed"],
        statistic=132845.5,
        p_value=p_value,
        effect_size=effect_size,
        effect_size_measure="rank_biserial_r",
        confidence_interval_95=(-2.93, 0.0),
        sample_sizes={"Fear": 50, "Greed": 50},
        normality_rejected=True,
        underpowered=False,
    )


# ── Tests ─────────────────────────────────────────────────────────────


class TestBuildInsight:
    """Tests for :func:`build_insight`."""

    def test_build_insight_populates_observation(self) -> None:
        """Constructor should auto-generate a plain observation."""
        evidence = _make_evidence()
        insight = build_insight(
            evidence=evidence,
            interpretation="Traders may underperform during Fear.",
            recommendation="Consider position-size limits in Fear regimes.",
            limitation="Correlation, not causation. Confounders may apply.",
        )
        assert isinstance(insight.observation, str)
        assert len(insight.observation) > 0
        assert "Closed PnL" in insight.observation
        assert "statistically significant" in insight.observation

    def test_build_insight_non_significant_observation(self) -> None:
        """Observation should reflect non-significant results correctly."""
        evidence = _make_evidence(p_value=0.45, effect_size=0.02)
        insight = build_insight(
            evidence=evidence,
            interpretation="No clear signal.",
            recommendation="No action needed.",
            limitation="Underpowered sample.",
        )
        assert "No statistically significant" in insight.observation

    def test_build_insight_sets_report_ready(self) -> None:
        """build_insight should set report_ready correctly."""
        evidence = _make_evidence(effect_size=0.3)
        insight = build_insight(
            evidence=evidence,
            interpretation="Test interpretation.",
            recommendation="Test recommendation.",
            limitation="Test limitation.",
        )
        assert insight.report_ready is True


class TestValidateInsight:
    """Tests for :func:`validate_insight`."""

    def test_validate_insight_requires_all_fields(self) -> None:
        """Should return False if any of the five components are empty."""
        evidence = _make_evidence()
        insight = BusinessInsight(
            insight_id="INS-01",
            observation="Test observation.",
            statistical_evidence=evidence,
            business_interpretation="",
            practical_recommendation="Valid recommendation.",
            limitation="Valid limitation.",
            report_ready=False,
        )
        assert validate_insight(insight) is False

    def test_validate_insight_requires_effect_size(self) -> None:
        """Should return False if effect_size is None."""
        evidence = _make_evidence(effect_size=0.5)
        insight = BusinessInsight(
            insight_id="INS-01",
            observation="Test observation.",
            statistical_evidence=evidence,
            business_interpretation="Valid interpretation.",
            practical_recommendation="Valid recommendation.",
            limitation="Valid limitation.",
            report_ready=False,
        )
        evidence.__dict__["effect_size"] = None  # bypass Pydantic validation
        assert validate_insight(insight) is False

    def test_validate_insight_all_valid(self) -> None:
        """Should return True when all components are present."""
        evidence = _make_evidence(effect_size=0.5)
        insight = BusinessInsight(
            insight_id="INS-01",
            observation="Valid observation.",
            statistical_evidence=evidence,
            business_interpretation="Valid interpretation.",
            practical_recommendation="Valid recommendation.",
            limitation="Valid limitation.",
            report_ready=False,
        )
        assert validate_insight(insight) is True

    def test_validate_empty_business_interpretation(self) -> None:
        """Should return False when business_interpretation is blank."""
        evidence = _make_evidence(effect_size=0.3)
        insight = BusinessInsight(
            insight_id="INS-01",
            observation="Obs.",
            statistical_evidence=evidence,
            business_interpretation="   ",
            practical_recommendation="Rec.",
            limitation="Lim.",
            report_ready=False,
        )
        assert validate_insight(insight) is False


class TestExportInsights:
    """Tests for :func:`export_insights_to_markdown`."""

    def test_export_insights_formatting(self, tmp_path: Path) -> None:
        """Markdown export should correctly header the five parts."""
        evidence = _make_evidence(effect_size=0.5)
        insight = build_insight(
            evidence=evidence,
            interpretation="Test interpretation.",
            recommendation="Test recommendation.",
            limitation="Test limitation.",
        )
        output_path = tmp_path / "insights_draft.md"
        export_insights_to_markdown([insight], output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "## Observation" in content
        assert "## Statistical Evidence" in content
        assert "## Business Interpretation" in content
        assert "## Practical Recommendation" in content
        assert "## Limitation" in content

    def test_export_raises_on_no_report_ready(self, tmp_path: Path) -> None:
        """Export should raise if no insights are report-ready."""
        evidence = _make_evidence(effect_size=0.5)
        evidence.__dict__["effect_size"] = None  # bypass Pydantic validation
        insight = BusinessInsight(
            insight_id="INS-01",
            observation="Obs.",
            statistical_evidence=evidence,
            business_interpretation="Interp.",
            practical_recommendation="Rec.",
            limitation="Lim.",
            report_ready=False,
        )
        output_path = tmp_path / "insights_draft.md"
        with pytest.raises(ValueError, match="no report-ready insights"):
            export_insights_to_markdown([insight], output_path)
