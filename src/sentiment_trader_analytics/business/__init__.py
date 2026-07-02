"""Business insight synthesis layer for the sentiment trader pipeline.

Translates statistical test results from Phase 06 into structured
business insights with the mandatory five-part format. Provides
validation and Markdown export for stakeholder reporting.
"""

from sentiment_trader_analytics.business.insight_generator import (
    BusinessInsight,
    build_insight,
    export_insights_to_markdown,
    validate_insight,
)

__all__ = [
    "BusinessInsight",
    "build_insight",
    "export_insights_to_markdown",
    "validate_insight",
]
