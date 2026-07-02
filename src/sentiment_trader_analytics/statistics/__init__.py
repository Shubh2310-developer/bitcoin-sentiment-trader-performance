"""Statistical analysis layer for the sentiment trader pipeline.

Provides hypothesis testing (normality, two-group, multi-group, chi-square),
correlation analysis, and multiple testing correction facilities.
"""

from sentiment_trader_analytics.statistics.correlation_analysis import (
    CorrelationResult,
    compute_correlation,
    compute_correlation_matrix,
)
from sentiment_trader_analytics.statistics.hypothesis_tests import (
    NormalityResult,
    StatTestResult,
    apply_multiple_testing_correction,
    check_normality,
    chi_square_test,
    compare_multiple_groups,
    compare_two_groups,
)

__all__ = [
    "CorrelationResult",
    "NormalityResult",
    "StatTestResult",
    "apply_multiple_testing_correction",
    "check_normality",
    "chi_square_test",
    "compare_multiple_groups",
    "compare_two_groups",
    "compute_correlation",
    "compute_correlation_matrix",
]
