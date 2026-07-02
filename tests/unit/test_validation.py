"""Unit tests for the validation layer.

Tests cover Pandera schema definitions, schema-check wrappers, and
data quality check functions for both Fear & Greed and Trader History
datasets.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data.metadata.schemas.fear_greed_schema import fear_greed_schema
from data.metadata.schemas.trader_history_schema import trader_history_schema
from sentiment_trader_analytics.config import (
    AppConfig,
    IngestionConfig,
    ValidationConfig,
)
from sentiment_trader_analytics.validation.quality_checks import (
    QualityReport,
    check_temporal_coverage,
    check_trader_account_cardinality,
    check_value_distribution,
)
from sentiment_trader_analytics.validation.schema_checks import (
    ValidationResult,
    validate_fear_greed,
    validate_trader_history,
)

FIXTURES = Path("tests/fixtures")


# ── helpers ──────────────────────────────────────────────────────────


def _fg_ingestion_config(tmp_path: Path, filename: str) -> IngestionConfig:
    return IngestionConfig(
        fear_greed_path=str(FIXTURES / filename),
        lineage_output_dir=str(tmp_path / "lineage"),
    )


def _th_ingestion_config(tmp_path: Path, filename: str) -> IngestionConfig:
    return IngestionConfig(
        trader_history_path=str(FIXTURES / filename),
        lineage_output_dir=str(tmp_path / "lineage"),
    )


# ── Schema Definitions ──────────────────────────────────────────────


class TestFearGreedSchema:
    """Tests for ``fear_greed_schema`` structure."""

    def test_schema_imports(self) -> None:
        assert fear_greed_schema is not None
        assert "timestamp" in fear_greed_schema.columns
        assert "value" in fear_greed_schema.columns
        assert "classification" in fear_greed_schema.columns

    def test_timestamp_column_constraints(self) -> None:
        col = fear_greed_schema.columns["timestamp"]
        assert col.nullable is False
        assert col.unique is True

    def test_value_column_constraints(self) -> None:
        col = fear_greed_schema.columns["value"]
        assert col.nullable is False

    def test_classification_column_constraints(self) -> None:
        col = fear_greed_schema.columns["classification"]
        assert col.nullable is False

    def test_schema_has_cross_column_check(self) -> None:
        check_names = [c.name for c in fear_greed_schema.checks]
        assert "classification_value_consistency" in check_names


class TestTraderHistorySchema:
    """Tests for ``trader_history_schema`` structure."""

    REQUIRED_COLS = [
        "Trade ID",
        "Account",
        "Timestamp",
        "Side",
        "Direction",
        "Size USD",
        "Execution Price",
        "Closed PnL",
        "Fee",
    ]

    def test_schema_imports(self) -> None:
        assert trader_history_schema is not None
        for col in self.REQUIRED_COLS:
            assert col in trader_history_schema.columns, f"Missing column: {col}"

    def test_trade_id_unique(self) -> None:
        col = trader_history_schema.columns["Trade ID"]
        assert col.unique is True
        assert col.nullable is False

    def test_non_nullable_columns(self) -> None:
        non_nullable = [
            "Trade ID",
            "Account",
            "Timestamp",
            "Side",
            "Direction",
            "Size USD",
            "Execution Price",
        ]
        for name in non_nullable:
            assert (
                trader_history_schema.columns[name].nullable is False
            ), f"{name} should be non-nullable"

    def test_nullable_columns(self) -> None:
        nullable = ["Closed PnL", "Fee", "Leverage"]
        for name in nullable:
            col = trader_history_schema.columns[name]
            assert col.nullable is True, f"{name} should be nullable"

    def test_leverage_optional(self) -> None:
        col = trader_history_schema.columns["Leverage"]
        assert col.required is False


# ── Schema Checks: validate_fear_greed ──────────────────────────────


class TestValidateFearGreed:
    """Tests for :func:`validate_fear_greed` using fixture files."""

    def test_valid_data_passes(self, tmp_path: Path) -> None:
        cfg = _fg_ingestion_config(tmp_path, "fear_greed_valid.csv")
        df = pd.read_csv(cfg.fear_greed_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["value"] = df["value"].astype(np.int64)
        df["classification"] = df["classification"].astype(
            pd.CategoricalDtype(
                categories=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
                ordered=True,
            )
        )

        result = validate_fear_greed(df, fear_greed_schema)

        assert result.passed is True, f"Expected pass, got violations: {result.violations}"
        assert result.violation_count == 0
        assert result.violations == []
        assert result.validated_df is not None
        assert len(result.validated_df) == 10

    def test_invalid_data_fails(self, tmp_path: Path) -> None:
        cfg = _fg_ingestion_config(tmp_path, "fear_greed_invalid.csv")
        df = pd.read_csv(cfg.fear_greed_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["value"] = df["value"].astype(np.int64)
        df["classification"] = df["classification"].astype(
            pd.CategoricalDtype(
                categories=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
                ordered=True,
            )
        )

        result = validate_fear_greed(df, fear_greed_schema)

        assert result.passed is False
        assert result.violation_count > 0
        assert result.validated_df is None

        violation_rules = {str(v["rule"]) for v in result.violations}
        violation_columns = {str(v["column"]) for v in result.violations}

        has_value_violation = "value" in violation_columns or any(
            "in_range" in r for r in violation_rules
        )
        has_classification_violation = (
            "classification" in violation_columns
            or "classification_value_consistency" in violation_rules
            or any("consistency" in r for r in violation_rules)
        )
        assert has_value_violation, f"Expected value range violation. rules={violation_rules}"
        assert (
            has_classification_violation
        ), f"Expected classification consistency violation. rules={violation_rules}"

    def test_wrong_dtype_fails(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2018-02-01", "2018-02-02"], utc=True),
                "value": pd.Series([30, 101], dtype=np.int64),
                "classification": pd.Series(["Fear", "Extreme Greed"], dtype=object),
            }
        )
        result = validate_fear_greed(df, fear_greed_schema)
        assert result.passed is False
        assert result.violation_count > 0

    def test_empty_dataframe_passes(self) -> None:
        df = pd.DataFrame(columns=["timestamp", "value", "classification"]).astype(
            {"timestamp": "datetime64[ns, UTC]", "value": np.int64, "classification": object}
        )
        result = validate_fear_greed(df, fear_greed_schema)
        assert result.passed is True
        assert result.violation_count == 0


# ── Schema Checks: validate_trader_history ──────────────────────────


class TestValidateTraderHistory:
    """Tests for :func:`validate_trader_history` using fixture files."""

    @staticmethod
    def _load_trader_df(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, keep_default_na=False)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype(np.float64), unit="ms", utc=True)
        df["Account"] = df["Account"].astype(str)
        float_cols = ["Size USD", "Execution Price", "Closed PnL", "Fee"]
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
        if "Leverage" in df.columns:
            df["Leverage"] = pd.to_numeric(df["Leverage"], errors="coerce").astype(np.float64)
        return df

    def test_valid_data_passes(self, tmp_path: Path) -> None:
        cfg = _th_ingestion_config(tmp_path, "trader_history_valid.csv")
        df = self._load_trader_df(cfg.trader_history_path)

        result = validate_trader_history(df, trader_history_schema)

        assert result.passed is True, f"Expected pass, got violations: {result.violations[:3]}"
        assert result.violation_count == 0
        assert result.violations == []
        assert result.validated_df is not None
        assert len(result.validated_df) == 10

    def test_invalid_data_fails(self, tmp_path: Path) -> None:
        cfg = _th_ingestion_config(tmp_path, "trader_history_invalid.csv")
        df = self._load_trader_df(cfg.trader_history_path)

        result = validate_trader_history(df, trader_history_schema)

        assert result.passed is False
        assert result.violation_count > 0
        assert result.validated_df is None

        violation_columns = {str(v["column"]) for v in result.violations}
        violation_rules = {str(v["rule"]) for v in result.violations}

        assert "Trade ID" in violation_columns or any(
            "unique" in r for r in violation_rules
        ), f"Expected duplicate Trade ID. cols={violation_columns}"

        assert (
            "Account" in violation_columns
        ), f"Expected Account violation. cols={violation_columns}"

        assert "Side" in violation_columns, f"Expected Side violation. cols={violation_columns}"

        assert (
            "Direction" in violation_columns
        ), f"Expected Direction violation. cols={violation_columns}"

        has_size_violation = "Size USD" in violation_columns or any(
            "Size USD" in str(v.get("value", "")) for v in result.violations
        )
        assert has_size_violation, f"Expected Size USD violation. cols={violation_columns}"


# ── Quality Checks ──────────────────────────────────────────────────


class TestQualityChecksTemporalCoverage:
    """Tests for :func:`check_temporal_coverage`."""

    def test_passes_with_sufficient_coverage(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC"),
            }
        )
        config = AppConfig(validation=ValidationConfig(min_temporal_coverage_days=365))
        report = check_temporal_coverage(df, config)

        assert report.passed is True
        assert report.metric_value is not None
        assert report.metric_value >= 365.0

    def test_fails_with_insufficient_coverage(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC"),
            }
        )
        config = AppConfig(validation=ValidationConfig(min_temporal_coverage_days=365))
        report = check_temporal_coverage(df, config)

        assert report.passed is False

    def test_fails_on_missing_column(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        config = AppConfig()
        report = check_temporal_coverage(df, config)

        assert report.passed is False


class TestQualityChecksValueDistribution:
    """Tests for :func:`check_value_distribution`."""

    def test_passes_with_diverse_values_in_range(self) -> None:
        df = pd.DataFrame({"value": [10, 20, 30, 40, 50]})
        config = AppConfig()
        report = check_value_distribution(df, "value", (0, 100), config)

        assert report.passed is True

    def test_fails_on_zero_variance(self) -> None:
        df = pd.DataFrame({"value": [50, 50, 50, 50]})
        config = AppConfig()
        report = check_value_distribution(df, "value", (0, 100), config)

        assert report.passed is False
        assert "variance" in report.details.lower()

    def test_fails_on_missing_column(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        config = AppConfig()
        report = check_value_distribution(df, "missing_col", (0, 100), config)

        assert report.passed is False

    def test_fails_on_all_null(self) -> None:
        df = pd.DataFrame({"value": [None, None, None]})
        config = AppConfig()
        report = check_value_distribution(df, "value", (0, 100), config)

        assert report.passed is False


class TestQualityChecksAccountCardinality:
    """Tests for :func:`check_trader_account_cardinality`."""

    def test_passes_with_sufficient_accounts(self) -> None:
        df = pd.DataFrame({"Account": ["a", "b", "c", "d", "e"]})
        config = AppConfig(validation=ValidationConfig(min_distinct_accounts=3))
        report = check_trader_account_cardinality(df, config)

        assert report.passed is True
        assert report.metric_value == 5.0

    def test_fails_with_insufficient_accounts(self) -> None:
        df = pd.DataFrame({"Account": ["a", "a", "a"]})
        config = AppConfig(validation=ValidationConfig(min_distinct_accounts=3))
        report = check_trader_account_cardinality(df, config)

        assert report.passed is False

    def test_fails_on_missing_column(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        config = AppConfig()
        report = check_trader_account_cardinality(df, config)

        assert report.passed is False


class TestValidationResultModel:
    """Tests for the :class:`ValidationResult` Pydantic model."""

    def test_passing_result(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = ValidationResult(passed=True, validated_df=df)

        assert result.passed is True
        assert result.violation_count == 0
        assert result.violations == []
        assert result.validated_df is not None

    def test_failing_result(self) -> None:
        violations = [
            {"column": "value", "index": 5, "value": "101", "rule": "in_range"},
        ]
        result = ValidationResult(
            passed=False,
            violation_count=1,
            violations=violations,
        )

        assert result.passed is False
        assert result.violation_count == 1
        assert len(result.violations) == 1
        assert result.validated_df is None


class TestQualityReportModel:
    """Tests for the :class:`QualityReport` dataclass."""

    def test_default_construction(self) -> None:
        report = QualityReport(check_name="test_check", passed=True)

        assert report.check_name == "test_check"
        assert report.passed is True
        assert report.metric_value is None
        assert report.threshold is None
        assert report.details == ""
