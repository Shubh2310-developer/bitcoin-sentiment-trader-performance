"""Configuration models for the sentiment trader analytics pipeline."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    """Configuration for the data ingestion pipeline stage.

    Attributes:
        fear_greed_path: Path to the Fear & Greed Index CSV file.
        trader_history_path: Path to the Trader History CSV file.
        chunk_size: Number of rows per chunk for large file reading.
        lineage_output_dir: Directory for writing lineage metadata.
    """

    fear_greed_path: Path = Field(
        default=Path("data/raw/fear_greed/fear_greed_index.csv"),
        description="Path to the Fear & Greed Index CSV file",
    )
    trader_history_path: Path = Field(
        default=Path("data/raw/trader_history/historical_data.csv"),
        description="Path to the Trader History CSV file",
    )
    chunk_size: int = Field(
        default=50000,
        description="Number of rows per chunk for chunked CSV reading",
        ge=1000,
    )
    lineage_output_dir: Path = Field(
        default=Path("data/metadata/lineage"),
        description="Directory for writing lineage/checksum metadata",
    )


class ValidationConfig(BaseModel):
    """Configuration for the validation pipeline stage.

    Attributes:
        min_temporal_coverage_days: Minimum number of days of data required.
        min_distinct_accounts: Minimum number of distinct trader accounts required.
        expected_value_range: Expected (min, max) range for sentiment values.
    """

    min_temporal_coverage_days: int = Field(
        default=365,
        ge=1,
        description="Minimum days of data coverage required",
    )
    min_distinct_accounts: int = Field(
        default=3,
        ge=1,
        description="Minimum distinct trader accounts required",
    )
    expected_value_range: tuple[int, int] = Field(
        default=(0, 100),
        description="Expected range for sentiment values",
    )


class PreprocessingConfig(BaseModel):
    """Configuration for the preprocessing pipeline stage.

    Attributes:
        interim_output_format: Output format for interim data (parquet or csv).
        processed_output_path: Path for processed output directory.
        fan_out_tolerance_fraction: Allowed fractional row increase from join.
        null_strategy_fear_greed: Null handling for Fear & Greed (drop or flag).
        null_strategy_trader: Null handling for trader history (drop, flag, halt).
    """

    interim_output_format: str = Field(
        default="parquet",
        pattern="^(parquet|csv)$",
        description="Output format for interim cleaned data",
    )
    processed_output_path: Path = Field(
        default=Path("data/processed"),
        description="Directory for processed output",
    )
    fan_out_tolerance_fraction: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Tolerable fractional row increase from merge fan-out",
    )
    null_strategy_fear_greed: str = Field(
        default="drop",
        pattern="^(drop|flag)$",
        description="Null handling strategy for Fear & Greed fields",
    )
    null_strategy_trader: str = Field(
        default="drop",
        pattern="^(drop|flag|halt)$",
        description="Null handling strategy for trader history fields",
    )


class FeatureConfig(BaseModel):
    """Configuration for the feature engineering pipeline stage.

    Attributes:
        sentiment_lag_days: Number of days for sentiment lag (default 1).
        sentiment_rolling_window: Rolling window size for sentiment mean (default 7).
        sentiment_regime_encoding: Ordinal encoding map for sentiment classifications.
        trader_rolling_windows: Dictionary of feature -> window size in days.
        trader_leverage_window_hours: Hours for leverage rolling average (default 24).
        trader_volatility_window_days: Days for PnL volatility window (default 14).
        output_path: Directory for feature store output.
        run_id: Unique identifier for this feature engineering run.
    """

    sentiment_lag_days: int = Field(default=1, ge=1, description="Lag in days for sentiment value")
    sentiment_rolling_window: int = Field(
        default=7, ge=1, description="Rolling window for sentiment mean"
    )
    sentiment_regime_encoding: dict[str, int] = Field(
        default={
            "Extreme Fear": 0,
            "Fear": 1,
            "Neutral": 2,
            "Greed": 3,
            "Extreme Greed": 4,
        },
        description="Ordinal encoding of sentiment regimes",
    )
    trader_rolling_windows: dict[str, int] = Field(
        default={
            "win_rate": 7,
            "pnl_7d": 7,
            "pnl_30d": 30,
            "trade_count": 7,
            "avg_size": 7,
            "pnl_volatility": 14,
        },
        description="Window sizes in days for trader rolling features",
    )
    trader_leverage_window_hours: int = Field(
        default=24, ge=1, description="Hours for leverage rolling window"
    )
    output_path: Path = Field(
        default=Path("data/features"), description="Feature store output directory"
    )
    run_id: str = Field(default="", description="Run identifier (auto-generated if empty)")


class EDAConfig(BaseModel):
    """Configuration for the EDA pipeline stage.

    Attributes:
        figures_dir: Directory for saving EDA figures.
        tables_dir: Directory for saving EDA tables.
        outlier_method: Method for outlier detection (iqr or zscore).
        outlier_iqr_multiplier: IQR multiplier for outlier threshold.
        correlation_method: Correlation method (pearson or spearman).
        numeric_features: List of numeric columns to include in correlation analysis.
    """

    figures_dir: Path = Field(
        default=Path("outputs/figures/eda"),
        description="Directory for EDA figure output",
    )
    tables_dir: Path = Field(
        default=Path("outputs/tables/eda"),
        description="Directory for EDA table output",
    )
    outlier_method: str = Field(
        default="iqr",
        pattern="^(iqr|zscore)$",
        description="Outlier detection method",
    )
    outlier_iqr_multiplier: float = Field(
        default=1.5,
        ge=0.0,
        description="IQR multiplier for outlier threshold",
    )
    correlation_method: str = Field(
        default="spearman",
        pattern="^(pearson|spearman|kendall)$",
        description="Correlation method",
    )
    numeric_features: list[str] = Field(
        default=[
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
        ],
        description="Numeric features for correlation analysis",
    )


class StatConfig(BaseModel):
    """Configuration for the statistical analysis pipeline stage.

    Attributes:
        alpha: Significance level for hypothesis tests (default 0.05).
        correction_method: Method for multiple testing correction
            (bonferroni or fdr_bh).
        min_sample_size: Minimum sample size per group; tests below this
            are flagged as underpowered.
        min_expected_frequency: Minimum expected cell frequency for chi-square;
            below this, Fisher's exact test is used.
        tables_dir: Directory for saving statistical output tables.
        run_extended_analyses: Whether to run ET-01 through ET-07 extended tests.
        regime_duration_buckets: Upper bounds for regime duration bucket groupings.
        trader_segment_k_min: Minimum k for k-means trader segmentation.
        trader_segment_k_max: Maximum k for k-means trader segmentation.
        max_lag_days: Maximum lag in days for ET-05 extended lagged correlations.
        proportion_test_alpha: Significance level for proportion tests in ET-07.
    """

    alpha: float = Field(default=0.05, ge=0.0, le=1.0, description="Significance level")
    correction_method: str = Field(
        default="bonferroni",
        pattern="^(bonferroni|fdr_bh)$",
        description="Multiple testing correction method",
    )
    min_sample_size: int = Field(
        default=30,
        ge=1,
        description="Minimum sample size per group",
    )
    min_expected_frequency: float = Field(
        default=5.0,
        ge=0.0,
        description="Minimum expected frequency for chi-square",
    )
    tables_dir: Path = Field(
        default=Path("outputs/tables/statistics"),
        description="Directory for statistical output tables",
    )
    run_extended_analyses: bool = Field(
        default=True,
        description="Whether to run extended analyses ET-01 through ET-07",
    )
    regime_duration_buckets: list[int] = Field(
        default=[1, 3, 7, 14],
        description="Upper bounds for regime duration bucket groupings (days)",
    )
    trader_segment_k_min: int = Field(
        default=2,
        ge=2,
        description="Minimum k for k-means trader segmentation",
    )
    trader_segment_k_max: int = Field(
        default=6,
        ge=2,
        description="Maximum k for k-means trader segmentation",
    )
    max_lag_days: int = Field(
        default=14,
        ge=1,
        description="Maximum lag in days for extended lagged correlation analysis",
    )
    proportion_test_alpha: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Significance level for proportion tests in ET-07",
    )


class MLConfig(BaseModel):
    """Configuration for the machine learning pipeline stage.

    Attributes:
        random_seed: Global random seed for reproducibility.
        train_cutoff_date: Chronological split cutoff date (YYYY-MM-DD).
        cv_folds: Number of TimeSeriesSplit folds.
        tracking_uri: Path to MLflow tracking directory.
        experiment_name: MLflow experiment name.
    """

    random_seed: int = Field(default=42, ge=0, description="Global random seed for reproducibility")
    train_cutoff_date: str = Field(
        default="2025-04-01",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Chronological train/test split cutoff date (YYYY-MM-DD)",
    )
    cv_folds: int = Field(default=5, ge=2, le=10, description="Number of TimeSeriesSplit folds")
    tracking_uri: Path = Field(
        default=Path("experiments/mlruns"),
        description="MLflow tracking URI",
    )
    experiment_name: str = Field(
        default="ml_experiments",
        description="MLflow experiment name",
    )


class AppConfig(BaseModel):
    """Top-level application configuration.

    Wraps pipeline-specific configs under top-level keys.
    """

    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    feature_engineering: FeatureConfig = Field(default_factory=FeatureConfig)
    eda: EDAConfig = Field(default_factory=EDAConfig)
    statistics: StatConfig = Field(default_factory=StatConfig)
    ml: MLConfig = Field(default_factory=MLConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate application configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A validated AppConfig instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
