"""Feature engineering module for the sentiment trader analytics pipeline.

Provides pure, stateless functions for constructing engineered features
across three domains:
    - Sentiment features (:mod:`sentiment_features`)
    - Trader performance features (:mod:`trader_features`)
    - Time-based features (:mod:`time_features`)

All feature functions follow the signature
``f(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame``.
"""

from sentiment_trader_analytics.feature_engineering.sentiment_features import (
    add_sentiment_fear_greed_flags,
    add_sentiment_lag,
    add_sentiment_regime_encoding,
    add_sentiment_rolling_mean,
)
from sentiment_trader_analytics.feature_engineering.time_features import (
    add_day_of_week,
    add_month,
    add_time_of_day,
)
from sentiment_trader_analytics.feature_engineering.trader_features import (
    add_trader_avg_size,
    add_trader_leverage_avg,
    add_trader_pnl_rolling,
    add_trader_pnl_volatility,
    add_trader_trade_count,
    add_trader_win_rate,
    flag_cold_start_rows,
)

__all__ = [
    "add_sentiment_lag",
    "add_sentiment_regime_encoding",
    "add_sentiment_rolling_mean",
    "add_sentiment_fear_greed_flags",
    "add_trader_win_rate",
    "add_trader_pnl_rolling",
    "add_trader_leverage_avg",
    "add_trader_pnl_volatility",
    "add_trader_trade_count",
    "add_trader_avg_size",
    "flag_cold_start_rows",
    "add_time_of_day",
    "add_day_of_week",
    "add_month",
]
