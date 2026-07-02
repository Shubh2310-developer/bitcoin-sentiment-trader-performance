# Data Dictionary
## Repository: bitcoin-sentiment-trader-performance
### Feature Engineering & Documentation Standards

---

## 1. Purpose

This document catalogs every dataset column and engineered feature used in the research pipeline. For each field, it specifies: name, source, data type, description, valid range, units, known limitations, and lineage.

This document is the authoritative reference for field names throughout the codebase. Any discrepancy between code and this document should be resolved by updating this document and the code simultaneously via a tracked commit.

---

## 2. Raw Datasets

### 2.1 Fear & Greed Index (`data/raw/fear_greed/fear_greed_index.csv`)

| Column | Type | Nullable | Description | Valid Range / Domain | Units | Notes |
|---|---|---|---|---|---|---|
| `timestamp` | `datetime64[UTC]` | No | Date of the sentiment reading (normalized to UTC at ingestion) | Parseable ISO date string | UTC date | Must be unique per row |
| `value` | `int64` | No | Numeric Fear & Greed Index value | [0, 100] | Dimensionless | 0 = Extreme Fear, 100 = Extreme Greed |
| `classification` | `str` (categorical) | No | Text classification of the sentiment value | `{"Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"}` | — | Must be consistent with `value` per regime thresholds |

**Validation rules:**
- `timestamp` must be parseable; duplicates by date are dropped with logging
- `value` must be in [0, 100]; values outside this range halt the pipeline
- `classification` must be one of the five valid categories and must be consistent with the value range for that category

---

### 2.2 Trader History (`data/raw/trader_history/historical_data.csv`)

| Column | Type | Nullable | Description | Valid Range / Domain | Units | Notes |
|---|---|---|---|---|---|---|
| `Trade ID` | `str` / `int64` | No | Unique trade identifier (primary key) | Unique; non-null | — | Duplicate Trade IDs halt the pipeline |
| `Account` | `str` | No | Trader account identifier | Non-null, non-empty string | — | Primary grouping key for trader-level analysis |
| `Timestamp` | `datetime64[UTC]` | No | Trade execution timestamp (normalized to UTC) | Parseable; UTC | — | Naive datetimes are rejected at ingestion |
| `Side` | `str` (categorical) | No | Trade side | `{"Long", "Short"}` | — | — |
| `Direction` | `str` (categorical) | No | Trade direction (open/close) | `{"Open", "Close"}` | — | — |
| `Size USD` | `float64` | No | Notional trade size in USD | ≥ 0.0 | USD | Negative values rejected at validation |
| `Execution Price` | `float64` | No | Trade execution price | > 0.0 | USD per unit | Zero or negative rejected at validation |
| `Closed PnL` | `float64` | Yes | Realized profit/loss for closed positions | Any real number | USD | Null for open positions; not imputed |
| `Fee` | `float64` | Yes | Trading fee paid | ≥ 0.0 | USD | — |
| `Leverage` | `float64` | Yes | Leverage multiplier at time of trade | > 0.0 | x (multiplier) | — |

---

## 3. Processed Dataset (`data/processed/`)

The processed dataset is the validated, cleaned, timezone-normalized, and joined output of both raw datasets. It contains all raw columns from both sources (with column name disambiguation where needed) plus the following join-derived columns:

| Column | Type | Nullable | Description | Source | Notes |
|---|---|---|---|---|---|
| `trade_date_utc` | `date` | No | UTC calendar date extracted from `Timestamp` | Derived from `Trader History.Timestamp` | Join key against sentiment date |
| `sentiment_value` | `int64` | Yes | Fear & Greed value for the trade's date | `Fear & Greed.value` | Null if no sentiment record for that date |
| `sentiment_classification` | `str` | Yes | Regime classification for the trade's date | `Fear & Greed.classification` | Null if no sentiment record for that date |

---

## 4. Engineered Features (`data/features/`)

All features are computed by pure, stateless functions in `src/sentiment_trader_analytics/feature_engineering/`. Naming convention: `<domain>_<metric>_<window>` (all lowercase, snake_case).

### 4.1 Sentiment Features (`sentiment_features.py`)

| Feature | Type | Description | Formula / Logic | Window | Look-Ahead Risk | Units |
|---|---|---|---|---|---|---|
| `sentiment_value_lag_1d` | `float64` | Sentiment value from 1 day prior | `sentiment_value.shift(1)` by date | 1-day lag | None — strict lag | Dimensionless [0, 100] |
| `sentiment_regime_encoded` | `int8` | Ordinal encoding of sentiment classification | `{Extreme Fear: 0, Fear: 1, Neutral: 2, Greed: 3, Extreme Greed: 4}` | — | None | Integer ordinal |
| `sentiment_value_rolling_7d` | `float64` | 7-day rolling mean of sentiment value | `sentiment_value.rolling(7, min_periods=1).mean()` | 7 days | None — historical window only | Dimensionless |
| `sentiment_is_fear` | `bool` | Binary indicator: regime is Fear or Extreme Fear | `sentiment_classification in {"Fear", "Extreme Fear"}` | — | None | Boolean |
| `sentiment_is_greed` | `bool` | Binary indicator: regime is Greed or Extreme Greed | `sentiment_classification in {"Greed", "Extreme Greed"}` | — | None | Boolean |

### 4.2 Trader Features (`trader_features.py`)

All trader features operate per-account (`.groupby("Account")`) and use strictly historical windows (`closed="left"`) to prevent look-ahead bias.

| Feature | Type | Description | Formula / Logic | Window | Look-Ahead Risk | Units |
|---|---|---|---|---|---|---|
| `trader_win_rate_7d` | `float64` | Rolling 7-day win rate per account | `is_close & (Closed PnL > 0)` → rolling fraction, per account, `closed="left"` | 7 rows | None — `closed="left"` | Ratio [0.0, 1.0] |
| `trader_pnl_rolling_7d` | `float64` | Rolling 7-day total realized PnL per account | `Closed PnL.rolling(7, closed="left").sum()` per account | 7 rows | None — `closed="left"` | USD |
| `trader_pnl_rolling_30d` | `float64` | Rolling 30-day total realized PnL per account | `Closed PnL.rolling(30, closed="left").sum()` per account | 30 rows | None — `closed="left"` | USD |
| `trader_leverage_avg_24h` | `float64` | Rolling 5-row average leverage per account | `Leverage.rolling(5, closed="left").mean()` per account. Leverage derived as `(Size Tokens × Execution Price) / Size USD` if not present | 5 rows | None — `closed="left"` | x (multiplier) |
| `trader_pnl_volatility_14d` | `float64` | Rolling 14-day std dev of closed PnL per account | `Closed PnL.rolling(14, closed="left").std()` per account | 14 rows | None — `closed="left"` | USD |
| `trader_trade_count_7d` | `float64` | Rolling 7-day trade count per account | Per-account row count over rolling 7 rows, `closed="left"` | 7 rows | None — `closed="left"` | Count |
| `trader_avg_size_usd_7d` | `float64` | Rolling 7-day mean position size per account | `Size USD.rolling(7, closed="left").mean()` per account | 7 rows | None — `closed="left"` | USD |

**Cold-start handling:** All rolling trader features produce `NaN` for the first window of each account's history. These rows are **retained** and flagged via the `feature_cold_start` boolean column. Statistical analyses downstream should exclude cold-start rows when computing regime comparisons.

| Feature | Type | Description | Formula / Logic | Window | Look-Ahead Risk | Units |
|---|---|---|---|---|---|---|
| `feature_cold_start` | `bool` | Flag indicating rolling trader features have not yet warmed up | `True` if ANY `trader_*` column is `NaN` for that row | — | None | Boolean |

### 4.3 Time Features (`time_features.py`)

| Feature | Type | Description | Formula / Logic | Window | Look-Ahead Risk | Units |
|---|---|---|---|---|---|---|
| `time_hour_utc` | `int8` | UTC hour of the trade timestamp | `Timestamp.hour` | — | None | Integer [0, 23] |
| `time_day_of_week` | `int8` | Day of week (0=Monday, 6=Sunday) | `Timestamp.dayofweek` | — | None | Integer [0, 6] |
| `time_is_weekend` | `bool` | Whether the trade occurred on a weekend | `day_of_week in {5, 6}` | — | None | Boolean |
| `time_month` | `int8` | Month of the trade | `Timestamp.month` | — | None | Integer [1, 12] |

---

## 5. Known Limitations and Caveats

| Issue | Affected Features/Datasets | Impact | Mitigation |
|---|---|---|---|
| Fear & Greed is published once daily | All sentiment features | Intraday variation in sentiment is not captured | Acknowledged in all reports; all sentiment analysis is daily-grain |
| Trader dataset covers only Hyperliquid | All trader features | Findings do not generalize to other venues without validation | Explicitly disclosed in executive summary |
| Rolling features have cold-start periods | All rolling window features | Early rows in each account's history have `NaN` or unreliable estimates | `cold-start` rows are retained and flagged via `feature_cold_start` column; excluded from statistical tests where sample validity requires a minimum window |
| Closed PnL is null for open positions | `trader_pnl_*` features | Win rate and PnL metrics exclude open/in-progress trades | Documented; analysis is restricted to completed trades unless otherwise specified |
| Leverage derived from trade fields | `trader_leverage_avg_24h` | No `Leverage` column exists in raw data; derived as `(Size Tokens × Execution Price) / Size USD` | This is the standard back-calculation for perpetual futures margin. Documented limitation: approximates per-trade leverage |
| Time-based rolling not feasible | `trader_leverage_avg_24h` | Processed dataset has only 2–4 unique timestamps per account after dedup | Fallback to 5-row rolling window (row-count-based, `closed="left"`) instead of time-based 24h |

---

## 6. Metadata Generation

Per every pipeline run producing `data/processed/` or `data/features/` artifacts writes the following to `data/metadata/`:

| Artifact | Location | Content |
|---|---|---|
| Schema snapshot | `data/metadata/schemas/<dataset>_<run_id>.json` | Column names, dtypes, constraints |
| Row count / null count | `data/metadata/data_dictionary/<dataset>_<run_id>.json` | Row count, null counts by column, generation timestamp |
| Lineage record | `data/metadata/lineage/<dataset>_<run_id>.json` | Source files, config hash, code version (git SHA) |

---


