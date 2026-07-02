# Analytical Methodology
## Repository: bitcoin-sentiment-trader-performance
### Documentation Standards

---

## 1. Purpose of This Document

This document records every non-obvious analytical decision made during the research pipeline — from data cleaning strategies to statistical test selection to model design choices. All entries are traceable to a specific pipeline stage, git commit, or experiment run. Results are auditable months after the fact.

---

## 2. Data Engineering Decisions

### 2.1 Raw Data Immutability
All files under `data/raw/` are treated as immutable. No in-place corrections, renaming, or overwriting is permitted. All corrections are applied downstream via documented pipeline code. This preserves the ability to re-derive any output from scratch.



### 2.2 Checksum Verification
Every raw file ingested has a SHA-256 checksum computed and recorded in `data/metadata/lineage/`. This checksum is re-verified at the start of each full pipeline run to detect silent data corruption or unauthorized modification.



### 2.3 Timezone Normalization
All timestamps in both datasets (Fear & Greed: daily date strings; Trader History: Unix or ISO timestamps) are parsed to timezone-aware `pandas.Timestamp` objects and normalized to **UTC** at ingestion. No naive datetimes persist past the ingestion stage. Any local-time display in reports is a presentation-layer conversion only.



### 2.4 Missing Value Strategy

All decisions below are implemented as **drop** (not impute) because:
- Sentiment indices are daily point-in-time observations — interpolation would fabricate non-existent data.
- Trade records with missing fundamental fields (Account, execution details) are incomplete and cannot be meaningfully recovered.
- Imputation would introduce look-ahead bias or distributional assumptions that violate the repository's data integrity standards (§2).

| Dataset | Field(s) | Strategy | Rationale |
|---|---|---|---|
| Fear & Greed | `value`, `classification` | Drop row; log count & proportion at WARNING | Sentiment is point-in-time; interpolation would introduce fabricated data |
| Trader History | `Account` | Drop row; log count & proportion at WARNING | Account is the primary grouping key; null Account is unresolvable |
| Trader History | `Timestamp` | Drop row; log count & proportion at WARNING | Temporal ordering is fundamental; unparseable timestamps cannot be imputed |
| Trader History | `Size USD` | Drop row; log count & proportion at WARNING | Null/zero size constitutes an invalid trade record |
| Trader History | `Execution Price` | Drop row; log count & proportion at WARNING | Null/zero price constitutes an invalid trade record |

All drop decisions are logged with exact row counts and proportions to `logs/pipeline.log` at WARNING level.



### 2.5 Duplicate Handling
- **Trader History:** Duplicates are detected by `Trade ID` (primary key). Exact duplicates are dropped with a logged count. Non-exact duplicates sharing a `Trade ID` are flagged as a data quality issue and halt the pipeline pending investigation.
- **Fear & Greed:** Duplicates are detected by `timestamp` (date). Exact duplicates are dropped with a logged count.



### 2.6 Dataset Join Strategy
Sentiment and trader datasets are joined on the **UTC calendar date** of each trade's timestamp against the sentiment date. This is a left join (trader-left, sentiment-right) since trade records are the primary unit of analysis. Days with trades but no available sentiment record are flagged in the join output and excluded from sentiment-conditioned analyses with a warning logged.



---

## 3. Feature Engineering Decisions

### 3.1 Look-Ahead Bias Prevention
All rolling-window features use **strictly historical windows** — the current row's timestamp is excluded from its own window. All lag features use a minimum lag of 1 day. Feature code is reviewed against this checklist as a mandatory pre-merge step.



### 3.2 Sentiment Regime Classification
The Fear & Greed Index is classified into five regimes based on the value field:
- `[0, 24]` → Extreme Fear
- `[25, 44]` → Fear
- `[45, 55]` → Neutral
- `[56, 74]` → Greed
- `[75, 100]` → Extreme Greed

These thresholds are sourced from the canonical Alternative.me classification scheme present in the raw data's `classification` column. Any deviation from these thresholds requires an explicit ADR entry below.

### 3.3 Trader Metric Windows
Trader performance metrics (e.g., `trader_win_rate_7d`, `trader_leverage_avg_24h`) use the following default window sizes sourced from `configs/pipelines/feature_engineering.yaml`:

| Feature Family | Default Window | Rationale |
|---|---|---|
| Win rate | 7 days | Balances recency with sufficient sample for a reliable estimate |
| Leverage average | 24 hours | Captures intraday leverage patterns |
| PnL rolling sum | 7 days, 30 days | Both short and medium-term performance horizons |
| Volatility (PnL std dev) | 14 days | Standard two-week lookback used in risk management |



---

## 4. Statistical Methodology

### 4.1 Significance Level
**α = 0.05** is the global significance threshold, configurable in `configs/pipelines/statistics.yaml`. Any deviation from this default for a specific test must be logged here with justification.



### 4.2 Normality Testing Protocol
Before any parametric test is applied, normality is assessed:
- **n ≤ 50**: Shapiro-Wilk test
- **n > 50**: D'Agostino-Pearson (K-squared) test

The chosen test, its statistic, and p-value are logged. If normality is rejected (p < α), a non-parametric alternative is used. This choice is recorded in the statistical output table.



### 4.3 Two-Group Comparison Protocol (e.g., PnL: Fear vs. Greed)
1. Test normality in each group (§4.2 above).
2. If both groups are normal: test variance homogeneity with Levene's test.
3. If Levene's p ≥ α: use independent samples t-test.
4. If Levene's p < α: use Welch's t-test.
5. If either group fails normality: use Mann-Whitney U test.
6. Report: test name, statistic, p-value, Cohen's d (or rank-biserial r for Mann-Whitney), 95% CI on the difference.



### 4.4 Multi-Group Comparison Protocol (e.g., PnL across all 5 regimes)
1. Test normality in each group.
2. If all groups are normal and Levene's test p ≥ α: one-way ANOVA.
3. Otherwise: Kruskal-Wallis H test.
4. If omnibus test is significant: post-hoc pairwise tests.
   - ANOVA → Tukey HSD
   - Kruskal-Wallis → Dunn's test with Holm-Bonferroni correction
5. Report: omnibus test result, post-hoc comparisons, effect size (η² for ANOVA, ε² for Kruskal-Wallis).



### 4.5 Correlation Analysis Protocol
- **Pearson r**: only if both variables are approximately normal and the relationship is expected to be linear.
- **Spearman ρ**: default for all trading/sentiment data given non-normality expectations.
- Always report: coefficient, p-value, sample size, and 95% CI (via Fisher z-transformation for Pearson; bootstrap for Spearman).



### 4.6 Multiple Testing Correction
When testing more than one hypothesis on related outcomes (e.g., all pairwise sentiment regime comparisons for a single metric), corrections are applied as follows:
- **Bonferroni**: for small numbers of pre-specified comparisons (≤ 10).
- **Benjamini-Hochberg (FDR)**: for larger numbers of exploratory comparisons.
The correction method and adjusted threshold are reported in the output table.



### 4.7 Effect Size Reporting
Every statistical test result includes its associated effect size measure. Effect sizes without p-values, and p-values without effect sizes, are incomplete and do not meet the reporting standard.

| Test | Effect Size Measure |
|---|---|
| t-test / Welch's t-test | Cohen's d |
| Mann-Whitney U | Rank-biserial correlation (r) or Cliff's delta |
| ANOVA | η² (eta-squared) |
| Kruskal-Wallis | ε² (epsilon-squared) |
| Chi-square | Cramér's V |
| Pearson correlation | r (the coefficient itself) |
| Spearman correlation | ρ (the coefficient itself) |



---

## 5. Machine Learning Methodology (Optional Module)

### 5.1 Prerequisite Gate
The ML module is invoked **only** after Phase 06 (Statistical Analysis) has produced a statistically defensible signal — i.e., at least one hypothesis test with p < α and a non-negligible effect size across sentiment regimes. A statistically null result does not warrant a predictive model.



### 5.2 Train/Test Split
A **chronological hard cutoff** is used — training data precedes the test set temporally. No random shuffling. The cutoff date is defined in `configs/pipelines/ml.yaml`. The exact split date and resulting training/test sample sizes are logged at the start of every training run.



### 5.3 Cross-Validation
`TimeSeriesSplit` with a configurable number of folds (default: 5) is the standard CV approach. If the task is trader-level generalization (predicting profitability for unseen accounts), grouped CV by `Account` is used instead to prevent information leakage from the same account appearing in both train and validation folds.



### 5.4 Baseline-First Protocol
A trivial baseline model is always established first:
- **Regression** (e.g., PnL prediction): predict the training set mean PnL for all test samples.
- **Classification** (e.g., profitable vs. unprofitable): predict the majority class.
Lift above baseline is the primary success criterion, not raw metric values.



### 5.5 Random Seed
All stochastic operations (train/test split, model initialization, bootstrap sampling) use the global `random_seed` defined in `configs/pipelines/ml.yaml`. This value is logged as part of every MLflow experiment run.



### 5.6 Class Imbalance Handling
The classification target (`profitable_trade` = Closed PnL > 0) exhibits near-balanced distribution (~49% profitable), but class imbalance may vary across different temporal splits and account groups. The following strategies are employed:

1. **class_weight='balanced'** on the Random Forest Classifier: This automatically adjusts weights inversely proportional to class frequencies, penalizing misclassification of the minority class more heavily. This is the primary defense against imbalance.

2. **Baseline comparison via majority-class classifier (DummyClassifier strategy='most_frequent'):** The baseline represents the trivial "always predict majority class" strategy. Any candidate model must demonstrably beat this baseline in terms of precision, recall, F1, and ROC-AUC to be considered useful.

3. **ROC-AUC as a threshold-independent metric:** Unlike accuracy, ROC-AUC evaluates model ranking quality across all classification thresholds and is robust to moderate class imbalance.

4. **Reporting all metrics:** Accuracy, precision, recall, F1, and ROC-AUC are all reported and logged to MLflow. Lift above baseline is computed for each metric individually, so underperformance on one axis (e.g., low recall) is explicitly visible and not masked by acceptable accuracy.

5. **No SMOTE/oversampling used:** Synthetic oversampling was considered but not adopted because the temporal structure of the data makes synthetic interpolation between time points potentially leakage-prone. The `class_weight='balanced'` approach is preferred as it operates entirely within the original data distribution.

The average profit rate across the full dataset is approximately 49.3%, making the class distribution nearly balanced overall. Class weight adjustment serves as insurance against split-induced imbalance.<hr>



---

## 6. Business Insight Reporting Methodology

All insights promoted to the final report follow the mandatory five-part structure per 

1. **Observation** — the data pattern, stated plainly.
2. **Statistical Evidence** — test name, statistic, p-value, effect size, CI.
3. **Business Interpretation** — what this means for trading/product decisions.
4. **Practical Recommendation** — concrete, actionable suggestion.
5. **Limitation** — confounders, correlation-vs-causation caveat, data coverage gaps.

Insights that do not satisfy all five components are returned to the statistical analysis stage.



---

## 7. Amendment Log

| Date | Section Modified | Change Description | Author / Commit |
|---|---|---|---|---|
| 2026-07-01 | §2.4 | Initial missing value strategy documented (drop, not impute) | Phase 10 |
| 2026-07-01 | §2.5 | Duplicate handling protocol documented (Trade ID dedup) | Phase 10 |
| 2026-07-01 | §4.1–§4.7 | Statistical test selection protocols documented | Phase 10 |
| 2026-07-01 | §5 | ML methodology documented (baseline-first, class imbalance) | Phase 10 |
| 2026-07-01 | §6 | Business insight five-part reporting methodology added | Phase 10 |

*Add a row to this table whenever a methodology decision is updated, added, or overridden.*

---

