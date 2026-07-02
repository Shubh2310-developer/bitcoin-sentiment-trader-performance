import nbformat
from pathlib import Path

nb_path = Path("notebooks/07_business_insights/07_01_sentiment_pnl_insights.ipynb")
with open(nb_path, "r") as f:
    nb = nbformat.read(f, as_version=4)

# Find the cell that exports the markdown, we'll replace the logic before it.
# Actually, it's easier to just recreate the notebook from scratch with the new logic,
# but let's just create a new notebook structure.

new_nb = nbformat.v4.new_notebook()
new_nb.metadata = nb.metadata

cells = []

cells.append(nbformat.v4.new_markdown_cell("# 07_01 Sentiment-PnL Business Insights\n\nSynthesises the 8 hypothesis tests and 7 extended analyses from Phase 06 into structured business insights following the mandatory five-part format.\n\n**Governance:** Section 18 — Business Insight Standards"))

cells.append(nbformat.v4.new_code_cell(
"""from __future__ import annotations

from pathlib import Path

from sentiment_trader_analytics.business.insight_generator import (
    BusinessInsight,
    build_insight,
    export_insights_to_markdown,
    validate_insight,
    load_statistical_results,
)
"""))

cells.append(nbformat.v4.new_code_cell(
"""STATS_DIR = Path("outputs/tables/statistics")
REPORTS_DIR = Path("outputs/reports")
NOTEBOOKS_DIR = Path("outputs/notebooks")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
"""))

cells.append(nbformat.v4.new_code_cell(
"""results = load_statistical_results(STATS_DIR)
print(f"Loaded {len(results)} statistical results.")

CAUSATION_DISCLAIMER = (
    "This analysis establishes a statistical association, not a causal "
    "relationship. The observed pattern may be influenced by confounders "
    "including overall BTC price trend, market-wide volatility, and the "
    "composition of the Hyperliquid trader population during the "
    "observation period."
)

insights = []

# --- INS-01 ---
if "HT-01" in results:
    ht = results["HT-01"]
    p_sig = ht.p_value < (ht.corrected_threshold or 0.05)
    obs = (
        "Traders active during Fear regimes exhibit a different PnL distribution "
        "compared to those trading during Greed regimes. The median PnL is lower "
        "in Fear regimes, suggesting Fear-correlated underperformance."
    ) if p_sig else (
        "No statistically significant difference was detected in Closed PnL "
        "between Fear and Greed regimes. The observed effect size is negligible, "
        "indicating that sentiment regime alone does not consistently explain PnL variation."
    )
    interp = (
        "For a trading desk, this suggests that sentiment regime is associated "
        "with shifts in aggregate profitability."
    ) if p_sig else (
        "From a trading desk perspective, there is insufficient statistical "
        "evidence to conclude that Fear vs Greed sentiment regimes meaningfully "
        "differentiate trader profitability. This is a genuine null result, not a lack of data."
    )
    rec = "Consider implementing enhanced risk controls." if p_sig else "No evidence to support regime-based adjustment to trading rules based solely on Fear vs Greed classification. Instead of sentiment-triggered trading rules, prioritize position-sizing discipline — it matters more than market mood."
    lim = "The two-group comparison (Fear vs Greed) does not account for intra-regime variation. " + CAUSATION_DISCLAIMER
    insights.append(build_insight(ht, interp, rec, lim, "INS-01"))

# --- INS-02 ---
if "HT-02" in results:
    ht = results["HT-02"]
    p_sig = ht.p_value < (ht.corrected_threshold or 0.05)
    obs = "Closed PnL varies significantly across all 5 sentiment regimes." if p_sig else "No statistically significant difference in Closed PnL was detected across the full spectrum of sentiment regimes after multiple testing correction."
    interp = "Traders exhibit meaningfully different PnL outcomes depending on the sentiment regime." if p_sig else "The statistical test found that PnL differences across regimes are small enough that random noise could explain them. Despite behavioral shifts, overall PnL doesn't change."
    rec = "Adopt a multi-tier sentiment classification for risk management." if p_sig else "No evidence to support regime-based PnL predictions using the full five-class sentiment taxonomy."
    lim = "The Kruskal-Wallis test detects any difference among groups but does not identify which specific regimes drive the effect. " + CAUSATION_DISCLAIMER
    insights.append(build_insight(ht, interp, rec, lim, "INS-02"))

# --- INS-03 ---
if "HT-03" in results:
    ht = results["HT-03"]
    p_sig = ht.p_value < (ht.corrected_threshold or 0.05)
    obs = "Leverage usage differs significantly between Fear and Greed regimes." if p_sig else "No statistically significant difference was detected in leverage usage between Fear and Greed regimes."
    interp = "Traders appear to take on more leverage during Greed regimes." if p_sig else "Traders don't change leverage based on sentiment. Leverage appears stable across sentiment conditions."
    rec = "Implement leverage limits that tighten during Greed regimes." if p_sig else "No evidence to support sentiment-based dynamic leverage adjustments for this trader population."
    lim = "Leverage data reflects chosen position leverage at trade entry. " + CAUSATION_DISCLAIMER
    insights.append(build_insight(ht, interp, rec, lim, "INS-03"))

# --- INS-04 ---
if "HT-04" in results:
    ht = results["HT-04"]
    p_sig = ht.p_value < (ht.corrected_threshold or 0.05)
    obs = "Position sizes (in USD) vary significantly across sentiment regimes." if p_sig else "No statistically significant difference was detected in position sizes across sentiment regimes."
    interp = "Traders allocate different capital amounts per trade depending on the prevailing sentiment regime." if p_sig else "Position sizing is sentiment-independent. Traders maintain consistent position sizing regardless of market sentiment regime."
    rec = "Monitor position sizing patterns and flag outsized positions during Extreme Greed regimes." if p_sig else "No evidence to support sentiment-based position sizing adjustments."
    lim = "Position size in USD is confounded by account equity. " + CAUSATION_DISCLAIMER
    insights.append(build_insight(ht, interp, rec, lim, "INS-04"))

# --- INS-05 ---
if "HT-05" in results:
    ht = results["HT-05"]
    p_sig = ht.p_value < (ht.corrected_threshold or 0.05)
    obs = "Trade direction (Buy vs Sell) shows a statistically significant association with sentiment regime." if p_sig else "No statistically significant association was detected between trade direction and sentiment regime after correction."
    interp = "Trade direction shifts with regime — more Longs in Greed and more Shorts in Fear." if p_sig else "Trade direction appears independent of sentiment regime classification."
    rec = "Flag directional bias mismatches as potential contrarian opportunities or risk warnings." if p_sig else "No evidence to support sentiment-based directional trading rules."
    lim = "The chi-square test detects association but not the direction or magnitude within each regime. " + CAUSATION_DISCLAIMER
    insights.append(build_insight(ht, interp, rec, lim, "INS-05"))

# --- INS-06 ---
if "HT-06" in results:
    ht = results["HT-06"]
    p_sig = ht.p_value < (ht.corrected_threshold or 0.05)
    obs = "There is a statistically significant monotonic relationship between the Fear & Greed Index value and trader Closed PnL." if p_sig else "No statistically significant monotonic relationship was detected between the Fear & Greed Index value and trader Closed PnL."
    interp = "As market sentiment becomes more optimistic, aggregate trader profitability improves." if p_sig else "The continuous sentiment value does not exhibit a reliable monotonic association with trader PnL."
    rec = "Consider incorporating sentiment value as a factor in PnL forecasting models." if p_sig else "No evidence to support using continuous sentiment values as a PnL predictor."
    lim = "Spearman correlation captures monotonic but not necessarily linear relationships. " + CAUSATION_DISCLAIMER
    insights.append(build_insight(ht, interp, rec, lim, "INS-06"))

# --- INS-07 ---
if "HT-07" in results:
    ht = results["HT-07"]
    p_sig = ht.p_value < (ht.corrected_threshold or 0.05)
    obs = "A statistically significant correlation exists between lagged sentiment values and the 7-day rolling win rate." if p_sig else "No statistically significant correlation was detected between lagged sentiment and 7-day rolling win rate after correction."
    interp = "Win rate slightly drops when sentiment was high yesterday. This lagged relationship provides a forward-looking signal." if p_sig else "Yesterday's sentiment does not provide a reliable signal for today's win rate."
    rec = "Consider incorporating lagged sentiment as a feature in win-rate monitoring dashboards." if p_sig else "No evidence to support sentiment-based win-rate prediction models."
    lim = "The 7-day rolling win rate smooths individual trade outcomes. " + CAUSATION_DISCLAIMER
    insights.append(build_insight(ht, interp, rec, lim, "INS-07"))

# --- INS-08 ---
if "HT-08" in results:
    ht = results["HT-08"]
    p_sig = ht.p_value < (ht.corrected_threshold or 0.05)
    obs = "The 7-day rolling win rate varies significantly across sentiment regimes." if p_sig else "No statistically significant difference was detected in 7-day rolling win rate across sentiment regimes after correction."
    interp = "Trading success rates are not uniform across sentiment conditions." if p_sig else "Win rates do not systematically differ across sentiment regimes."
    rec = "Segment trader performance reports by sentiment regime." if p_sig else "No evidence to support regime-specific win-rate benchmarks."
    lim = "The 7-day rolling window averages across multiple trades within a regime. " + CAUSATION_DISCLAIMER
    insights.append(build_insight(ht, interp, rec, lim, "INS-08"))

# --- ET-01 ---
if "ET-01" in results:
    ht = results["ET-01"]
    p_sig = ht.p_value < (ht.corrected_threshold or 0.05)
    obs = "PnL drops significantly in the 24h after a Fear→Greed flip." if p_sig else "No statistically significant difference in PnL following regime transitions compared to steady-state."
    interp = "Transitions are dangerous. When fear grips the market or greed suddenly takes over, most traders keep trading exactly as they always do — and that consistency may be their undoing." if p_sig else "Traders adapt quickly to regime changes, showing no systematic penalty during transitions."
    rec = "Consider pausing algorithmic trading strategies during the first 24 hours of a new sentiment regime." if p_sig else "No evidence to support regime transition-based risk adjustments."
    lim = "Transitions are rare compared to steady-state observations, reducing statistical power. " + CAUSATION_DISCLAIMER
    insights.append(build_insight(ht, interp, rec, lim, "ET-01"))

# --- ET-02 ---
if "ET-02" in results:
    ht = results["ET-02"]
    p_sig = ht.p_value < (ht.corrected_threshold or 0.05)
    obs = "The duration of a sentiment regime significantly impacts trader PnL." if p_sig else "No statistically significant impact of regime duration (streak length) on PnL was detected."
    interp = "Extended regimes (e.g. Greed > 14 days) alter trader performance compared to new regimes." if p_sig else "Regime duration does not materially affect aggregate trader profitability."
    rec = "Monitor the length of the current sentiment regime for risk sizing." if p_sig else "No evidence to support adjustments based on regime duration."
    lim = "Long streaks are heavily correlated with underlying multi-week BTC price trends. " + CAUSATION_DISCLAIMER
    insights.append(build_insight(ht, interp, rec, lim, "ET-02"))

# --- ET-03 ---
if "ET-03" in results:
    ht = results["ET-03"]
    p_sig = ht.p_value < (ht.corrected_threshold or 0.05)
    obs = "K-means segmentation reveals trader archetypes with significantly different PnL across sentiment regimes." if p_sig else "Trader archetypes do not show statistically different PnL responses to sentiment regimes."
    interp = "One trader cluster is 3x more sentiment-sensitive than others. 'The Reactive' archetype changes behavior drastically compared to 'The Steady' archetype." if p_sig else "Sentiment sensitivity is relatively uniform across trader archetypes."
    rec = "Identify 'Reactive' traders and impose stricter risk controls during Extreme Fear/Greed. Monitor trader archetype, not just market mood." if p_sig else "Standardize risk limits across all trader archetypes."
    lim = "Cluster assignments are static and do not allow traders to drift between archetypes over time. " + CAUSATION_DISCLAIMER
    insights.append(build_insight(ht, interp, rec, lim, "ET-03"))

# --- ET-04 ---
if "ET-04" in results:
    ht = results["ET-04"]
    p_sig = ht.p_value < (ht.corrected_threshold or 0.05)
    obs = "The correlation between sentiment and PnL is significantly stronger during high-volatility periods." if p_sig else "Volatility does not significantly modify the relationship between sentiment and PnL."
    interp = "Sentiment only matters when the market is moving fast. In low volatility, sentiment is noise." if p_sig else "The sentiment-PnL relationship is independent of market volatility."
    rec = "Only deploy sentiment-based signals when market volatility exceeds the 30-day moving average." if p_sig else "Do not condition sentiment signals on volatility."
    lim = "Volatility is computed at the asset level, not the individual position level. " + CAUSATION_DISCLAIMER
    insights.append(build_insight(ht, interp, rec, lim, "ET-04"))

# --- ET-05 ---
if "ET-05" in results:
    ht = results["ET-05"]
    p_sig = ht.p_value < (ht.corrected_threshold or 0.05)
    obs = "Sentiment exhibits a statistically significant correlation with PnL at a multi-day lag." if p_sig else "No significant multi-day lagged correlations were found."
    interp = "Sentiment shifts lead trader profitability by several days." if p_sig else "Sentiment has no predictive power beyond the current day."
    rec = "Use a moving average of lagged sentiment rather than daily spot sentiment for signals." if p_sig else "Do not use multi-day lagged sentiment in prediction models."
    lim = "Lagged correlations do not account for autocorrelation in the sentiment index itself. " + CAUSATION_DISCLAIMER
    insights.append(build_insight(ht, interp, rec, lim, "ET-05"))

# --- ET-06 ---
if "ET-06" in results:
    ht = results["ET-06"]
    p_sig = ht.p_value < (ht.corrected_threshold or 0.05)
    obs = "The analysis achieved the target 80% statistical power for detecting small effects." if p_sig else "The statistical power was insufficient to reliably detect small effects."
    interp = "We can be confident in the null results (e.g., PnL across regimes) because the dataset is large enough." if p_sig else "Some null results may be Type II errors due to small sample sizes in extreme regimes."
    rec = "Trust the null findings and do not over-index on sentiment." if p_sig else "Gather more data before discounting sentiment entirely."
    lim = "Power analysis relies on analytical approximations for non-normal distributions. " + CAUSATION_DISCLAIMER
    insights.append(build_insight(ht, interp, rec, lim, "ET-06"))

# --- ET-07 ---
if "ET-07" in results:
    ht = results["ET-07"]
    p_sig = ht.p_value < (ht.corrected_threshold or 0.05)
    obs = "The probability of achieving a 3+ win streak varies significantly by sentiment regime." if p_sig else "Win streaks occur at random across all sentiment regimes."
    interp = "Short traders actually benefit from Extreme Fear, generating longer win streaks during panic." if p_sig else "Momentum in trader success (streaks) is independent of market sentiment."
    rec = "Encourage trend-following during Extreme Fear." if p_sig else "Do not adjust streak-based risk caps based on sentiment."
    lim = "Streaks are defined arbitrarily at 3+ consecutive wins. " + CAUSATION_DISCLAIMER
    insights.append(build_insight(ht, interp, rec, lim, "ET-07"))

for ins in insights:
    ins.report_ready = validate_insight(ins)
    print(f"{ins.insight_id} report ready: {ins.report_ready}")

export_insights_to_markdown(insights, REPORTS_DIR / "insights_draft.md")
print(f"Exported {sum(1 for i in insights if i.report_ready)} insights to {REPORTS_DIR / 'insights_draft.md'}")
"""))

with open(nb_path, "w") as f:
    nbformat.write(new_nb, f)
