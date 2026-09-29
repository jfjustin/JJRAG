# Quant dashboard — cross-sectional multi-factor equity model

A local Streamlit dashboard for analyzing individual China A-share stocks,
built on **AKShare** — a free, open-source, credential-free market-data
library — with a multi-factor model that leans heavily on **cross-sectional
linkage (截面联动类)** factors: how a stock moves *relative to its peers*
at each point in time, not just its own price history in isolation.

This replaces an earlier version of this project (`emquant/`) that depended
on the paid East Money Choice **EMQuantAPI** SDK and a login. That's gone —
everything here runs with zero account, zero API key, zero cost.

## Quick start

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Type a code (e.g. `300274.SZ`, `600519.SH`) into the box, click **生成报告
Analyze**, then switch between the view buttons: Overview, Price Chart,
Cross-Sectional Linkage, Capital Flow, Factor Radar. Every button click
re-renders from data already fetched for that run — nothing re-hits the
network until you click Analyze again.

Prefer a terminal? `python -m quant.cli 300274.SZ` prints the same report
as text, no Streamlit needed.

## Why AKShare, not EastMoney's API directly / not EMQuantAPI

[AKShare](https://github.com/akfamily/akshare) is the highest-starred,
most actively maintained open-source China-market data library, and the
most comprehensive free option available: A-share/HK/US quotes, index and
industry/concept board data, fundamentals, capital-flow, Stock-Connect
(northbound) holdings, margin trading, and more, aggregated from multiple
public sources (East Money's public data center, Sina, 同花顺, and the
exchanges directly) behind one consistent API. No account, no login, no
paid Choice/EMQuantAPI subscription.

One honest caveat: a number of AKShare's A-share endpoints do, under the
hood, call East Money's own *public, unauthenticated* JSON data-center
endpoints (as one of several backends it wraps) — that's simply where a lot
of free, comprehensive Chinese market data lives. This is unrelated to, and
far lighter-weight than, the paid EMQuantAPI/Choice terminal product this
project used previously: no account, no credentials, no cost, and the
dependency lives in one file (`quant/data.py`) if you ever want to swap a
specific endpoint for a different backend.

## A note on the referenced article

This model's factor selection was meant to incorporate the methodology
from a specific Zhihu article
(`zhuanlan.zhihu.com/p/2043763197556036891`). That URL was **not
reachable** while building this — Zhihu is blocked by this environment's
network egress policy, and the post isn't indexed anywhere else searchable.
So the cross-sectional linkage factors below follow standard, well-
established quant treatment of that factor category (industry/market beta,
peer correlation networks, cross-sectional percentile ranking, capital-flow
co-movement) rather than that article's specific claims. If you paste the
article's content in, the factors and weights in `quant/factors.py` /
`quant/model.py` can be tuned to match it more precisely.

## Model design

Five factor categories, each producing a 0–100 sub-score
(`quant/factors.py` computes raw values, `quant/model.py` scores them):

| Category | Weight | What it captures |
| --- | --- | --- |
| Technical | 25% | MA trend, RSI, MACD, momentum (1m/3m/6m/12m, 12-1), realized volatility, 52-week range |
| Valuation | 15% | PE/PB level, plus **cross-sectional** cheapness rank (see below) |
| Growth & quality | 20% | Revenue/profit YoY growth, gross/net margin, ROE, EPS trend |
| Capital flow | 15% | Main-fund net inflow (5d/20d), northbound (Stock-Connect) holding change, margin balance |
| **Cross-sectional linkage** | 25% | See below |

Category scores are averaged (ignoring factors that came back NaN for this
stock, so missing data shrinks the average rather than scoring as 0), then
combined by `CATEGORY_WEIGHTS` into one composite 0–100 score and a stance
label (Strong Bullish / Bullish / Neutral / Bearish / Strong Bearish).
**This is a transparent, hand-weighted scoring model, not a backtested or
fitted one** — treat the composite as a structured summary of the factor
readings, not a calibrated probability or investment advice.

### Cross-sectional linkage factors (截面联动类)

This is the category the brief asked to lean on hardest. Instead of
describing a stock only by its own time series, these factors place it
inside its peer cross-section — industry board and whole market — at each
point in time:

- **Market beta / correlation / R²** — trailing-120-day OLS beta,
  correlation, and R² of the stock's daily returns against the CSI 300.
  R² is "how much of this stock's variance is systemic market risk."
- **Industry beta / correlation / R²** — same, against its own industry
  board index (isolates sector-specific co-movement from broad-market
  co-movement).
- **Return percentile within industry / within market** — this stock's
  percentile rank (0–100) on today's return and 60-day return, computed
  against the *live* industry-peer and whole-market cross-sections — not
  against its own history. This is the literal cross-sectional treatment:
  percentile-ranking against peers rather than using raw absolute numbers.
- **Value rank within industry / within market** — same idea applied to
  valuation: this stock's PE percentile among its peers (inverted, so 100 =
  cheapest in the group).
- **Peer correlation network** — a full pairwise return-correlation matrix
  across the stock and its largest industry peers (rendered as a heatmap in
  the dashboard), plus the stock's average correlation with that peer group
  and its 3 most-correlated peers. High average correlation = the stock
  trades mostly as part of the sector herd; low = mostly idiosyncratic.
- **Fund-flow / sector fund-flow correlation** — same-day correlation
  between the stock's own main-fund net inflow and its industry's aggregate
  net inflow, capturing whether sector-wide capital rotation is pulling (or
  pushing) this name.

## Caching

Every AKShare call goes through `quant/cache.py` (disk cache, pickle,
TTL-based — 15 min for the whole-market snapshot, 30 min for price/flow
history, 1 hour for index/industry history, 1 day for slow-changing things
like industry classification and fundamentals). This keeps the dashboard
responsive across button clicks. Delete `quant/.cache/` (or call
`quant.cache.clear()`) to force fresh data.

## Offline self-test

AKShare has no test/sandbox mode — it's live scrapers of public endpoints.
`python -m quant.selftest` mocks every AKShare call with synthetic data
shaped like the real schemas and runs the full pipeline end to end
(data → factors → composite score), so you can sanity-check the install
without hitting the network or waiting on rate limits.

## Known limitations

- **Margin trading** (`margin_snapshot`) is published by the exchanges as a
  per-day, all-symbols snapshot with no per-stock history endpoint, so this
  reports only the most recent available day's balance, not a time series.
- **Northbound (Stock-Connect) holding** is empty for stocks outside the
  Connect universe — this is expected, not a bug.
- **Concept-board (概念板块) linkage** was left out: AKShare has no cheap
  reverse lookup from a stock to its concept-board memberships, and
  reconstructing it by scanning every board is too slow for an interactive
  dashboard. Industry-board linkage (a stricter, single classification) is
  used instead.
- Weights in `model.py` encode a reasonable prior, not a fitted or
  backtested result. Tune `CATEGORY_WEIGHTS` and the per-factor scoring
  bands in `quant/model.py` for your own view.
