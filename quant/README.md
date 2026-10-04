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
Analyze**, then switch between the view buttons:

| View | Shows |
| --- | --- |
| 综合评分 Overview | Composite score, category bars, snapshot, full factor detail |
| K线走势 Price Chart | Candlestick + moving averages, volume, geometric-SDE drift/diffusion |
| 截面联动分析 Cross-Sectional Linkage | Market/industry beta, percentile ranks, same-day peer correlation heatmap |
| 时滞联动 Lead-Lag Network | Who leads this stock and by how long, the leading-peer signal, and a real-linkage-vs-noise scatter |
| 资金流向 Capital Flow | Daily and cumulative main-fund inflow, northbound, margin |
| 因子雷达图 Factor Radar | The five category scores as a radar chart |

Every button click re-renders from data already fetched for that run —
nothing re-hits the network until you click Analyze again.

Prefer a terminal? `python -m quant.cli 300274.SZ` prints the same report
as text, no Streamlit needed.

### Timeframes and narrow windows

The **周期 Timeframe** selector switches between daily bars and 60-minute
bars (A-share hourly bars close at 10:30, 11:30, 14:00 and 15:00 — four per
session), and **回看天数 Lookback** sets the window in calendar days. The
preset buttons set both in one click:

| Preset | Timeframe | Window | MAs (bars) |
| --- | --- | --- | --- |
| 近两月·日线 2M daily | daily | 61 days (~44 bars) | 5 / 20 |
| 近两月·小时 2M hourly | 60-minute | 61 days (~170 bars) | 20 / 60 |
| 一年半·日线 18M daily | daily | 548 days | 20 / 60 |

Same from the terminal — `--freq both` runs daily then hourly:

```bash
python -m quant.cli 300476.SZ --days 61 --freq both
```

On hourly bars every bar-based measure is an hourly measure (MAs, RSI,
MACD, betas against the CSI 300 and industry-board minute bars), lead-lag
reads in **trading hours**, and volatility/SDE figures are annualized on
four bars a day so they stay comparable with the daily run. Factors that
need more history than the window holds (3/6/12-month returns, 12-1
momentum, 52-week range) come back n/a rather than being silently computed
over the shorter window; the window's own return and range are reported
instead. Money-flow data is only published daily, so it is cut to the same
window whichever timeframe you pick.

**Use the hourly run for lead-lag on a short window.** Two months of daily
bars is ~44 observations, and on the benchmark below only 27% of genuine
links could be detected at a controlled false-positive rate on a sample
that short (54% on two months of hourly bars).

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

## Source methodology: DGNSDE

The factor design follows the QuantML write-up of **DGNSDE** — *Delay-Aware
Graph Neural Stochastic Differential Equations for Financial Time Series
Modeling and Forecasting* (WWW'26). Its argument is that conventional
cross-sectional graph models rest on two assumptions that don't hold in
real markets: that information reaches every stock **simultaneously**, and
that a stock's state evolves **deterministically**. DGNSDE drops both.

What carries over to a factor model, and what doesn't:

| DGNSDE component | Here |
| --- | --- |
| Two-stage delay-aware correlation estimator | **Stage 2 implemented as specified; stage 1 replaced** after it failed a known-lag benchmark — see below |
| Fractional-day alignment via cubic Hermite interpolation | **Implemented** — `hermite_sample`, used for both the delay shift and reading a leader's past state |
| Continuous control path via cubic spline (handles halts/irregular sampling) | **Implemented** — `continuous_path` |
| Geometric SDE drift/diffusion decomposition | **Implemented classically** — `factors.sde_factors` fits GBM by moments; the paper's μ and σ are neural, these are estimated on the realized path |
| Time-aligned graph message aggregation | **Distilled, not reproduced** — `leadlag.leading_peer_signal` aggregates leading peers' already-made moves weighted by delay-aligned co-movement, which is the tradeable content of the mechanism without the GNN |
| Neural training, dual-task loss, attention pooling | **Not implemented** — requires training infrastructure well beyond a dashboard |

So this is a *factor-model reading* of the paper, not a reimplementation of
it. The paper's reported numbers (RankIC 0.058, IR 2.92 on CSI300) belong
to the trained neural model and say nothing about the scores this
dashboard produces.

### The lead-lag layer (`quant/leadlag.py`)

**Stage 1 — delay measurement (departs from the paper).** The paper
DTW-aligns the two stocks' standardized price *levels* and averages the
warping path's offsets. This project originally did the same, and it
failed a benchmark built for the purpose: 52 synthetic peers with known
lags of -3 to +3 bars sharing a sector factor, with idiosyncratic noise at
realistic levels (true-lag return correlation 0.25-0.39).

| Stage-1 estimator | 6 months daily | 2 months daily | 2 months hourly |
| --- | --- | --- | --- |
| DTW on price levels (paper) | 21% within ±1 bar | 19% | 10% |
| Weighted return cross-correlation (used) | 92% | 79% | 88% |

A price level is dominated by each stock's own cumulative random walk, so
level-DTW warps to match idiosyncratic drift rather than the shared factor;
its sign was right about as often as a coin flip. The replacement is
recency-weighted cross-correlation of log returns (stationary, so the
shared factor is visible) over integer lags within a range scaled to the
sample size, with the peak refined to a fractional lag by parabolic
interpolation. Recency weighting is kept from the paper.

**Stage 2 — trend similarity after delay alignment.** Shift the peer by
that fractional delay through a monotone cubic Hermite (PCHIP) spline, take
the overlap, log-difference both into slope series, z-score them so price
level drops out, and run a second band-constrained DTW. The normalized
distance maps into [0, 1] — smaller means the two names trace the same
trend once the offset is removed.

**Sign convention:** `lead_days > 0` means the **peer leads the target**.

**Significance gating.** Picking the best of 2L+1 lags inflates the
winning correlation even for pure noise, so a fixed cutoff lets unrelated
peers in. A peer counts as `linked` only when its peak correlation clears a
family-wise threshold (α = 0.10 across the lags searched), computed on the
effective sample size left after recency weighting. Only linked peers feed
the signal and the leader/follower summary. Tuned on the same benchmark
plus 100 unrelated peers:

| Window | Real links detected | Noise peers admitted |
| --- | --- | --- |
| 6 months daily | 100% | 5% |
| 2 months daily | 27% | 3% |
| 2 months hourly | 54% | 3% |

Lag search is capped at ±10 days on daily bars and ±16 hours on hourly
bars, and further limited to about one-eighth of the sample.

**The signal.** For every linked peer leading by at least half a bar, read
the move that peer has already made over exactly its (fractional) lead
window — the move this stock has not yet followed — and average across
peers weighted by lag correlation. Positive means leaders have risen and
this stock hasn't caught up.

These benchmark figures come from synthetic data with a known answer; they
show the estimator can recover a lag when one exists at this noise level,
not how often real stocks have exploitable ones.

## Model design

Five factor categories, each producing a 0–100 sub-score
(`quant/factors.py` computes raw values, `quant/model.py` scores them):

| Category | Weight | What it captures |
| --- | --- | --- |
| Technical | 25% | MA trend, RSI, MACD, momentum (1m/3m/6m/12m, 12-1), realized volatility, 52-week range, **geometric-SDE drift/diffusion** |
| Valuation | 15% | PE/PB level, plus **cross-sectional** cheapness rank (see below) |
| Growth & quality | 20% | Revenue/profit YoY growth, gross/net margin, ROE, EPS trend |
| Capital flow | 10% | Main-fund net inflow (5d/20d), northbound (Stock-Connect) holding change, margin balance |
| **Cross-sectional linkage** | 30% | See below |

Linkage carries the largest weight because DGNSDE's ablation found the
delay-aware graph to be its single biggest contributor (removing it cost
more than removing the SDE, the message passing, or the reconstruction
loss: CSI300 annualized return 60.9% → 40.0%, RankIC 0.058 → 0.027). That
is evidence from the paper's trained model, not from a backtest of this
scoring model.

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
- **Lead-lag network** (see the DGNSDE section above) — who moves before
  this stock and by how many fractional days, the move those leaders have
  already made that it hasn't yet followed (`leading_peer_signal_pct`,
  double-weighted within the category as the only forward-looking factor
  in the set), and whether the stock is a sector bellwether or a follower
  (`target_leadership_days`).

Note that the same-day correlation matrix and the lead-lag network are
deliberately both kept: the first is the conventional simultaneous-
information picture, the second is what you see once that assumption is
dropped. Comparing them is informative — a stock can look weakly correlated
same-day yet be tightly linked at a 2-day offset.

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
without hitting the network or waiting on rate limits. It runs three
scenarios — 18-month daily, two-month daily, two-month hourly — and checks,
among other things, that linked peers' lags come out within one bar of the
lags the synthetic universe was built with, and that long-horizon factors
are n/a on a two-month window.

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
- Weights in `model.py` encode a reasonable prior informed by DGNSDE's
  ablation, not a fitted or backtested result for this scoring model. Tune
  `CATEGORY_WEIGHTS` and the per-factor scoring bands in `quant/model.py`
  for your own view.
- **Lead-lag resolution is one bar.** The fractional refinement is a
  parabolic fit to the correlation peak, good to roughly ±1 bar on the
  benchmark. On daily bars that is ±1 day; use hourly bars when the
  question is "a few hours or a day?".
- **No RankIC/ICIR validation is included.** Measuring it properly needs
  point-in-time factor recomputation across a universe and many rebalance
  dates — every factor here is computed as-of-now only. Without that, the
  composite score is a structured summary of current readings, and there is
  no evidence in this repo about its predictive power. That would be the
  natural next thing to build.
