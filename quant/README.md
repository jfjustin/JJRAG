# Quant dashboard — cross-sectional multi-factor equity model

A local Streamlit dashboard for analyzing individual China A-share stocks
on daily, 60-, 30- and 15-minute bars, with a multi-factor model that leans
on **cross-sectional linkage (截面联动类)** factors: how a stock moves
*relative to its peers*, and which peers move *before* it.

All data comes from free sources that need no account and no API key, and
none of it goes through East Money (see [Data sources](#data-sources)).

## Quick start

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Type a code (e.g. `300476.SZ`, `600519.SH`) into the box, pick a preset or
set the timeframe and lookback yourself, click **生成报告 Analyze**, then
switch between the view buttons:

| View | Shows |
| --- | --- |
| 综合评分 Overview | Composite score, category bars, snapshot, full factor detail |
| K线走势 Price Chart | Candlestick + moving averages, volume, window range, geometric-SDE drift/diffusion |
| 截面联动分析 Cross-Sectional Linkage | Market/industry beta, percentile ranks, same-day peer correlation heatmap |
| 时滞联动 Lead-Lag Network | Who leads this stock and by how long, the leading-peer signal, and a real-linkage-vs-noise scatter |
| 资金流向 Capital Flow | THS net inflow over 3/5/10/20 sessions and its industry/market rank, the industry's 5-day inflow league, a bar-direction flow estimate, margin balance |
| 因子雷达图 Factor Radar | The five category scores as a radar chart |

Every button click re-renders from data already fetched for that run;
nothing re-hits the network until you click Analyze again.

From a terminal, `python -m quant.cli 300476.SZ` prints the same report as
text, no Streamlit needed.

### Timeframes and narrow windows

| Preset | Timeframe | Window | Bars | MAs (bars) |
| --- | --- | --- | --- | --- |
| 近两月·日线 2M daily | daily | 61 days | ~44 | 5 / 20 |
| 近两月·60分 2M 60m | 60-minute | 61 days | ~176 | 20 / 60 |
| 近两月·30分 2M 30m | 30-minute | 61 days | ~352 | 20 / 60 |
| 近两月·15分 2M 15m | 15-minute | 61 days | ~704 | 20 / 60 |
| 一年半·日线 18M daily | daily | 548 days | ~370 | 20 / 60 |

A-share sessions run four hours, so a day is 4 hourly, 8 half-hour or 16
quarter-hour bars. From the terminal, `--freq` takes a comma list or `all`:

```bash
python -m quant.cli 300476.SZ --days 61 --freq all        # daily, 60m, 30m, 15m
python -m quant.cli 300476.SZ --days 61 --freq daily,15m
```

On intraday bars every bar-based measure is computed at that timeframe
(MAs, RSI, MACD, betas against CSI 300 and industry minute bars), lead-lag
reads in **trading hours**, and volatility/SDE figures are annualized on
the right number of bars per day so they stay comparable with the daily
run. Factors that need more history than the window holds (3/6/12-month
returns, 12-1 momentum, 52-week range) come back n/a rather than being
silently computed over the shorter window; the window's own return and
range are reported instead.

**Use an intraday run for lead-lag on a short window.** Two months of daily
bars is ~44 observations; on the benchmark below only 27% of genuine links
could be detected at a controlled false-positive rate on a sample that
short, against 54% on two months of hourly bars. At 15 and 30 minutes the
sample is larger still, at the cost of shorter-horizon, noisier bars.

## Data sources

| Data | Source | Notes |
| --- | --- | --- |
| Daily bars, PE (TTM), PB, turnover | **BaoStock** (证券宝) | Forward-adjusted; suspended sessions dropped |
| 15/30/60-minute bars | **BaoStock** | History back years; end-of-day updates |
| CSI 300 daily | **BaoStock** | |
| CSI 300 minute bars | **Sina Finance** | BaoStock serves no index minute bars; Sina returns the latest ~1,970 bars (≈4 months of 15-minute) |
| CSRC industry for every stock | **BaoStock** | One call; defines the peer universe |
| Whole-market cross-section | **BaoStock** per-date query | Newer BaoStock releases only — see fallback below |
| Profitability & growth | **BaoStock** quarterly | ROE annualized from year-to-date; revenue YoY vs the same quarter a year earlier |
| Money flow (3/5/10/20 sessions) | **同花顺 THS** | Whole-market rankings |
| Margin balance | **SSE / SZSE** | The exchanges' own daily detail files |

**Why BaoStock as the primary source.** It is free with an anonymous login
(no registration, no token, no points quota), runs its own servers, and is
the one free source that covers daily *and* 5/15/30/60-minute A-share bars
with years of history, valuation fields, industry classification and
fundamentals behind a single stable API. The alternatives considered:

- **Tushare Pro** — needs registration and a token, and minute bars sit
  behind its paid points tiers.
- **JoinQuant JQData** — a free *trial* account with a daily quota; good
  data, but it needs credentials and the trial expires.
- **Sina / Tencent quote endpoints** — free, but undocumented and capped at
  recent bars, with no valuation or fundamentals. Sina is used here only
  for what BaoStock lacks (index minute bars).
- **East Money's public endpoints** — excluded by request.

**Trade-offs worth knowing.** BaoStock is end-of-day: a session's bars
arrive that evening, and there is no live intraday feed, so this is a
post-close analysis tool. The whole-market cross-section depends on
BaoStock's `query_daily_history_k_AStock`, which only newer releases have;
without it, industry ranks fall back to a 13-stock peer sample, market-
wide ranks are skipped, and the report says so. BaoStock has no industry
*index*, so the industry benchmark is an equal-weighted composite of the
stock's most-traded CSRC-industry peers. Northbound (Stock-Connect)
holdings are gone: per-stock northbound data is no longer published daily,
and the only free per-stock series came through East Money.

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

Lag search is capped at ±10 days on daily bars and at four sessions on
intraday bars (±16 bars at 60m, ±32 at 30m, ±64 at 15m), and further
limited to about one-eighth of the sample.

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
| Capital flow | 10% | THS net inflow over 5/20 sessions, scaled by trading value and percentile-ranked within industry and market; bar-direction flow estimate; margin balance |
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
inside its peer cross-section — CSRC industry and whole market — at each
point in time:

- **Market beta / correlation / R²** — OLS beta, correlation and R² of the
  stock's bar returns against the CSI 300 over the timeframe's window
  (120 daily bars, or ~40 sessions of intraday bars). R² is "how much of
  this stock's variance is systemic market risk."
- **Industry beta / correlation / R²** — same, against an equal-weighted
  composite of its most-traded CSRC-industry peers (isolates sector co-
  movement from broad-market co-movement).
- **Return percentile within industry / within market** — this stock's
  percentile rank (0–100) on the latest session's return and 60-session
  return, computed against the industry and whole-market cross-sections — not
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
- **Money-flow rank** (scored under capital flow) — THS net inflow over 5
  and 20 sessions, divided by the stock's trading value so company size
  doesn't dominate, percentile-ranked against its industry and the whole
  market.
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

Every fetch goes through `quant/cache.py` (disk cache, pickle, TTL-based).
BaoStock data changes once a day, so bars cache for an hour, the market
cross-section for six hours, and industry classification and fundamentals
for a day; THS money flow caches for an hour. Delete `quant/.cache/` (or
call `quant.cache.clear()`) to force fresh data. The first run on a new
code fetches the stock plus 12 peers and takes longest.

## Offline self-test

None of the sources has a test mode. `python -m quant.selftest` replaces
them with synthetic data shaped like the real responses — BaoStock is faked
at the library level with string fields, `YYYYMMDDHHMMSSsss` minute
timestamps and decimal ratios; THS tables carry `1.23亿`-style amounts and
integer codes with lost leading zeros — and runs the whole pipeline. Six
scenarios: 18-month daily; two-month daily, 60m, 30m and 15m; and a run
with the whole-market query unavailable, which must fall back to the peer
sample. Among other things it checks that linked peers' lags land within
one bar of the lags the synthetic universe was built with, that bar times
match each timeframe's session schedule, that annualized volatility agrees
across timeframes, and that long-horizon factors are n/a on two months.

## Known limitations

- **End-of-day data.** BaoStock publishes after the close, so intraday
  timeframes analyze completed sessions, not the live one.
- **Margin trading** is published by the exchanges as a per-day,
  all-symbols file with no per-stock history, so this reports only the
  most recent available day's balance.
- **THS money flow** is a ranking snapshot per horizon, not a per-stock
  daily history, so there is no money-flow time series; the dashboard's
  daily flow chart is the bar-direction estimate, labelled as such.
- **Concept-board (概念板块) linkage** is not included; the CSRC industry
  (a single, stricter classification) defines peers.
- Weights in `model.py` encode a reasonable prior informed by DGNSDE's
  ablation, not a fitted or backtested result for this scoring model. Tune
  `CATEGORY_WEIGHTS` and the per-factor scoring bands in `quant/model.py`
  for your own view.
- **Lead-lag resolution is one bar.** The fractional refinement is a
  parabolic fit to the correlation peak, good to roughly ±1 bar on the
  benchmark: ±1 day on daily bars, ±15 minutes on 15-minute bars.
- **No RankIC/ICIR validation is included.** Measuring it properly needs
  point-in-time factor recomputation across a universe and many rebalance
  dates; every factor here is computed as-of-now only. The composite score
  is a structured summary of current readings, and there is no evidence in
  this repo about its predictive power. That would be the natural next
  thing to build.
