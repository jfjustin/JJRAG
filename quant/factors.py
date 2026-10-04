"""Factor computation.

Five categories. The first four are the usual multi-factor building blocks;
the fifth — cross-sectional linkage (截面联动类) — is the one this model
leans on hardest per the brief: instead of only describing a stock by its
own time series, it places the stock inside its peer cross-section at each
point in time (industry board, whole market) and measures how tightly its
returns and money flow move together with that cross-section.

Every `*_factors` function returns a flat dict of {factor_name: value}.
Missing upstream data yields NaN for that one factor, never an exception —
one broken factor should not take down the whole report.
"""

import numpy as np
import pandas as pd

from . import data, leadlag

TRADING_DAYS = 252


def _covers(prices, days, slack_days=7):
    """True if the history reaches back `days` calendar days (allowing a
    week of slack for weekends and holidays at the start)."""
    if prices.empty:
        return False
    span = prices["date"].iloc[-1] - prices["date"].iloc[0]
    return span >= pd.Timedelta(days=days - slack_days)


def _pct_return(prices, days):
    """Trailing simple return over `days` calendar days, using the closest
    available bar on/after `today - days`. NaN when the history is shorter
    than the span — otherwise a two-month window would report its own
    two-month return under a 12-month label."""
    if prices.empty or not _covers(prices, days):
        return float("nan")
    last = prices.iloc[-1]
    cutoff = last["date"] - pd.Timedelta(days=days)
    window = prices[prices["date"] >= cutoff]
    if window.empty:
        return float("nan")
    base = window.iloc[0]["close"]
    if not base:
        return float("nan")
    return (last["close"] / base - 1) * 100


def _rolling_beta_corr(stock_ret, bench_ret):
    """Align two return series on index, OLS beta + correlation + R^2."""
    joined = pd.concat([stock_ret, bench_ret], axis=1, join="inner").dropna()
    if len(joined) < 20:
        return float("nan"), float("nan"), float("nan")
    y = joined.iloc[:, 0].values
    x = joined.iloc[:, 1].values
    if np.std(x) == 0:
        return float("nan"), float("nan"), float("nan")
    beta, _intercept = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]
    return float(beta), float(corr), float(corr ** 2)


# ---------------------------------------------------------------------------
# A. Technical (time-series, own history only)
# ---------------------------------------------------------------------------

def technical_factors(prices, short=20, long=60, bars_per_day=1):
    """Indicators are computed in bars of whatever timeframe `prices` is
    (MA/RSI/MACD on 60-minute bars are 60-minute indicators). Volatility
    windows and annualization are converted to trading days via
    `bars_per_day` so vol_20d means 20 trading days at any timeframe."""
    if prices.empty or len(prices) < long + 5:
        return {}
    p = prices.sort_values("date").reset_index(drop=True)
    close = p["close"]
    ret = close.pct_change()

    ma_short = close.rolling(short).mean()
    ma_long = close.rolling(long).mean()
    ma5 = close.rolling(5).mean()
    ma120 = close.rolling(120).mean() if len(p) >= 120 else pd.Series([np.nan] * len(p))

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi14 = 100 - 100 / (1 + rs)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    boll_mid = close.rolling(20).mean()
    boll_std = close.rolling(20).std()
    boll_pct_b = (close - (boll_mid - 2 * boll_std)) / (4 * boll_std).replace(0, np.nan)

    annualizer = np.sqrt(TRADING_DAYS * bars_per_day)
    vol20 = ret.tail(20 * bars_per_day).std() * annualizer * 100
    vol60 = (ret.tail(60 * bars_per_day).std() * annualizer * 100
             if len(ret) >= 60 * bars_per_day else np.nan)

    turnover_ma20 = p["turnover"].rolling(20).mean().iloc[-1] if "turnover" in p else np.nan
    has_year = _covers(p, 365)

    return {
        "ma_short": ma_short.iloc[-1],
        "ma_long": ma_long.iloc[-1],
        "ma_trend_signal": int(ma_short.iloc[-1] > ma_long.iloc[-1]),
        "price_vs_ma5_pct": (close.iloc[-1] / ma5.iloc[-1] - 1) * 100 if pd.notna(ma5.iloc[-1]) else np.nan,
        "price_vs_ma120_pct": (close.iloc[-1] / ma120.iloc[-1] - 1) * 100 if pd.notna(ma120.iloc[-1]) else np.nan,
        "rsi14": rsi14.iloc[-1],
        "macd_hist": macd_hist.iloc[-1],
        "macd_hist_rising": int(macd_hist.iloc[-1] > macd_hist.iloc[-2]) if len(macd_hist) > 1 else np.nan,
        "boll_pct_b": boll_pct_b.iloc[-1],
        "ret_5d": _pct_return(p, 5),
        "ret_1m": _pct_return(p, 30),
        "ret_3m": _pct_return(p, 91),
        "ret_6m": _pct_return(p, 182),
        "ret_12m": _pct_return(p, 365),
        "momentum_12_1": _pct_return(p, 365) - _pct_return(p, 30) if pd.notna(_pct_return(p, 365)) else np.nan,
        "vol_20d_annualized_pct": vol20,
        "vol_60d_annualized_pct": vol60,
        "turnover_ma20_pct": turnover_ma20,
        "hi_52w": p[p["date"] >= p["date"].max() - pd.Timedelta(days=365)]["close"].max() if has_year else np.nan,
        "lo_52w": p[p["date"] >= p["date"].max() - pd.Timedelta(days=365)]["close"].min() if has_year else np.nan,
        "hi_window": close.max(),
        "lo_window": close.min(),
        "ret_window": (close.iloc[-1] / close.iloc[0] - 1) * 100,
    }


# ---------------------------------------------------------------------------
# B. Valuation (levels only here; cross-sectional percentiles live in
#    linkage_factors, since "cheap vs. what" is inherently a cross-section
#    question)
# ---------------------------------------------------------------------------

def valuation_factors(snapshot_row):
    if snapshot_row is None:
        return {}
    return {
        "pe_ttm": snapshot_row.get("pe_ttm"),
        "pb": snapshot_row.get("pb"),
        "total_mkt_cap": snapshot_row.get("total_mkt_cap"),
        "float_mkt_cap": snapshot_row.get("float_mkt_cap"),
    }


# ---------------------------------------------------------------------------
# C. Growth & quality (fundamentals)
# ---------------------------------------------------------------------------

def growth_quality_factors(fin_abstract):
    if fin_abstract is None or fin_abstract.empty:
        return {}
    latest = fin_abstract.iloc[0]  # most recent report period first
    out = {
        "revenue_yoy_pct": latest.get("revenue_yoy"),
        "net_profit_yoy_pct": latest.get("net_profit_yoy"),
        "gross_margin_pct": latest.get("gross_margin"),
        "net_margin_pct": latest.get("net_margin"),
        "roe_pct": latest.get("roe"),
    }
    if len(fin_abstract) >= 4 and "eps" in fin_abstract:
        recent_eps = fin_abstract["eps"].head(4).astype(float)
        if recent_eps.notna().sum() >= 2:
            x = np.arange(len(recent_eps))[::-1]  # oldest->newest order
            y = recent_eps.values[::-1]
            mask = ~np.isnan(y)
            if mask.sum() >= 2:
                slope, _ = np.polyfit(x[mask], y[mask], 1)
                out["eps_trend_slope"] = float(slope)
    return out


# ---------------------------------------------------------------------------
# D. Capital flow
# ---------------------------------------------------------------------------

def _flow_intensity(table, horizon_days, market_snap):
    """Net inflow scaled by the stock's typical trading value over the
    horizon, so a large cap's routine inflow doesn't outrank a small cap's
    heavy one. Falls back to raw inflow when no snapshot turnover exists."""
    t = table[["code", "net_inflow"]].copy()
    if market_snap is not None and not market_snap.empty and "amount" in market_snap:
        t = t.merge(market_snap[["code", "amount"]], on="code", how="left")
        scale = t["amount"] * horizon_days
        t["intensity"] = np.where(scale > 0, t["net_inflow"] / scale, np.nan)
        if t["intensity"].notna().sum() > 10:
            return t[["code", "intensity"]], True
    return t.rename(columns={"net_inflow": "intensity"})[["code", "intensity"]], False


def _pctile_of(values, code):
    """Percentile rank (0-100) of `code` within the `values` frame."""
    v = values.dropna(subset=["intensity"])
    if code not in v["code"].values or len(v) < 5:
        return float("nan")
    return float(v["intensity"].rank(pct=True)[v["code"] == code].iloc[0] * 100)


def bar_direction_flow(prices):
    """Estimated net flow from the bars themselves: each bar's traded value
    signed by whether it closed above its open, summed and divided by total
    traded value over the window. In [-1, 1]. A crude stand-in for order-
    size-based money flow, but computed from our own data at any timeframe
    and available for every stock."""
    if prices is None or prices.empty or "amount" not in prices:
        return float("nan")
    amt = prices["amount"].fillna(0)
    total = amt.sum()
    if total <= 0:
        return float("nan")
    sign = np.sign(prices["close"] - prices["open"]).fillna(0)
    return float((amt * sign).sum() / total)


def capital_flow_factors(code, flow_values, flow_tables, industry_codes,
                         market_snap, margin, prices=None):
    """THS money flow as levels and as cross-sectional standing.

    For the 5- and 20-session horizons, the stock's net inflow is scaled by
    its trading value and percentile-ranked against its industry and the
    whole market — 截面 treatment of money flow, rather than a raw yuan
    figure that mostly reflects company size.
    """
    symbol = data.normalize_code(code)[0]
    out = {f"net_inflow_{h}": v for h, v in flow_values.items()}
    for h, days in (("5d", 5), ("20d", 20)):
        table = flow_tables.get(h)
        if table is None or table.empty:
            continue
        scaled, normalized = _flow_intensity(table, days, market_snap)
        out[f"inflow_{h}_pctile_market"] = _pctile_of(scaled, symbol)
        if industry_codes:
            out[f"inflow_{h}_pctile_industry"] = _pctile_of(
                scaled[scaled["code"].isin(industry_codes)], symbol)
        out["inflow_scaled_by_turnover"] = normalized
    if prices is not None:
        out["bar_direction_flow"] = bar_direction_flow(prices)
    if margin:
        out["margin_balance"] = margin.get("margin_balance")
    return out


# ---------------------------------------------------------------------------
# D2. Geometric SDE decomposition
# ---------------------------------------------------------------------------

def sde_factors(prices, recent_days=20, bars_per_day=1):
    """Split the price path into drift and diffusion under a geometric SDE.

    DGNSDE evolves each stock's hidden state as dh = mu(.)h dt + sigma(.)h dW —
    a deterministic trend plus a Brownian noise term, in geometric
    (proportional) form. The model's own drift/diffusion are neural, but
    the same decomposition estimated classically on the realized path is
    directly usable as a factor: fit geometric Brownian motion by moments
    on log returns and report the annualized drift, the annualized
    diffusion, and their ratio.

    The drift/diffusion ratio is the signal-to-noise of the price path —
    how much of the move is trend versus how much is churn. `vol_regime` is
    recent diffusion over long-run diffusion: >1 means volatility is
    expanding relative to its own history.
    """
    if prices.empty or len(prices) < 30:
        return {}
    close = prices.sort_values("date")["close"].astype(float)
    close = close[close > 0]
    if len(close) < 30:
        return {}
    log_ret = np.diff(np.log(close.values))
    if len(log_ret) < 20 or not np.isfinite(log_ret).all():
        return {}

    bars_per_year = TRADING_DAYS * bars_per_day
    recent_window = recent_days * bars_per_day
    sigma = float(np.std(log_ret, ddof=1) * np.sqrt(bars_per_year))
    # GBM drift: the Ito correction turns the mean log return into the
    # arithmetic drift mu of dS = mu*S*dt + sigma*S*dW.
    mu = float(np.mean(log_ret) * bars_per_year + 0.5 * sigma ** 2)
    out = {
        "sde_drift_annual": mu,
        "sde_diffusion_annual": sigma,
        "sde_drift_diffusion_ratio": mu / sigma if sigma > 0 else float("nan"),
    }
    if len(log_ret) > recent_window:
        recent_sigma = float(np.std(log_ret[-recent_window:], ddof=1) * np.sqrt(bars_per_year))
        out["sde_recent_diffusion_annual"] = recent_sigma
        out["sde_vol_regime"] = recent_sigma / sigma if sigma > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# E. Cross-sectional linkage (截面联动类) — the centerpiece
# ---------------------------------------------------------------------------

def industry_composite(peer_prices):
    """Equal-weighted industry index from peers' bar returns, rebased to 1.

    BaoStock publishes no industry-board indices, so the industry benchmark
    is built from the same peers used everywhere else in this category. The
    target stock must not be in `peer_prices` or its own beta is inflated.
    Returns a date/close frame, or empty if fewer than 3 peers have bars.
    """
    rets = {}
    for code, h in peer_prices.items():
        if h is None or h.empty:
            continue
        rets[code] = h.sort_values("date").set_index("date")["close"].pct_change()
    if len(rets) < 3:
        return pd.DataFrame(columns=["date", "close"])
    panel = pd.concat(rets, axis=1).sort_index()
    # Require at least half the peers on a bar, so one stray stock's
    # session doesn't become an index print.
    avg = panel.mean(axis=1).where(panel.notna().sum(axis=1) >= max(3, len(rets) // 2))
    level = (1 + avg.fillna(0)).cumprod()
    level = level[avg.notna() | (level.index == level.index[0])]
    return pd.DataFrame({"date": level.index, "close": level.values}).reset_index(drop=True)


def _pctile(frame, col, symbol, higher_is_better=True, positive_only=False):
    f = frame.dropna(subset=[col])
    if positive_only:
        f = f[f[col] > 0]
    if symbol not in f["code"].values or len(f) < 3:
        return float("nan")
    p = float(f[col].rank(pct=True)[f["code"] == symbol].iloc[0] * 100)
    return p if higher_is_better else 100 - p


def linkage_factors(code, prices, market_snap, industry_peers, peer_prices,
                    market_index_prices, industry_index_prices,
                    beta_window=120, leadlag_window=leadlag.DEFAULT_WINDOW,
                    max_lead=leadlag.MAX_LEAD_DAYS):
    """Cross-sectional and co-movement factors:

    - market_beta / market_corr / market_r2 — how much of this stock's
      variance is systemic (CSI 300) risk
    - industry_beta / industry_corr / industry_r2 — same, against the
      equal-weighted composite of its CSRC-industry peers (isolates
      sector co-movement from broad-market co-movement)
    - *_pctile_industry / *_pctile_market — percentile rank (0-100) on
      return and valuation against the industry and whole-market
      cross-sections (not against its own history)
    - peer_avg_corr / peer_top_correlated / peer_corr_matrix — same-day
      return correlation with its most-traded industry peers
    - leadlag_network / leading_peer_signal_pct / target_leadership_days —
      the lead-lag layer (leadlag.py), which drops the assumption that
      peers move simultaneously

    `peer_prices` maps peer code -> bars at the same timeframe as `prices`;
    `industry_peers` supplies names and cross-sectional fields.
    """
    out = {}
    symbol, _ = data.normalize_code(code)
    p = prices.sort_values("date").set_index("date")
    stock_ret = p["close"].pct_change().rename("stock")

    for name, bench in (("market", market_index_prices), ("industry", industry_index_prices)):
        if bench is None or bench.empty:
            continue
        bench_ret = bench.set_index("date")["close"].pct_change().rename(name)
        beta, corr, r2 = _rolling_beta_corr(stock_ret.tail(beta_window),
                                            bench_ret.tail(beta_window))
        out[f"{name}_beta"], out[f"{name}_corr"], out[f"{name}_r2"] = beta, corr, r2

    # Cross-sectional standing: where this stock sits among its peers on
    # the latest session, not versus its own past.
    if industry_peers is not None and not industry_peers.empty:
        ip = industry_peers
        out["industry_size"] = int(len(ip))
        out["ret_today_pctile_industry"] = _pctile(ip, "pct_chg", symbol)
        out["ret_60d_pctile_industry"] = _pctile(ip, "ret_60d", symbol)
        # 100 = cheapest; loss-makers (PE <= 0) are left out of the ranking.
        out["value_rank_industry"] = _pctile(ip, "pe", symbol, higher_is_better=False,
                                             positive_only=True)
    if market_snap is not None and not market_snap.empty:
        out["ret_60d_pctile_market"] = _pctile(market_snap, "ret_60d", symbol)
        out["value_rank_market"] = _pctile(market_snap, "pe_ttm", symbol,
                                           higher_is_better=False, positive_only=True)

    # Peer network. Two layers over the same peer set:
    #   (a) same-day return correlation — the conventional picture, which
    #       assumes information reaches every name simultaneously;
    #   (b) the lead-lag network, which measures who moves first.
    names = {}
    if industry_peers is not None and not industry_peers.empty:
        names = dict(zip(industry_peers["code"], industry_peers["name"]))
    target_label = f"{symbol} (本股 target)"
    return_series = {target_label: stock_ret}
    close_series = {}
    for peer_code, peer_bars in peer_prices.items():
        if peer_code == symbol or peer_bars is None or peer_bars.empty:
            continue
        peer_close = peer_bars.sort_values("date").set_index("date")["close"]
        label = f"{peer_code} {names.get(peer_code, '')}".strip()
        return_series[label] = peer_close.pct_change()
        close_series[label] = peer_close

    if len(return_series) >= 3:
        returns_df = pd.concat(return_series, axis=1, join="inner").dropna(how="all")
        returns_df = returns_df.dropna(axis=1, thresh=30)
        if target_label in returns_df.columns and returns_df.shape[1] >= 3:
            corr_matrix = returns_df.corr()
            target_corrs = corr_matrix[target_label].drop(target_label).dropna()
            if not target_corrs.empty:
                out["peer_avg_corr"] = float(target_corrs.mean())
                out["peer_top_correlated"] = [
                    (name, float(c)) for name, c in
                    target_corrs.sort_values(ascending=False).head(3).items()]
                out["peer_corr_matrix"] = corr_matrix

    if close_series:
        network = leadlag.build_leadlag_network(p["close"], close_series,
                                                window=leadlag_window, max_lead=max_lead)
        if not network.empty:
            out["leadlag_network"] = network
            out["leadlag_peer_closes"] = close_series
            out.update(leadlag.network_summary(network))
            out.update(leadlag.leading_peer_signal(network, close_series))

    return out
