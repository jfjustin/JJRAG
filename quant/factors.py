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

from . import data


def _pct_return(prices, days):
    """Trailing simple return over `days` calendar days, using the closest
    available bar on/after `today - days`."""
    if prices.empty:
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

def technical_factors(prices, short=20, long=60):
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

    vol20 = ret.tail(20).std() * np.sqrt(250) * 100
    vol60 = ret.tail(60).std() * np.sqrt(250) * 100

    turnover_ma20 = p["turnover"].rolling(20).mean().iloc[-1] if "turnover" in p else np.nan

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
        "hi_52w": p[p["date"] >= p["date"].max() - pd.Timedelta(days=365)]["close"].max(),
        "lo_52w": p[p["date"] >= p["date"].max() - pd.Timedelta(days=365)]["close"].min(),
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

def capital_flow_factors(fund_flow, northbound, margin):
    out = {}
    if fund_flow is not None and not fund_flow.empty:
        out["main_inflow_5d_sum"] = fund_flow["main_net_inflow"].tail(5).sum()
        out["main_inflow_20d_sum"] = fund_flow["main_net_inflow"].tail(20).sum()
        out["main_inflow_pct_latest"] = fund_flow["main_net_inflow_pct"].iloc[-1]
    if northbound is not None and not northbound.empty and len(northbound) >= 2:
        out["northbound_hold_ratio_latest"] = northbound["hold_ratio"].iloc[-1]
        out["northbound_hold_ratio_chg_20d"] = (
            northbound["hold_ratio"].iloc[-1] - northbound["hold_ratio"].iloc[max(0, len(northbound) - 20)]
        )
    if margin:
        out["margin_balance"] = margin.get("margin_balance")
    return out


# ---------------------------------------------------------------------------
# E. Cross-sectional linkage (截面联动类) — the centerpiece
# ---------------------------------------------------------------------------

def linkage_factors(code, prices, industry_name, snapshot_row, market_snap,
                    industry_peers, fund_flow, sector_flow,
                    market_index_prices, industry_index_prices,
                    peer_history_fn, max_peers=12):
    """Cross-sectional and co-movement factors:

    - market_beta / market_corr / market_r2 — how much of this stock's
      variance is systemic (whole-market) risk
    - industry_beta / industry_corr / industry_r2 — same, vs its own
      industry board index (isolates sector-specific co-movement from
      broad-market co-movement)
    - *_percentile_industry / *_percentile_market — this stock's percentile
      rank (0-100) on return and valuation, computed against the live
      industry-peer and whole-market cross-sections (not its own history)
    - peer_avg_corr / peer_top_correlated — pairwise return-correlation
      network against its largest industry peers: how much it trades as
      part of the herd vs. idiosyncratically, and who its closest movers are
    - fundflow_industry_corr — same-day correlation between this stock's
      own main-fund net inflow and its industry's aggregate net inflow
      (captures whether sector-wide capital rotation is pulling this name
      along)
    """
    out = {}
    p = prices.sort_values("date").set_index("date")
    stock_ret = p["close"].pct_change().rename("stock")

    if market_index_prices is not None and not market_index_prices.empty:
        mkt_ret = (market_index_prices.set_index("date")["close"]
                  .pct_change().rename("market"))
        beta, corr, r2 = _rolling_beta_corr(stock_ret.tail(120), mkt_ret.tail(120))
        out["market_beta"] = beta
        out["market_corr"] = corr
        out["market_r2"] = r2

    if industry_index_prices is not None and not industry_index_prices.empty:
        ind_ret = (industry_index_prices.set_index("date")["close"]
                  .pct_change().rename("industry"))
        beta, corr, r2 = _rolling_beta_corr(stock_ret.tail(120), ind_ret.tail(120))
        out["industry_beta"] = beta
        out["industry_corr"] = corr
        out["industry_r2"] = r2

    # Cross-sectional percentile ranks: where does this stock sit *today*
    # among its peers, not "vs its own past".
    symbol, _ = data.normalize_code(code)
    if industry_peers is not None and not industry_peers.empty:
        ip = industry_peers.copy()
        if (ip["pct_chg"].notna().sum() >= 3) and (symbol in ip["code"].values):
            out["ret_today_pctile_industry"] = float(
                ip["pct_chg"].rank(pct=True)[ip["code"] == symbol].iloc[0] * 100)
        if (ip["pe"].notna().sum() >= 3) and (symbol in ip["code"].values):
            # cheaper = lower percentile of raw PE; report as "value rank"
            # where 100 = cheapest in the industry, for consistent scoring
            valid = ip[ip["pe"] > 0]
            if symbol in valid["code"].values:
                out["value_rank_industry"] = float(
                    100 - valid["pe"].rank(pct=True)[valid["code"] == symbol].iloc[0] * 100)

    if market_snap is not None and not market_snap.empty and symbol in market_snap["code"].values:
        ms = market_snap
        out["ret_60d_pctile_market"] = float(
            ms["ret_60d"].rank(pct=True)[ms["code"] == symbol].iloc[0] * 100
        ) if ms["ret_60d"].notna().sum() > 10 else np.nan
        valid = ms[ms["pe_ttm"] > 0]
        if symbol in valid["code"].values:
            out["value_rank_market"] = float(
                100 - valid["pe_ttm"].rank(pct=True)[valid["code"] == symbol].iloc[0] * 100)

    # Peer correlation network: pairwise return-correlation matrix across
    # this stock and the largest names in the same industry board — a
    # cross-sectional relationship structure, not just a stock-vs-peer list.
    if industry_peers is not None and not industry_peers.empty:
        peers = (industry_peers[industry_peers["code"] != symbol]
                .sort_values("amount", ascending=False)
                .head(max_peers))
        target_label = f"{symbol} (本股 target)"
        return_series = {target_label: stock_ret}
        for _, peer in peers.iterrows():
            try:
                peer_prices = peer_history_fn(peer["code"])
            except Exception:
                continue
            if peer_prices is None or peer_prices.empty:
                continue
            peer_ret = (peer_prices.sort_values("date").set_index("date")["close"]
                       .pct_change())
            return_series[f"{peer['code']} {peer['name']}"] = peer_ret

        if len(return_series) >= 3:
            returns_df = pd.concat(return_series, axis=1, join="inner").dropna(how="all")
            # Require enough overlapping history for a meaningful correlation.
            returns_df = returns_df.dropna(axis=1, thresh=30)
            if target_label in returns_df.columns and returns_df.shape[1] >= 3:
                corr_matrix = returns_df.corr()
                target_corrs = corr_matrix[target_label].drop(target_label).dropna()
                if not target_corrs.empty:
                    out["peer_avg_corr"] = float(target_corrs.mean())
                    out["peer_top_correlated"] = [
                        (name, float(c)) for name, c in
                        target_corrs.sort_values(ascending=False).head(3).items()
                    ]
                    out["peer_corr_matrix"] = corr_matrix

    # Capital-flow linkage: does sector-wide money flow move with this
    # stock's own money flow (same-day correlation)?
    if (fund_flow is not None and not fund_flow.empty
            and sector_flow is not None and not sector_flow.empty):
        joined = pd.merge(
            fund_flow[["date", "main_net_inflow"]].rename(columns={"main_net_inflow": "stock_flow"}),
            sector_flow[["date", "main_net_inflow"]].rename(columns={"main_net_inflow": "sector_flow"}),
            on="date", how="inner",
        )
        if len(joined) >= 20:
            out["fundflow_industry_corr"] = float(
                joined["stock_flow"].corr(joined["sector_flow"]))

    return out
