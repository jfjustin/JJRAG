"""Orchestrates data.py + factors.py + model.py into one report for a code.

This is the single entry point both the CLI and the dashboard call.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from . import data, factors, model

MARKET_INDEX_SYMBOL = "sh000300"  # CSI 300 — broad-market beta proxy
MARKET_INDEX_CODE = "000300"      # same index, bare code for minute bars

# Everything that changes with bar frequency lives here. Windows are in
# bars of that timeframe; bars_per_day converts them back to trading days.
# A-share 60-minute bars close at 10:30, 11:30, 14:00 and 15:00 — four per
# session — so one hourly bar is one hour of trading time and a lead-lag
# measured on them comes out in trading hours.
TIMEFRAMES = {
    "daily": {
        "label": "日线 Daily",
        "bars_per_day": 1,
        "bar_unit": "day",
        "minute_period": None,
        "beta_window": 120,
        "leadlag_window": 120,
        "max_lead": 10,
    },
    "60m": {
        "label": "60分钟 Hourly",
        "bars_per_day": 4,
        "bar_unit": "hour",
        "minute_period": "60",
        "beta_window": 160,
        "leadlag_window": 160,
        # Hours to a few days is where intraday transmission lives; four
        # sessions is the outer edge before a warp stops meaning anything.
        "max_lead": 16,
    },
}

# Lead-lag outputs measured in bars that need converting to days.
_BAR_UNIT_KEYS = ("peer_mean_lead_days", "target_leadership_days",
                  "leadlag_abs_days", "leading_peer_mean_lead_days")


@dataclass
class StockReport:
    code: str
    name: str
    industry: str
    as_of: str
    prices: pd.DataFrame
    snapshot_row: dict
    fund_flow: pd.DataFrame
    market_index_prices: pd.DataFrame
    industry_index_prices: pd.DataFrame
    industry_peers: pd.DataFrame
    financials: pd.DataFrame
    freq: str = "daily"
    bar_unit: str = "day"
    bars_per_day: int = 1
    start: str = ""
    end: str = ""
    short_ma: int = 20
    long_ma: int = 60
    technical: dict = field(default_factory=dict)
    valuation: dict = field(default_factory=dict)
    growth_quality: dict = field(default_factory=dict)
    capital_flow: dict = field(default_factory=dict)
    linkage: dict = field(default_factory=dict)
    composite_score: float = float("nan")
    stance: str = ""
    breakdown: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def _trim(df, start, end):
    if df is None or df.empty or "date" not in df:
        return df
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)
    return df[(df["date"] >= lo) & (df["date"] < hi)].reset_index(drop=True)


def _fit_ma_windows(n_bars, short_ma, long_ma, warnings):
    """Shrink MA windows that the window can't support, rather than letting
    the whole technical category come back empty."""
    if n_bars >= long_ma + 5:
        return short_ma, long_ma
    long_fit = max(5, n_bars - 5)
    short_fit = max(2, min(short_ma, long_fit // 3))
    if n_bars >= 10:
        warnings.append(
            f"Only {n_bars} bars in the window — MA{short_ma}/MA{long_ma} "
            f"shrunk to MA{short_fit}/MA{long_fit}.")
    return short_fit, long_fit


def _convert_lead_units(link, bars_per_day, bar_unit):
    """Lead-lag runs in bars. Convert the summary scalars to days and, on
    intraday bars, keep the native hours alongside."""
    if bars_per_day == 1:
        return link
    for key in _BAR_UNIT_KEYS:
        if key in link and link[key] == link[key]:
            bars = link[key]
            link[key] = bars / bars_per_day
            if bar_unit == "hour":
                link[key.replace("_days", "_hours")] = bars
    net = link.get("leadlag_network")
    if net is not None and not net.empty:
        net = net.rename(columns={"lead_days": "lead_bars"})
        net["lead_days"] = net["lead_bars"] / bars_per_day
        if bar_unit == "hour":
            net["lead_hours"] = net["lead_bars"]
        link["leadlag_network"] = net
    return link


def build_report(code, lookback_days=548, short_ma=20, long_ma=60, freq="daily"):
    if freq not in TIMEFRAMES:
        raise ValueError(f"freq must be one of {list(TIMEFRAMES)}, got {freq!r}")
    tf = TIMEFRAMES[freq]
    bpd = tf["bars_per_day"]

    warnings = []
    symbol, market = data.normalize_code(code)
    code_disp = data.display_code(symbol, market)

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    basic = data.stock_basic_info(code_disp)
    name = basic.get("name") or code_disp
    industry = basic.get("industry") or ""

    if tf["minute_period"]:
        def history(c):
            return data.intraday_history(c, start, end, period=tf["minute_period"])
    else:
        def history(c):
            return data.daily_history(c, start, end)

    prices = history(code_disp)
    if prices.empty:
        warnings.append("No price history returned — check the code and try again.")

    try:
        market_snap = data.market_snapshot()
    except Exception as e:
        market_snap = pd.DataFrame()
        warnings.append(f"Whole-market snapshot unavailable: {e}")
    snapshot_row = {}
    if not market_snap.empty:
        row = market_snap[market_snap["code"] == symbol]
        if not row.empty:
            snapshot_row = row.iloc[0].to_dict()

    industry_peers = pd.DataFrame()
    industry_index_prices = pd.DataFrame()
    sector_flow = pd.DataFrame()
    if industry:
        try:
            industry_peers = data.industry_constituents(industry)
        except Exception as e:
            warnings.append(f"Industry peer list unavailable: {e}")
        try:
            if tf["minute_period"]:
                industry_index_prices = data.industry_intraday(
                    industry, start, end, period=tf["minute_period"])
            else:
                industry_index_prices = data.industry_index_history(industry, start, end)
        except Exception as e:
            warnings.append(f"Industry index history unavailable: {e}")
        try:
            sector_flow = _trim(data.sector_fund_flow(industry), start, end)
        except Exception as e:
            warnings.append(f"Sector fund-flow history unavailable: {e}")
    else:
        warnings.append("Industry classification unavailable — "
                        "cross-sectional peer factors will be skipped.")

    try:
        if tf["minute_period"]:
            market_index_prices = data.index_intraday(
                MARKET_INDEX_CODE, start, end, period=tf["minute_period"])
        else:
            market_index_prices = data.index_history(MARKET_INDEX_SYMBOL, start, end)
    except Exception as e:
        market_index_prices = pd.DataFrame()
        warnings.append(f"Market index history unavailable: {e}")

    # Money-flow data is published daily whatever the bar timeframe; it is
    # cut to the same lookback so a narrow window means narrow everywhere.
    try:
        fund_flow = _trim(data.stock_fund_flow(code_disp), start, end)
    except Exception as e:
        fund_flow = pd.DataFrame()
        warnings.append(f"Stock fund-flow history unavailable: {e}")

    try:
        northbound = data.northbound_holding(code_disp, start, end)
    except Exception as e:
        northbound = pd.DataFrame()
        warnings.append(f"Northbound holding data unavailable: {e}")

    try:
        margin = data.margin_snapshot(code_disp)
    except Exception as e:
        margin = None
        warnings.append(f"Margin-trading snapshot unavailable: {e}")

    try:
        financials = data.financial_abstract(code_disp)
    except Exception as e:
        financials = pd.DataFrame()
        warnings.append(f"Financial fundamentals unavailable: {e}")

    short_ma, long_ma = _fit_ma_windows(len(prices), short_ma, long_ma, warnings)

    tech = (factors.technical_factors(prices, short=short_ma, long=long_ma,
                                      bars_per_day=bpd)
            if not prices.empty else {})
    # Geometric-SDE drift/diffusion live alongside the other path factors.
    tech.update(factors.sde_factors(prices, bars_per_day=bpd) if not prices.empty else {})
    val = factors.valuation_factors(snapshot_row)
    gq = factors.growth_quality_factors(financials)
    cf = factors.capital_flow_factors(fund_flow, northbound, margin)
    link = {}
    if not prices.empty:
        link = factors.linkage_factors(
            code_disp, prices, industry, snapshot_row, market_snap,
            industry_peers, fund_flow, sector_flow,
            market_index_prices, industry_index_prices, history,
            beta_window=tf["beta_window"],
            leadlag_window=tf["leadlag_window"],
            max_lead=tf["max_lead"],
        )
        link = _convert_lead_units(link, bpd, tf["bar_unit"])

    composite, stance, breakdown = model.composite_score(tech, val, gq, cf, link)

    as_of_fmt = "%Y-%m-%d %H:%M" if tf["minute_period"] else "%Y-%m-%d"
    return StockReport(
        code=code_disp, name=name, industry=industry,
        as_of=prices["date"].max().strftime(as_of_fmt) if not prices.empty else end,
        prices=prices, snapshot_row=snapshot_row, fund_flow=fund_flow,
        market_index_prices=market_index_prices,
        industry_index_prices=industry_index_prices,
        industry_peers=industry_peers, financials=financials,
        freq=freq, bar_unit=tf["bar_unit"], bars_per_day=bpd,
        start=start, end=end, short_ma=short_ma, long_ma=long_ma,
        technical=tech, valuation=val, growth_quality=gq,
        capital_flow=cf, linkage=link,
        composite_score=composite, stance=stance, breakdown=breakdown,
        warnings=warnings,
    )
