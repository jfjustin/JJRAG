"""Orchestrates data.py + factors.py + model.py into one report for a code.

This is the single entry point both the CLI and the dashboard call.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from . import data, factors, model

MARKET_INDEX_BS = "sh.000300"   # CSI 300, BaoStock daily
MARKET_INDEX_SINA = "sh000300"  # CSI 300, Sina minute bars
MAX_PEERS = 12

# Everything that changes with bar frequency lives here. Windows are in
# bars of that timeframe and all cover roughly the same ~40 trading days
# on intraday bars; bars_per_day converts back to trading days. A-share
# sessions run 4 hours, so a day is 4 hourly, 8 half-hour or 16
# quarter-hour bars. The lead cap is four sessions at every intraday
# timeframe — hours to a few days is where transmission lives.
TIMEFRAMES = {
    "daily": {"label": "日线 Daily", "bars_per_day": 1, "bar_minutes": None,
              "beta_window": 120, "leadlag_window": 120, "max_lead": 10},
    "60m": {"label": "60分钟 Hourly", "bars_per_day": 4, "bar_minutes": 60,
            "beta_window": 160, "leadlag_window": 160, "max_lead": 16},
    "30m": {"label": "30分钟 30-min", "bars_per_day": 8, "bar_minutes": 30,
            "beta_window": 320, "leadlag_window": 320, "max_lead": 32},
    "15m": {"label": "15分钟 15-min", "bars_per_day": 16, "bar_minutes": 15,
            "beta_window": 640, "leadlag_window": 640, "max_lead": 64},
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
    market_index_prices: pd.DataFrame
    industry_index_prices: pd.DataFrame
    industry_peers: pd.DataFrame
    financials: pd.DataFrame
    freq: str = "daily"
    bar_unit: str = "day"
    bars_per_day: int = 1
    bar_minutes: int = None
    start: str = ""
    end: str = ""
    short_ma: int = 20
    long_ma: int = 60
    flow_values: dict = field(default_factory=dict)
    flow_peers: pd.DataFrame = field(default_factory=pd.DataFrame)
    cross_section: str = ""
    technical: dict = field(default_factory=dict)
    valuation: dict = field(default_factory=dict)
    growth_quality: dict = field(default_factory=dict)
    capital_flow: dict = field(default_factory=dict)
    linkage: dict = field(default_factory=dict)
    composite_score: float = float("nan")
    stance: str = ""
    breakdown: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


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


def _convert_lead_units(link, bars_per_day, bar_minutes):
    """Lead-lag runs in bars. Convert the summary scalars to days and, on
    intraday bars, add trading hours alongside."""
    if bars_per_day == 1:
        return link
    hours_per_bar = bar_minutes / 60
    for key in _BAR_UNIT_KEYS:
        if key in link and link[key] == link[key]:
            bars = link[key]
            link[key] = bars / bars_per_day
            link[key.replace("_days", "_hours")] = bars * hours_per_bar
    net = link.get("leadlag_network")
    if net is not None and not net.empty:
        net = net.rename(columns={"lead_days": "lead_bars"})
        net["lead_days"] = net["lead_bars"] / bars_per_day
        net["lead_hours"] = net["lead_bars"] * hours_per_bar
        link["leadlag_network"] = net
    return link


def _select_peers(industry_peers, symbol, n=MAX_PEERS):
    """Most-traded peers when the cross-section has turnover; otherwise the
    first n listed (an arbitrary sample, which the report warns about)."""
    others = industry_peers[industry_peers["code"] != symbol]
    if others["amount"].notna().any():
        others = others.sort_values("amount", ascending=False, na_position="last")
    return others.head(n)["code"].tolist()


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

    def bars(c):
        return data.history(c, start, end, freq)

    prices = bars(code_disp)
    if prices.empty:
        warnings.append("No price history returned — check the code, or BaoStock "
                        "may not have published the latest session yet.")

    # Daily bars carry this stock's own valuation regardless of timeframe.
    own_daily = prices if freq == "daily" else data.daily_history(
        code_disp, (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d"), end)

    # --- Cross-section: whole market if BaoStock serves it, else peers ---
    market_snap = data.market_snapshot()
    industry_peers = pd.DataFrame(columns=["code", "name", "pct_chg", "pe", "pb",
                                           "amount", "ret_60d", "turnover"])
    if industry:
        try:
            industry_peers = data.industry_constituents(industry, market_snap)
        except Exception as e:
            warnings.append(f"Industry peer list unavailable: {e}")
    else:
        warnings.append("Industry classification unavailable — cross-sectional "
                        "peer factors will be skipped.")

    peer_codes = _select_peers(industry_peers, symbol) if not industry_peers.empty else []
    peer_prices = {}
    for pc in peer_codes:
        try:
            peer_prices[pc] = bars(pc)
        except Exception:
            continue

    if market_snap.empty:
        cross_section = "peer sample"
        if not industry_peers.empty:
            warnings.append(
                "Whole-market snapshot unavailable (needs a BaoStock release with "
                "query_daily_history_k_AStock) — industry ranks use a "
                f"{len(peer_codes) + 1}-stock peer sample, market ranks are skipped, "
                "and the peer sample is not ranked by turnover.")
            daily_start = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
            sample = {pc: data.daily_history(pc, daily_start, end) for pc in peer_codes}
            sample[symbol] = own_daily
            snap = data.peer_snapshot_from_history(sample)
            industry_peers = (industry_peers[["code", "name"]]
                              .merge(snap, on="code", how="inner"))
    else:
        cross_section = f"whole market ({len(market_snap)} stocks)"

    snapshot_row = {}
    if not market_snap.empty and symbol in market_snap["code"].values:
        snapshot_row = market_snap[market_snap["code"] == symbol].iloc[0].to_dict()
    elif not own_daily.empty:
        last = own_daily.iloc[-1]
        snapshot_row = {"price": last["close"], "pct_chg": last["pct_chg"],
                        "pe_ttm": last["pe_ttm"], "pb": last["pb"],
                        "turnover": last["turnover"], "amount": last["amount"]}

    # --- Benchmarks ---------------------------------------------------------
    try:
        if tf["bar_minutes"]:
            market_index_prices = data.index_intraday(MARKET_INDEX_SINA, start, end,
                                                      period=str(tf["bar_minutes"]))
        else:
            market_index_prices = data.index_history(MARKET_INDEX_BS, start, end)
    except Exception as e:
        market_index_prices = pd.DataFrame()
        warnings.append(f"CSI 300 bars unavailable: {e}")
    industry_index_prices = factors.industry_composite(peer_prices)

    # --- Money flow, margin, fundamentals -----------------------------------
    try:
        flow_values, flow_tables = data.money_flow(code_disp)
    except Exception as e:
        flow_values, flow_tables = {}, {}
        warnings.append(f"THS money-flow rankings unavailable: {e}")
    if not flow_tables:
        warnings.append("No THS money-flow data — capital-flow ranks are skipped.")
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

    # --- Factors ------------------------------------------------------------
    short_ma, long_ma = _fit_ma_windows(len(prices), short_ma, long_ma, warnings)
    tech = (factors.technical_factors(prices, short=short_ma, long=long_ma, bars_per_day=bpd)
            if not prices.empty else {})
    tech.update(factors.sde_factors(prices, bars_per_day=bpd) if not prices.empty else {})
    val = factors.valuation_factors(snapshot_row)
    gq = factors.growth_quality_factors(financials)
    industry_codes = industry_peers["code"].tolist() if not industry_peers.empty else []
    cf = factors.capital_flow_factors(code_disp, flow_values, flow_tables, industry_codes,
                                      market_snap, margin, prices)
    link = {}
    if not prices.empty:
        link = factors.linkage_factors(
            code_disp, prices, market_snap, industry_peers, peer_prices,
            market_index_prices, industry_index_prices,
            beta_window=tf["beta_window"], leadlag_window=tf["leadlag_window"],
            max_lead=tf["max_lead"])
        link = _convert_lead_units(link, bpd, tf["bar_minutes"])

    flow_peers = pd.DataFrame()
    if "5d" in flow_tables and industry_codes:
        t = flow_tables["5d"]
        flow_peers = (t[t["code"].isin(industry_codes)]
                      .merge(industry_peers[["code", "name"]], on="code", how="left")
                      .sort_values("net_inflow", ascending=False))

    composite, stance, breakdown = model.composite_score(tech, val, gq, cf, link)

    as_of_fmt = "%Y-%m-%d %H:%M" if tf["bar_minutes"] else "%Y-%m-%d"
    return StockReport(
        code=code_disp, name=name, industry=industry,
        as_of=prices["date"].max().strftime(as_of_fmt) if not prices.empty else end,
        prices=prices, snapshot_row=snapshot_row,
        market_index_prices=market_index_prices,
        industry_index_prices=industry_index_prices,
        industry_peers=industry_peers, financials=financials,
        freq=freq, bar_unit="day" if freq == "daily" else freq,
        bars_per_day=bpd, bar_minutes=tf["bar_minutes"],
        start=start, end=end, short_ma=short_ma, long_ma=long_ma,
        flow_values=flow_values, flow_peers=flow_peers, cross_section=cross_section,
        technical=tech, valuation=val, growth_quality=gq,
        capital_flow=cf, linkage=link,
        composite_score=composite, stance=stance, breakdown=breakdown,
        warnings=warnings,
    )
