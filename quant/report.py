"""Orchestrates data.py + factors.py + model.py into one report for a code.

This is the single entry point both the CLI and the dashboard call.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from . import data, factors, model

MARKET_INDEX_SYMBOL = "sh000300"  # CSI 300 — broad-market beta proxy


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
    technical: dict = field(default_factory=dict)
    valuation: dict = field(default_factory=dict)
    growth_quality: dict = field(default_factory=dict)
    capital_flow: dict = field(default_factory=dict)
    linkage: dict = field(default_factory=dict)
    composite_score: float = float("nan")
    stance: str = ""
    breakdown: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def build_report(code, lookback_days=548, short_ma=20, long_ma=60):
    warnings = []
    symbol, market = data.normalize_code(code)
    code_disp = data.display_code(symbol, market)

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    basic = data.stock_basic_info(code_disp)
    name = basic.get("name") or code_disp
    industry = basic.get("industry") or ""

    prices = data.daily_history(code_disp, start, end)
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
            industry_index_prices = data.industry_index_history(industry, start, end)
        except Exception as e:
            warnings.append(f"Industry index history unavailable: {e}")
        try:
            sector_flow = data.sector_fund_flow(industry)
        except Exception as e:
            warnings.append(f"Sector fund-flow history unavailable: {e}")
    else:
        warnings.append("Industry classification unavailable — "
                        "cross-sectional peer factors will be skipped.")

    try:
        market_index_prices = data.index_history(MARKET_INDEX_SYMBOL, start, end)
    except Exception as e:
        market_index_prices = pd.DataFrame()
        warnings.append(f"Market index history unavailable: {e}")

    try:
        fund_flow = data.stock_fund_flow(code_disp)
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

    def peer_history_fn(peer_code):
        return data.daily_history(peer_code, start, end)

    tech = factors.technical_factors(prices, short=short_ma, long=long_ma) if not prices.empty else {}
    # Geometric-SDE drift/diffusion live alongside the other path factors.
    tech.update(factors.sde_factors(prices) if not prices.empty else {})
    val = factors.valuation_factors(snapshot_row)
    gq = factors.growth_quality_factors(financials)
    cf = factors.capital_flow_factors(fund_flow, northbound, margin)
    link = {}
    if not prices.empty:
        link = factors.linkage_factors(
            code_disp, prices, industry, snapshot_row, market_snap,
            industry_peers, fund_flow, sector_flow,
            market_index_prices, industry_index_prices, peer_history_fn,
        )

    composite, stance, breakdown = model.composite_score(tech, val, gq, cf, link)

    return StockReport(
        code=code_disp, name=name, industry=industry,
        as_of=prices["date"].max().strftime("%Y-%m-%d") if not prices.empty else end,
        prices=prices, snapshot_row=snapshot_row, fund_flow=fund_flow,
        market_index_prices=market_index_prices,
        industry_index_prices=industry_index_prices,
        industry_peers=industry_peers, financials=financials,
        technical=tech, valuation=val, growth_quality=gq,
        capital_flow=cf, linkage=link,
        composite_score=composite, stance=stance, breakdown=breakdown,
        warnings=warnings,
    )
