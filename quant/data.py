"""AKShare data access layer.

All functions here return clean, English-column pandas DataFrames/dicts —
every bit of "which Chinese column name does this endpoint use this month"
lives in this one file, behind `_pick` (which tolerates upstream renames by
trying several candidate labels instead of hard-failing). Every call is
disk-cached (see cache.py) because AKShare hits free public JSON endpoints
(East Money, Sina, THS, the exchanges) with no auth and no SLA — a few are
slow (the full-market snapshot is ~5,000 rows) and none should be re-fetched
on every dashboard click.

AKShare (https://github.com/akfamily/akshare) was chosen over the paid
Choice/EMQuantAPI SDK this project used previously: it's free, needs no
account or credentials, is the most-starred and most actively maintained
open-source China-market data library, and aggregates multiple public
sources (East Money's public data center, Sina, 同花顺, the exchanges
directly) rather than depending on any single paid vendor.
"""

import re
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd

from . import cache

# ---------------------------------------------------------------------------
# Code normalization
# ---------------------------------------------------------------------------

_MARKET_SUFFIX = {"SH": "sh", "SZ": "sz", "BJ": "bj"}


def normalize_code(code):
    """'300274.SZ' -> ('300274', 'sz'). Also accepts bare 6-digit codes,
    inferring the market from the leading digit (a standard A-share
    convention: 6xxxxx=SH, 0xxxxx/3xxxxx=SZ, 8xxxxx/4xxxxx=BJ)."""
    code = code.strip().upper()
    m = re.match(r"^(\d{6})\.(SH|SZ|BJ)$", code)
    if m:
        return m.group(1), _MARKET_SUFFIX[m.group(2)]
    m = re.match(r"^(SH|SZ|BJ)(\d{6})$", code)
    if m:
        return m.group(2), _MARKET_SUFFIX[m.group(1)]
    m = re.match(r"^\d{6}$", code)
    if m:
        digit = code[0]
        if digit == "6":
            market = "sh"
        elif digit in ("8", "4"):
            market = "bj"
        else:
            market = "sz"
        return code, market
    raise ValueError(f"Unrecognized stock code format: {code!r}")


def display_code(symbol, market):
    return f"{symbol}.{market.upper()}"


def _pick(row_or_df, *candidates, default=None):
    """Return the VALUE at the first present key/column from `candidates`
    (for a row/Series/dict). For picking a DataFrame's column NAME instead,
    use `_pick_col`."""
    for c in candidates:
        try:
            if c in row_or_df:
                return row_or_df[c]
        except TypeError:
            pass
    return default


def _pick_col(df, *candidates, default=None):
    """Return the first column NAME from `candidates` present in df.columns."""
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _to_num(series):
    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------------

def daily_history(code, start, end, adjust="qfq", ttl=1800):
    """Daily OHLCV bars, forward-adjusted by default. Returns columns:
    date, open, close, high, low, volume, amount, amplitude, pct_chg,
    chg, turnover — sorted ascending by date."""
    symbol, _ = normalize_code(code)
    start_c = start.replace("-", "")
    end_c = end.replace("-", "")
    key = f"hist:{symbol}:{start_c}:{end_c}:{adjust}"

    def fetch():
        return ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start_c, end_date=end_c, adjust=adjust,
        )

    df = cache.get_or_fetch(key, fetch, ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "close", "high", "low",
                                     "volume", "amount", "amplitude",
                                     "pct_chg", "chg", "turnover"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["日期"]),
        "open": _to_num(df["开盘"]),
        "close": _to_num(df["收盘"]),
        "high": _to_num(df["最高"]),
        "low": _to_num(df["最低"]),
        "volume": _to_num(df["成交量"]),
        "amount": _to_num(df["成交额"]),
        "amplitude": _to_num(df["振幅"]),
        "pct_chg": _to_num(df["涨跌幅"]),
        "chg": _to_num(df["涨跌额"]),
        "turnover": _to_num(df["换手率"]),
    }).sort_values("date").reset_index(drop=True)
    return out


def intraday_history(code, start, end, period="60", adjust="qfq", ttl=900):
    """Minute bars (period in {"5","15","30","60"}) between two dates, in the
    same column layout as daily_history with `date` holding the bar's
    timestamp. 60-minute A-share bars close at 10:30, 11:30, 14:00, 15:00 —
    four per session."""
    symbol, _ = normalize_code(code)
    key = f"min:{symbol}:{period}:{start}:{end}:{adjust}"

    def fetch():
        return ak.stock_zh_a_hist_min_em(
            symbol=symbol, period=period, adjust=adjust,
            start_date=f"{start} 09:30:00", end_date=f"{end} 15:00:00",
        )

    df = cache.get_or_fetch(key, fetch, ttl_seconds=ttl)
    cols = ["date", "open", "close", "high", "low", "volume", "amount",
            "amplitude", "pct_chg", "chg", "turnover"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["时间"]),
        "open": _to_num(df["开盘"]),
        "close": _to_num(df["收盘"]),
        "high": _to_num(df["最高"]),
        "low": _to_num(df["最低"]),
        "volume": _to_num(df["成交量"]),
        "amount": _to_num(df["成交额"]),
        "amplitude": _to_num(_pick(df, "振幅", default=pd.Series(dtype=float))),
        "pct_chg": _to_num(_pick(df, "涨跌幅", default=pd.Series(dtype=float))),
        "chg": _to_num(_pick(df, "涨跌额", default=pd.Series(dtype=float))),
        "turnover": _to_num(_pick(df, "换手率", default=pd.Series(dtype=float))),
    })
    return (out.dropna(subset=["close"]).sort_values("date")
            .reset_index(drop=True))


def index_intraday(index_code, start, end, period="60", ttl=900):
    """Index minute bars; index_code is the bare 6-digit code, e.g. '000300'."""
    key = f"idxmin:{index_code}:{period}:{start}:{end}"

    def fetch():
        return ak.index_zh_a_hist_min_em(
            symbol=index_code, period=period,
            start_date=f"{start} 09:30:00", end_date=f"{end} 15:00:00",
        )

    df = cache.get_or_fetch(key, fetch, ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    return (pd.DataFrame({"date": pd.to_datetime(df["时间"]),
                          "close": _to_num(df["收盘"])})
            .dropna().sort_values("date").reset_index(drop=True))


def industry_intraday(industry_name, start, end, period="60", ttl=900):
    """Industry-board minute bars. The endpoint has no date filter, so the
    full available history is cached and the window is cut here."""
    key = f"indmin:{industry_name}:{period}"

    def fetch():
        return ak.stock_board_industry_hist_min_em(symbol=industry_name,
                                                   period=period)

    df = cache.get_or_fetch(key, fetch, ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    date_col = _pick_col(df, "日期时间", "时间")
    out = (pd.DataFrame({"date": pd.to_datetime(df[date_col]),
                         "close": _to_num(df["收盘"])})
           .dropna().sort_values("date"))
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)
    return out[(out["date"] >= lo) & (out["date"] < hi)].reset_index(drop=True)


def index_history(index_symbol, start, end, ttl=1800):
    """Broad index daily history, e.g. index_symbol='sh000300' (CSI 300)."""
    key = f"idx:{index_symbol}:{start}:{end}"

    def fetch():
        return ak.stock_zh_index_daily_em(
            symbol=index_symbol,
            start_date=start.replace("-", ""), end_date=end.replace("-", ""),
        )

    df = cache.get_or_fetch(key, fetch, ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    date_col = _pick_col(df, "date", "日期")
    close_col = _pick_col(df, "close", "收盘")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
        "close": _to_num(df[close_col]),
    }).sort_values("date").reset_index(drop=True)
    return out


def industry_index_history(industry_name, start, end, ttl=1800):
    key = f"indhist:{industry_name}:{start}:{end}"

    def fetch():
        return ak.stock_board_industry_hist_em(
            symbol=industry_name, period="日k",
            start_date=start.replace("-", ""), end_date=end.replace("-", ""),
        )

    df = cache.get_or_fetch(key, fetch, ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["日期"]),
        "close": _to_num(df["收盘"]),
    }).sort_values("date").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Basic info / industry classification
# ---------------------------------------------------------------------------

def stock_basic_info(code, ttl=86400):
    symbol, _ = normalize_code(code)
    key = f"info:{symbol}"

    def fetch():
        return ak.stock_individual_info_em(symbol=symbol)

    df = cache.get_or_fetch(key, fetch, ttl_seconds=ttl)
    info = {}
    if df is not None and not df.empty:
        for _, row in df.iterrows():
            info[row["item"]] = row["value"]
    return {
        "code": info.get("股票代码", symbol),
        "name": info.get("股票简称", ""),
        "industry": info.get("行业", ""),
        "total_shares": info.get("总股本"),
        "float_shares": info.get("流通股"),
        "total_mkt_cap": info.get("总市值"),
        "float_mkt_cap": info.get("流通市值"),
        "list_date": info.get("上市时间"),
        "latest_price": info.get("最新"),
    }


def industry_constituents(industry_name, ttl=86400):
    """Peer universe for an industry board: code, name, price, pct_chg,
    pe, pb, turnover, amount — one row per constituent."""
    key = f"indcons:{industry_name}"

    def fetch():
        return ak.stock_board_industry_cons_em(symbol=industry_name)

    df = cache.get_or_fetch(key, fetch, ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame(columns=["code", "name", "price", "pct_chg",
                                     "pe", "pb", "turnover", "amount"])
    out = pd.DataFrame({
        "code": df["代码"].astype(str),
        "name": df["名称"],
        "price": _to_num(df["最新价"]),
        "pct_chg": _to_num(df["涨跌幅"]),
        "pe": _to_num(df["市盈率-动态"]),
        "pb": _to_num(df["市净率"]) if "市净率" in df.columns else float("nan"),
        "turnover": _to_num(df["换手率"]),
        "amount": _to_num(df["成交额"]),
    })
    return out


def market_snapshot(ttl=900):
    """Whole-market cross-section (all ~5,000 A-shares), one row/stock.
    This is the backbone of every cross-sectional percentile-rank factor."""
    key = "spot:all"

    def fetch():
        return ak.stock_zh_a_spot_em()

    df = cache.get_or_fetch(key, fetch, ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "code": df["代码"].astype(str),
        "name": df["名称"],
        "price": _to_num(df["最新价"]),
        "pct_chg": _to_num(df["涨跌幅"]),
        "turnover": _to_num(df["换手率"]),
        "pe_ttm": _to_num(df["市盈率-动态"]),
        "pb": _to_num(df["市净率"]),
        "volume_ratio": _to_num(df["量比"]),
        "total_mkt_cap": _to_num(df["总市值"]),
        "float_mkt_cap": _to_num(df["流通市值"]),
        "ret_60d": _to_num(df["60日涨跌幅"]),
        "ret_ytd": _to_num(df["年初至今涨跌幅"]),
    })
    return out


# ---------------------------------------------------------------------------
# Capital flow
# ---------------------------------------------------------------------------

def stock_fund_flow(code, ttl=1800):
    symbol, market = normalize_code(code)
    key = f"flow:{symbol}"

    def fetch():
        return ak.stock_individual_fund_flow(stock=symbol, market=market)

    df = cache.get_or_fetch(key, fetch, ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "main_net_inflow",
                                     "main_net_inflow_pct"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["日期"]),
        "main_net_inflow": _to_num(df["主力净流入-净额"]),
        "main_net_inflow_pct": _to_num(df["主力净流入-净占比"]),
    }).sort_values("date").reset_index(drop=True)
    return out


def sector_fund_flow(industry_name, ttl=1800):
    key = f"secflow:{industry_name}"

    def fetch():
        return ak.stock_sector_fund_flow_hist(symbol=industry_name)

    df = cache.get_or_fetch(key, fetch, ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "main_net_inflow",
                                     "main_net_inflow_pct"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["日期"]),
        "main_net_inflow": _to_num(df["主力净流入-净额"]),
        "main_net_inflow_pct": _to_num(df["主力净流入-净占比"]),
    }).sort_values("date").reset_index(drop=True)
    return out


def northbound_holding(code, start, end, ttl=3600):
    """Stock-Connect (沪深港通) foreign holding history for a stock.
    Empty result is normal for names outside the Connect universe."""
    symbol, _ = normalize_code(code)
    key = f"hsgt:{symbol}:{start}:{end}"

    def fetch():
        try:
            return ak.stock_hsgt_individual_detail_em(
                symbol=symbol,
                start_date=start.replace("-", ""), end_date=end.replace("-", ""),
            )
        except Exception:
            return pd.DataFrame()

    df = cache.get_or_fetch(key, fetch, ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "hold_shares", "hold_ratio",
                                     "hold_mkt_cap"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["持股日期"]),
        "hold_shares": _to_num(df["持股数量"]),
        "hold_ratio": _to_num(df["持股数量占A股百分比"]),
        "hold_mkt_cap": _to_num(df["持股市值"]),
    }).sort_values("date").reset_index(drop=True)
    return out


def margin_snapshot(code, lookback_days=10):
    """Best-effort latest margin-trading balance for this stock. Margin
    data is published per-day for the whole exchange (no per-symbol history
    endpoint), so this walks back a few trading days looking for the most
    recent day the exchange has published, and returns just that one row —
    not a full time series."""
    symbol, market = normalize_code(code)
    fetch_fn = {"sh": ak.stock_margin_detail_sse,
                "sz": ak.stock_margin_detail_szse}.get(market)
    if fetch_fn is None:
        return None
    code_col_candidates = ["标的证券代码", "证券代码"]
    for delta in range(lookback_days):
        day = (datetime.now() - timedelta(days=delta)).strftime("%Y%m%d")
        key = f"margin:{market}:{day}"
        try:
            df = cache.get_or_fetch(key, lambda d=day: fetch_fn(date=d),
                                    ttl_seconds=86400)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        code_col = next((c for c in code_col_candidates if c in df.columns), None)
        if code_col is None:
            continue
        row = df[df[code_col].astype(str) == symbol]
        if row.empty:
            continue
        row = row.iloc[0]
        return {
            "date": day,
            "margin_balance": _pick(row, "融资余额"),
            "margin_buy": _pick(row, "融资买入额"),
            "short_balance": _pick(row, "融券余量"),
        }
    return None


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def financial_abstract(code, ttl=86400):
    """Quarterly fundamentals (revenue/profit growth, margins, ROE, EPS).
    Returns a DataFrame indexed by report period, most recent first, with
    whatever of these columns THS actually published this run — callers
    must not assume every column is present."""
    symbol, _ = normalize_code(code)
    ths_symbol = symbol  # stock_financial_abstract_ths takes bare 6-digit code
    key = f"finabs:{ths_symbol}"

    def fetch():
        try:
            return ak.stock_financial_abstract_ths(symbol=ths_symbol,
                                                    indicator="按报告期")
        except Exception:
            return pd.DataFrame()

    df = cache.get_or_fetch(key, fetch, ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame()

    def col(*names):
        for n in names:
            if n in df.columns:
                return _to_num(df[n].astype(str).str.replace("%", "", regex=False))
        return pd.Series([float("nan")] * len(df))

    out = pd.DataFrame({
        "report_period": df.get("报告期"),
        "revenue_yoy": col("营业总收入同比增长率", "营业收入同比增长率"),
        "net_profit_yoy": col("净利润同比增长率"),
        "gross_margin": col("销售毛利率", "毛利率"),
        "net_margin": col("销售净利率", "净利率"),
        "roe": col("净资产收益率", "净资产收益率-摊薄"),
        "eps": col("基本每股收益", "每股收益"),
    })
    return out.dropna(how="all", subset=[c for c in out.columns
                                         if c != "report_period"])
