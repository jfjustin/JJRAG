"""Market-data access layer — no East Money endpoints.

Sources, chosen for being free, credential-free, and independent of East
Money's data center (which every previous version of this file leaned on,
directly or through AKShare):

  BaoStock (证券宝, via quant/bsapi.py) — the primary source. Daily bars
    with peTTM / pbMRQ / turnover, 5/15/30/60-minute bars, CSI 300 daily,
    CSRC industry classification for every listed stock in one call, and
    quarterly profitability/growth. Its own servers; anonymous login.
  Sina Finance (via AKShare's stock_zh_a_minute) — CSI 300 minute bars,
    because BaoStock serves no minute bars for indices.
  同花顺 / THS (via AKShare's stock_fund_flow_individual) — whole-market
    money-flow rankings over 3/5/10/20 sessions.
  SSE / SZSE (via AKShare) — margin-trading detail straight from the
    exchanges.

Northbound (Stock-Connect) holdings are gone: per-stock northbound data is
no longer published daily, and the only free per-stock series came through
East Money.

Every function returns clean English-column DataFrames/dicts, and every
fetch goes through the disk cache — BaoStock data only changes once a day,
so most TTLs are long.
"""

import re
from datetime import datetime, timedelta

import akshare as ak
import numpy as np
import pandas as pd

from . import bsapi, cache

# ---------------------------------------------------------------------------
# Code normalization
# ---------------------------------------------------------------------------

_MARKET_SUFFIX = {"SH": "sh", "SZ": "sz", "BJ": "bj"}

BAR_COLUMNS = ["date", "open", "close", "high", "low", "volume", "amount",
               "pct_chg", "turnover"]


def normalize_code(code):
    """'300274.SZ' -> ('300274', 'sz'). Also accepts 'sz.300274', 'SZ300274'
    and bare 6-digit codes, inferring the market from the leading digit
    (6xxxxx=SH, 0xxxxx/3xxxxx=SZ, 8xxxxx/4xxxxx/9xxxxx=BJ)."""
    code = code.strip().upper()
    m = re.match(r"^(\d{6})\.(SH|SZ|BJ)$", code)
    if m:
        return m.group(1), _MARKET_SUFFIX[m.group(2)]
    m = re.match(r"^(SH|SZ|BJ)\.?(\d{6})$", code)
    if m:
        return m.group(2), _MARKET_SUFFIX[m.group(1)]
    m = re.match(r"^\d{6}$", code)
    if m:
        digit = code[0]
        if digit == "6":
            market = "sh"
        elif digit in ("8", "4", "9"):
            market = "bj"
        else:
            market = "sz"
        return code, market
    raise ValueError(f"Unrecognized stock code format: {code!r}")


def display_code(symbol, market):
    return f"{symbol}.{market.upper()}"


def _bs(code):
    return bsapi.bs_code(*normalize_code(code))


def _pick(row_or_df, *candidates, default=None):
    """Return the VALUE at the first present key/column from `candidates`."""
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


_CN_UNITS = {"亿": 1e8, "万": 1e4}


def parse_cn_number(value):
    """'1.23亿' -> 1.23e8, '-4567.8万' -> -4.5678e7, '12.3%' -> 12.3."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float, np.number)):
        return float(value)
    s = str(value).strip().replace(",", "").replace("%", "")
    if s in ("", "-", "--"):
        return float("nan")
    scale = 1.0
    for unit, mult in _CN_UNITS.items():
        if s.endswith(unit):
            s, scale = s[:-len(unit)], mult
            break
    try:
        return float(s) * scale
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Price history (BaoStock)
# ---------------------------------------------------------------------------

def daily_history(code, start, end, adjust="qfq", ttl=3600):
    """Daily bars, forward-adjusted by default, suspended sessions dropped.
    Columns: BAR_COLUMNS + pe_ttm, pb."""
    key = f"bs:d:{_bs(code)}:{start}:{end}:{adjust}"
    df = cache.get_or_fetch(key, lambda: bsapi.history(_bs(code), start, end, "d", adjust),
                            ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS + ["pe_ttm", "pb"])
    if "tradestatus" in df:
        df = df[df["tradestatus"].astype(str) != "0"]
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "open": df["open"], "close": df["close"],
        "high": df["high"], "low": df["low"],
        "volume": df["volume"], "amount": df["amount"],
        "pct_chg": df["pctChg"], "turnover": df["turn"],
        "pe_ttm": df["peTTM"], "pb": df["pbMRQ"],
    })
    return out.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def _parse_bs_time(series):
    """BaoStock minute `time` is the bar close as YYYYMMDDHHMMSSsss."""
    return pd.to_datetime(series.astype(str).str[:14], format="%Y%m%d%H%M%S")


def intraday_history(code, start, end, period="60", adjust="qfq", ttl=3600):
    """5/15/30/60-minute bars; `date` holds each bar's close timestamp.
    Per session: 60m = 4 bars (10:30 11:30 14:00 15:00), 30m = 8, 15m = 16."""
    if period not in bsapi.MINUTE_FREQS:
        raise ValueError(f"period must be one of {sorted(bsapi.MINUTE_FREQS)}")
    key = f"bs:{period}:{_bs(code)}:{start}:{end}:{adjust}"
    df = cache.get_or_fetch(key, lambda: bsapi.history(_bs(code), start, end, period, adjust),
                            ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    close = df["close"]
    out = pd.DataFrame({
        "date": _parse_bs_time(df["time"]),
        "open": df["open"], "close": close,
        "high": df["high"], "low": df["low"],
        "volume": df["volume"], "amount": df["amount"],
        "pct_chg": close.pct_change() * 100,
        "turnover": np.nan,
    })
    # Zero-volume bars are suspended sessions BaoStock fills in flat.
    out = out[out["volume"].fillna(0) > 0]
    return out.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def history(code, start, end, freq="daily"):
    """Dispatch on timeframe: 'daily' or '5m'/'15m'/'30m'/'60m'."""
    if freq == "daily":
        return daily_history(code, start, end)
    return intraday_history(code, start, end, period=freq.rstrip("m"))


def index_history(index_code, start, end, ttl=3600):
    """Index daily closes from BaoStock, e.g. index_code='sh.000300'."""
    key = f"bs:idx:{index_code}:{start}:{end}"
    df = cache.get_or_fetch(key, lambda: bsapi.history(index_code, start, end, "d", ""),
                            ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    return (pd.DataFrame({"date": pd.to_datetime(df["date"]), "close": df["close"]})
            .dropna().sort_values("date").reset_index(drop=True))


def index_intraday(sina_symbol, start, end, period="60", ttl=1800):
    """Index minute bars from Sina, e.g. sina_symbol='sh000300'. Sina
    returns the most recent ~1,970 bars, i.e. about 4 months of 15-minute
    bars and more at coarser periods; the window is cut here."""
    key = f"sina:idxmin:{sina_symbol}:{period}"
    df = cache.get_or_fetch(
        key, lambda: ak.stock_zh_a_minute(symbol=sina_symbol, period=period, adjust=""),
        ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    out = pd.DataFrame({"date": pd.to_datetime(df[_pick_col(df, "day", "date")]),
                        "close": _to_num(df["close"])}).dropna()
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    out = out[(out["date"] >= lo) & (out["date"] < hi)]
    return out.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Classification, peers and the market cross-section (BaoStock)
# ---------------------------------------------------------------------------

def industry_map(ttl=86400):
    """CSRC industry for every listed stock: code (6-digit), name, industry."""
    df = cache.get_or_fetch("bs:industry:all", lambda: bsapi.stock_industry(""),
                            ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame(columns=["code", "name", "industry"])
    out = pd.DataFrame({
        "code": df["code"].astype(str).str.split(".").str[-1],
        "name": df["code_name"],
        "industry": df["industry"].fillna(""),
    })
    return out[out["industry"] != ""].reset_index(drop=True)


def stock_basic_info(code, ttl=86400):
    symbol, _ = normalize_code(code)
    name, industry = "", ""
    try:
        imap = industry_map()
        row = imap[imap["code"] == symbol]
        if not row.empty:
            name, industry = row.iloc[0]["name"], row.iloc[0]["industry"]
    except Exception:
        pass
    if not name:
        basic = cache.get_or_fetch(f"bs:basic:{_bs(code)}",
                                   lambda: bsapi.stock_basic(_bs(code)), ttl_seconds=ttl)
        if basic is not None and not basic.empty:
            name = basic.iloc[0].get("code_name", "")
    return {"code": symbol, "name": name, "industry": industry}


def _latest_trading_dates(n_back=60):
    """(latest session, the session n_back sessions before it)."""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=int(n_back * 1.6) + 20)).strftime("%Y-%m-%d")
    days = cache.get_or_fetch(f"bs:tradedates:{start}:{end}",
                              lambda: bsapi.trade_dates(start, end), ttl_seconds=21600)
    if not days or len(days) <= n_back:
        return None, None
    return days, days[-1 - n_back]


def market_snapshot(ttl=21600):
    """Whole-market cross-section from BaoStock's per-date query: one row
    per stock with close, pct_chg, turnover, pe_ttm, pb, amount, and
    ret_60d (60-session return, from a second whole-market query).

    Best-effort: needs a BaoStock release with the whole-market query, and
    walks back to the latest session the server has published (today's
    bars arrive in the evening). Returns an empty frame if unavailable —
    callers then fall back to a peer-sample cross-section.
    """
    def fetch():
        days, _ = _latest_trading_dates()
        if not days:
            return pd.DataFrame()
        for i in range(1, 4):
            latest = days[-i]
            now = bsapi.all_stocks_on(latest)
            if now is not None and not now.empty:
                base_day = days[-i - 60] if len(days) > i + 60 else None
                base = bsapi.all_stocks_on(base_day) if base_day else pd.DataFrame()
                return _build_snapshot(now, base)
        return pd.DataFrame()

    try:
        return cache.get_or_fetch("bs:snapshot", fetch, ttl_seconds=ttl)
    except Exception:
        return pd.DataFrame()


def _build_snapshot(now, base):
    code_col = _pick_col(now, "code")
    out = pd.DataFrame({"code": now[code_col].astype(str).str.split(".").str[-1]})
    for dst, *src in [("price", "close"), ("pct_chg", "pctChg"), ("turnover", "turn"),
                      ("pe_ttm", "peTTM"), ("pb", "pbMRQ"), ("amount", "amount")]:
        col = _pick_col(now, *src)
        out[dst] = _to_num(now[col]).values if col else np.nan
    out["ret_60d"] = np.nan
    if base is not None and not base.empty and _pick_col(base, "close"):
        b = pd.DataFrame({"code": base[_pick_col(base, "code")].astype(str).str.split(".").str[-1],
                          "base_close": _to_num(base["close"])})
        out = out.merge(b, on="code", how="left")
        out["ret_60d"] = (out["price"] / out["base_close"] - 1) * 100
        out = out.drop(columns="base_close")
    try:
        out = out.merge(industry_map()[["code", "name", "industry"]], on="code", how="left")
    except Exception:
        out["name"], out["industry"] = "", ""
    return out


def industry_constituents(industry_name, market_snap=None):
    """Peers in the same CSRC industry, joined with whatever the market
    snapshot carries (pct_chg, pe, pb, amount, ret_60d)."""
    imap = industry_map()
    peers = imap[imap["industry"] == industry_name][["code", "name"]].copy()
    if market_snap is not None and not market_snap.empty:
        cols = [c for c in ("pct_chg", "pe_ttm", "pb", "amount", "ret_60d", "turnover")
                if c in market_snap]
        peers = peers.merge(market_snap[["code"] + cols], on="code", how="left")
    for c in ("pct_chg", "pe_ttm", "pb", "amount", "ret_60d", "turnover"):
        if c not in peers:
            peers[c] = np.nan
    return peers.rename(columns={"pe_ttm": "pe"}).reset_index(drop=True)


def peer_snapshot_from_history(peer_histories):
    """Fallback cross-section built from peers' own daily bars, for when
    the whole-market query isn't available. Only as wide as the peer
    sample, which the caller should say."""
    rows = []
    for code, h in peer_histories.items():
        if h is None or h.empty:
            continue
        last = h.iloc[-1]
        ret_60d = (last["close"] / h["close"].iloc[-61] - 1) * 100 if len(h) > 60 else np.nan
        rows.append({"code": code, "pct_chg": last.get("pct_chg"),
                     "pe": last.get("pe_ttm"), "pb": last.get("pb"),
                     "amount": last.get("amount"), "ret_60d": ret_60d})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Money flow (同花顺 THS) and margin (exchanges)
# ---------------------------------------------------------------------------

THS_HORIZONS = {"3d": "3日排行", "5d": "5日排行", "10d": "10日排行", "20d": "20日排行"}


def money_flow_table(horizon="5d", ttl=3600):
    """Whole-market THS money-flow ranking over `horizon` sessions:
    code, net_inflow (CNY), period_return (%), turnover (%)."""
    key = f"ths:flow:{horizon}"
    df = cache.get_or_fetch(key, lambda: ak.stock_fund_flow_individual(symbol=THS_HORIZONS[horizon]),
                            ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame(columns=["code", "net_inflow", "period_return", "turnover"])
    # read_html parses codes as integers, dropping leading zeros.
    out = pd.DataFrame({
        "code": df["股票代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6),
        "net_inflow": df["资金流入净额"].map(parse_cn_number),
        "period_return": df["阶段涨跌幅"].map(parse_cn_number),
        "turnover": df["连续换手率"].map(parse_cn_number),
    })
    return out.drop_duplicates("code").reset_index(drop=True)


def money_flow(code, horizons=("3d", "5d", "10d", "20d")):
    """This stock's THS net inflow per horizon, plus each table for
    cross-sectional ranking. Returns ({horizon: inflow}, {horizon: table})."""
    symbol, _ = normalize_code(code)
    values, tables = {}, {}
    for h in horizons:
        try:
            t = money_flow_table(h)
        except Exception:
            continue
        tables[h] = t
        row = t[t["code"] == symbol]
        values[h] = float(row["net_inflow"].iloc[0]) if not row.empty else float("nan")
    return values, tables


def margin_snapshot(code, lookback_days=10):
    """Best-effort latest margin-trading balance, straight from the
    exchange's per-day detail file (no per-symbol history endpoint exists),
    so this is the most recent published day only."""
    symbol, market = normalize_code(code)
    fetch_fn = {"sh": ak.stock_margin_detail_sse,
                "sz": ak.stock_margin_detail_szse}.get(market)
    if fetch_fn is None:
        return None
    for delta in range(lookback_days):
        day = (datetime.now() - timedelta(days=delta)).strftime("%Y%m%d")
        try:
            df = cache.get_or_fetch(f"margin:{market}:{day}",
                                    lambda d=day: fetch_fn(date=d), ttl_seconds=86400)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        code_col = _pick_col(df, "标的证券代码", "证券代码")
        if code_col is None:
            continue
        row = df[df[code_col].astype(str).str.zfill(6) == symbol]
        if row.empty:
            continue
        row = row.iloc[0]
        return {"date": day,
                "margin_balance": _pick(row, "融资余额"),
                "margin_buy": _pick(row, "融资买入额"),
                "short_balance": _pick(row, "融券余量")}
    return None


# ---------------------------------------------------------------------------
# Fundamentals (BaoStock)
# ---------------------------------------------------------------------------

def _recent_quarters(n=8):
    now = datetime.now()
    year, quarter = now.year, (now.month - 1) // 3 + 1
    out = []
    for _ in range(n):
        out.append((year, quarter))
        year, quarter = (year, quarter - 1) if quarter > 1 else (year - 1, 4)
    return out


def financial_abstract(code, quarters=8, ttl=86400):
    """Quarterly fundamentals, most recent published quarter first:
    report_period, revenue_yoy, net_profit_yoy, gross_margin, net_margin,
    roe (annualized), eps (TTM) — all percentages except eps.

    BaoStock reports ratios as decimals and ROE / revenue as year-to-date
    figures, so ROE is annualized by quarter and revenue growth is computed
    against the same quarter a year earlier.
    """
    bcode = _bs(code)

    def fetch():
        rows = []
        for year, q in _recent_quarters(quarters + 4):
            try:
                p = bsapi.profit(bcode, year, q)
            except Exception:
                p = pd.DataFrame()
            if p is None or p.empty:
                continue
            try:
                g = bsapi.growth(bcode, year, q)
            except Exception:
                g = pd.DataFrame()
            pr = p.iloc[0]
            gr = g.iloc[0] if g is not None and not g.empty else {}
            rows.append({"year": year, "quarter": q,
                         "report_period": pr.get("statDate"),
                         "revenue": pr.get("MBRevenue"),
                         "gross_margin": pr.get("gpMargin"),
                         "net_margin": pr.get("npMargin"),
                         "roe_ytd": pr.get("roeAvg"),
                         "eps": pr.get("epsTTM"),
                         "net_profit_yoy": _pick(gr, "YOYNI")})
        return pd.DataFrame(rows)

    df = cache.get_or_fetch(f"bs:fin:{bcode}", fetch, ttl_seconds=ttl)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    prior = {(r["year"], r["quarter"]): r["revenue"] for _, r in df.iterrows()}
    df["revenue_yoy"] = [
        (r["revenue"] / prior[(r["year"] - 1, r["quarter"])] - 1) * 100
        if prior.get((r["year"] - 1, r["quarter"])) not in (None, 0)
        and pd.notna(prior.get((r["year"] - 1, r["quarter"]))) and pd.notna(r["revenue"])
        else np.nan
        for _, r in df.iterrows()
    ]
    out = pd.DataFrame({
        "report_period": df["report_period"],
        "revenue_yoy": df["revenue_yoy"],
        "net_profit_yoy": _to_num(df["net_profit_yoy"]) * 100,
        "gross_margin": _to_num(df["gross_margin"]) * 100,
        "net_margin": _to_num(df["net_margin"]) * 100,
        "roe": _to_num(df["roe_ytd"]) * 100 * 4 / df["quarter"],
        "eps": _to_num(df["eps"]),
    })
    return out.head(quarters).reset_index(drop=True)
