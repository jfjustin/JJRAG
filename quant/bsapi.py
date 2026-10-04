"""Thin adapter over BaoStock (证券宝, http://baostock.com).

BaoStock is free and needs no registration — `login()` with no arguments
is an anonymous session. It serves A-share daily bars with valuation
fields (peTTM, pbMRQ, turnover), 5/15/30/60-minute bars, CSRC industry
classification and quarterly fundamentals from its own servers, with
history back to the 1990s. Data is end-of-day: bars for a session land
after that evening's update, there is no intraday real-time feed.

Everything here returns pandas DataFrames with numeric columns already
converted (BaoStock returns every field as a string). The session is
opened lazily once per process and reused.
"""

import threading

import pandas as pd

try:
    import baostock as bs
except ImportError:  # pragma: no cover
    bs = None

_lock = threading.Lock()
_logged_in = False

DAILY_FIELDS = ("date,code,open,high,low,close,preclose,volume,amount,"
                "turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,isST")
# Minute bars carry no valuation fields; `time` is the bar's close as
# YYYYMMDDHHMMSSsss.
MINUTE_FIELDS = "date,time,code,open,high,low,close,volume,amount"
MINUTE_FREQS = {"5", "15", "30", "60"}
ADJUST = {"hfq": "1", "qfq": "2", "": "3"}

# Fields that look numeric but must stay strings: `time` is a timestamp,
# the rest are codes and flags compared as text.
_KEEP_AS_TEXT = {"time", "code", "tradestatus", "isST", "type", "status"}


class BaoStockError(RuntimeError):
    pass


def _ensure_login():
    global _logged_in
    if bs is None:
        raise BaoStockError("baostock is not installed — pip install baostock")
    with _lock:
        if not _logged_in:
            lg = bs.login()
            if lg.error_code != "0":
                raise BaoStockError(f"BaoStock login failed: {lg.error_msg}")
            _logged_in = True


def _collect(rs, what):
    if rs.error_code != "0":
        raise BaoStockError(f"{what}: {rs.error_msg}")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    for col in df.columns:
        if col in _KEEP_AS_TEXT:
            continue
        # Convert only when every non-empty value parses: dates, names and
        # industry labels stay text without having to be listed by name.
        present = df[col].astype(str).str.strip().ne("")
        converted = pd.to_numeric(df[col].where(present), errors="coerce")
        if converted[present].notna().all():
            df[col] = converted
    return df


def bs_code(symbol, market):
    """('300476', 'sz') -> 'sz.300476'."""
    return f"{market}.{symbol}"


def history(code, start, end, frequency="d", adjust="qfq"):
    """Daily (frequency='d') or minute ('5','15','30','60') bars."""
    _ensure_login()
    fields = DAILY_FIELDS if frequency == "d" else MINUTE_FIELDS
    rs = bs.query_history_k_data_plus(code, fields, start_date=start,
                                      end_date=end, frequency=frequency,
                                      adjustflag=ADJUST[adjust])
    return _collect(rs, f"history {code} {frequency}")


def stock_industry(code=""):
    """CSRC industry for one code, or for every listed stock when code=''."""
    _ensure_login()
    return _collect(bs.query_stock_industry(code=code), "stock_industry")


def stock_basic(code=""):
    _ensure_login()
    return _collect(bs.query_stock_basic(code=code), "stock_basic")


def profit(code, year, quarter):
    _ensure_login()
    return _collect(bs.query_profit_data(code=code, year=year, quarter=quarter),
                    f"profit {code} {year}Q{quarter}")


def growth(code, year, quarter):
    _ensure_login()
    return _collect(bs.query_growth_data(code=code, year=year, quarter=quarter),
                    f"growth {code} {year}Q{quarter}")


def trade_dates(start, end):
    _ensure_login()
    df = _collect(bs.query_trade_dates(start_date=start, end_date=end), "trade_dates")
    return df[df["is_trading_day"] == 1]["calendar_date"].tolist()


def all_stocks_on(date):
    """Every A-share's daily bar on one date, in one call. Newer BaoStock
    releases only; the columns are defined server-side, so callers must
    treat any of them as optional."""
    _ensure_login()
    fn = getattr(bs, "query_daily_history_k_AStock", None)
    if fn is None:
        raise BaoStockError("this baostock version has no whole-market daily query")
    return _collect(fn(date=date), f"all stocks on {date}")
