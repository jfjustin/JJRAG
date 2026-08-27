"""Quantitative share-market helpers built on EMQuantAPI.

All functions assume an active session (see ``em_client.session``) and
return pandas DataFrames ready for quant workflows.
"""

import pandas as pd

from . import em_client
from .em_client import EmQuantError


def _api():
    if em_client.c is None:
        raise ImportError(em_client._SDK_MISSING_MSG)
    return em_client.c


def _check(result, what):
    if result.ErrorCode != 0:
        raise EmQuantError(f"{what} failed ({result.ErrorCode}): {result.ErrorMsg}")
    return result


def index_constituents(index_sector_code="009006256", trade_date=None):
    """Return the constituent codes of an index sector on a trade date.

    The default sector code 009006256 is the CSI 300 universe on Choice;
    look up other universes in the Choice terminal's sector browser.
    """
    trade_date = trade_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    result = _check(_api().sector(index_sector_code, trade_date), "sector")
    # Data alternates code, name, code, name, ...
    codes = result.Data[::2]
    names = result.Data[1::2]
    return pd.DataFrame({"code": codes, "name": names})


def daily_history(codes, start, end, indicators="OPEN,HIGH,LOW,CLOSE,VOLUME,AMOUNT",
                  options="period=1,adjustflag=3,curtype=1,order=1,market=CNSESH"):
    """Fetch daily bars for one or more securities via csd.

    ``codes`` may be a string ("600519.SH") or list of strings. Returns a
    long-format DataFrame: date, code, <indicator columns>.
    """
    if isinstance(codes, (list, tuple)):
        codes = ",".join(codes)
    result = _check(_api().csd(codes, indicators, start, end, options), "csd")

    # Data[code] is a list with one entry per indicator, each an array of
    # values aligned with result.Dates.
    frames = []
    indicator_list = [i.strip() for i in indicators.split(",")]
    for code, per_indicator in result.Data.items():
        frame = pd.DataFrame(dict(zip(indicator_list, per_indicator)))
        frame.insert(0, "date", pd.to_datetime(result.Dates))
        frame.insert(1, "code", code)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def snapshot(codes, indicators="NAME,CLOSE,PE,PB,MV", options=""):
    """Cross-sectional snapshot for a list of securities via css."""
    if isinstance(codes, (list, tuple)):
        codes = ",".join(codes)
    result = _check(_api().css(codes, indicators, options), "css")
    indicator_list = [i.strip() for i in indicators.split(",")]
    rows = {code: dict(zip(indicator_list, values))
            for code, values in result.Data.items()}
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "code"
    return frame.reset_index()


def moving_average_signal(prices, short=20, long=60):
    """Classic dual moving-average crossover signal on a close-price series.

    ``prices`` is a DataFrame from ``daily_history`` for a single code.
    Returns the frame with ma_short, ma_long, and signal (+1 long / 0 flat).
    """
    out = prices.sort_values("date").copy()
    out["ma_short"] = out["CLOSE"].rolling(short).mean()
    out["ma_long"] = out["CLOSE"].rolling(long).mean()
    out["signal"] = (out["ma_short"] > out["ma_long"]).astype(int)
    return out
