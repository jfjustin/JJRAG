"""Offline self-test: mocks every AKShare call with synthetic data shaped
like the real API responses, then runs the full report pipeline end to end.

This exists because AKShare talks to live public endpoints with no test
mode — this is the only way to check the data.py -> factors.py -> model.py
-> report.py wiring is correct without a network call. It is NOT a
validation of AKShare's real-world data; run `python -m quant.cli <code>`
against the network for that.

    python -m quant.selftest

Three scenarios run: the default ~18-month daily report, a two-month
daily window, and a two-month 60-minute window. The mocks honor the
start/end dates they're asked for, on a calendar ending today, so the
narrow scenarios really do get narrow data.

`install_mocks()` is reusable — e.g. to drive the dashboard offline.
"""

import sys
import zlib
from datetime import datetime
from unittest import mock

import numpy as np
import pandas as pd

TARGET_CODE = "300274"
TARGET_INDUSTRY = "电源设备"
N_DAYS = 400                       # synthetic daily history length (bars)
HOURLY_CLOSES = ("10:30", "11:30", "14:00", "15:00")

# Deterministic lead assignment (in bars): the target sits at 0, some
# peers lead it, some lag, and two carry no sector loading at all.
LEADS = {"300000": 3.0, "300001": 2.0, "300002": 4.0,
         "300003": -2.0, "300004": -3.0}
NOISE_ONLY = {"300005", "300006"}


def _seed(symbol):
    # crc32, not hash(): str hashing is salted per process, which would
    # make this test's data — and its assertions — differ run to run.
    return zlib.crc32(symbol.encode()) % (2 ** 31)


def _true_lead(label):
    """The lead (in bars) the mock universe assigned to a peer label."""
    code = label.split()[0]
    return LEADS.get(code, float((_seed(code) % 7) - 3))


def _daily_calendar(n=N_DAYS):
    return pd.bdate_range(end=pd.Timestamp(datetime.now().date()), periods=n)


def _hourly_calendar(n_days=N_DAYS):
    stamps = [pd.Timestamp(f"{d.date()} {t}")
              for d in _daily_calendar(n_days) for t in HOURLY_CLOSES]
    return pd.DatetimeIndex(stamps)


def _hist_from_close(close, seed=0, dates=None, time_col="日期", fmt="%Y-%m-%d"):
    """Wrap a close-price array in the column layout AKShare returns."""
    n = len(close)
    rng = np.random.default_rng(seed)
    if dates is None:
        dates = _daily_calendar(n)
    close = np.asarray(close, dtype=float)
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(1_000_000, 8_000_000, n)
    return pd.DataFrame({
        time_col: dates.strftime(fmt),
        "开盘": open_.round(2), "收盘": close.round(2),
        "最高": high.round(2), "最低": low.round(2),
        "成交量": volume, "成交额": (volume * close).round(0),
        "振幅": rng.uniform(1, 5, n).round(2),
        "涨跌幅": (pd.Series(close).pct_change().fillna(0) * 100).round(2),
        "涨跌额": pd.Series(close).diff().fillna(0).round(2),
        "换手率": rng.uniform(0.5, 6, n).round(2),
    })


def _sector_universe(n, max_shift=6, seed=5, step_scale=0.018):
    """A sector factor plus per-symbol lead/lag offsets.

    Symbol i's returns are the sector factor read `lead_i` bars *ahead* of
    the target's, so a symbol with lead_i > 0 genuinely moves before the
    target — which is what the lead-lag estimator has to recover.
    """
    rng = np.random.default_rng(seed)
    pad = max_shift + 2
    sector = (pd.Series(rng.normal(0.0004, 1.0, n + 2 * pad))
              .rolling(4).mean().fillna(0).values * step_scale)

    def series_for(lead, beta, noise_seed, start_price, noise=0.012):
        r = np.random.default_rng(noise_seed)
        start = pad + int(round(lead))
        rets = sector[start:start + n] * beta + r.normal(0, noise, n)
        return start_price * np.exp(np.cumsum(rets))

    return series_for


def _close_for(symbol, series_for, noise=0.012):
    seed = _seed(symbol)
    if symbol == TARGET_CODE:
        return series_for(0.0, 1.0, seed, 110.0, noise)
    if symbol in NOISE_ONLY:
        return series_for(0.0, 0.0, seed, 60.0, noise)
    lead = LEADS.get(symbol, float((seed % 7) - 3))
    return series_for(lead, 0.9, seed, float(50 + seed % 150), noise)


def _between(df, col, start, end):
    """Filter like the real endpoints do: on a date or datetime column."""
    t = pd.to_datetime(df[col])
    lo = pd.Timestamp(start) if start else t.min()
    hi = pd.Timestamp(end) if end else t.max()
    if len(str(end or "")) <= 8:  # 'YYYYMMDD' means through end of day
        hi = hi + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[(t >= lo) & (t <= hi)].reset_index(drop=True)


def _synthetic_spot(n=60, seed=1):
    rng = np.random.default_rng(seed)
    codes = [TARGET_CODE] + [f"{300000 + i:06d}" for i in range(n - 1)]
    return pd.DataFrame({
        "代码": codes,
        "名称": [f"公司{i}" for i in range(n)],
        "最新价": rng.uniform(8, 200, n).round(2),
        "涨跌幅": rng.normal(0, 3, n).round(2),
        "成交量": rng.integers(1e5, 5e6, n),
        "成交额": rng.uniform(1e7, 5e8, n),
        "振幅": rng.uniform(1, 6, n).round(2),
        "换手率": rng.uniform(0.3, 8, n).round(2),
        "市盈率-动态": rng.uniform(8, 90, n).round(2),
        "量比": rng.uniform(0.5, 3, n).round(2),
        "总市值": rng.uniform(2e9, 5e11, n),
        "流通市值": rng.uniform(1e9, 4e11, n),
        "市净率": rng.uniform(0.8, 15, n).round(2),
        "60日涨跌幅": rng.normal(0, 15, n).round(2),
        "年初至今涨跌幅": rng.normal(0, 25, n).round(2),
    })


def install_mocks():
    """Patch every AKShare call the package makes. Returns a function that
    removes the patches again."""
    daily_universe = _sector_universe(N_DAYS)
    hourly_universe = _sector_universe(N_DAYS * 4, step_scale=0.009)
    daily_cal, hourly_cal = _daily_calendar(), _hourly_calendar()
    daily_cache, hourly_cache = {}, {}

    def fake_stock_zh_a_hist(symbol, start_date=None, end_date=None, **_):
        if symbol not in daily_cache:
            daily_cache[symbol] = _hist_from_close(
                _close_for(symbol, daily_universe), _seed(symbol), daily_cal)
        return _between(daily_cache[symbol], "日期", start_date, end_date)

    def fake_stock_min(symbol, start_date=None, end_date=None, period="60", **_):
        if symbol not in hourly_cache:
            hourly_cache[symbol] = _hist_from_close(
                _close_for(symbol, hourly_universe, noise=0.006), _seed(symbol),
                hourly_cal, time_col="时间", fmt="%Y-%m-%d %H:%M:%S")
        return _between(hourly_cache[symbol], "时间", start_date, end_date)

    def fake_individual_info_em(symbol, **_):
        rows = [("股票代码", symbol), ("股票简称", "阳光电源"),
                ("行业", TARGET_INDUSTRY), ("总股本", 2e9), ("流通股", 1.9e9),
                ("总市值", 2.1e11), ("流通市值", 2.0e11),
                ("上市时间", "20111102"), ("最新", 110.5)]
        return pd.DataFrame(rows, columns=["item", "value"])

    spot_df = _synthetic_spot()

    def fake_industry_cons_em(symbol):
        peers = spot_df.sample(n=15, random_state=2).copy()
        if TARGET_CODE not in peers["代码"].values:
            peers.iloc[0, peers.columns.get_loc("代码")] = TARGET_CODE
        return peers[["代码", "名称", "最新价", "涨跌幅", "市盈率-动态",
                      "市净率", "换手率", "成交额"]]

    sector_daily = _hist_from_close(daily_universe(0.0, 1.0, 99, 1000.0, 0.004), 99, daily_cal)
    sector_hourly = _hist_from_close(hourly_universe(0.0, 1.0, 99, 1000.0, 0.002), 99,
                                     hourly_cal, time_col="日期时间",
                                     fmt="%Y-%m-%d %H:%M:%S")
    index_daily = _hist_from_close(daily_universe(0.0, 0.6, 42, 3800.0, 0.006), 42, daily_cal)
    index_hourly = _hist_from_close(hourly_universe(0.0, 0.6, 42, 3800.0, 0.003), 42,
                                    hourly_cal, time_col="时间", fmt="%Y-%m-%d %H:%M:%S")

    def fake_industry_hist_em(symbol, start_date=None, end_date=None, **_):
        return _between(sector_daily, "日期", start_date, end_date)

    def fake_industry_min(symbol, period="60"):
        return sector_hourly  # real endpoint has no date filter either

    def fake_index_daily_em(symbol, start_date=None, end_date=None):
        raw = _between(index_daily, "日期", start_date, end_date)
        return pd.DataFrame({"date": raw["日期"], "open": raw["开盘"],
                             "close": raw["收盘"], "high": raw["最高"],
                             "low": raw["最低"], "volume": raw["成交量"],
                             "amount": raw["成交额"]})

    def fake_index_min(symbol, period="60", start_date=None, end_date=None):
        return _between(index_hourly, "时间", start_date, end_date)

    def _flow(seed, scale):
        rng = np.random.default_rng(seed)
        dates = daily_cal[-120:]
        return pd.DataFrame({
            "日期": dates.strftime("%Y-%m-%d"),
            "主力净流入-净额": rng.normal(0, scale, len(dates)).round(0),
            "主力净流入-净占比": rng.normal(0, 5, len(dates)).round(2),
        })

    def fake_margin_szse(date):
        return pd.DataFrame({"证券代码": [TARGET_CODE], "证券简称": ["阳光电源"],
                             "融资买入额": [5.2e7], "融资余额": [3.1e9],
                             "融券卖出量": [10000], "融券余量": [200000]})

    def fake_fin_abstract(symbol, **_):
        return pd.DataFrame({
            "报告期": ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"],
            "营业总收入同比增长率": ["-18.26%", "-12.0%", "35.0%", "40.0%"],
            "净利润同比增长率": ["-40.12%", "-20.0%", "45.0%", "50.0%"],
            "销售毛利率": ["28.5%", "29.1%", "31.0%", "30.5%"],
            "销售净利率": ["14.7%", "15.2%", "18.0%", "17.5%"],
            "净资产收益率": ["9.8%", "5.1%", "22.0%", "18.0%"],
            "基本每股收益": ["1.10", "0.55", "5.20", "3.90"],
        })

    targets = {
        "stock_zh_a_hist": fake_stock_zh_a_hist,
        "stock_zh_a_hist_min_em": fake_stock_min,
        "stock_individual_info_em": fake_individual_info_em,
        "stock_zh_a_spot_em": lambda: spot_df,
        "stock_board_industry_cons_em": fake_industry_cons_em,
        "stock_board_industry_hist_em": fake_industry_hist_em,
        "stock_board_industry_hist_min_em": fake_industry_min,
        "stock_zh_index_daily_em": fake_index_daily_em,
        "index_zh_a_hist_min_em": fake_index_min,
        "stock_individual_fund_flow": lambda stock, market: _flow(7, 3e7),
        "stock_sector_fund_flow_hist": lambda symbol: _flow(8, 2e8),
        # Empty = "not a Stock-Connect name" — must not crash.
        "stock_hsgt_individual_detail_em": lambda symbol, **_: pd.DataFrame(),
        "stock_margin_detail_sse": lambda date: pd.DataFrame(),
        "stock_margin_detail_szse": fake_margin_szse,
        "stock_financial_abstract_ths": fake_fin_abstract,
    }
    patches = [mock.patch(f"akshare.{name}", side_effect=fn)
               for name, fn in targets.items()]
    for p in patches:
        p.start()

    def stop():
        for p in patches:
            p.stop()
    return stop


def _print_report(result):
    print(f"\n{'=' * 72}\n{result.code}  {result.name}  [{result.freq}]  "
          f"{result.start} → {result.end}  ({len(result.prices)} bars, "
          f"MA{result.short_ma}/MA{result.long_ma})")
    print(f"Composite score: {result.composite_score:.1f}  ->  {result.stance}")
    for cat, (score, _detail) in result.breakdown.items():
        print(f"  {cat:15s} {score:.1f}" if score == score else f"  {cat:15s} n/a")
    net = result.linkage.get("leadlag_network")
    if net is not None and not net.empty:
        cols = [c for c in ("lead_hours", "lead_days", "lag_corr", "threshold", "linked") if c in net]
        print("Lead-lag network (positive = peer leads this stock):")
        print(net[cols].sort_values(cols[0], ascending=False).round(2).to_string())
    for w in result.warnings:
        print(f"  ! {w}")


def run():
    stop = install_mocks()
    try:
        from quant import cache, report
        cache.clear()
        full = report.build_report(f"{TARGET_CODE}.SZ")
        narrow_daily = report.build_report(f"{TARGET_CODE}.SZ", lookback_days=61,
                                           short_ma=5, long_ma=20, freq="daily")
        narrow_hourly = report.build_report(f"{TARGET_CODE}.SZ", lookback_days=61,
                                            short_ma=20, long_ma=60, freq="60m")
    finally:
        stop()

    for r in (full, narrow_daily, narrow_hourly):
        _print_report(r)

    # --- Full daily report ------------------------------------------------
    r = full
    assert r.composite_score == r.composite_score and 0 <= r.composite_score <= 100
    for key in ("industry_beta", "market_beta", "ret_today_pctile_industry",
                "peer_avg_corr"):
        assert key in r.linkage, f"full: missing {key}"
    assert "sde_drift_diffusion_ratio" in r.technical
    assert r.technical.get("ret_12m") == r.technical.get("ret_12m"), \
        "full: 12m return should exist on ~18 months of data"
    net = r.linkage.get("leadlag_network")
    assert net is not None and not net.empty, "full: lead-lag network empty"
    assert r.linkage.get("n_leading_peers", 0) > 0, "full: no leading peers found"
    # Every synthetic peer here is sector-loaded with |lead| <= 3 bars, so on
    # ~6 months of daily bars most should be detected, and their lags read
    # close to the truth.
    linked = net[net["linked"].astype(bool)]
    assert len(linked) >= len(net) * 0.6, f"full: only {len(linked)}/{len(net)} peers linked"
    true = {label: _true_lead(label) for label in linked.index}
    err = np.abs(linked["lead_days"] - pd.Series(true))
    assert (err <= 1.0).mean() >= 0.75, f"full: lag recovery too poor: {err.round(2).to_dict()}"

    # --- Two-month daily --------------------------------------------------
    r = narrow_daily
    assert 35 <= len(r.prices) <= 46, f"narrow daily: {len(r.prices)} bars"
    assert r.technical, "narrow daily: technical category empty"
    t = r.technical
    for key in ("ret_12m", "ret_6m", "ret_3m", "momentum_12_1", "hi_52w"):
        assert t.get(key) != t.get(key), \
            f"narrow daily: {key} must be NaN on two months, got {t.get(key)}"
    assert t.get("ret_window") == t.get("ret_window"), "narrow daily: window return missing"
    assert len(r.fund_flow) <= 46, "narrow daily: fund flow not trimmed to window"

    # --- Two-month hourly -------------------------------------------------
    r = narrow_hourly
    assert r.bar_unit == "hour" and r.bars_per_day == 4
    assert 140 <= len(r.prices) <= 184, f"narrow hourly: {len(r.prices)} bars"
    assert r.technical, "narrow hourly: technical category empty"
    assert "market_beta" in r.linkage, "narrow hourly: no market beta from index minute bars"
    assert "industry_beta" in r.linkage, "narrow hourly: no industry beta from board minute bars"
    net = r.linkage.get("leadlag_network")
    assert net is not None and not net.empty, "narrow hourly: lead-lag network empty"
    assert {"lead_hours", "lead_days", "lead_bars"} <= set(net.columns)
    assert np.allclose(net["lead_days"] * 4, net["lead_hours"]), \
        "narrow hourly: hours/days conversion inconsistent"
    assert r.linkage.get("n_linked_peers", 0) > 0, "narrow hourly: no peers linked"
    # Hourly vol must be annualized on 4 bars/day, not as if each bar were a day.
    daily_vol = narrow_daily.technical["sde_diffusion_annual"]
    hourly_vol = r.technical["sde_diffusion_annual"]
    assert 0.2 < hourly_vol / daily_vol < 5, \
        f"hourly vs daily annualized diffusion out of line: {hourly_vol:.3f} vs {daily_vol:.3f}"

    print("\nSelf-test passed (full daily, two-month daily, two-month hourly).")


if __name__ == "__main__":
    run()
    sys.exit(0)
