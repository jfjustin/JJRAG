"""Offline self-test: replaces every data source with synthetic data shaped
like the real responses, then runs the full report pipeline end to end.

None of the sources has a test mode, so this is the only way to check the
bsapi -> data -> factors -> model -> report wiring without the network. It
is NOT a validation of real market data; run `python -m quant.cli <code>`
for that.

    python -m quant.selftest

BaoStock is faked at the library level — result sets hand back string
fields, minute bars carry YYYYMMDDHHMMSSsss timestamps, ratios are
decimals — so the adapter's parsing is exercised too. THS tables carry
'1.23亿'-style amounts and integer codes with lost leading zeros, as
read_html produces them.

Scenarios: 18-month daily; two-month daily, 60m, 30m and 15m; and a
two-month daily run with BaoStock's whole-market query unavailable, which
must fall back to a peer-sample cross-section. The synthetic universe has
known lead-lags, and the test checks they are recovered.

`install_mocks()` is reusable — e.g. to drive the dashboard offline.
"""

import sys
import zlib
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

TARGET_CODE = "300476"
TARGET_NAME = "测试标的"
INDUSTRY = "C39计算机、通信和其他电子设备制造业"
OTHER_INDUSTRY = "C38电气机械和器材制造业"
N_DAYS = 400
N_INTRADAY_DAYS = 90

# Bar close times per session, per minute frequency.
SESSION_BARS = {
    "60": ["10:30", "11:30", "14:00", "15:00"],
    "30": ["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00"],
    "15": ["09:45", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30",
           "13:15", "13:30", "13:45", "14:00", "14:15", "14:30", "14:45", "15:00"],
}
# Per-bar sector step and idiosyncratic noise, scaled by sqrt(bar length)
# so every timeframe has the same signal-to-noise (true-lag corr ~0.3).
SCALES = {"d": (0.018, 0.012), "60": (0.009, 0.006), "30": (0.0064, 0.0042),
          "15": (0.0045, 0.003)}

LEADS = {"300000": 3.0, "300001": 2.0, "300002": 4.0,
         "300003": -2.0, "300004": -3.0}
NOISE_ONLY = {"300005", "300006"}
UNIVERSE = [TARGET_CODE] + [f"{300000 + i:06d}" for i in range(59)]


def _seed(symbol):
    # crc32, not hash(): str hashing is salted per process.
    return zlib.crc32(symbol.encode()) % (2 ** 31)


def _true_lead(label):
    """The lead (in bars) the synthetic universe assigned to a peer label."""
    code = label.split()[0]
    return LEADS.get(code, float((_seed(code) % 7) - 3))


def _industry_of(code):
    return INDUSTRY if code == TARGET_CODE or int(code) % 3 != 2 else OTHER_INDUSTRY


def _trading_days(n):
    return pd.bdate_range(end=pd.Timestamp(datetime.now().date()), periods=n)


def _sector_universe(n, step, max_shift=6, seed=5):
    rng = np.random.default_rng(seed)
    pad = max_shift + 2
    sector = pd.Series(rng.normal(0.0004, 1.0, n + 2 * pad)).rolling(4).mean().fillna(0).values * step

    def series_for(lead, beta, noise_seed, start_price, noise):
        r = np.random.default_rng(noise_seed)
        s = pad + int(round(lead))
        return start_price * np.exp(np.cumsum(sector[s:s + n] * beta + r.normal(0, noise, n)))
    return series_for


def _close_for(symbol, series_for, noise):
    seed = _seed(symbol)
    if symbol == TARGET_CODE:
        return series_for(0.0, 1.0, seed, 110.0, noise)
    if symbol in NOISE_ONLY:
        return series_for(0.0, 0.0, seed, 60.0, noise)
    return series_for(_true_lead(symbol), 0.9, seed, float(50 + seed % 150), noise)


def _bs_code(code):
    return f"{'sh' if code.startswith('6') else 'sz'}.{code}"


class _ResultSet:
    """Mimics baostock.data.resultset.ResultData."""

    def __init__(self, df, error_code="0", error_msg="success"):
        self.error_code, self.error_msg = error_code, error_msg
        self.fields = list(df.columns)
        self._rows = df.astype(str).values.tolist()
        self._i = -1

    def next(self):
        self._i += 1
        return self._i < len(self._rows)

    def get_row_data(self):
        return self._rows[self._i]


def _fmt(x, d=4):
    return f"{x:.{d}f}"


def _build_daily(code, days):
    step, noise = SCALES["d"]
    close = _close_for(code, _build_daily.universe, noise)
    rng = np.random.default_rng(_seed(code) + 1)
    pre = np.concatenate([[close[0]], close[:-1]])
    vol = rng.integers(1_000_000, 8_000_000, len(close))
    pe = (rng.uniform(10, 80) * close / close[-1]).round(4)
    return pd.DataFrame({
        "date": days.strftime("%Y-%m-%d"), "code": _bs_code(code),
        "open": [_fmt(c * (1 + rng.normal(0, 0.003))) for c in close],
        "high": [_fmt(c * 1.01) for c in close], "low": [_fmt(c * 0.99) for c in close],
        "close": [_fmt(c) for c in close], "preclose": [_fmt(c) for c in pre],
        "volume": vol.astype(str), "amount": [_fmt(v * c, 2) for v, c in zip(vol, close)],
        "turn": [_fmt(t) for t in rng.uniform(0.5, 6, len(close))],
        "tradestatus": "1", "pctChg": [_fmt(x) for x in (close / pre - 1) * 100],
        "peTTM": [_fmt(x) for x in pe], "pbMRQ": _fmt(rng.uniform(1, 12)),
        "psTTM": _fmt(rng.uniform(1, 10)), "isST": "0",
    })


def _build_minute(code, freq, days):
    step, noise = SCALES[freq]
    stamps = [pd.Timestamp(f"{d.date()} {t}") for d in days for t in SESSION_BARS[freq]]
    close = _close_for(code, _build_minute.universes[freq], noise)[-len(stamps):]
    rng = np.random.default_rng(_seed(code) + int(freq))
    vol = rng.integers(50_000, 900_000, len(close))
    return pd.DataFrame({
        "date": [s.strftime("%Y-%m-%d") for s in stamps],
        "time": [s.strftime("%Y%m%d%H%M%S") + "000" for s in stamps],
        "code": _bs_code(code),
        "open": [_fmt(c * (1 + rng.normal(0, 0.002))) for c in close],
        "high": [_fmt(c * 1.004) for c in close], "low": [_fmt(c * 0.996) for c in close],
        "close": [_fmt(c) for c in close], "volume": vol.astype(str),
        "amount": [_fmt(v * c, 2) for v, c in zip(vol, close)],
    })


def _ths_amount(v):
    return f"{v / 1e8:.2f}亿" if abs(v) >= 1e8 else f"{v / 1e4:.2f}万"


def install_mocks(whole_market=True):
    """Patch BaoStock, Sina, THS and the exchange margin fetchers. Returns a
    function that removes the patches. whole_market=False simulates a
    BaoStock release without the per-date whole-market query."""
    from quant import bsapi

    days = _trading_days(N_DAYS)
    intraday_days = days[-N_INTRADAY_DAYS:]
    _build_daily.universe = _sector_universe(N_DAYS, SCALES["d"][0])
    _build_minute.universes = {
        f: _sector_universe(N_INTRADAY_DAYS * len(SESSION_BARS[f]), SCALES[f][0])
        for f in SESSION_BARS}
    daily = {c: _build_daily(c, days) for c in UNIVERSE}
    minute = {}

    def query_history(code, fields, start_date=None, end_date=None, frequency="d", adjustflag="3"):
        sym = code.split(".")[1]
        if sym == "000300":  # CSI 300 daily
            step, noise = SCALES["d"]
            close = _build_daily.universe(0.0, 0.6, 42, 3800.0, noise * 0.5)
            df = pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "code": code,
                               "close": [_fmt(c) for c in close]})
        elif frequency == "d":
            df = daily[sym]
        else:
            if (sym, frequency) not in minute:
                minute[(sym, frequency)] = _build_minute(sym, frequency, intraday_days)
            df = minute[(sym, frequency)]
        d = pd.to_datetime(df["date"])
        keep = (d >= pd.Timestamp(start_date)) & (d <= pd.Timestamp(end_date))
        return _ResultSet(df[keep][[f for f in fields.split(",") if f in df.columns]])

    industry_df = pd.DataFrame({
        "updateDate": "2026-09-01", "code": [_bs_code(c) for c in UNIVERSE],
        "code_name": [TARGET_NAME if c == TARGET_CODE else f"公司{int(c) - 300000}" for c in UNIVERSE],
        "industry": [_industry_of(c) for c in UNIVERSE],
        "industryClassification": "证监会行业分类"})

    def query_stock_industry(code="", date=""):
        return _ResultSet(industry_df if not code else industry_df[industry_df["code"] == code])

    def query_stock_basic(code="", code_name=""):
        row = industry_df[industry_df["code"] == code][["code", "code_name"]]
        return _ResultSet(row)

    def query_trade_dates(start_date=None, end_date=None):
        cal = pd.date_range(start_date, end_date)
        return _ResultSet(pd.DataFrame({
            "calendar_date": cal.strftime("%Y-%m-%d"),
            "is_trading_day": ["1" if d in days else "0" for d in cal]}))

    def query_daily_history_k_AStock(date=""):
        rows = [daily[c][daily[c]["date"] == date] for c in UNIVERSE]
        return _ResultSet(pd.concat(rows))

    published = [q for q in (datetime.now().year * 4 + (datetime.now().month - 1) // 3 - k
                             for k in range(1, 12))]

    def _quarter_ok(year, quarter):
        return year * 4 + quarter - 1 in published

    def query_profit_data(code, year=None, quarter=None):
        if not _quarter_ok(year, quarter):
            return _ResultSet(pd.DataFrame(columns=["code"]))
        k = year * 4 + quarter
        return _ResultSet(pd.DataFrame([{
            "code": code, "pubDate": f"{year}-{quarter * 3:02d}-28",
            "statDate": f"{year}-{quarter * 3:02d}-30",
            "roeAvg": _fmt(0.03 * quarter + 0.001 * (k % 5)),
            "npMargin": "0.142", "gpMargin": "0.215", "netProfit": "1.0e9",
            "epsTTM": _fmt(1.0 + 0.05 * (k % 8)),
            "MBRevenue": _fmt(5e9 * quarter * (1 + 0.04 * (year - 2024)), 0),
            "totalShare": "8.6e8", "liqaShare": "8.5e8"}]))

    def query_growth_data(code, year=None, quarter=None):
        if not _quarter_ok(year, quarter):
            return _ResultSet(pd.DataFrame(columns=["code"]))
        return _ResultSet(pd.DataFrame([{
            "code": code, "pubDate": "", "statDate": "", "YOYEquity": "0.1",
            "YOYAsset": "0.12", "YOYNI": "0.35", "YOYEPSBasic": "0.3", "YOYPNI": "0.34"}]))

    fake_bs = SimpleNamespace(
        login=lambda *a, **k: SimpleNamespace(error_code="0", error_msg="success"),
        logout=lambda *a, **k: None,
        query_history_k_data_plus=query_history,
        query_stock_industry=query_stock_industry,
        query_stock_basic=query_stock_basic,
        query_trade_dates=query_trade_dates,
        query_profit_data=query_profit_data,
        query_growth_data=query_growth_data,
    )
    if whole_market:
        fake_bs.query_daily_history_k_AStock = query_daily_history_k_AStock

    def sina_minute(symbol, period="1", adjust=""):
        stamps = [pd.Timestamp(f"{d.date()} {t}") for d in intraday_days
                  for t in SESSION_BARS[period]]
        step, noise = SCALES[period]
        close = _build_minute.universes[period](0.0, 0.6, 42, 3800.0, noise * 0.5)[-len(stamps):]
        return pd.DataFrame({"day": [s.strftime("%Y-%m-%d %H:%M:%S") for s in stamps],
                             "open": close, "high": close, "low": close,
                             "close": [_fmt(c) for c in close], "volume": 1})[-1970:]

    def ths_flow(symbol="即时"):
        rng = np.random.default_rng(_seed(symbol))
        n = len(UNIVERSE) + 1
        codes = [int(c) for c in UNIVERSE] + [1]  # 000001 loses its zeros in read_html
        return pd.DataFrame({
            "序号": range(1, n + 1), "股票代码": codes, "股票简称": "x", "最新价": 10.0,
            "阶段涨跌幅": [f"{x:.2f}%" for x in rng.normal(0, 6, n)],
            "连续换手率": [f"{x:.2f}%" for x in rng.uniform(2, 40, n)],
            "资金流入净额": [_ths_amount(x) for x in rng.normal(0, 2e8, n)]})

    def margin_szse(date):
        return pd.DataFrame({"证券代码": [TARGET_CODE], "证券简称": [TARGET_NAME],
                             "融资买入额": [5.2e7], "融资余额": [3.1e9],
                             "融券卖出量": [10000], "融券余量": [200000]})

    patches = [
        mock.patch.object(bsapi, "bs", fake_bs),
        mock.patch.object(bsapi, "_logged_in", False),
        mock.patch("akshare.stock_zh_a_minute", side_effect=sina_minute),
        mock.patch("akshare.stock_fund_flow_individual", side_effect=ths_flow),
        mock.patch("akshare.stock_margin_detail_sse", side_effect=lambda date: pd.DataFrame()),
        mock.patch("akshare.stock_margin_detail_szse", side_effect=margin_szse),
    ]
    for p in patches:
        p.start()

    def stop():
        for p in patches:
            p.stop()
    return stop


def _print_report(r):
    print(f"\n{'=' * 76}\n{r.code} {r.name} [{r.freq}] {r.start} → {r.end}  "
          f"{len(r.prices)} bars  MA{r.short_ma}/MA{r.long_ma}  cross-section: {r.cross_section}")
    print(f"Composite {r.composite_score:.1f} -> {r.stance}   " + "  ".join(
        f"{c}={s:.0f}" if s == s else f"{c}=n/a" for c, (s, _) in r.breakdown.items()))
    net = r.linkage.get("leadlag_network")
    if net is not None and not net.empty:
        print(f"  linked {r.linkage.get('n_linked_peers')}/{len(net)}, leaders "
              f"{r.linkage.get('n_leading_peers', 0)}, signal "
              f"{r.linkage.get('leading_peer_signal_pct', float('nan')):+.2f}%")
    for w in r.warnings:
        print(f"  ! {w}")


def _run(freq, days, short, long, whole_market=True):
    from quant import cache, report
    stop = install_mocks(whole_market=whole_market)
    try:
        cache.clear()
        return report.build_report(f"{TARGET_CODE}.SZ", lookback_days=days,
                                   short_ma=short, long_ma=long, freq=freq)
    finally:
        stop()
        cache.clear()


def run():
    from quant import data

    # Unit checks on parsing that every scenario relies on.
    assert data.parse_cn_number("1.23亿") == 1.23e8
    assert data.parse_cn_number("-4567.8万") == -4.5678e7
    assert data.parse_cn_number("12.3%") == 12.3
    assert data.parse_cn_number("--") != data.parse_cn_number("--")  # NaN

    full = _run("daily", 548, 20, 60)
    narrow = {f: _run(f, 61, *((5, 20) if f == "daily" else (20, 60)))
              for f in ("daily", "60m", "30m", "15m")}
    fallback = _run("daily", 61, 5, 20, whole_market=False)

    for r in [full, *narrow.values(), fallback]:
        _print_report(r)

    # --- Full daily -------------------------------------------------------
    r = full
    assert 0 <= r.composite_score <= 100
    assert r.cross_section.startswith("whole market")
    for key in ("market_beta", "industry_beta", "ret_60d_pctile_market",
                "ret_60d_pctile_industry", "value_rank_market", "value_rank_industry",
                "peer_avg_corr"):
        assert r.linkage.get(key) == r.linkage.get(key), f"full: {key} missing"
    assert r.technical.get("ret_12m") == r.technical.get("ret_12m"), "full: ret_12m missing"
    cf = r.capital_flow
    assert set(r.flow_values) == {"3d", "5d", "10d", "20d"}, r.flow_values
    assert len(set(r.flow_values.values())) == 4, \
        f"each THS horizon should be read from its own table: {r.flow_values}"
    for key in ("inflow_5d_pctile_industry", "inflow_5d_pctile_market", "bar_direction_flow"):
        assert cf.get(key) == cf.get(key), f"full: {key} missing"
    assert cf.get("inflow_scaled_by_turnover") is True
    assert cf.get("margin_balance") == 3.1e9
    fin = r.financials
    assert not fin.empty and fin["roe"].between(3, 30).all(), \
        f"ROE should be annualized percent: {fin['roe'].tolist()}"
    assert fin["net_profit_yoy"].iloc[0] == 35.0
    assert fin["revenue_yoy"].notna().any(), "revenue YoY never computed"
    net = r.linkage["leadlag_network"]
    linked = net[net["linked"].astype(bool)]
    assert len(linked) >= len(net) * 0.6, f"full: only {len(linked)}/{len(net)} linked"
    err = (linked["lead_days"] - pd.Series({k: _true_lead(k) for k in linked.index})).abs()
    assert (err <= 1.0).mean() >= 0.75, f"full: lag recovery too poor {err.round(2).to_dict()}"
    assert all(TARGET_CODE not in label for label in net.index), "target in its own peer set"

    # --- Two-month daily --------------------------------------------------
    r = narrow["daily"]
    assert 35 <= len(r.prices) <= 46, f"2m daily: {len(r.prices)} bars"
    for key in ("ret_12m", "ret_6m", "ret_3m", "momentum_12_1", "hi_52w"):
        v = r.technical.get(key)
        assert v != v, f"2m daily: {key} must be NaN, got {v}"

    # --- Two-month intraday -----------------------------------------------
    daily_sigma = narrow["daily"].technical["sde_diffusion_annual"]
    for freq, per_day in (("60m", 4), ("30m", 8), ("15m", 16)):
        r = narrow[freq]
        assert r.bars_per_day == per_day and r.bar_minutes == int(freq[:-1])
        n = len(r.prices)
        assert 40 * per_day <= n <= 46 * per_day, f"{freq}: {n} bars"
        assert r.prices["date"].dt.time.astype(str).isin(
            [t + ":00" for t in SESSION_BARS[freq[:-1]]]).all(), f"{freq}: bad bar times"
        assert r.technical, f"{freq}: technical empty"
        assert "market_beta" in r.linkage, f"{freq}: no market beta (Sina index bars)"
        assert "industry_beta" in r.linkage, f"{freq}: no industry beta (peer composite)"
        net = r.linkage.get("leadlag_network")
        assert net is not None and not net.empty, f"{freq}: lead-lag network empty"
        assert np.allclose(net["lead_hours"], net["lead_bars"] * int(freq[:-1]) / 60)
        assert np.allclose(net["lead_days"], net["lead_bars"] / per_day)
        assert r.linkage.get("n_linked_peers", 0) >= len(net) * 0.5, \
            f"{freq}: only {r.linkage.get('n_linked_peers')}/{len(net)} linked"
        ratio = r.technical["sde_diffusion_annual"] / daily_sigma
        assert 0.2 < ratio < 5, f"{freq}: annualized diffusion off vs daily ({ratio:.2f}x)"

    # --- Whole-market query unavailable -------------------------------------
    r = fallback
    assert r.cross_section == "peer sample"
    assert any("Whole-market snapshot unavailable" in w for w in r.warnings)
    assert r.linkage.get("ret_60d_pctile_market") is None
    assert r.linkage.get("value_rank_industry") == r.linkage.get("value_rank_industry"), \
        "fallback: industry value rank should come from the peer sample"
    assert len(r.industry_peers) <= 13

    print("\nSelf-test passed: full daily; two-month daily/60m/30m/15m; "
          "peer-sample fallback.")


if __name__ == "__main__":
    run()
    sys.exit(0)
