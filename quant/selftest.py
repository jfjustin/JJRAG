"""Offline self-test: mocks every AKShare call with synthetic data shaped
like the real API responses, then runs the full report pipeline end to end.

This exists because AKShare talks to live public endpoints with no test
mode — this is the only way to check the data.py -> factors.py -> model.py
-> report.py wiring is correct without a network call. It is NOT a
validation of AKShare's real-world data; run `python -m quant.cli <code>`
against the network for that.

    python -m quant.selftest
"""

import sys
import zlib
from unittest import mock

import numpy as np
import pandas as pd


def _hist_from_close(close, seed=0):
    """Wrap a close-price array in the column layout stock_zh_a_hist returns."""
    n = len(close)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n)
    close = np.asarray(close, dtype=float)
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(1_000_000, 8_000_000, n)
    amount = volume * close
    return pd.DataFrame({
        "日期": dates.strftime("%Y-%m-%d"),
        "开盘": open_.round(2), "收盘": close.round(2),
        "最高": high.round(2), "最低": low.round(2),
        "成交量": volume, "成交额": amount.round(0),
        "振幅": (rng.uniform(1, 5, n)).round(2),
        "涨跌幅": (pd.Series(close).pct_change().fillna(0) * 100).round(2),
        "涨跌额": pd.Series(close).diff().fillna(0).round(2),
        "换手率": rng.uniform(0.5, 6, n).round(2),
    })


def _sector_universe(n=400, max_shift=6, seed=5):
    """A sector factor plus per-symbol lead/lag offsets.

    Symbol i's returns are the sector factor read `lead_i` days *ahead* of
    the target's, so a symbol with lead_i > 0 genuinely moves before the
    target — which is what the lead-lag estimator has to recover. Two
    symbols get zero loading on the factor (pure noise) to confirm they are
    filtered out rather than scored.
    """
    rng = np.random.default_rng(seed)
    pad = max_shift + 2
    sector = pd.Series(rng.normal(0.0004, 1.0, n + 2 * pad)).rolling(4).mean().fillna(0).values * 0.018

    def series_for(lead, beta, noise_seed, start_price):
        r = np.random.default_rng(noise_seed)
        start = pad + int(round(lead))
        rets = sector[start:start + n] * beta + r.normal(0, 0.012, n)
        return start_price * np.exp(np.cumsum(rets))

    return series_for


def _synthetic_hist(n=400, seed=0, start_price=100.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n)
    rets = rng.normal(0.0003, 0.02, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(1_000_000, 8_000_000, n)
    amount = volume * close
    return pd.DataFrame({
        "日期": dates.strftime("%Y-%m-%d"),
        "开盘": open_.round(2), "收盘": close.round(2),
        "最高": high.round(2), "最低": low.round(2),
        "成交量": volume, "成交额": amount.round(0),
        "振幅": (rng.uniform(1, 5, n)).round(2),
        "涨跌幅": (pd.Series(close).pct_change().fillna(0) * 100).round(2),
        "涨跌额": pd.Series(close).diff().fillna(0).round(2),
        "换手率": rng.uniform(0.5, 6, n).round(2),
    })


def _synthetic_spot(target_code, target_industry, n=60, seed=1):
    rng = np.random.default_rng(seed)
    codes = [target_code] + [f"{300000 + i:06d}" for i in range(n - 1)]
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
        "5分钟涨跌": rng.normal(0, 0.5, n).round(2),
        "最高": rng.uniform(8, 200, n).round(2),
        "最低": rng.uniform(8, 200, n).round(2),
        "今开": rng.uniform(8, 200, n).round(2),
        "昨收": rng.uniform(8, 200, n).round(2),
        "总市值": rng.uniform(2e9, 5e11, n),
        "流通市值": rng.uniform(1e9, 4e11, n),
        "涨速": rng.normal(0, 0.3, n).round(2),
        "市净率": rng.uniform(0.8, 15, n).round(2),
        "60日涨跌幅": rng.normal(0, 15, n).round(2),
        "年初至今涨跌幅": rng.normal(0, 25, n).round(2),
    })


def run():
    target_code = "300274"
    target_disp = "300274.SZ"
    target_industry = "电源设备"

    hist_cache = {}
    series_for = _sector_universe()
    # Deterministic lead assignment: the target sits at 0, some peers lead
    # it by 2-4 days, some lag, and two carry no sector loading at all.
    LEADS = {"300000": 3.0, "300001": 2.0, "300002": 4.0,
             "300003": -2.0, "300004": -3.0}
    NOISE_ONLY = {"300005", "300006"}

    def fake_stock_zh_a_hist(symbol, **kwargs):
        if symbol not in hist_cache:
            # crc32, not hash(): str hashing is salted per process, which
            # would make this test's data — and its assertions — differ
            # from run to run.
            seed = zlib.crc32(symbol.encode()) % (2 ** 31)
            if symbol == target_code:
                close = series_for(0.0, 1.0, seed, 110.0)
            elif symbol in NOISE_ONLY:
                close = series_for(0.0, 0.0, seed, 60.0)
            else:
                lead = LEADS.get(symbol, float((seed % 7) - 3))
                close = series_for(lead, 0.9, seed, float(50 + seed % 150))
            hist_cache[symbol] = _hist_from_close(close, seed=seed)
        return hist_cache[symbol]

    def fake_individual_info_em(symbol, **kwargs):
        rows = [
            ("股票代码", symbol), ("股票简称", "阳光电源"),
            ("行业", target_industry), ("总股本", 2e9), ("流通股", 1.9e9),
            ("总市值", 2.1e11), ("流通市值", 2.0e11),
            ("上市时间", "20111102"), ("最新", 110.5),
        ]
        return pd.DataFrame(rows, columns=["item", "value"])

    spot_df = _synthetic_spot(target_code, target_industry)

    def fake_spot_em():
        return spot_df

    def fake_industry_cons_em(symbol):
        # Reuse a subset of the synthetic spot universe as "industry peers"
        peers = spot_df.sample(n=15, random_state=2).copy()
        if target_code not in peers["代码"].values:
            peers.iloc[0, peers.columns.get_loc("代码")] = target_code
        return peers.rename(columns={"市盈率-动态": "市盈率-动态"})[
            ["代码", "名称", "最新价", "涨跌幅", "市盈率-动态", "市净率", "换手率", "成交额"]
        ]

    def fake_industry_hist_em(symbol, **kwargs):
        raw = _synthetic_hist(seed=99, start_price=1000.0)
        return raw

    def fake_index_daily_em(symbol, **kwargs):
        raw = _synthetic_hist(seed=42, start_price=3800.0)
        return pd.DataFrame({
            "date": raw["日期"], "open": raw["开盘"], "close": raw["收盘"],
            "high": raw["最高"], "low": raw["最低"],
            "volume": raw["成交量"], "amount": raw["成交额"],
        })

    def fake_fund_flow(stock, market):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2025-06-01", periods=60)
        return pd.DataFrame({
            "日期": dates.strftime("%Y-%m-%d"),
            "主力净流入-净额": rng.normal(0, 3e7, 60).round(0),
            "主力净流入-净占比": rng.normal(0, 5, 60).round(2),
        })

    def fake_sector_flow(symbol):
        rng = np.random.default_rng(8)
        dates = pd.bdate_range("2025-06-01", periods=60)
        return pd.DataFrame({
            "日期": dates.strftime("%Y-%m-%d"),
            "主力净流入-净额": rng.normal(0, 2e8, 60).round(0),
            "主力净流入-净占比": rng.normal(0, 3, 60).round(2),
        })

    def fake_hsgt_detail(symbol, **kwargs):
        return pd.DataFrame()  # simulate "not a Stock-Connect name" — must not crash

    def fake_margin_sse(date):
        return pd.DataFrame()

    def fake_margin_szse(date):
        return pd.DataFrame({
            "证券代码": [target_code], "证券简称": ["阳光电源"],
            "融资买入额": [5.2e7], "融资余额": [3.1e9],
            "融券卖出量": [10000], "融券余量": [200000],
        })

    def fake_fin_abstract(symbol, **kwargs):
        periods = ["2025-06-30", "2025-03-31", "2024-12-31", "2024-09-30"]
        return pd.DataFrame({
            "报告期": periods,
            "营业总收入同比增长率": ["-18.26%", "-12.0%", "35.0%", "40.0%"],
            "净利润同比增长率": ["-40.12%", "-20.0%", "45.0%", "50.0%"],
            "销售毛利率": ["28.5%", "29.1%", "31.0%", "30.5%"],
            "销售净利率": ["14.7%", "15.2%", "18.0%", "17.5%"],
            "净资产收益率": ["9.8%", "5.1%", "22.0%", "18.0%"],
            "基本每股收益": ["1.10", "0.55", "5.20", "3.90"],
        })

    patches = [
        mock.patch("akshare.stock_zh_a_hist", side_effect=fake_stock_zh_a_hist),
        mock.patch("akshare.stock_individual_info_em", side_effect=fake_individual_info_em),
        mock.patch("akshare.stock_zh_a_spot_em", side_effect=fake_spot_em),
        mock.patch("akshare.stock_board_industry_cons_em", side_effect=fake_industry_cons_em),
        mock.patch("akshare.stock_board_industry_hist_em", side_effect=fake_industry_hist_em),
        mock.patch("akshare.stock_zh_index_daily_em", side_effect=fake_index_daily_em),
        mock.patch("akshare.stock_individual_fund_flow", side_effect=fake_fund_flow),
        mock.patch("akshare.stock_sector_fund_flow_hist", side_effect=fake_sector_flow),
        mock.patch("akshare.stock_hsgt_individual_detail_em", side_effect=fake_hsgt_detail),
        mock.patch("akshare.stock_margin_detail_sse", side_effect=fake_margin_sse),
        mock.patch("akshare.stock_margin_detail_szse", side_effect=fake_margin_szse),
        mock.patch("akshare.stock_financial_abstract_ths", side_effect=fake_fin_abstract),
    ]
    for p in patches:
        p.start()
    try:
        from quant import cache, report
        cache.clear()
        result = report.build_report(target_disp)
    finally:
        for p in patches:
            p.stop()

    print(f"{result.code}  {result.name}  ({result.industry})  as of {result.as_of}")
    print(f"Composite score: {result.composite_score:.1f}  ->  {result.stance}")
    print("\nCategory breakdown:")
    for cat, (score, detail) in result.breakdown.items():
        score_str = f"{score:.1f}" if score == score else "n/a"
        print(f"  {cat:15s} {score_str}")
        for k, v in detail.items():
            if isinstance(v, float):
                print(f"      {k:28s} {v:.1f}" if v == v else f"      {k:28s} n/a")
    bulky = ("peer_corr_matrix", "peer_top_correlated",
             "leadlag_network", "leadlag_peer_closes")
    print("\nLinkage factors:")
    for k, v in result.linkage.items():
        if k not in bulky:
            print(f"  {k:32s} {v:.4f}" if isinstance(v, float) else f"  {k:32s} {v}")
    if "peer_top_correlated" in result.linkage:
        print("Top correlated peers:", result.linkage["peer_top_correlated"])

    net = result.linkage.get("leadlag_network")
    if net is not None and not net.empty:
        print("\nLead-lag network (positive lead_days = peer leads this stock):")
        print(net[["lead_days", "aligned_corr", "trend_similarity", "overlap"]]
              .sort_values("lead_days", ascending=False).round(3).to_string())
    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    assert result.composite_score == result.composite_score, "composite score is NaN"
    assert 0 <= result.composite_score <= 100
    assert not result.prices.empty
    assert "industry_beta" in result.linkage
    assert "ret_today_pctile_industry" in result.linkage
    assert "peer_avg_corr" in result.linkage
    assert "sde_drift_annual" in result.technical
    assert "sde_drift_diffusion_ratio" in result.technical

    # The lead-lag layer must actually fire on data built to contain leaders.
    assert net is not None and not net.empty, "lead-lag network is empty"
    assert result.linkage.get("n_leading_peers", 0) > 0, \
        "no leading peers found despite synthetic leaders in the universe"
    signal = result.linkage.get("leading_peer_signal_pct")
    assert signal is not None and signal == signal, \
        "leading_peer_signal_pct did not compute"
    # Sector-loaded peers should read as more linked than the noise-only ones.
    linked = net["aligned_corr"].dropna()
    assert linked.max() > 0.3, f"no peer showed real linkage (max corr {linked.max():.2f})"
    print("\nSelf-test passed.")


if __name__ == "__main__":
    run()
    sys.exit(0)
