"""Local Streamlit dashboard for the quant model.

    streamlit run dashboard.py

One text box for a stock code, a handful of buttons that switch the view,
each rendering its own Plotly visualization(s). All data comes from
AKShare (free, no login) via quant/report.py.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant import report

st.set_page_config(page_title="截面联动 Quant Dashboard", layout="wide")

VIEWS = [
    ("overview", "📊 综合评分 Overview"),
    ("price", "📈 K线走势 Price Chart"),
    ("linkage", "🔗 截面联动分析 Cross-Sectional Linkage"),
    ("leadlag", "⏱️ 时滞联动 Lead-Lag Network"),
    ("flow", "💰 资金流向 Capital Flow"),
    ("radar", "🎯 因子雷达图 Factor Radar"),
]

FREQ_LABELS = {key: tf["label"] for key, tf in report.TIMEFRAMES.items()}

# (label, freq, lookback days, short MA, long MA). MAs are in bars: on a
# two-month daily window only ~44 bars exist, so 5/20 is what fits; on
# hourly bars 20/60 is roughly one and three weeks of trading.
PRESETS = [
    ("近两月·日线 2M daily", "daily", 61, 5, 20),
    ("近两月·小时 2M hourly", "60m", 61, 20, 60),
    ("一年半·日线 18M daily", "daily", 548, 20, 60),
]

for key, default in [("report", None), ("view", "overview"), ("error", None),
                     ("run_now", False), ("freq", "daily"), ("days", 548),
                     ("short_ma", 20), ("long_ma", 60)]:
    if key not in st.session_state:
        st.session_state[key] = default


def apply_preset(freq, days, short, long):
    """Runs before the rerun, so it may set widget values."""
    st.session_state.freq = freq
    st.session_state.days = days
    st.session_state.short_ma = short
    st.session_state.long_ma = long
    st.session_state.run_now = True


def run_analysis(code, freq, days, short_ma, long_ma):
    st.session_state.error = None
    try:
        with st.spinner(f"Fetching & scoring {code} on {FREQ_LABELS[freq]} bars, "
                        f"{days}-day window ... several live AKShare endpoints, ~10-60s"):
            st.session_state.report = report.build_report(
                code, lookback_days=days, short_ma=short_ma, long_ma=long_ma,
                freq=freq)
    except Exception as e:
        st.session_state.report = None
        st.session_state.error = str(e)


# ---------------------------------------------------------------------------
# Top bar: the one code box, timeframe/window, Analyze, presets
# ---------------------------------------------------------------------------

st.title("截面联动 Quant Dashboard")
st.caption("Free AKShare data · cross-sectional-linkage-weighted multi-factor model · runs entirely on your machine")

top = st.columns([3, 1.3, 1, 1, 1, 1.3])
with top[0]:
    code_input = st.text_input("股票代码 Stock code", value="300274.SZ",
                               placeholder="e.g. 300274.SZ, 600519.SH")
with top[1]:
    st.selectbox("周期 Timeframe", options=list(FREQ_LABELS),
                 format_func=FREQ_LABELS.get, key="freq")
with top[2]:
    st.number_input("回看天数 Lookback (days)", min_value=20, max_value=1500, key="days")
with top[3]:
    st.number_input("短均线 Short MA", min_value=2, max_value=120, key="short_ma",
                    help="In bars of the chosen timeframe.")
with top[4]:
    st.number_input("长均线 Long MA", min_value=5, max_value=250, key="long_ma",
                    help="In bars of the chosen timeframe. Shrunk automatically "
                         "if the window has too few bars.")
with top[5]:
    st.write("")
    st.write("")
    analyze_clicked = st.button("🔍 生成报告 Analyze", type="primary", use_container_width=True)

presets = st.columns(len(PRESETS) + 2)
presets[0].caption("快捷 Presets →")
for i, (label, freq, days, short, long) in enumerate(PRESETS, start=1):
    presets[i].button(label, use_container_width=True, on_click=apply_preset,
                      args=(freq, days, short, long))

if (analyze_clicked or st.session_state.run_now) and code_input.strip():
    st.session_state.run_now = False
    run_analysis(code_input.strip(), st.session_state.freq, int(st.session_state.days),
                 int(st.session_state.short_ma), int(st.session_state.long_ma))

if st.session_state.error:
    st.error(f"Failed to build report: {st.session_state.error}")

r = st.session_state.report

if r is not None:
    nav = st.columns(len(VIEWS))
    for i, (key, label) in enumerate(VIEWS):
        with nav[i]:
            if st.button(label, use_container_width=True,
                        type="primary" if st.session_state.view == key else "secondary"):
                st.session_state.view = key

    st.divider()
    header = st.columns([2, 1, 1, 1])
    header[0].metric("股票 Stock", f"{r.name} ({r.code})")
    header[1].metric("行业 Industry", r.industry or "n/a")
    score_str = f"{r.composite_score:.1f}" if r.composite_score == r.composite_score else "n/a"
    header[2].metric("综合评分 Composite Score", f"{score_str} / 100")
    header[3].metric("信号 Stance", r.stance)
    st.caption(f"**{FREQ_LABELS[r.freq]}** · window {r.start} → {r.end} · "
               f"{len(r.prices)} bars · last bar {r.as_of} · "
               f"MA{r.short_ma}/MA{r.long_ma} ({r.bar_unit} bars)")

    if r.warnings:
        with st.expander(f"⚠️ {len(r.warnings)} data warning(s)"):
            for w in r.warnings:
                st.write(f"- {w}")

    view = st.session_state.view

    # -----------------------------------------------------------------
    # Overview
    # -----------------------------------------------------------------
    if view == "overview":
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("分类得分 Category scores")
            cats, scores = [], []
            for cat, (score, _detail) in r.breakdown.items():
                cats.append(cat)
                scores.append(score if score == score else 0)
            fig = go.Figure(go.Bar(
                x=scores, y=cats, orientation="h",
                marker_color=["#2E86AB" if s >= 50 else "#C1443C" for s in scores],
                text=[f"{s:.0f}" for s in scores], textposition="outside",
            ))
            fig.update_layout(xaxis_range=[0, 100], height=320,
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("基本信息 Snapshot")
            snap = r.snapshot_row or {}
            info_rows = {
                "最新价 Price": snap.get("price"),
                "涨跌幅 % Chg": snap.get("pct_chg"),
                "市盈率 PE (TTM)": snap.get("pe_ttm"),
                "市净率 PB": snap.get("pb"),
                "总市值 Mkt Cap": snap.get("total_mkt_cap"),
                "60日涨跌幅 60d %": snap.get("ret_60d"),
                "年初至今 YTD %": snap.get("ret_ytd"),
            }
            st.table(pd.DataFrame(info_rows.items(), columns=["指标", "值"]))

        st.subheader("因子明细 Factor detail")
        for cat, (score, detail) in r.breakdown.items():
            score_str = f"{score:.1f}" if score == score else "n/a"
            with st.expander(f"{cat}  —  {score_str}/100"):
                st.json({k: (round(v, 3) if isinstance(v, float) and v == v else v)
                        for k, v in detail.items()})

    # -----------------------------------------------------------------
    # Price chart
    # -----------------------------------------------------------------
    elif view == "price":
        p = r.prices
        if p.empty:
            st.warning("No price history available.")
        else:
            p = p.sort_values("date")
            ma_s = p["close"].rolling(r.short_ma).mean()
            ma_l = p["close"].rolling(r.long_ma).mean()
            intraday = r.bar_unit != "day"
            # Intraday bars on a time axis leave huge overnight/weekend gaps;
            # a category axis puts consecutive bars side by side instead.
            x = p["date"].dt.strftime("%m-%d %H:%M") if intraday else p["date"]
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=x, open=p["open"], high=p["high"],
                low=p["low"], close=p["close"], name=r.code,
            ))
            fig.add_trace(go.Scatter(x=x, y=ma_s, name=f"MA{r.short_ma}",
                                     line=dict(width=1.3, color="#F5A623")))
            fig.add_trace(go.Scatter(x=x, y=ma_l, name=f"MA{r.long_ma}",
                                     line=dict(width=1.3, color="#7B61FF")))
            fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                              margin=dict(l=10, r=10, t=30, b=10),
                              legend=dict(orientation="h"))
            if intraday:
                fig.update_xaxes(type="category", nticks=12)
            st.plotly_chart(fig, use_container_width=True)

            vol_fig = go.Figure(go.Bar(x=x, y=p["volume"], marker_color="#8892A0"))
            vol_fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10),
                                  title="成交量 Volume")
            if intraday:
                vol_fig.update_xaxes(type="category", nticks=12)
            st.plotly_chart(vol_fig, use_container_width=True)

            t = r.technical
            w = st.columns(4)
            def _pct(v):
                return f"{v:+.2f}%" if v == v and v is not None else "n/a"
            w[0].metric("区间涨跌 Window return", _pct(t.get("ret_window")))
            w[1].metric("区间高点 Window high", f"{t.get('hi_window', float('nan')):.2f}")
            w[2].metric("区间低点 Window low", f"{t.get('lo_window', float('nan')):.2f}")
            w[3].metric("RSI14", f"{t.get('rsi14', float('nan')):.1f}",
                        help=f"On {r.bar_unit} bars.")

            t = r.technical
            st.subheader("几何SDE分解 Geometric-SDE decomposition")
            st.caption("Fitting dS = μ·S·dt + σ·S·dW to the realized path: μ is the "
                      "deterministic trend, σ the stochastic diffusion. Their ratio is "
                      "the path's signal-to-noise.")
            s = st.columns(4)
            def _sde(col, label, key, fmt, helptext=None):
                v = t.get(key, float("nan"))
                col.metric(label, fmt.format(v) if v == v else "n/a", help=helptext)
            _sde(s[0], "漂移项 Drift μ (annual)", "sde_drift_annual", "{:+.1%}")
            _sde(s[1], "扩散项 Diffusion σ (annual)", "sde_diffusion_annual", "{:.1%}")
            _sde(s[2], "信噪比 μ/σ", "sde_drift_diffusion_ratio", "{:+.2f}",
                 "Trend per unit of noise — a Sharpe-like reading of the path.")
            _sde(s[3], "波动区制 Vol regime", "sde_vol_regime", "{:.2f}",
                 "Recent diffusion / long-run diffusion. >1 = volatility expanding.")

    # -----------------------------------------------------------------
    # Cross-sectional linkage
    # -----------------------------------------------------------------
    elif view == "linkage":
        link = r.linkage
        st.subheader("联动指标 Co-movement metrics")
        m = st.columns(4)
        m[0].metric("大盘 Beta (Market)", f"{link.get('market_beta', float('nan')):.2f}"
                    if link.get("market_beta") == link.get("market_beta") else "n/a")
        m[1].metric("大盘相关性 Market Corr", f"{link.get('market_corr', float('nan')):.2f}"
                    if link.get("market_corr") == link.get("market_corr") else "n/a")
        m[2].metric("行业 Beta (Industry)", f"{link.get('industry_beta', float('nan')):.2f}"
                    if link.get("industry_beta") == link.get("industry_beta") else "n/a")
        m[3].metric("行业 R² (variance explained)", f"{link.get('industry_r2', float('nan')):.2f}"
                    if link.get("industry_r2") == link.get("industry_r2") else "n/a")

        m2 = st.columns(4)
        m2[0].metric("今日行业内百分位 Today's rank in industry",
                     f"{link.get('ret_today_pctile_industry', float('nan')):.0f}%ile"
                     if link.get("ret_today_pctile_industry") == link.get("ret_today_pctile_industry") else "n/a")
        m2[1].metric("60日全市场百分位 60d rank in market",
                     f"{link.get('ret_60d_pctile_market', float('nan')):.0f}%ile"
                     if link.get("ret_60d_pctile_market") == link.get("ret_60d_pctile_market") else "n/a")
        m2[2].metric("行业内估值排名 Value rank vs industry",
                     f"{link.get('value_rank_industry', float('nan')):.0f}%ile"
                     if link.get("value_rank_industry") == link.get("value_rank_industry") else "n/a")
        m2[3].metric("资金流-行业相关性 Fund-flow vs sector corr",
                     f"{link.get('fundflow_industry_corr', float('nan')):.2f}"
                     if link.get("fundflow_industry_corr") == link.get("fundflow_industry_corr") else "n/a")

        st.caption("Beta/corr/R² are computed on the trailing ~120 trading days of daily returns. "
                  "R² measures how much of this stock's variance is explained by its benchmark — "
                  "high = tightly linked / systemic, low = idiosyncratic / stock-specific moves.")

        st.subheader("同行业相关性网络 Peer correlation network")
        corr_matrix = link.get("peer_corr_matrix")
        if corr_matrix is not None and not corr_matrix.empty:
            fig = go.Figure(go.Heatmap(
                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                colorbar=dict(title="corr"),
            ))
            fig.update_layout(height=max(400, 28 * len(corr_matrix)),
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            if link.get("peer_top_correlated"):
                st.write("**最相关同行 Most correlated peers:**")
                for name, corr in link["peer_top_correlated"]:
                    st.write(f"- {name}: corr = {corr:.2f}")
        else:
            st.info("Peer correlation network unavailable (industry classification or peer "
                    "history missing for this code).")

    # -----------------------------------------------------------------
    # Lead-lag network (delay-aware layer, quant/leadlag.py)
    # -----------------------------------------------------------------
    elif view == "leadlag":
        link = r.linkage
        net = link.get("leadlag_network")
        hourly = r.bar_unit == "hour"
        unit, col = ("h", "lead_hours") if hourly else ("d", "lead_days")
        st.subheader("时滞联动 Lead-lag structure")
        st.caption(
            f"Per peer: recency-weighted cross-correlation of returns finds the "
            f"(fractional) lag at which the two move together; the peer is then "
            f"shifted by that lag to measure trend similarity. Measured on "
            f"**{FREQ_LABELS[r.freq]}** bars, so lags read in "
            f"{'trading hours' if hourly else 'trading days'}. "
            f"**Positive lead = that peer moves before this stock.** "
            f"A peer only counts if its correlation clears a significance "
            f"threshold that accounts for how many lags were searched — "
            f"otherwise its 'lead' is noise."
        )

        def _lead_fmt(key):
            v = link.get(key, float("nan"))
            if v != v:
                return "n/a"
            return (f"{link.get(key.replace('_days', '_hours')):+.1f}h"
                    if hourly else f"{v:+.1f}d")

        m = st.columns(4)
        sig = link.get("leading_peer_signal_pct", float("nan"))
        m[0].metric("领先股信号 Leading-peer signal",
                    f"{sig:+.2f}%" if sig == sig else "n/a",
                    help="Move the linked leading peers have already made that "
                         "this stock has not yet followed. Positive = bullish pressure.")
        m[1].metric("显著联动 Linked peers",
                    f"{link.get('n_linked_peers', 0)} / {link.get('n_peers_tested', 0)}",
                    help="Peers whose lagged correlation passes significance.")
        m[2].metric("领先股 Linked leaders", link.get("n_leading_peers", 0),
                    delta=f"mean lead {_lead_fmt('leading_peer_mean_lead_days')}"
                    if link.get("n_leading_peers") else None, delta_color="off")
        leadership = link.get("target_leadership_days", float("nan"))
        m[3].metric("本股地位 This stock's role",
                    ("领先 Leader" if leadership > 0.5 / r.bars_per_day else
                     "滞后 Follower" if leadership < -0.5 / r.bars_per_day else "同步 In-step")
                    if leadership == leadership else "n/a",
                    delta=(f"{_lead_fmt('target_leadership_days')} vs linked peers"
                           if leadership == leadership else None))

        if r.freq == "daily" and len(r.prices) < 80:
            st.warning(f"Only {len(r.prices)} daily bars in this window. On a sample "
                       "this short few genuine links can reach significance — "
                       "switch to 60分钟 Hourly for a usable lead-lag read.")

        if net is not None and not net.empty:
            plot_df = net.sort_values(col)
            linked = plot_df["linked"].astype(bool)
            fig = go.Figure(go.Bar(
                x=plot_df[col], y=plot_df.index, orientation="h",
                marker=dict(color=plot_df["lag_corr"], colorscale="RdBu", cmid=0,
                            cmin=-1, cmax=1, colorbar=dict(title="lag<br>corr"),
                            opacity=[1.0 if v else 0.25 for v in linked]),
                customdata=np.stack([plot_df["lag_corr"], plot_df["threshold"],
                                     linked.map({True: "linked", False: "not significant"})], axis=1),
                hovertemplate=("%{y}<br>lead: %{x:.2f}" + unit +
                               "<br>corr: %{customdata[0]:.2f} (needs %{customdata[1]:.2f})"
                               "<br>%{customdata[2]}<extra></extra>"),
            ))
            fig.add_vline(x=0, line_width=1, line_color="#555")
            fig.update_layout(
                height=max(380, 30 * len(plot_df)),
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title=f"← 滞后 lags this stock    |    领先 leads this stock →   ({unit})",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Faded bars did not pass significance and are ignored by the signal.")

            st.subheader("真实联动 vs 噪声 Real linkage vs noise")
            thr = float(net["threshold"].median())
            scat = go.Figure()
            for is_linked, name, color in [(True, "linked", "#2E86AB"),
                                           (False, "not significant", "#B0B7C3")]:
                part = net[net["linked"].astype(bool) == is_linked]
                if part.empty:
                    continue
                scat.add_trace(go.Scatter(
                    x=part[col], y=part["lag_corr"], mode="markers+text", name=name,
                    text=[label.split()[0] for label in part.index],
                    textposition="top center",
                    marker=dict(size=11, color=color)))
            scat.add_hline(y=thr, line_dash="dash", line_color="#888",
                           annotation_text=f"significance threshold ≈ {thr:.2f}")
            scat.add_vline(x=0, line_width=1, line_color="#555")
            scat.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title=f"lead ({unit}, positive = peer leads)",
                               yaxis_title="peak lagged return correlation",
                               legend=dict(orientation="h"))
            st.plotly_chart(scat, use_container_width=True)

            with st.expander("完整时滞表 Full lead-lag table"):
                st.dataframe(net.sort_values(["linked", "lag_corr"], ascending=False).round(3),
                             use_container_width=True)
        else:
            st.info("Lead-lag network unavailable — needs an industry peer list "
                    "and at least 30 overlapping bars per peer.")

    # -----------------------------------------------------------------
    # Capital flow
    # -----------------------------------------------------------------
    elif view == "flow":
        ff = r.fund_flow
        if ff.empty:
            st.warning("No fund-flow history available for this stock.")
        else:
            ff = ff.sort_values("date").copy()
            ff["cum_main_inflow"] = ff["main_net_inflow"].cumsum()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ff["date"], y=ff["main_net_inflow"],
                                 name="主力净流入 Main net inflow (daily)",
                                 marker_color=np.where(ff["main_net_inflow"] >= 0, "#C1443C", "#2E86AB")))
            fig.add_trace(go.Scatter(x=ff["date"], y=ff["cum_main_inflow"],
                                     name="累计 Cumulative", yaxis="y2",
                                     line=dict(color="#F5A623", width=2)))
            fig.update_layout(
                height=420, margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(title="Daily net inflow (CNY)"),
                yaxis2=dict(title="Cumulative", overlaying="y", side="right"),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

        cf = r.capital_flow
        m = st.columns(3)
        m[0].metric("5日主力净流入 5d main inflow",
                    f"{cf.get('main_inflow_5d_sum', 0):,.0f}" if cf.get("main_inflow_5d_sum") == cf.get("main_inflow_5d_sum") else "n/a")
        m[1].metric("20日主力净流入 20d main inflow",
                    f"{cf.get('main_inflow_20d_sum', 0):,.0f}" if cf.get("main_inflow_20d_sum") == cf.get("main_inflow_20d_sum") else "n/a")
        m[2].metric("融资余额 Margin balance",
                    f"{cf.get('margin_balance', 0):,.0f}" if cf.get("margin_balance") == cf.get("margin_balance") else "n/a")

        if "northbound_hold_ratio_latest" in cf:
            st.caption(f"北向资金持股占比 Northbound holding ratio: "
                      f"{cf['northbound_hold_ratio_latest']:.2f}% "
                      f"(20d change: {cf.get('northbound_hold_ratio_chg_20d', float('nan')):+.2f}pp)")

    # -----------------------------------------------------------------
    # Factor radar
    # -----------------------------------------------------------------
    elif view == "radar":
        cats, scores = [], []
        for cat, (score, _detail) in r.breakdown.items():
            cats.append(cat)
            scores.append(score if score == score else 0)
        cats.append(cats[0])
        scores.append(scores[0])
        fig = go.Figure(go.Scatterpolar(r=scores, theta=cats, fill="toself",
                                        line_color="#2E86AB"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                          height=520, showlegend=False,
                          margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    csv_cols = st.columns(2)
    with csv_cols[0]:
        if not r.prices.empty:
            st.download_button("⬇️ 下载价格数据 Download price history CSV",
                              r.prices.to_csv(index=False), file_name=f"{r.code}_prices.csv")
    with csv_cols[1]:
        summary = {**{f"technical.{k}": v for k, v in r.technical.items()},
                  **{f"valuation.{k}": v for k, v in r.valuation.items()},
                  **{f"growth_quality.{k}": v for k, v in r.growth_quality.items()},
                  **{f"capital_flow.{k}": v for k, v in r.capital_flow.items()},
                  **{k: v for k, v in r.linkage.items()
                     if k not in ("peer_corr_matrix", "leadlag_network",
                                  "leadlag_peer_closes", "peer_top_correlated")}}
        summary_df = pd.DataFrame(summary.items(), columns=["factor", "value"])
        st.download_button("⬇️ 下载因子数据 Download factor summary CSV",
                          summary_df.to_csv(index=False),
                          file_name=f"{r.code}_factors.csv")

else:
    st.info("Enter a stock code above (e.g. `300274.SZ`) and click **生成报告 Analyze** to start.")
