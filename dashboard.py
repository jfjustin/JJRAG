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
    ("flow", "💰 资金流向 Capital Flow"),
    ("radar", "🎯 因子雷达图 Factor Radar"),
]

if "report" not in st.session_state:
    st.session_state.report = None
if "view" not in st.session_state:
    st.session_state.view = "overview"
if "error" not in st.session_state:
    st.session_state.error = None


def run_analysis(code, short_ma, long_ma):
    st.session_state.error = None
    try:
        with st.spinner(f"Fetching & scoring {code} ... this pulls several "
                        f"live AKShare endpoints, ~10-30s"):
            st.session_state.report = report.build_report(
                code, short_ma=short_ma, long_ma=long_ma)
    except Exception as e:
        st.session_state.report = None
        st.session_state.error = str(e)


# ---------------------------------------------------------------------------
# Top bar: the one code box + Analyze + view buttons
# ---------------------------------------------------------------------------

st.title("截面联动 Quant Dashboard")
st.caption("Free AKShare data · cross-sectional-linkage-weighted multi-factor model · runs entirely on your machine")

top = st.columns([3, 1, 1, 1])
with top[0]:
    code_input = st.text_input("股票代码 Stock code", value="300274.SZ",
                               label_visibility="visible",
                               placeholder="e.g. 300274.SZ, 600519.SH")
with top[1]:
    short_ma = st.number_input("短均线 Short MA", min_value=2, max_value=60, value=20)
with top[2]:
    long_ma = st.number_input("长均线 Long MA", min_value=5, max_value=250, value=60)
with top[3]:
    st.write("")
    st.write("")
    analyze_clicked = st.button("🔍 生成报告 Analyze", type="primary", use_container_width=True)

if analyze_clicked and code_input.strip():
    run_analysis(code_input.strip(), short_ma, long_ma)

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
            ma_s = p["close"].rolling(int(short_ma)).mean()
            ma_l = p["close"].rolling(int(long_ma)).mean()
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=p["date"], open=p["open"], high=p["high"],
                low=p["low"], close=p["close"], name=r.code,
            ))
            fig.add_trace(go.Scatter(x=p["date"], y=ma_s, name=f"MA{short_ma}",
                                     line=dict(width=1.3, color="#F5A623")))
            fig.add_trace(go.Scatter(x=p["date"], y=ma_l, name=f"MA{long_ma}",
                                     line=dict(width=1.3, color="#7B61FF")))
            fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                              margin=dict(l=10, r=10, t=30, b=10),
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

            vol_fig = go.Figure(go.Bar(x=p["date"], y=p["volume"],
                                       marker_color="#8892A0"))
            vol_fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10),
                                  title="成交量 Volume")
            st.plotly_chart(vol_fig, use_container_width=True)

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
                  **{k: v for k, v in r.linkage.items() if k != "peer_corr_matrix"}}
        summary_df = pd.DataFrame(summary.items(), columns=["factor", "value"])
        st.download_button("⬇️ 下载因子数据 Download factor summary CSV",
                          summary_df.to_csv(index=False),
                          file_name=f"{r.code}_factors.csv")

else:
    st.info("Enter a stock code above (e.g. `300274.SZ`) and click **生成报告 Analyze** to start.")
