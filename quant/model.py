"""Composite multi-factor scoring.

Every factor is squashed to a 0-100 "goodness" score (100 = most bullish
reading for that factor), grouped into five categories, averaged within
each category (ignoring factors that came back NaN — missing data shrinks
the average's denominator, it never silently scores as 0), then combined
across categories with CATEGORY_WEIGHTS into one composite score.

This is a transparent, editable linear scoring model, not a fitted
statistical one — there's no historical backtest behind these weights, they
encode a reasonable prior (cross-sectional standing and technical trend
carry the most weight; capital flow the least, since it's the noisiest).
Treat the composite as a structured summary of the factor readings, not a
calibrated probability.
"""

import numpy as np

CATEGORY_WEIGHTS = {
    "technical": 0.25,
    "valuation": 0.15,
    "growth_quality": 0.20,
    "capital_flow": 0.15,
    "linkage": 0.25,
}

STANCE_BANDS = [
    (75, "强烈看多 Strong Bullish"),
    (60, "看多 Bullish"),
    (40, "中性 Neutral"),
    (25, "看空 Bearish"),
    (0, "强烈看空 Strong Bearish"),
]


def _clip_linear(value, low, high, invert=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    score = (value - low) / (high - low) * 100
    score = max(0.0, min(100.0, score))
    return 100 - score if invert else score


def _band_score(value, center, width):
    """Score peaks at `center`, decays linearly to 0 over +/- width."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    dist = abs(value - center)
    return max(0.0, 100 - dist / width * 100)


def _avg(scores):
    valid = [s for s in scores if s is not None and not (isinstance(s, float) and np.isnan(s))]
    return float(np.mean(valid)) if valid else float("nan")


def score_technical(f):
    scores = {
        "trend": 100.0 if f.get("ma_trend_signal") == 1 else 0.0 if "ma_trend_signal" in f else np.nan,
        "price_vs_ma5": _clip_linear(f.get("price_vs_ma5_pct"), -8, 8),
        "price_vs_ma120": _clip_linear(f.get("price_vs_ma120_pct"), -20, 20),
        "rsi14": _band_score(f.get("rsi14"), center=58, width=35),
        "macd_hist_rising": 100.0 if f.get("macd_hist_rising") == 1 else 0.0 if f.get("macd_hist_rising") == 0 else np.nan,
        "momentum_12_1": _clip_linear(f.get("momentum_12_1"), -30, 30),
        "ret_1m": _clip_linear(f.get("ret_1m"), -15, 15),
        "ret_3m": _clip_linear(f.get("ret_3m"), -25, 25),
        "low_volatility": _clip_linear(f.get("vol_20d_annualized_pct"), 20, 80, invert=True),
    }
    return _avg(scores.values()), scores


def score_valuation(f, linkage):
    scores = {
        "value_rank_industry": linkage.get("value_rank_industry", np.nan),
        "value_rank_market": linkage.get("value_rank_market", np.nan),
        "pe_sanity": _band_score(f.get("pe_ttm"), center=25, width=60) if f.get("pe_ttm", 0) and f.get("pe_ttm") > 0 else np.nan,
    }
    return _avg(scores.values()), scores


def score_growth_quality(f):
    scores = {
        "revenue_yoy": _clip_linear(f.get("revenue_yoy_pct"), -20, 30),
        "net_profit_yoy": _clip_linear(f.get("net_profit_yoy_pct"), -40, 40),
        "gross_margin": _clip_linear(f.get("gross_margin_pct"), 5, 45),
        "roe": _clip_linear(f.get("roe_pct"), 0, 25),
        "eps_trend": _clip_linear(f.get("eps_trend_slope"), -0.2, 0.2) if "eps_trend_slope" in f else np.nan,
    }
    return _avg(scores.values()), scores


def score_capital_flow(f):
    scores = {
        "main_inflow_5d": _clip_linear(f.get("main_inflow_5d_sum"), -2e8, 2e8),
        "main_inflow_20d": _clip_linear(f.get("main_inflow_20d_sum"), -5e8, 5e8),
        "northbound_chg_20d": _clip_linear(f.get("northbound_hold_ratio_chg_20d"), -0.5, 0.5),
    }
    return _avg(scores.values()), scores


def score_linkage(linkage):
    scores = {
        "ret_today_pctile_industry": linkage.get("ret_today_pctile_industry", np.nan),
        "ret_60d_pctile_market": linkage.get("ret_60d_pctile_market", np.nan),
        "industry_r2_moderate": _band_score(linkage.get("industry_r2"), center=0.35, width=0.5),
        "fundflow_industry_corr": _clip_linear(linkage.get("fundflow_industry_corr"), -1, 1),
        "peer_avg_corr_moderate": _band_score(linkage.get("peer_avg_corr"), center=0.5, width=0.7),
    }
    return _avg(scores.values()), scores


def stance_for(score):
    if score is None or np.isnan(score):
        return "数据不足 Insufficient data"
    for threshold, label in STANCE_BANDS:
        if score >= threshold:
            return label
    return STANCE_BANDS[-1][1]


def composite_score(technical, valuation, growth_quality, capital_flow, linkage):
    """Run all five category scorers and combine into one composite.

    Returns (composite_score, stance_label, category_breakdown) where
    category_breakdown maps category name -> (score, {sub-factor: score}).
    """
    tech_score, tech_detail = score_technical(technical)
    val_score, val_detail = score_valuation(valuation, linkage)
    gq_score, gq_detail = score_growth_quality(growth_quality)
    cf_score, cf_detail = score_capital_flow(capital_flow)
    link_score, link_detail = score_linkage(linkage)

    breakdown = {
        "technical": (tech_score, tech_detail),
        "valuation": (val_score, val_detail),
        "growth_quality": (gq_score, gq_detail),
        "capital_flow": (cf_score, cf_detail),
        "linkage": (link_score, link_detail),
    }

    weighted_sum, weight_total = 0.0, 0.0
    for cat, (score, _detail) in breakdown.items():
        if score is not None and not np.isnan(score):
            w = CATEGORY_WEIGHTS[cat]
            weighted_sum += score * w
            weight_total += w
    composite = weighted_sum / weight_total if weight_total > 0 else float("nan")

    return composite, stance_for(composite), breakdown
