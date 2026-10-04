"""Terminal report for a single stock (no dashboard needed).

    python -m quant.cli 300274.SZ
    python -m quant.cli 300476.SZ --days 61 --freq both    # 2-month daily + hourly
"""

import argparse

from . import report


def _fmt(v, suffix=""):
    if v is None or v != v:  # NaN check without importing numpy here
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2f}{suffix}"
    return f"{v}{suffix}"


def _lead(link, key, unit):
    """Format a lead-lag scalar in days, plus native hours on hourly bars."""
    days = link.get(key)
    if days is None or days != days:
        return "n/a"
    if unit == "hour":
        hours = link.get(key.replace("_days", "_hours"))
        return f"{days:+.2f}d ({hours:+.1f} trading hours)"
    return f"{days:+.2f}d"


def print_report(r):
    tf = report.TIMEFRAMES[r.freq]["label"]
    print(f"\n{r.code}  {r.name}  |  行业 Industry: {r.industry or 'n/a'}")
    print(f"周期 Timeframe: {tf}  |  window {r.start} → {r.end}  "
          f"|  {len(r.prices)} bars  |  last bar {r.as_of}")
    print("=" * 78)
    print(f"COMPOSITE SCORE: {_fmt(r.composite_score)} / 100   ->   {r.stance}")
    print("-" * 78)
    for cat, (score, detail) in r.breakdown.items():
        print(f"{cat:16s} {_fmt(score):>8s}   " +
              "  ".join(f"{k}={_fmt(v)}" for k, v in detail.items()))

    t = r.technical
    print(f"\n--- Path over the window (MA{r.short_ma}/MA{r.long_ma} in {r.bar_unit} bars) ---")
    print(f"  Window return {_fmt(t.get('ret_window'), '%')}  |  range "
          f"{_fmt(t.get('lo_window'))} – {_fmt(t.get('hi_window'))}  |  "
          f"RSI14 {_fmt(t.get('rsi14'))}  |  MA trend "
          f"{'up' if t.get('ma_trend_signal') == 1 else 'down' if t.get('ma_trend_signal') == 0 else 'n/a'}")
    print(f"  SDE drift μ {_fmt(t.get('sde_drift_annual'))}  diffusion σ "
          f"{_fmt(t.get('sde_diffusion_annual'))}  μ/σ "
          f"{_fmt(t.get('sde_drift_diffusion_ratio'))}  vol regime {_fmt(t.get('sde_vol_regime'))}")

    print("\n--- Cross-sectional linkage (截面联动类) ---")
    link = r.linkage
    print(f"  Market beta/corr/R^2   : {_fmt(link.get('market_beta'))} / "
          f"{_fmt(link.get('market_corr'))} / {_fmt(link.get('market_r2'))}")
    print(f"  Industry beta/corr/R^2 : {_fmt(link.get('industry_beta'))} / "
          f"{_fmt(link.get('industry_corr'))} / {_fmt(link.get('industry_r2'))}")
    print(f"  Today's return percentile within industry : {_fmt(link.get('ret_today_pctile_industry'))}")
    print(f"  60d return percentile within whole market  : {_fmt(link.get('ret_60d_pctile_market'))}")
    print(f"  Value rank (cheapness) within industry      : {_fmt(link.get('value_rank_industry'))}")
    print(f"  Avg same-day correlation with peers         : {_fmt(link.get('peer_avg_corr'))}")
    print(f"  Stock-vs-sector fund-flow correlation       : {_fmt(link.get('fundflow_industry_corr'))}")

    print("\n--- Lead-lag network (时滞联动) ---")
    print(f"  Peers linked          : {link.get('n_linked_peers', 0)} of "
          f"{link.get('n_peers_tested', 0)} pass significance")
    print(f"  Leading-peer signal   : {_fmt(link.get('leading_peer_signal_pct'), '%')} "
          f"from {link.get('n_leading_peers', 0)} linked leader(s)")
    print(f"  Leaders' mean lead    : {_lead(link, 'leading_peer_mean_lead_days', r.bar_unit)}")
    print(f"  This stock vs peers   : {_lead(link, 'target_leadership_days', r.bar_unit)}  "
          f"(positive = it leads its sector)")
    net = link.get("leadlag_network")
    if net is not None and not net.empty:
        col = "lead_hours" if r.bar_unit == "hour" else "lead_days"
        unit = "h" if r.bar_unit == "hour" else "d"
        shown = net.sort_values(["linked", "lag_corr"], ascending=False)
        for label, row in shown.iterrows():
            if row["linked"]:
                tag = ("LEADS  " if row[col] >= 0.5 else
                       "lags   " if row[col] <= -0.5 else "in-step")
                note = ""
            else:
                tag, note = "  —    ", "  (not significant — ignored)"
            print(f"    {label:22s} {tag} {row[col]:+6.1f}{unit}  "
                  f"corr={row['lag_corr']:+.2f} (needs {row['threshold']:.2f}){note}")
    if r.freq == "daily" and len(r.prices) < 80:
        print("  Note: on fewer than ~80 daily bars few real links can reach "
              "significance — the 60-minute run is the better read on a short window.")

    if r.warnings:
        print("\nWarnings:")
        for w in r.warnings:
            print(f"  - {w}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("code", help="Stock code, e.g. 300274.SZ")
    parser.add_argument("--days", type=int, default=548,
                        help="Calendar-day lookback (default 548; ~61 for two months)")
    parser.add_argument("--freq", choices=["daily", "60m", "both"], default="daily",
                        help="Bar timeframe; 'both' runs daily then 60-minute")
    parser.add_argument("--short", type=int, default=None, help="Short MA (bars)")
    parser.add_argument("--long", type=int, default=None, help="Long MA (bars)")
    args = parser.parse_args()

    # Sensible per-timeframe defaults: on hourly bars 20/60 is ~1 and ~3
    # weeks of trading; on a narrow daily window 5/20 is what fits.
    narrow = args.days < 120
    defaults = {"daily": (5, 20) if narrow else (20, 60), "60m": (20, 60)}

    freqs = ["daily", "60m"] if args.freq == "both" else [args.freq]
    for freq in freqs:
        short, long = defaults[freq]
        r = report.build_report(args.code, lookback_days=args.days,
                                short_ma=args.short or short,
                                long_ma=args.long or long, freq=freq)
        print_report(r)


if __name__ == "__main__":
    main()
