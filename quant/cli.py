"""Terminal report for a single stock (no dashboard needed).

    python -m quant.cli 300274.SZ
"""

import argparse

from . import report


def _fmt(v, suffix=""):
    if v is None or v != v:  # NaN check without importing numpy here
        return "n/a"
    if isinstance(v, float):
        return f"{v:.2f}{suffix}"
    return f"{v}{suffix}"


def print_report(r):
    print(f"\n{r.code}  {r.name}  |  行业 Industry: {r.industry or 'n/a'}  |  as of {r.as_of}")
    print("=" * 70)
    print(f"COMPOSITE SCORE: {_fmt(r.composite_score)} / 100   ->   {r.stance}")
    print("-" * 70)
    for cat, (score, detail) in r.breakdown.items():
        print(f"{cat:16s} {_fmt(score):>8s}   " +
              "  ".join(f"{k}={_fmt(v)}" for k, v in detail.items()))

    print("\n--- Cross-sectional linkage (截面联动类) ---")
    link = r.linkage
    print(f"  Market beta/corr/R^2   : {_fmt(link.get('market_beta'))} / "
          f"{_fmt(link.get('market_corr'))} / {_fmt(link.get('market_r2'))}")
    print(f"  Industry beta/corr/R^2 : {_fmt(link.get('industry_beta'))} / "
          f"{_fmt(link.get('industry_corr'))} / {_fmt(link.get('industry_r2'))}")
    print(f"  Today's return percentile within industry : {_fmt(link.get('ret_today_pctile_industry'))}")
    print(f"  60d return percentile within whole market  : {_fmt(link.get('ret_60d_pctile_market'))}")
    print(f"  Value rank (cheapness) within industry      : {_fmt(link.get('value_rank_industry'))}")
    print(f"  Value rank (cheapness) within whole market   : {_fmt(link.get('value_rank_market'))}")
    print(f"  Avg correlation with top industry peers      : {_fmt(link.get('peer_avg_corr'))}")
    if link.get("peer_top_correlated"):
        print("  Most correlated peers:")
        for name, corr in link["peer_top_correlated"]:
            print(f"    {name:20s} corr={corr:.2f}")
    print(f"  Stock-vs-sector fund-flow correlation         : {_fmt(link.get('fundflow_industry_corr'))}")

    if r.warnings:
        print("\nWarnings:")
        for w in r.warnings:
            print(f"  - {w}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("code", help="Stock code, e.g. 300274.SZ")
    parser.add_argument("--short", type=int, default=20, help="Short MA window")
    parser.add_argument("--long", type=int, default=60, help="Long MA window")
    args = parser.parse_args()

    r = report.build_report(args.code, short_ma=args.short, long_ma=args.long)
    print_report(r)


if __name__ == "__main__":
    main()
