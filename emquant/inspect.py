"""Inspect a security: pull daily bars via EMQuantAPI and run the MA model.

Usage (requires the EmQuantAPI SDK and your Choice credentials, see README):

    python -m emquant.inspect 300274.SZ
    python -m emquant.inspect 600519.SH --start 2025-01-01 --short 10 --long 30
"""

import argparse

import pandas as pd

from .em_client import session
from .market_data import daily_history, moving_average_signal, snapshot


def inspect_report(bars, short=20, long=60):
    """Compute the signal and summary stats for one code's daily bars."""
    sig = moving_average_signal(bars, short=short, long=long)
    last = sig.iloc[-1]
    year = sig[sig["date"] >= sig["date"].max() - pd.Timedelta(days=365)]

    def trailing_return(days):
        window = sig[sig["date"] >= sig["date"].max() - pd.Timedelta(days=days)]
        return (last["CLOSE"] / window.iloc[0]["CLOSE"] - 1) * 100

    daily_ret = sig["CLOSE"].pct_change().dropna()
    flips = sig.dropna(subset=["ma_long"]).copy()
    flips = flips[flips["signal"].diff().fillna(0) != 0]

    return sig, {
        "as_of": last["date"].date(),
        "close": last["CLOSE"],
        "stance": "LONG" if last["signal"] else "FLAT",
        f"ma{short}": last["ma_short"],
        f"ma{long}": last["ma_long"],
        "ret_1m_pct": trailing_return(30),
        "ret_3m_pct": trailing_return(91),
        "ret_12m_pct": trailing_return(365),
        "hi_52w": year["CLOSE"].max(),
        "lo_52w": year["CLOSE"].min(),
        "ann_vol_pct": daily_ret.tail(250).std() * (250 ** 0.5) * 100,
        "crossovers": [
            (row["date"].date(), "LONG" if row["signal"] else "FLAT",
             row["CLOSE"])
            for _, row in flips.iterrows()
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("code", help="Security code, e.g. 300274.SZ")
    parser.add_argument("--start", default=None,
                        help="History start date (default: ~18 months back)")
    parser.add_argument("--short", type=int, default=20)
    parser.add_argument("--long", type=int, default=60)
    args = parser.parse_args()

    start = args.start or (pd.Timestamp.today()
                           - pd.Timedelta(days=548)).strftime("%Y-%m-%d")
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    with session():
        print(snapshot(args.code).to_string(index=False))
        bars = daily_history(args.code, start, end)

    sig, report = inspect_report(bars, short=args.short, long=args.long)

    print(f"\n{args.code} — {len(sig)} bars through {report['as_of']}")
    print(f"Close {report['close']:.2f} | 52w range "
          f"{report['lo_52w']:.2f}–{report['hi_52w']:.2f} | "
          f"ann. vol {report['ann_vol_pct']:.1f}%")
    print(f"Returns: 1m {report['ret_1m_pct']:+.1f}%  "
          f"3m {report['ret_3m_pct']:+.1f}%  12m {report['ret_12m_pct']:+.1f}%")
    print(f"MA{args.short} {report[f'ma{args.short}']:.2f} vs "
          f"MA{args.long} {report[f'ma{args.long}']:.2f} → "
          f"stance: {report['stance']}")
    if report["crossovers"]:
        print("Crossovers:")
        for date, stance, close in report["crossovers"]:
            print(f"  {date}  → {stance:5s} @ {close:.2f}")


if __name__ == "__main__":
    main()
