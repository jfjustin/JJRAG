"""End-to-end demo: log in, pull share-market data, compute a simple signal.

Run from the repository root after installing EmQuantAPI and setting your
credentials (see emquant/README.md):

    python -m emquant.demo_quant
"""

from emquant.em_client import session
from emquant.market_data import daily_history, moving_average_signal, snapshot


def main():
    with session():
        # Cross-sectional snapshot of a few large caps.
        codes = ["600519.SH", "000858.SZ", "601318.SH"]
        print("Snapshot:")
        print(snapshot(codes).to_string(index=False))

        # Six months of daily bars for Kweichow Moutai.
        bars = daily_history("600519.SH", "2026-01-01", "2026-06-30")
        print(f"\nFetched {len(bars)} daily bars for 600519.SH")

        # Dual moving-average crossover signal.
        signal = moving_average_signal(bars, short=20, long=60)
        latest = signal.dropna().tail(1)
        if not latest.empty:
            row = latest.iloc[0]
            stance = "LONG" if row["signal"] else "FLAT"
            print(f"Latest signal ({row['date'].date()}): {stance} "
                  f"(MA20={row['ma_short']:.2f}, MA60={row['ma_long']:.2f})")


if __name__ == "__main__":
    main()
