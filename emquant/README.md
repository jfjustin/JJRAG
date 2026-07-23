# EMQuantAPI setup — quantitative share-market data

This directory wires the **East Money Choice EMQuantAPI** into the project for
quantitative equity work: index constituents, daily bars, cross-sectional
snapshots, and a sample moving-average signal.

## 1. Install the EmQuantAPI package

EmQuantAPI is **not on PyPI**. Download the Python SDK from the Choice quant
platform ([quantapi.eastmoney.com](https://quantapi.eastmoney.com/)), unzip it,
and run its bundled installer:

```bash
cd EmQuantAPI_Python/python3
python installEmQuantAPI.py
```

Also install the local requirements:

```bash
pip install pandas
```

## 2. Configure your account (never commit it)

Use the username and password issued with your Choice quant account. Two
options — environment variables win if both are set:

**Environment variables (recommended):**

```bash
export EMQUANT_USERNAME="your_username"
export EMQUANT_PASSWORD="your_password"
```

**Or a local config file:**

```bash
cp emquant/credentials.example.ini emquant/credentials.ini
# then edit emquant/credentials.ini with your account
```

`credentials.ini` is listed in `.gitignore`, so it stays on your machine.
If your credentials were ever pasted into a chat, file, or commit, treat them
as exposed and reset the password in the Choice terminal.

## 3. Run the demo

```bash
python -m emquant.demo_quant
```

It logs in, prints a snapshot of a few large caps, downloads six months of
daily bars, and reports the latest 20/60-day moving-average crossover stance.

## Module overview

| File | Purpose |
| --- | --- |
| `em_client.py` | Credential loading + `session()` context manager (login/logout) |
| `market_data.py` | `index_constituents`, `daily_history` (csd), `snapshot` (css), `moving_average_signal` |
| `demo_quant.py` | Runnable end-to-end example |
| `credentials.example.ini` | Template for the gitignored `credentials.ini` |

## Notes

- `session()` uses `ForceLogin=1` so a stale session elsewhere doesn't block
  login; pass `force=False` to disable.
- Quota: csd/css calls are metered by Choice; batch codes into one call where
  possible (both helpers accept lists).
- All data functions raise `EmQuantError` with the Choice error code and
  message when a call fails.
