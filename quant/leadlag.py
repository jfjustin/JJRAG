"""Lead-lag (先导-滞后) estimation between stocks, via two-stage DTW.

Implements the delay-aware correlation estimator from DGNSDE ("Delay-Aware
Graph Neural Stochastic Differential Equations for Financial Time Series
Modeling and Forecasting", WWW'26), minus the neural parts — the pieces
that carry over to a factor model are the two-stage DTW and the
fractional-day alignment:

  Stage 1 — elastic alignment & delay measurement. DTW the two stocks'
  standardized close-price series over a trailing window, walk the warping
  path, and read the index offset (i - j) at each aligned point. Weight
  those offsets by recency (alignment points nearer today count for more)
  and average → the pair's mean lead-lag in days, which is generally
  fractional.

  Stage 2 — trend similarity after delay alignment. Shift the peer by that
  (fractional) delay using a cubic Hermite spline, take the overlap,
  log-difference both into slope (instantaneous-return) series, z-score
  them so price level drops out, and run a second band-constrained DTW.
  The normalized distance maps to [0, 1] — smaller means the two names
  trace the same trend once the time offset is removed.

Sign convention, used everywhere below: **lead_days > 0 means the PEER
LEADS the TARGET** by that many trading days (the peer moves first, the
target follows). Negative means the target leads the peer.

Complexity note: DTW is O(n*m) with a sequential inner recurrence, so it
stays a Python loop. At the default 120-day window that's ~14k cell
updates per pair — a few milliseconds — but it's why the peer set is
capped rather than run against a whole index.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

DEFAULT_WINDOW = 120
DEFAULT_DECAY = 3.0
DEFAULT_BAND = 5
# Real lead-lag transmission runs hours to a few days; past this a large
# estimate is a warping artifact, not information flow.
MAX_LEAD_DAYS = 10.0


def _zscore(a):
    a = np.asarray(a, dtype=float)
    sd = np.std(a)
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(a)
    return (a - np.mean(a)) / sd


def dtw(x, y, band=None):
    """Dynamic time warping between two 1-D series.

    Returns (distance, path) where path is a list of (i, j) index pairs,
    oldest first. `band` applies a Sakoe-Chiba constraint of that radius
    (in index units); None means unconstrained.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float("inf"), []

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        if band is None:
            j_lo, j_hi = 1, m
        else:
            # Keep the band proportional when the two series differ in length.
            centre = int(round(i * m / n))
            j_lo = max(1, centre - band)
            j_hi = min(m, centre + band)
        xi = x[i - 1]
        for j in range(j_lo, j_hi + 1):
            d = abs(xi - y[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    if not np.isfinite(cost[n, m]):
        return float("inf"), []

    # Backtrack the warping path.
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return float(cost[n, m]), path


def hermite_sample(values, positions):
    """Evaluate a series at fractional index positions via a monotone cubic
    Hermite (PCHIP) spline — the interpolation DGNSDE uses to read a
    neighbour's state at a fractional-day offset. Out-of-range positions
    are clamped to the series' ends."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return np.full(len(np.atleast_1d(positions)), np.nan)
    spline = PchipInterpolator(np.arange(n), values, extrapolate=False)
    pos = np.clip(np.atleast_1d(positions), 0, n - 1)
    return spline(pos)


def continuous_path(dates, values, freq="D"):
    """Resample an irregularly sampled series (trading halts, holidays)
    onto a uniform grid via the same monotone cubic spline, giving the
    continuous control path the SDE formulation assumes. Returns
    (grid_dates, grid_values)."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return dates, values
    dates, values = dates[mask].reset_index(drop=True), values[mask]
    day_index = (dates - dates.iloc[0]).dt.total_seconds().values / 86400.0
    grid = pd.date_range(dates.iloc[0], dates.iloc[-1], freq=freq)
    grid_pos = (grid - dates.iloc[0]).total_seconds() / 86400.0
    spline = PchipInterpolator(day_index, values, extrapolate=False)
    return grid, spline(grid_pos)


def estimate_lead_lag(target_close, peer_close, window=DEFAULT_WINDOW,
                      decay=DEFAULT_DECAY, band=DEFAULT_BAND,
                      max_lead=MAX_LEAD_DAYS):
    """Two-stage DTW lead-lag estimate for one target/peer pair.

    `target_close` and `peer_close` must already be aligned to a common set
    of trading dates (same length, same calendar). Returns a dict with
    lead_days (>0 = peer leads target), trend_distance in [0, 1] (smaller =
    same trend once shifted), trend_similarity = 1 - trend_distance,
    aligned_corr, and the raw stage-1 DTW distance — or None if there isn't
    enough overlap.

    Two guards against reading structure into noise. The delay is clamped
    to +/-`max_lead` days, since real information transmission runs hours
    to a few days and anything larger is a warping artifact. And
    `aligned_corr` — plain Pearson correlation of the delay-aligned slope
    series — is reported alongside the DTW-derived similarity because it is
    the better-calibrated of the two: it sits near 0 for unrelated names,
    whereas the DTW distance map stays middling. Weight by aligned_corr
    when you need noise peers to fall out on their own.
    """
    t = np.asarray(target_close, dtype=float)
    p = np.asarray(peer_close, dtype=float)
    n = min(len(t), len(p))
    if n < 30:
        return None
    t, p = t[-min(n, window):], p[-min(n, window):]
    if not (np.isfinite(t).all() and np.isfinite(p).all()):
        return None

    # --- Stage 1: elastic alignment, recency-weighted delay -------------
    dist, path = dtw(_zscore(t), _zscore(p))
    if not path:
        return None
    idx = np.array(path, dtype=float)          # columns: target i, peer j
    offsets = idx[:, 0] - idx[:, 1]            # >0 => peer leads target
    recency = idx[:, 0] / max(len(t) - 1, 1)   # 0 = oldest, 1 = today
    weights = np.exp(-decay * (1.0 - recency))
    lead_days = float(np.sum(weights * offsets) / np.sum(weights))
    lead_days = float(np.clip(lead_days, -max_lead, max_lead))

    # --- Stage 2: similarity of the delay-aligned slope series ----------
    # Shift the peer forward by lead_days so its move lines up with the
    # target's; fractional shifts are read off the Hermite spline.
    positions = np.arange(len(p)) - lead_days
    peer_shifted = hermite_sample(p, positions)
    valid = np.isfinite(peer_shifted) & np.isfinite(t)
    trend_distance = float("nan")
    aligned_corr = float("nan")
    if valid.sum() >= 20:
        t_ov, p_ov = t[valid], peer_shifted[valid]
        if (t_ov > 0).all() and (p_ov > 0).all():
            t_slope = np.diff(np.log(t_ov))
            p_slope = np.diff(np.log(p_ov))
            if len(t_slope) >= 10:
                d2, path2 = dtw(_zscore(t_slope), _zscore(p_slope), band=band)
                if path2 and np.isfinite(d2):
                    # Per-aligned-point cost in sigma units -> [0, 1].
                    mean_cost = d2 / len(path2)
                    trend_distance = float(1.0 - np.exp(-mean_cost))
                if np.std(t_slope) > 0 and np.std(p_slope) > 0:
                    aligned_corr = float(np.corrcoef(t_slope, p_slope)[0, 1])

    return {
        "lead_days": lead_days,
        "dtw_distance": dist,
        "trend_distance": trend_distance,
        "trend_similarity": (1.0 - trend_distance
                            if np.isfinite(trend_distance) else float("nan")),
        "aligned_corr": aligned_corr,
    }


def build_leadlag_network(target_close, peer_closes, window=DEFAULT_WINDOW,
                          decay=DEFAULT_DECAY, band=DEFAULT_BAND):
    """Run estimate_lead_lag over a dict of {peer_label: close Series}.

    Every series is inner-joined to the target's dates first, so trading
    halts on either side simply shorten the overlap instead of skewing the
    alignment. Returns a DataFrame indexed by peer label with columns
    lead_days, trend_similarity, trend_distance, dtw_distance, overlap.
    """
    rows = {}
    target = pd.Series(target_close).dropna()
    for label, peer in peer_closes.items():
        peer = pd.Series(peer).dropna()
        joined = pd.concat([target.rename("t"), peer.rename("p")],
                          axis=1, join="inner").dropna()
        if len(joined) < 30:
            continue
        est = estimate_lead_lag(joined["t"].values, joined["p"].values,
                                window=window, decay=decay, band=band)
        if est is None:
            continue
        est["overlap"] = len(joined)
        rows[label] = est
    if not rows:
        return pd.DataFrame(columns=["lead_days", "dtw_distance",
                                     "trend_distance", "trend_similarity",
                                     "aligned_corr", "overlap"])
    return pd.DataFrame.from_dict(rows, orient="index")


def leading_peer_signal(network, peer_closes, min_lead=0.5, min_corr=0.2):
    """The tradeable distillation of DGNSDE's time-aligned aggregation.

    For every peer that *leads* the target by at least `min_lead` days,
    read the move that peer has already made over exactly that (fractional)
    lead window — the move the target has not yet made — and average those
    across peers, weighted by delay-aligned co-movement.

    Weighting uses `aligned_corr` rather than the DTW `trend_similarity`:
    an unrelated peer lands near 0 and drops out of the average on its own,
    while the DTW map would still hand it a middling weight. `min_corr`
    additionally excludes peers whose aligned returns barely co-move at all.

    Positive = leading peers have risen and the target hasn't followed yet
    (bullish pressure); negative = the reverse. Returns a dict with the
    signal in percent, how many peers fed it, and their mean lead.
    """
    empty = {"leading_peer_signal_pct": float("nan"), "n_leading_peers": 0}
    if network is None or network.empty:
        return {}
    leaders = network[(network["lead_days"] >= min_lead)
                     & (network["aligned_corr"].fillna(0) >= min_corr)]
    if leaders.empty:
        return empty

    contributions, weights = [], []
    for label, row in leaders.iterrows():
        closes = pd.Series(peer_closes.get(label)).dropna()
        if len(closes) < 5:
            continue
        lead = float(row["lead_days"])
        last_idx = len(closes) - 1
        past = hermite_sample(closes.values, [last_idx - lead])[0]
        if not np.isfinite(past) or past <= 0:
            continue
        move_pct = (closes.values[-1] / past - 1.0) * 100.0
        w = float(row["aligned_corr"])
        if not np.isfinite(w) or w <= 0:
            continue
        contributions.append(move_pct * w)
        weights.append(w)

    if not contributions:
        return empty
    return {
        "leading_peer_signal_pct": float(np.sum(contributions) / np.sum(weights)),
        "n_leading_peers": int(len(contributions)),
        "leading_peer_mean_lead_days": float(leaders["lead_days"].mean()),
    }


def network_summary(network):
    """Where the stock sits in its sector's information flow."""
    if network is None or network.empty:
        return {}
    lead = network["lead_days"].dropna()
    sim = network["trend_similarity"].dropna()
    corr = network["aligned_corr"].dropna() if "aligned_corr" in network else pd.Series(dtype=float)
    if lead.empty:
        return {}
    return {
        "peer_mean_aligned_corr": float(corr.mean()) if not corr.empty else float("nan"),
        # >0 => peers lead this stock on average, i.e. it is a follower.
        "peer_mean_lead_days": float(lead.mean()),
        # Flip the sign for the more intuitive reading.
        "target_leadership_days": float(-lead.mean()),
        "n_leading_peers": int((lead >= 0.5).sum()),
        "n_lagging_peers": int((lead <= -0.5).sum()),
        "leadlag_abs_days": float(lead.abs().mean()),
        "peer_mean_trend_similarity": float(sim.mean()) if not sim.empty else float("nan"),
    }
