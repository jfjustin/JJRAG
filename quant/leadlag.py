"""Lead-lag (先导-滞后) estimation between stocks.

Follows the two-stage delay-aware estimator in DGNSDE ("Delay-Aware Graph
Neural Stochastic Differential Equations for Financial Time Series
Modeling and Forecasting", WWW'26), minus the neural parts — with one
deliberate departure in stage 1, explained below.

  Stage 1 — delay measurement. Recency-weighted cross-correlation of the
  two stocks' log returns over integer lags within a range scaled to the
  sample size, with the peak refined to a fractional lag by parabolic
  interpolation. The paper instead DTW-aligns standardized price *levels*
  and averages the warping path's offsets. Benchmarked against synthetic
  peers with known lags and realistic noise, that level-DTW recovered only
  10-21% of lags to within one bar and got the sign right about half the
  time: a price level is dominated by each stock's own cumulative random
  walk, so the warping chases idiosyncratic drift rather than the shared
  factor. Cross-correlating returns (which are stationary) recovered 79-92%
  to within one bar on the same data. Recency weighting is kept from the
  paper — observations nearer today count for more.

  Stage 2 — trend similarity after delay alignment, as in the paper. Shift
  the peer by the fractional delay through a monotone cubic Hermite (PCHIP)
  spline, log-difference both into slope series, z-score them, and run a
  band-constrained DTW; the normalized distance maps to [0, 1].

Significance gating. Picking the best of ~2L+1 lags inflates the winning
correlation even for pure noise, so a fixed cutoff lets noise peers in. A
pair counts as `linked` only when its peak correlation clears a
family-wise threshold at alpha=0.10 over the lags searched, using the
effective sample size left after recency weighting. On the benchmark that
admitted 3-5% of unrelated peers while detecting 100% of real links on
~6 months of daily bars, 54% on two months of hourly bars, and only 27% on
two months of daily bars — 44 daily bars simply can't establish lead-lag
reliably, which is the case for using intraday bars on short windows.

Sign convention: **lead_days > 0 means the PEER LEADS the TARGET** (the
peer moves first, the target follows). Negative means the target leads.

Units: every offset is in *bars of the input series* — days on daily bars,
trading hours on 60-minute bars. Callers on intraday bars convert for
display (see report._convert_lead_units).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

DEFAULT_WINDOW = 120
# Exponential recency decay across the window: weight runs from e^-DECAY on
# the oldest observation to 1 on the newest. 1.0 balanced recency against
# effective sample size best on the benchmark; at 3.0 the shrunken sample
# made genuine links fail significance.
DEFAULT_DECAY = 1.0
DEFAULT_BAND = 5
DEFAULT_ALPHA = 0.10
# Real lead-lag transmission runs hours to a few days; past this a large
# estimate is noise, not information flow.
MAX_LEAD_DAYS = 10.0
# Small per-bar penalty so a near-tie between a short and a long lag goes
# to the short one — long lags are where spurious peaks live.
LAG_PENALTY = 0.01

NETWORK_COLUMNS = ["lead_days", "lag_corr", "threshold", "linked",
                   "aligned_corr", "trend_similarity", "trend_distance",
                   "overlap"]


def _zscore(a):
    a = np.asarray(a, dtype=float)
    sd = np.std(a)
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(a)
    return (a - np.mean(a)) / sd


def _recency_weights(n, decay):
    return np.exp(-decay * (1.0 - np.arange(n) / max(n - 1, 1)))


def _weighted_corr(a, b, w):
    w = w / w.sum()
    ma, mb = np.sum(w * a), np.sum(w * b)
    va, vb = np.sum(w * (a - ma) ** 2), np.sum(w * (b - mb) ** 2)
    if va <= 0 or vb <= 0:
        return float("nan")
    return float(np.sum(w * (a - ma) * (b - mb)) / np.sqrt(va * vb))


def _lagged(target_ret, peer_ret, k):
    """Pair target[t] with peer[t-k]: k > 0 tests 'peer leads by k'."""
    if k > 0:
        return target_ret[k:], peer_ret[:-k]
    if k < 0:
        return target_ret[:k], peer_ret[-k:]
    return target_ret, peer_ret


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
            centre = int(round(i * m / n))
            j_lo = max(1, centre - band)
            j_hi = min(m, centre + band)
        xi = x[i - 1]
        for j in range(j_lo, j_hi + 1):
            d = abs(xi - y[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    if not np.isfinite(cost[n, m]):
        return float("inf"), []

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
    neighbour's state at a fractional-bar offset. Out-of-range positions
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


def lag_search_range(n_returns, max_lead=MAX_LEAD_DAYS):
    """Lags searched scale with the sample: scanning +/-10 lags on 40
    returns mostly finds noise peaks."""
    return int(min(max_lead, max(2, n_returns // 8)))


def estimate_lead_lag(target_close, peer_close, window=DEFAULT_WINDOW,
                      decay=DEFAULT_DECAY, band=DEFAULT_BAND,
                      max_lead=MAX_LEAD_DAYS, alpha=DEFAULT_ALPHA):
    """Lead-lag estimate for one target/peer pair.

    `target_close` and `peer_close` must already be aligned to a common set
    of bars. Returns a dict, or None if there isn't enough overlap:

    - lead_days: fractional lag in bars, >0 = peer leads target
    - lag_corr: recency-weighted return correlation at the peak lag
    - threshold, linked: significance cutoff for lag_corr and whether the
      pair clears it (see module docstring)
    - aligned_corr: plain correlation of returns after the fractional shift
    - trend_similarity / trend_distance: the paper's stage-2 DTW measure
    """
    t = np.asarray(target_close, dtype=float)
    p = np.asarray(peer_close, dtype=float)
    n = min(len(t), len(p), window)
    if n < 30:
        return None
    t, p = t[-n:], p[-n:]
    if not (np.isfinite(t).all() and np.isfinite(p).all()) or (t <= 0).any() or (p <= 0).any():
        return None

    # --- Stage 1: weighted cross-correlation over a sample-scaled range ---
    t_ret, p_ret = np.diff(np.log(t)), np.diff(np.log(p))
    span = lag_search_range(len(t_ret), max_lead)
    lags = np.arange(-span, span + 1)
    cc = np.array([_weighted_corr(*_lagged(t_ret, p_ret, k),
                                  _recency_weights(len(t_ret) - abs(k), decay))
                   for k in lags])
    if not np.isfinite(cc).any():
        return None
    i = int(np.nanargmax(np.where(np.isfinite(cc), cc - LAG_PENALTY * np.abs(lags), -np.inf)))
    frac = 0.0
    if 0 < i < len(cc) - 1 and np.isfinite(cc[i - 1:i + 2]).all():
        y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
        curvature = y0 - 2 * y1 + y2
        if curvature < 0:
            frac = float(np.clip(0.5 * (y0 - y2) / curvature, -0.5, 0.5))
    lead_days = float(lags[i] + frac)
    lag_corr = float(cc[i])

    w = _recency_weights(len(t_ret) - abs(int(lags[i])), decay)
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    threshold = float(norm.ppf(1 - alpha / (2 * len(lags))) / np.sqrt(max(n_eff - 3, 1)))

    # --- Stage 2: similarity of the delay-aligned slope series -----------
    peer_shifted = hermite_sample(p, np.arange(len(p)) - lead_days)
    valid = np.isfinite(peer_shifted) & (peer_shifted > 0)
    trend_distance = aligned_corr = float("nan")
    if valid.sum() >= 20:
        t_slope = np.diff(np.log(t[valid]))
        p_slope = np.diff(np.log(peer_shifted[valid]))
        d2, path2 = dtw(_zscore(t_slope), _zscore(p_slope), band=band)
        if path2 and np.isfinite(d2):
            trend_distance = float(1.0 - np.exp(-d2 / len(path2)))
        if np.std(t_slope) > 0 and np.std(p_slope) > 0:
            aligned_corr = float(np.corrcoef(t_slope, p_slope)[0, 1])

    return {
        "lead_days": lead_days,
        "lag_corr": lag_corr,
        "threshold": threshold,
        "linked": bool(lag_corr >= threshold),
        "aligned_corr": aligned_corr,
        "trend_distance": trend_distance,
        "trend_similarity": (1.0 - trend_distance
                             if np.isfinite(trend_distance) else float("nan")),
    }


def build_leadlag_network(target_close, peer_closes, window=DEFAULT_WINDOW,
                          decay=DEFAULT_DECAY, band=DEFAULT_BAND,
                          max_lead=MAX_LEAD_DAYS, alpha=DEFAULT_ALPHA):
    """Run estimate_lead_lag over a dict of {peer_label: close Series}.

    Every series is inner-joined to the target's timestamps first, so a
    trading halt on either side shortens the overlap instead of skewing the
    alignment. Returns a DataFrame indexed by peer label (NETWORK_COLUMNS).
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
                                window=window, decay=decay, band=band,
                                max_lead=max_lead, alpha=alpha)
        if est is None:
            continue
        est["overlap"] = min(len(joined), window)
        rows[label] = est
    if not rows:
        return pd.DataFrame(columns=NETWORK_COLUMNS)
    return pd.DataFrame.from_dict(rows, orient="index")[NETWORK_COLUMNS]


def leading_peer_signal(network, peer_closes, min_lead=0.5):
    """The tradeable distillation of DGNSDE's time-aligned aggregation.

    For every *linked* peer that leads the target by at least `min_lead`
    bars, read the move that peer has already made over exactly its
    (fractional) lead window — the move the target has not yet made — and
    average across peers weighted by lag correlation.

    Positive = linked leaders have risen and the target hasn't followed yet
    (bullish pressure); negative = the reverse.
    """
    empty = {"leading_peer_signal_pct": float("nan"), "n_leading_peers": 0}
    if network is None or network.empty:
        return {}
    leaders = network[network["linked"].astype(bool)
                      & (network["lead_days"] >= min_lead)]
    if leaders.empty:
        return empty

    contributions, weights, leads = [], [], []
    for label, row in leaders.iterrows():
        closes = pd.Series(peer_closes.get(label)).dropna()
        if len(closes) < 5:
            continue
        lead = float(row["lead_days"])
        past = hermite_sample(closes.values, [len(closes) - 1 - lead])[0]
        if not np.isfinite(past) or past <= 0:
            continue
        w = float(row["lag_corr"])
        contributions.append((closes.values[-1] / past - 1.0) * 100.0 * w)
        weights.append(w)
        leads.append(lead)

    if not contributions:
        return empty
    return {
        "leading_peer_signal_pct": float(np.sum(contributions) / np.sum(weights)),
        "n_leading_peers": int(len(contributions)),
        "leading_peer_mean_lead_days": float(np.mean(leads)),
    }


def network_summary(network):
    """Where the stock sits in its sector's information flow, measured on
    linked peers only — an unlinked peer's 'lead' is noise."""
    if network is None or network.empty:
        return {}
    linked = network[network["linked"].astype(bool)]
    out = {
        "n_peers_tested": int(len(network)),
        "n_linked_peers": int(len(linked)),
        "peer_mean_lag_corr": float(network["lag_corr"].mean()),
        "peer_mean_trend_similarity": float(network["trend_similarity"].mean()),
    }
    if linked.empty:
        return out
    lead = linked["lead_days"]
    out.update({
        # >0 => linked peers lead this stock on average, i.e. it follows.
        "peer_mean_lead_days": float(lead.mean()),
        "target_leadership_days": float(-lead.mean()),
        "n_lagging_peers": int((lead <= -0.5).sum()),
        "leadlag_abs_days": float(lead.abs().mean()),
    })
    return out
