"""Lightweight disk cache for slow/rate-limited market-data calls.

AKShare endpoints are free but unauthenticated scrapers of public sites —
some (the full-market snapshot, per-peer history loops) take several
seconds. A dashboard re-running the same fetch on every widget interaction
would be unusable, so every call in ``data.py`` goes through this cache.

Not thread-safe beyond what pandas/pickle already guarantee; fine for a
single-user local Streamlit process.
"""

import hashlib
import pickle
import time
from pathlib import Path

_CACHE_DIR = Path(__file__).with_name(".cache")


def _key_path(key):
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    safe = "".join(c if c.isalnum() else "_" for c in key)[:60]
    return _CACHE_DIR / f"{safe}_{digest}.pkl"


def get_or_fetch(key, fetch_fn, ttl_seconds=3600):
    """Return the cached value for ``key`` if fresh, else call fetch_fn()."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _key_path(key)
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl_seconds:
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass  # fall through to refetch on any corruption/version issue

    value = fetch_fn()
    try:
        with open(path, "wb") as fh:
            pickle.dump(value, fh)
    except Exception:
        pass  # caching is an optimization, never fatal
    return value


def clear():
    """Remove all cached entries."""
    if _CACHE_DIR.exists():
        for f in _CACHE_DIR.glob("*.pkl"):
            f.unlink(missing_ok=True)
