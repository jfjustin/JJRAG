"""Login/session management for the East Money Choice EMQuantAPI.

Credentials are read from (in order of precedence):

1. Environment variables ``EMQUANT_USERNAME`` / ``EMQUANT_PASSWORD``.
2. A ``credentials.ini`` file next to this module (see
   ``credentials.example.ini``). This file is gitignored — never commit it.

The official ``EmQuantAPI`` package is not on PyPI; it must be downloaded
from the Choice quant platform (https://quantapi.eastmoney.com/) and
installed with its bundled installer. See ``emquant/README.md``.
"""

import configparser
import os
from contextlib import contextmanager
from pathlib import Path

try:
    from EmQuantAPI import c
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The EmQuantAPI package is not installed. Download it from "
        "https://quantapi.eastmoney.com/ and run its installer, "
        "then retry. See emquant/README.md for step-by-step setup."
    ) from exc

_CREDENTIALS_FILE = Path(__file__).with_name("credentials.ini")


class EmQuantError(RuntimeError):
    """Raised when the Choice API reports a non-zero error code."""


def load_credentials():
    """Return (username, password), preferring env vars over credentials.ini."""
    username = os.environ.get("EMQUANT_USERNAME")
    password = os.environ.get("EMQUANT_PASSWORD")
    if username and password:
        return username, password

    if _CREDENTIALS_FILE.exists():
        parser = configparser.ConfigParser()
        parser.read(_CREDENTIALS_FILE, encoding="utf-8")
        section = parser["emquant"]
        return section["username"], section["password"]

    raise EmQuantError(
        "No EMQuantAPI credentials found. Set EMQUANT_USERNAME and "
        "EMQUANT_PASSWORD, or copy emquant/credentials.example.ini to "
        "emquant/credentials.ini and fill in your account."
    )


def _main_callback(quantdata):
    # Fired on disconnects and other out-of-band events.
    print(f"[EmQuant] callback: ErrorCode={quantdata.ErrorCode}, "
          f"ErrorMsg={quantdata.ErrorMsg}")
    return 0


def login(force=True):
    """Start an EMQuantAPI session with the configured account."""
    username, password = load_credentials()
    options = f"UserName={username},Password={password}"
    if force:
        options = "ForceLogin=1," + options
    result = c.start(options, "", _main_callback)
    if result.ErrorCode != 0:
        raise EmQuantError(
            f"EMQuantAPI login failed ({result.ErrorCode}): {result.ErrorMsg}"
        )
    return result


def logout():
    """Terminate the EMQuantAPI session."""
    c.stop()


@contextmanager
def session(force=True):
    """Context manager wrapping login/logout.

    Usage::

        from emquant.em_client import session
        with session():
            data = c.csd("000300.SH", "CLOSE", "2026-01-01", "2026-06-30", "")
    """
    login(force=force)
    try:
        yield c
    finally:
        logout()
