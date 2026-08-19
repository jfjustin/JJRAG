"""Network egress guard.

The whole point of this project is that document text never reaches a third
party. A configuration flag saying so is not evidence; this module makes it
mechanical. When ``security.enforce_local_only`` is set, every outbound TCP
connection attempt is checked against an allowlist (loopback plus the
configured local model host) and anything else raises :class:`EgressBlocked`
before a single byte leaves the process.

That means a dependency which quietly tries to phone home — a telemetry ping, a
model download, an accidentally re-added SaaS client — fails loudly instead of
silently exfiltrating text.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import threading
from urllib.parse import urlparse

logger = logging.getLogger("jjrag.security.egress")

_LOCK = threading.Lock()
_INSTALLED = False
_ALLOWED_HOSTS: set[str] = set()
_ORIGINAL_GETADDRINFO = socket.getaddrinfo
_ORIGINAL_CREATE_CONNECTION = socket.create_connection
_BLOCKED_ATTEMPTS: list[tuple[str, int | None]] = []


class EgressBlocked(RuntimeError):
    """Raised when code tries to open a connection outside the allowlist."""


def _host_from(value: str) -> str:
    """Accept a bare host or a URL and return the hostname."""
    value = (value or "").strip()
    if not value:
        return ""
    if "://" in value:
        return (urlparse(value).hostname or "").lower()
    return value.split(":")[0].lower()


def _is_loopback(host: str) -> bool:
    host = host.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain", ""}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_allowed(host: str) -> bool:
    host = _host_from(host)
    if _is_loopback(host):
        return True
    return host in _ALLOWED_HOSTS


def allowed_hosts() -> set[str]:
    return set(_ALLOWED_HOSTS)


def blocked_attempts() -> list[tuple[str, int | None]]:
    """Hosts that were refused — surfaced in the compliance endpoint."""
    return list(_BLOCKED_ATTEMPTS)


def _record_block(host: str, port: int | None) -> None:
    entry = (host, port)
    if entry not in _BLOCKED_ATTEMPTS:
        _BLOCKED_ATTEMPTS.append(entry)
    logger.error("egress blocked host=%s port=%s", host, port)


def install(allow_hosts: list[str] | None = None) -> None:
    """Patch the socket layer so only allowlisted hosts are reachable.

    Idempotent: calling it again just widens the allowlist. Docker deployments
    should *also* run the app on an internal-only network — this guard is
    defence in depth inside the process, not a replacement for a firewall.
    """
    global _INSTALLED

    with _LOCK:
        for entry in allow_hosts or []:
            host = _host_from(entry)
            if host and not _is_loopback(host):
                _ALLOWED_HOSTS.add(host)

        if _INSTALLED:
            return

        def guarded_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[no-untyped-def]
            name = _host_from(host if isinstance(host, str) else str(host or ""))
            if not is_allowed(name):
                _record_block(name, port if isinstance(port, int) else None)
                raise EgressBlocked(
                    f"Outbound connection to {name!r} blocked by JJRAG's local-only "
                    "policy. Document text must never leave this host. If this host "
                    "is genuinely required, add it to security.extra_allowed_hosts."
                )
            return _ORIGINAL_GETADDRINFO(host, port, *args, **kwargs)

        def guarded_create_connection(address, *args, **kwargs):  # type: ignore[no-untyped-def]
            host = _host_from(str(address[0])) if address else ""
            port = address[1] if address and len(address) > 1 else None
            if not is_allowed(host):
                _record_block(host, port if isinstance(port, int) else None)
                raise EgressBlocked(
                    f"Outbound connection to {host!r} blocked by JJRAG's local-only "
                    "policy."
                )
            return _ORIGINAL_CREATE_CONNECTION(address, *args, **kwargs)

        socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
        socket.create_connection = guarded_create_connection  # type: ignore[assignment]
        _INSTALLED = True
        logger.info(
            "local-only egress guard installed (allowlist: loopback + %s)",
            sorted(_ALLOWED_HOSTS) or "nothing else",
        )


def uninstall() -> None:
    """Restore the stock socket functions (tests, and nothing else)."""
    global _INSTALLED
    with _LOCK:
        socket.getaddrinfo = _ORIGINAL_GETADDRINFO  # type: ignore[assignment]
        socket.create_connection = _ORIGINAL_CREATE_CONNECTION  # type: ignore[assignment]
        _ALLOWED_HOSTS.clear()
        _BLOCKED_ATTEMPTS.clear()
        _INSTALLED = False


def is_installed() -> bool:
    return _INSTALLED


def apply_policy(settings) -> None:  # type: ignore[no-untyped-def]
    """Install the guard according to a :class:`~jjrag.config.Settings`."""
    if not settings.security.enforce_local_only:
        logger.warning(
            "security.enforce_local_only is OFF — outbound connections are "
            "unrestricted. This is not a compliant configuration."
        )
        return
    hosts = [settings.llm.host, *settings.security.extra_allowed_hosts]
    install(hosts)
