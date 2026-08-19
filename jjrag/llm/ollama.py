"""Local model client (Ollama).

This is the **only** generation path in JJRAG. There is deliberately no hosted
provider client anywhere in the codebase — not behind a flag, not as an
optional import — because "the cloud path was disabled" is a weaker compliance
statement than "the cloud path does not exist".

The client refuses to talk to a non-local host unless that host has been
explicitly allowlisted in ``security.extra_allowed_hosts``, which is the same
list the socket-level egress guard enforces. A self-hosted Ollama on another
machine inside your own network is therefore possible, but only as a deliberate,
recorded decision.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from urllib.parse import urlparse

import requests

from ..config import LLMSettings
from ..security import egress

logger = logging.getLogger("jjrag.llm.ollama")


class LocalModelError(RuntimeError):
    pass


class ModelNotInstalled(LocalModelError):
    pass


class OllamaClient:
    def __init__(self, settings: LLMSettings, allowed_hosts: Sequence[str] = ()) -> None:
        self.settings = settings
        self.host = settings.host.rstrip("/")
        hostname = (urlparse(self.host).hostname or "").lower()

        if not egress.is_allowed(hostname) and hostname not in {
            h.lower() for h in allowed_hosts
        }:
            raise LocalModelError(
                f"refusing to use model host {hostname!r}: it is neither local "
                "nor in security.extra_allowed_hosts. JJRAG only generates with "
                "models you control."
            )
        self._session = requests.Session()

    # -- health -------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            self._session.get(f"{self.host}/api/tags", timeout=3).raise_for_status()
            return True
        except Exception:  # noqa: BLE001 - unreachable is a normal state here
            return False

    def list_models(self) -> list[dict]:
        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("cannot list local models: %s", exc)
            return []
        return [
            {
                "name": m.get("name", ""),
                "size_bytes": m.get("size", 0),
                "family": (m.get("details") or {}).get("family", ""),
                "parameter_size": (m.get("details") or {}).get("parameter_size", ""),
                "quantization": (m.get("details") or {}).get("quantization_level", ""),
            }
            for m in models
        ]

    def model_names(self) -> list[str]:
        return [m["name"] for m in self.list_models()]

    def ensure_model(self, model: str | None = None) -> str:
        """Confirm a model is installed, tolerating the ``:latest`` suffix."""
        model = model or self.settings.model
        installed = self.model_names()
        if not installed:
            raise LocalModelError(
                f"no local model server reachable at {self.host}. Start Ollama "
                f"(`ollama serve`) and install a model (`ollama pull {model}`)."
            )
        if model in installed:
            return model
        base = model.split(":")[0]
        for candidate in installed:
            if candidate.split(":")[0] == base:
                return candidate
        raise ModelNotInstalled(
            f"model {model!r} is not installed. Available: "
            f"{', '.join(installed)}. Install it with: ollama pull {model}"
        )

    # -- generation ---------------------------------------------------------
    def _options(self, **overrides) -> dict:
        options = {
            "temperature": self.settings.temperature,
            "num_ctx": self.settings.num_ctx,
            "num_predict": self.settings.max_tokens,
        }
        options.update({k: v for k, v in overrides.items() if v is not None})
        return options

    def generate(
        self, prompt: str, *, system: str | None = None, model: str | None = None,
        **options,
    ) -> str:
        return "".join(
            self.stream(prompt, system=system, model=model, **options)
        )

    def stream(
        self, prompt: str, *, system: str | None = None, model: str | None = None,
        **options,
    ) -> Iterator[str]:
        """Yield generated text as it is produced by the local model."""
        payload = {
            "model": model or self.settings.model,
            "prompt": prompt,
            "system": system or self.settings.system_prompt,
            "stream": True,
            "options": self._options(**options),
        }
        try:
            with self._session.post(
                f"{self.host}/api/generate", json=payload, stream=True,
                timeout=self.settings.request_timeout_s,
            ) as response:
                if response.status_code == 404:
                    raise ModelNotInstalled(
                        f"model {payload['model']!r} is not installed on the "
                        f"local server. Run: ollama pull {payload['model']}"
                    )
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("error"):
                        raise LocalModelError(event["error"])
                    token = event.get("response")
                    if token:
                        yield token
                    if event.get("done"):
                        break
        except requests.exceptions.ConnectionError as exc:
            raise LocalModelError(
                f"cannot reach the local model at {self.host}. Is Ollama "
                f"running? ({exc})"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LocalModelError(
                f"the local model did not respond within "
                f"{self.settings.request_timeout_s}s"
            ) from exc
