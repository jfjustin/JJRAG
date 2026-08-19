"""ASGI entry point: ``uvicorn jjrag.api.main:app``."""

from __future__ import annotations

from ..config import get_settings
from .app import create_app

app = create_app(get_settings())
