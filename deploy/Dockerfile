# JJRAG application image.
#
# Two-stage build so model weights are downloaded once, at build time, and the
# running container never needs outbound network access. That is what lets the
# runtime enforce the local-only egress policy without breaking itself.

# ── Stage 1: build the virtualenv and pre-fetch the embedding model ────────
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
# CPU-only torch keeps the image ~2 GB smaller than the CUDA default.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY pyproject.toml README.md ./
COPY jjrag ./jjrag
RUN pip install --no-deps .

# Bake the embedding weights into the image (~90 MB for all-MiniLM-L6-v2).
ENV HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    HF_HUB_DISABLE_TELEMETRY=1
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"

# ── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    JJRAG_CONFIG_FILE=/app/config/jjrag.yaml

# Run as a non-root user; uploads are untrusted input by definition.
RUN groupadd --gid 10001 jjrag \
 && useradd --uid 10001 --gid jjrag --create-home --shell /usr/sbin/nologin jjrag

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models /opt/models

WORKDIR /app
COPY --chown=jjrag:jjrag jjrag ./jjrag
COPY --chown=jjrag:jjrag web ./web
COPY --chown=jjrag:jjrag config ./config
COPY --chown=jjrag:jjrag pyproject.toml README.md ./

RUN mkdir -p /app/data && chown -R jjrag:jjrag /app/data /opt/models
VOLUME ["/app/data"]

USER jjrag
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "\
import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "jjrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
