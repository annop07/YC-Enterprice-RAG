FROM python:3.12-slim

# uv resolves and installs from the lockfile; no pip, no virtualenv juggling.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

WORKDIR /app

# Dependencies first, so editing application code does not re-resolve them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app

# fastembed downloads its ONNX models on first use. Baking them in keeps the
# first request from taking thirty seconds and the container from needing the
# network at run time.
#
# FASTEMBED_CACHE_PATH, not HF_HOME: fastembed does not read HF_HOME at all. It
# calls `define_cache_dir()`, which reads this variable and otherwise falls back
# to $TMPDIR/fastembed_cache — so with HF_HOME set the models were being baked
# into /tmp, which happened to survive into the image and would stop surviving
# the moment /tmp became a tmpfs or the build was cleaned between layers.
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN uv run python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
import os; \
TextEmbedding(os.environ.get('EMBED_MODEL', 'BAAI/bge-small-en-v1.5')); \
TextCrossEncoder(os.environ.get('RERANK_MODEL', 'Xenova/ms-marco-MiniLM-L-6-v2'))"

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
