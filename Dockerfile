# One image for api and ui (different commands). Multi-stage: build the venv with uv, ship it slim.
FROM python:3.11-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.5.29 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
COPY mcp_server ./mcp_server
RUN uv sync --frozen --no-dev

FROM python:3.11-slim
RUN useradd --create-home --uid 1000 copilot
WORKDIR /app
COPY --from=build /app /app
COPY ui ./ui
COPY data ./data
COPY scripts ./scripts
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER copilot
EXPOSE 8000 8501
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
