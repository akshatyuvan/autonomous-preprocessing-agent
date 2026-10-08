# Dockerfile -- the API and the full test suite in a reproducible Linux container.
#   Build:  docker build -t preprocessing-agent .
#   Tests:  docker run --rm preprocessing-agent python -m pytest -q
#   API:    docker run --rm -p 8000:8000 preprocessing-agent
FROM python:3.12-slim

# No .pyc files; logs print immediately; no pip download cache bloating the image.
# The LLM settings come from environment variables, because .env is never copied in.
# host.docker.internal = "the Mac running Docker Desktop", where Ollama runs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_PROVIDER=ollama \
    OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 \
    OLLAMA_MODEL=llama3.2:3b

WORKDIR /app

# Dependencies in their own layer, BEFORE the code: editing code doesn't trigger a
# full reinstall. The lock file pins the exact versions the tests passed with.
COPY requirements.lock.txt .
RUN pip install -r requirements.lock.txt

# Run as a non-root user: a compromised process can't modify the image's system files.
# chown /app itself: COPY --chown only covers copied files, and the pipeline must
# CREATE data/cleaned/ under /app to save snapshots.
RUN useradd --create-home app && chown app:app /app
COPY --chown=app:app . .
USER app

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]