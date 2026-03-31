# ─────────────────────────────────────────────────────────────────────────────
# Caravan Layout Optimizer — OpenEnv Environment
# Optimised for HuggingFace Spaces (port 7860) and local Docker.
#
# Build:  docker build -t caravan-optimizer .
# Run:    docker run -p 7860:7860 \
#           -e API_BASE_URL="https://api.openai.com/v1" \
#           -e MODEL_NAME="gpt-4o-mini" \
#           -e HF_TOKEN="your-key" \
#           caravan-optimizer
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="hackathon-submission"
LABEL description="Caravan Layout Optimizer — OpenEnv-compatible RL environment"
LABEL version="1.0.0"

# ── System dependencies ───────────────────────────────────────────────────────
# curl: used by Docker HEALTHCHECK
# No build tools needed — all deps are pure Python
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker layer cache is reused on code-only changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Runtime environment variables ─────────────────────────────────────────────
# These are overridden at runtime via -e flags or HF Secrets.
# Defaults are placeholders only — do NOT commit real keys here.
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""
ENV ENV_BASE_URL="http://localhost:7860"

# HuggingFace Spaces always uses port 7860
ENV PORT=7860
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
# Automated hackathon validator pings GET / — must return 200
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=15s \
    --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# ── Start server ──────────────────────────────────────────────────────────────
# Single worker (hackathon infra: 2 vCPU / 8 GB RAM constraint)
# --no-access-log reduces noise in HF Spaces logs
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--no-access-log"]
