# Single image serving both the API and the console, so the demo has one URL.
#
# Two stages: Node builds the console, Python runs everything. The Node
# toolchain never reaches the runtime image, which keeps it small and removes a
# large amount of surface area the running service has no use for.

# ---- stage 1: build the console -------------------------------------------
FROM node:22-alpine AS console

WORKDIR /console

# Dependencies first, so this layer stays cached until the lockfile actually
# changes; editing a component then rebuilds in seconds.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# No VITE_API_BASE here: the deployed console talks to the origin it is served
# from, so there is no cross-origin hop in production.
RUN npm run build


# ---- stage 2: runtime ------------------------------------------------------
FROM python:3.12-slim AS runtime

# PYTHONUNBUFFERED keeps logs streaming to the platform in real time rather than
# sitting in a buffer until the process exits.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY --from=console /console/dist ./frontend/dist

# The decision journal lives here. On a platform with an ephemeral filesystem
# this resets on redeploy; mount a disk at /app/data to keep history.
RUN mkdir -p /app/data

# Run unprivileged: the service needs no rights beyond its own files.
RUN useradd --create-home --uid 10001 desk && chown -R desk:desk /app
USER desk

# Safe defaults. Credentials are never baked into the image; the platform
# injects them as environment variables at run time.
ENV DATABASE_URL=sqlite:///./data/dispersion_desk.db \
    ALPACA_PAPER_TRADE=true \
    PROPOSE_ONLY=true \
    LLM_PROVIDER=mock \
    PORT=8000

EXPOSE 8000

# The platform supplies $PORT; defaulting to 8000 keeps `docker run` simple.
# One worker on purpose: the agent loop and its SQLite journal are stateful, and
# a second worker would mean two agents trading the same account.
CMD ["sh", "-c", "uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
