# Use official Python 3.10 slim image
FROM python:3.10-slim

# Prevent .pyc files and stdout buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Storage directories
ENV CHROMA_PERSIST_DIRECTORY=/app/data/chroma
ENV DATA_DIRECTORY=/app/data
ENV REPORTS_DIRECTORY=/app/reports

# This image installs requirements.txt, which excludes torch and
# sentence-transformers, so no local embedding model can be loaded. Default the
# embedding tier accordingly rather than letting it try, fail, and fall back
# silently. render.yaml sets this too; this keeps a bare `docker run` honest.
ENV EMBEDDING_BACKEND=hash

WORKDIR /app

# Install dependencies.
#
# Note there is no separate torch install step. An earlier version ran:
#   pip install torch ... --index-url https://download.pytorch.org/whl/cpu
# which broke the build: --index-url *replaces* PyPI, so when pip fell back to a
# source distribution it could not resolve that package's build backend
# (flit_core), and the layer failed. Since the free tier pins hash embeddings,
# torch was pure weight anyway — around 2 GB of image and a much slower cold
# start. If you do need semantic embeddings here, install
# requirements-local.txt and read the --extra-index-url note in that file.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Ensure writable data and report directories.
# Note: on hosts with an ephemeral filesystem these do not survive a restart, so
# run history and generated PDFs are lost. See the README.
RUN mkdir -p /app/data/chroma /app/reports

EXPOSE 8000

# Render supplies $PORT at runtime; 8000 is the local fallback.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
