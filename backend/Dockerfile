# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Define storage directories
ENV HF_HOME=/app/data/.cache/huggingface 
ENV CHROMA_PERSIST_DIRECTORY=/app/data/chroma
ENV DATA_DIRECTORY=/app/data
ENV REPORTS_DIRECTORY=/app/reports

WORKDIR /app

# Install system dependencies required for compilation (e.g., hnswlib used by chromadb)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .

# OPTIMIZATION: Install CPU-only version of PyTorch first to save ~2GB of space.
# Cloud platforms like Render Free don't have GPUs, so CUDA binaries are wasted space.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the application requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend directory into the container
COPY . .

# Ensure data and report directories exist with correct permissions
RUN mkdir -p /app/data/chroma /app/reports && chmod -R 777 /app/data /app/reports

# Render will provide the $PORT environment variable dynamically.
# We expose 8000 for local testing.
EXPOSE 8000

# Start the FastAPI app on 0.0.0.0 and bind to $PORT (or 8000 fallback)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
