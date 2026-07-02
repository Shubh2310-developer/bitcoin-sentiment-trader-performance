FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests and install
COPY pyproject.toml requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY pipelines/ pipelines/
COPY data/ data/

# Set default config path
ENV CONFIG_PATH=configs/base.yaml

ENTRYPOINT ["python", "pipelines/run_full_pipeline.py", "--config", "configs/base.yaml"]
