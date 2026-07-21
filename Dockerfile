# 1. Base Image
FROM python:3.12-slim

# 2. Working Directory
WORKDIR /app

# Install system dependencies required for Healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy requirements and install dependencies first (Optimized Layer Caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy Application & Model Artifacts
COPY app/ ./app/
COPY models/ ./models/

# 5. Expose Streamlit Port
EXPOSE 8501

# 7. Healthcheck for Streamlit server status
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 6. Run Streamlit Application
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
