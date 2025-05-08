FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and curl for healthcheck
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir sentence-transformers faiss-cpu openpyxl

# Create necessary directories
RUN mkdir -p /app/Data_real /app/Jsonfile

# Copy data files
COPY ["Data real/", "/app/Data_real/"]
COPY ["Jsonfile/", "/app/Jsonfile/"]

# Copy application code
COPY chat2.py .

# Create and set permissions for run script
RUN echo '#!/bin/bash\n\
    cd /app\n\
    streamlit run chat2.py --server.port=8502 --server.address=0.0.0.0' > /app/run.sh \
    && chmod +x /app/run.sh

# Copy remaining files
COPY . .

# Expose port
EXPOSE 8502

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8502/_stcore/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/run.sh"]