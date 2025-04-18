FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (for better caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies needed for vector stores
RUN pip install --no-cache-dir sentence-transformers faiss-cpu openpyxl

# Create necessary directories
RUN mkdir -p /app/Data_real /app/Jsonfile

# Copy data files (use underscore instead of space)
COPY ["Data real/", "/app/Data_real/"]
COPY ["Jsonfile/", "/app/Jsonfile/"]

# First, let's create the build_vector_stores.py file if it doesn't exist
COPY chat2.py /app/

# Create a simple run script
RUN echo '#!/bin/bash\ncd /app\nstreamlit run chat2.py --server.port=8502' > /app/run.sh
RUN chmod +x /app/run.sh

# Copy the rest of the application
COPY . /app/

EXPOSE 8502

HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health || exit 1

ENTRYPOINT ["/app/run.sh"]