# Dockerfile, Image, Container
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
RUN mkdir -p .streamlit
COPY .streamlit/config.toml .streamlit/
COPY datamover.py .
COPY main.py .

EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
