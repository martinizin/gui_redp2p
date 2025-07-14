# Multi-stage build for better optimization
FROM python:3.10-slim as builder

# Install system dependencies for gnpy and matplotlib
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with specific versions for stability
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files with proper ownership
COPY --chown=app:app app.py .
COPY --chown=app:app scenario*.py ./
COPY --chown=app:app utils.py .
COPY --chown=app:app templates/ templates/
COPY --chown=app:app static/ static/
COPY --chown=app:app data/ data/

# Copy .env file if it exists (for API keys and configuration)
COPY --chown=app:app .env* ./

# Create directory for uploaded files
RUN mkdir -p /app/uploads && chown app:app /app/uploads

# Switch to non-root user
USER app

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV TOPOLOGY_DIR=/app/data
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Run the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
