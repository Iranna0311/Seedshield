# SeedShield Docker Configuration
# Multi-stage build for optimized production image

# ============================================
# Stage 1: Build Frontend
# ============================================
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci --silent

# Copy source code
COPY frontend/ .

# Build production bundle
RUN npm run build

# ============================================
# Stage 2: Production Backend
# ============================================
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SEEDSHIELD_DEMO_MODE=false
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-train the seed classifier so it's ready at startup
RUN python3 -c "from inference.pipeline import SeedShieldPipeline; SeedShieldPipeline()" \
    && echo "Seed classifier pre-trained successfully"

# Copy built frontend
COPY --from=frontend-builder /app/frontend/build /app/static

# Create directories for models
RUN mkdir -p /app/models/weights/detector \
    && mkdir -p /app/models/weights/classifier

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
