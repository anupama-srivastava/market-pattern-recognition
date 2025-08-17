FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_phase3.txt .
RUN pip install --no-cache-dir -r requirements_phase3.txt

# Copy source code
COPY src/ /app/src/
COPY services/data-ingestion/app.py /app/

# Expose port
EXPOSE 8000

# Run the service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
