FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and corpus
COPY app/ ./app/
COPY corpus/ ./corpus/

# Set MOCK_LLM=1 for offline-first deployments
ENV MOCK_LLM=1
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
