FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY support_assistant/ ./support_assistant/

# Set MOCK_LLM=1 for graded baseline (offline-first)
ENV MOCK_LLM=1
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 7860

# Run FastAPI server
CMD ["python", "-m", "uvicorn", "support_assistant.app.main:app", "--host", "0.0.0.0", "--port", "7860"]

